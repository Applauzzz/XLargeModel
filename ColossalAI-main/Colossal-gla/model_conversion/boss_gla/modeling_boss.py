"""PyTorch BoSs model."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch_moskit


from transformers.activations import ACT2FN
from fla.modules.activations import swish
from einops import rearrange

from fla.modules import ShortConvolution

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from fla.models.utils import Cache

# I need to finish the cache
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import ModelOutput

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)

from .configuration_boss import BoSsConfig
from .cache import Cache as BossCache


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BoSsConfig"

@dataclass
class BoSsModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    balance_loss: Optional[torch.FloatTensor] =None

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    balance_loss: Optional[torch.FloatTensor] = None

class BoSsRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BoSsRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class BoSsRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class BoSsMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)    

class BoSsAttention(nn.Module):
    def __init__(self, config: BoSsConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_states = config.num_states
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        self.flash_varlen_qkv_fn = flash_attn_varlen_func
        self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = BoSsRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def boss_attention(self, q, k, v, sid, sid_current):
        '''
        q: [b q_len h d]
        k: [b k_len h d]
        v: [b k_len h d]
        sid: [b q_len]
        '''
        
        batch_size = k.shape[0]
        softmax_scale = k.shape[-1] ** (-0.5)
        
        if sid_current is not None:
            # generation
            ns_gen = self.num_states + 1
            idx, offs_k, _, state_sizes = torch_moskit.sid_to_indices(sid, ns_gen)
            idx = rearrange(idx, 'b l -> b l 1 1').contiguous().expand_as(v).to(torch.int64) # bs qlen -> bs qlen 1 1 -> [batch_size, q_len, num_heads, head_dim]
            offs_k = offs_k.to(torch.int32)
            k, v = [rearrange(torch.gather(src, 1, idx).contiguous(),
                            'b l h d -> (b l) h d') for src in [k, v]]
            max_seqlen_k = torch.max(state_sizes).item()
            
            q = rearrange(q, 'b l h d -> (b l) h d')
            mid_offs = []
            for i in range(1, batch_size):
                mid_offs += [i] * (ns_gen - sid_current[i-1] + sid_current[i]).item()
            if batch_size > 1:
                offs_q = torch.cat((torch.zeros(sid_current[0].item() + 1), torch.tensor(mid_offs), torch.full((ns_gen - sid_current[batch_size - 1].item(),), batch_size)))
            else:
                offs_q = torch.cat((torch.zeros(sid_current.item() + 1), torch.full((ns_gen - sid_current.item(),), batch_size)))
            offs_q = offs_q.to(torch.int32).to(offs_k.device)
        
            o = self.flash_varlen_qkv_fn(
                q, k, v,
                offs_q, offs_k,
                1, max_seqlen_k, 
                causal=True,
                window_size=(self.sliding_window, 0),
                softmax_scale=softmax_scale,
                dropout_p=self.attention_dropout if self.training else 0.0,)
            # [b, l, h, d]
            o = rearrange(o, '(b l) h d -> b l h d', b=batch_size)
        else:
            idx, offs, _, state_sizes = torch_moskit.sid_to_indices(sid, self.num_states if self.training else self.num_states + 1)
            idx = rearrange(idx, 'b l -> b l 1 1').contiguous().expand_as(v).to(torch.int64) # bs qlen -> bs qlen 1 1 -> [batch_size, q_len, num_heads, head_dim]
            offs = offs.to(torch.int32)
            # gather in
            q, k, v = [rearrange(torch.gather(src, 1, idx).contiguous(),
                            'b l h d -> (b l) h d') for src in [q, k, v]]
            max_seqlen = torch.max(state_sizes).item()
        
            o = self.flash_varlen_qkv_fn(
                q, k, v,
                offs, offs,
                max_seqlen, max_seqlen, 
                causal=True,
                window_size=(self.sliding_window, 0),
                softmax_scale=softmax_scale,
                dropout_p=self.attention_dropout if self.training else 0.0,)
            # [b, l, h, d]
            o = rearrange(o, '(b l) h d -> b l h d', b=batch_size)
            
            # scatter out
            o = o.scatter(1, idx, o).contiguous()
        
        return o
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        sid: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        # if isinstance(past_key_value, StaticCache):
        #     raise ValueError(
        #         "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
        #         "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        #     )
        # output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += cache_position[0]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # for generation
        sid_current = sid.squeeze() if q_len == 1 else None
        
        # handle attention mask when prefilling
        if attention_mask is not None:
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0 if past_key_value is not None else False
            if cache_has_contents:
                attention_mask = attention_mask[:, -1:]
            sid[attention_mask == 0] = self.num_states

        if past_key_value is not None:
            # cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states, sid = past_key_value.update([key_states, value_states, sid], self.layer_idx, "boss", q_len, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # input_dtype = query_states.dtype
        # if input_dtype == torch.float32:
        #     if torch.is_autocast_enabled():
        #         target_dtype = torch.get_autocast_gpu_dtype()
        #     # Handle the case where the model is quantized
        #     elif hasattr(self.config, "_pre_quantization_dtype"):
        #         target_dtype = self.config._pre_quantization_dtype
        #     else:
        #         target_dtype = self.q_proj.weight.dtype

        #     logger.warning_once(
        #         f"The input hidden states seems to be silently casted in float32, this might be related to"
        #         f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
        #         f" {target_dtype}."
        #     )

        #     query_states = query_states.to(target_dtype)
        #     key_states = key_states.to(target_dtype)
        #     value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        
        attn_output = self.boss_attention(query_states, key_states, value_states, sid, sid_current)

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)
        
        ######### [2024.10.9] Added by Yuqi ##########
        ########## Check for NaN with generation ##########
        # for obj in [('input', hidden_states), ('q', query_states), ('k', key_states), ('v', value_states), ('output', attn_output)]:
        #     if torch.isnan(obj[1]).any():
        #         print("===================")
        #         print("Find NaN in " + obj[0] + f" of boss part at layer {self.layer_idx}!")
        #         print(f"seen tokens: {past_key_value.get_seq_length(self.layer_idx)}.")
        #         exit()

        return attn_output, None, past_key_value
    
class MainAttention(nn.Module):
    def __init__(self, config: BoSsConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mode = "chunk"

        self.num_states = config.num_states
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.g_norm = BoSsRMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps)
        
        self.feature_map = ACT2FN['relu']

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.gate_low_rank_dim = 16
        self.gate_logit_normalizer = 16
        self.gk_proj = nn.Sequential(nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False),
                                     nn.Linear(self.gate_low_rank_dim, self.num_key_value_heads * self.head_dim, bias=True))

        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode
        
        recurrent_state = None
        if use_cache and past_key_value is not None:
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            recurrent_state = past_key_value[self.layer_idx][0][0] if cache_has_contents else None
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # dealing with left-padding
        if attention_mask is not None:
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0 if past_key_value is not None else False
            if cache_has_contents:
                attention_mask = attention_mask[:, -1:]
            value_states = value_states.mul_(attention_mask.unsqueeze(-1))

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        query_states = self.feature_map(query_states)
        key_states = self.feature_map(key_states)
        
        gk = self.gk_proj(hidden_states)
        gk = gk.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        gk = repeat_kv(gk, self.num_key_value_groups)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if mode == 'fused_recurrent':
            attn_output, recurrent_state = fused_recurrent_gla(query_states,
                                                               key_states,
                                                               value_states,
                                                               gk,
                                                               initial_state=recurrent_state,
                                                               output_final_state=use_cache)
        elif mode == 'chunk':
            attn_output, recurrent_state = chunk_gla(query_states,
                                                     key_states,
                                                     value_states,
                                                     gk,
                                                     initial_state=recurrent_state,
                                                     output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if past_key_value is not None:
            last_state = (recurrent_state,)
            past_key_value.update(last_state, self.layer_idx, "main", q_len)

        attn_output = attn_output.transpose(1, 2)
        attn_output = self.g_norm(attn_output)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)
        
        ######## [2024.10.9] Added by Yuqi ##########
        ######### Check for NaN with generation ##########
        # for obj in [('input', hidden_states), ('q', query_states), ('k', key_states), ('v', value_states),
        #             ('gk', gk), ('output', attn_output)]:
        #     if torch.isnan(obj[1]).any():
        #         print("===================")
        #         print("Find NaN in " + obj[0] + f" of main part at layer {self.layer_idx}!")
        #         print(f"seen tokens: {past_key_value.get_seq_length(self.layer_idx)}.")
        #         exit()

        return attn_output, None, past_key_value


# copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with Llama->Mistral, LLAMA->MISTRAL
# TODO(joao): add me back asap :)
class BoSsDecoderLayer(nn.Module):
    def __init__(self, config: BoSsConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attn_boss = BoSsAttention(config=config, layer_idx=layer_idx)
        self.attn_main = MainAttention(config=config, layer_idx=layer_idx)
        
        self.num_states = config.num_states
        self.gate = nn.Linear(self.hidden_size, self.num_states, bias=False)

        self.mlp = BoSsMLP(config)
        self.input_layernorm = BoSsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BoSsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Get the gate
        gates = self.gate(hidden_states)
        gates = F.softmax(gates, dim=-1)
        
        # compute the balance loss
        out_weight, indices1_s = gates.max(dim=-1)
        if self.training:
            mask1 = F.one_hot(indices1_s, num_classes=self.num_states)
            me = torch.mean(gates, dim=(0, 1))
            ce = torch.mean(mask1.float(), dim=(0, 1))
            l_aux = torch.sum(me * ce) * self.num_states
        else:
            l_aux = 0.0
        
        # Self Attention
        boss_part, _, past_key_value = self.attn_boss(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            sid=indices1_s,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        # boss_part = torch.zeros_like(residual, device=residual.device, dtype=residual.dtype)
        
        main_part, _, past_key_value = self.attn_main(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + main_part + boss_part * out_weight.to(boss_part.dtype).unsqueeze(-1)
        
        ########## [2024.9.25] Added by Yuqi ##########
        ########## Save out_weight and indices1_s ##########
        # unique_elements, counts = torch.unique(indices1_s, return_counts=True)
        # file_prefix = f"/gpfsdata/yuhong/gpt-neox/boss-save/{torch.distributed.get_rank()}/"
        # torch.save(unique_elements, f"{file_prefix}sid_layer{self.layer_idx}.pt")
        # torch.save(counts, f"{file_prefix}counts_layer{self.layer_idx}.pt")
        # torch.save(torch.max(out_weight.mean()), f"{file_prefix}outweight_layer{self.layer_idx}.pt")
        
        ######### [2024.10.9] Added by Yuqi ##########
        ########## Check for NaN with generation ##########
        # if torch.isnan(main_part).any():
        #     print("===================")
        #     print(f"Find NaN in main part at layer {self.layer_idx}!")
        #     exit()
        # if torch.isnan(boss_part).any():
        #     print("===================")
        #     print(f"Find NaN in boss part at layer {self.layer_idx}!")
        #     exit()
            
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, l_aux, )

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs
    

BoSs_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MistralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    BoSs_START_DOCSTRING,
)
class BoSsPreTrainedModel(PreTrainedModel):
    config_class = BoSsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BoSsDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


BoSs_INPUTS_DOCSTRING = r"""
    Nothing
"""


@add_start_docstrings(
    "BoSs",
    BoSs_START_DOCSTRING,
)
class BoSsModel(BoSsPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: BoSsConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        self.layers = nn.ModuleList(
            [BoSsDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = BoSsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(BoSs_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BoSsModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, BossCache) and not self.training:
            past_key_values = BossCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # attention_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        # )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # the balance loss
        balance_loss = 0.0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            l_aux = layer_outputs[1]
            balance_loss = balance_loss + l_aux

            if use_cache:
                next_decoder_cache = layer_outputs[3 if output_attentions else 2]

            
            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BoSsModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            balance_loss=balance_loss
        )


class BoSsForCausalLM(BoSsPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = BoSsModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @add_start_docstrings_to_model_forward(BoSs_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # # Upcast to float if we need to compute the loss to avoid potential precision issues
            # logits = logits.float()
            # # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # # Flatten the tokens
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # # Ensure tensors are on the same device
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(shift_logits, shift_labels)

            loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(logits.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            balance_loss=outputs.balance_loss,
        )
    
    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     cache_position=None,
    #     position_ids=None,
    #     use_cache=True,
    #     **kwargs,
    # ):
    #  # only last token for `inputs_ids` if the `past_key_values` is passed along.
    #     if past_key_values is not None:
    #         if not isinstance(past_key_values, Cache):
    #             past_key_values = Cache.from_legacy_cache(past_key_values, input_ids.shape[1] - 1)
    #         input_ids, attention_mask = input_ids[:, -1:], attention_mask[:, -1:]
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None, # (batch_size, seq_len) 
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask, #(batch_size, q_len, k_len, head_dim)
            }
        )
        return model_inputs