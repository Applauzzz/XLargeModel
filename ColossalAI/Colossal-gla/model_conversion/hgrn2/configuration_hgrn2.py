# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class HGRN2Config(PretrainedConfig):

    model_type = 'hgrn2'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        attn_mode="chunk",
        num_attention_heads=32,
        num_key_value_heads=8,
        expand_ratio=128,
        use_short_conv=False,
        conv_size=4,
        use_lower_bound=True,
        intermediate_size=14336,
        hidden_act="swish",
        max_position_embeddings=8192,
        elementwise_affine=True,
        norm_eps=1e-6,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        initializer_range=0.02,
        fuse_cross_entropy=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.expand_ratio = expand_ratio
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_lower_bound = use_lower_bound
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
