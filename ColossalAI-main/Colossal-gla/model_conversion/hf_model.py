from .gla import GLAConfig, GLAForCausalLM, GLAloraForCausalLM
from .hgrn2 import HGRN2Config, HGRN2ForCausalLM
from .boss_gla import BoSsForCausalLM, BoSsConfig
from .mhboss_gla import mhBoSsForCausalLM
from .loraboss_gla import LoRA2BoSsForCausalLM
from transformers import MistralForCausalLM, MistralConfig

def HuggingfaceModel(model_name, pretrained):
    '''
    Load Huggingface Model (Mistral) for (linear) conversion.

    Args:
        model_name (str): You can choose from [mistral, hgrn2, gla, boss-gla, loraboss-gla].
        pretrained (str): Path to pre-trained model checkpoint.
    
    Returns:
        model: A Huggingface type model.
    '''

    if model_name == "mistral":
        mistral = MistralForCausalLM.from_pretrained(pretrained)
        mistral.gradient_checkpointing_enable()
        return mistral
    elif model_name == "hgrn2":
        hgrn2 = HGRN2ForCausalLM(HGRN2Config())
        mistral = MistralForCausalLM.from_pretrained(pretrained)

        # 复制嵌入层参数
        hgrn2.model.embeddings.weight.data.copy_(mistral.model.embed_tokens.weight.data)
        # hgrn2.model.embed_tokens.weight.requires_grad = False  # 冻结参数

        # 复制输出层参数
        hgrn2.lm_head.weight.data.copy_(mistral.lm_head.weight.data)
        # hgrn2.lm_head.weight.requires_grad = False  # 冻结参数
        
        for idx in range(len(mistral.model.layers)):
            mistral_layer = mistral.model.layers[idx]
            hgrn2_layer = hgrn2.model.layers[idx]

            # 复制 MLP 参数
            hgrn2_layer.mlp.gate_proj.weight.data.copy_(mistral_layer.mlp.gate_proj.weight.data)
            hgrn2_layer.mlp.up_proj.weight.data.copy_(mistral_layer.mlp.up_proj.weight.data)
            hgrn2_layer.mlp.down_proj.weight.data.copy_(mistral_layer.mlp.down_proj.weight.data)

            # 复制注意力参数
            hgrn2_layer.attn.q_proj.weight.data.copy_(mistral_layer.self_attn.q_proj.weight.data)
            hgrn2_layer.attn.k_proj.weight.data.copy_(mistral_layer.self_attn.k_proj.weight.data)
            hgrn2_layer.attn.i_proj.weight.data.copy_(mistral_layer.self_attn.v_proj.weight.data)
            hgrn2_layer.attn.o_proj.weight.data.copy_(mistral_layer.self_attn.o_proj.weight.data)

            # 复制输入和输出的 RMSNorm 权重
            hgrn2_layer.attn_norm.weight.data.copy_(mistral_layer.input_layernorm.weight.data)
            hgrn2_layer.mlp_norm.weight.data.copy_(mistral_layer.post_attention_layernorm.weight.data)
        
        # 复制最终的归一化层参数
        hgrn2.model.norm.weight.data.copy_(mistral.model.norm.weight.data)

        hgrn2.gradient_checkpointing_enable()
        return hgrn2
    elif model_name == "gla":
        gla = GLAForCausalLM(GLAConfig())
        ### gla = GLAloraForCausalLM(GLAConfig())  ## lora setting
        ### lora.mark_only_lora_as_trainable(gla, bias='all')  ## lora setting
        mistral = MistralForCausalLM.from_pretrained(pretrained)

        gla.model.embeddings.weight.data.copy_(mistral.model.embed_tokens.weight.data)
        gla.lm_head.weight.data.copy_(mistral.lm_head.weight.data)
        
        for idx in range(len(mistral.model.layers)):
            mistral_layer = mistral.model.layers[idx]
            gla_layer = gla.model.layers[idx]

            gla_layer.mlp.gate_proj.weight.data.copy_(mistral_layer.mlp.gate_proj.weight.data)
            gla_layer.mlp.up_proj.weight.data.copy_(mistral_layer.mlp.up_proj.weight.data)
            gla_layer.mlp.down_proj.weight.data.copy_(mistral_layer.mlp.down_proj.weight.data)

            gla_layer.attn.q_proj.weight.data.copy_(mistral_layer.self_attn.q_proj.weight.data)
            gla_layer.attn.k_proj.weight.data.copy_(mistral_layer.self_attn.k_proj.weight.data)
            gla_layer.attn.v_proj.weight.data.copy_(mistral_layer.self_attn.v_proj.weight.data)
            gla_layer.attn.o_proj.weight.data.copy_(mistral_layer.self_attn.o_proj.weight.data)

            gla_layer.attn_norm.weight.data.copy_(mistral_layer.input_layernorm.weight.data)
            gla_layer.mlp_norm.weight.data.copy_(mistral_layer.post_attention_layernorm.weight.data)
        
        gla.model.norm.weight.data.copy_(mistral.model.norm.weight.data)
        gla.gradient_checkpointing_enable()
        return gla
    elif model_name == "boss-gla":
        boss = BoSsForCausalLM(BoSsConfig())  # vanilla boss-gla
        # boss = mhBoSsForCausalLM(BoSsConfig())  # multi-head boss-gla
        mistral = MistralForCausalLM.from_pretrained(pretrained)

        # 复制嵌入层参数并冻结
        boss.model.embed_tokens.weight.data.copy_(mistral.model.embed_tokens.weight.data)
        # boss.model.embed_tokens.weight.requires_grad = False  # 冻结参数

        # 复制输出层参数并冻结
        boss.lm_head.weight.data.copy_(mistral.lm_head.weight.data)
        # boss.lm_head.weight.requires_grad = False  # 冻结参数
        
        for idx in range(len(mistral.model.layers)):
            mistral_layer = mistral.model.layers[idx]
            boss_layer = boss.model.layers[idx]

            # 复制 MLP 参数
            boss_layer.mlp.gate_proj.weight.data.copy_(mistral_layer.mlp.gate_proj.weight.data)
            boss_layer.mlp.up_proj.weight.data.copy_(mistral_layer.mlp.up_proj.weight.data)
            boss_layer.mlp.down_proj.weight.data.copy_(mistral_layer.mlp.down_proj.weight.data)

            # 复制注意力参数（将 Mistral 的 self_attn 参数复制到 BoSs 的 attn_boss）
            boss_layer.attn_boss.q_proj.weight.data.copy_(mistral_layer.self_attn.q_proj.weight.data)
            boss_layer.attn_boss.k_proj.weight.data.copy_(mistral_layer.self_attn.k_proj.weight.data)
            boss_layer.attn_boss.v_proj.weight.data.copy_(mistral_layer.self_attn.v_proj.weight.data)
            boss_layer.attn_boss.o_proj.weight.data.copy_(mistral_layer.self_attn.o_proj.weight.data)

            # 同样复制 self_attn 参数到 BoSs 的 attn_main
            boss_layer.attn_main.q_proj.weight.data.copy_(mistral_layer.self_attn.q_proj.weight.data)
            boss_layer.attn_main.k_proj.weight.data.copy_(mistral_layer.self_attn.k_proj.weight.data)
            boss_layer.attn_main.v_proj.weight.data.copy_(mistral_layer.self_attn.v_proj.weight.data)
            boss_layer.attn_main.o_proj.weight.data.copy_(mistral_layer.self_attn.o_proj.weight.data)

            # 如果 BoSs 的 attn_main 中有 gk_proj，需要进行特殊处理
            # 由于 Mistral 中没有对应的参数，可能需要初始化或根据其他规则设置
            # 这里我们暂时保持 gk_proj 的原始初始化状态

            # 复制输入和输出的 RMSNorm 权重
            boss_layer.input_layernorm.weight.data.copy_(mistral_layer.input_layernorm.weight.data)
            boss_layer.post_attention_layernorm.weight.data.copy_(mistral_layer.post_attention_layernorm.weight.data)

        # 复制最终的归一化层参数
        boss.model.norm.weight.data.copy_(mistral.model.norm.weight.data)
        boss.gradient_checkpointing_enable()
        return boss
    elif model_name == "loraboss-gla":
        boss = LoRA2BoSsForCausalLM(BoSsConfig())  # boss-gla with lora^2
        mistral = MistralForCausalLM.from_pretrained(pretrained)

        # 复制嵌入层参数并冻结
        boss.model.embed_tokens.weight.data.copy_(mistral.model.embed_tokens.weight.data)
        # boss.model.embed_tokens.weight.requires_grad = False  # 冻结参数

        # 复制输出层参数并冻结
        boss.lm_head.weight.data.copy_(mistral.lm_head.weight.data)
        # boss.lm_head.weight.requires_grad = False  # 冻结参数
        
        for idx in range(len(mistral.model.layers)):
            mistral_layer = mistral.model.layers[idx]
            boss_layer = boss.model.layers[idx]

            # 复制 MLP 参数
            boss_layer.mlp.gate_proj.weight.data.copy_(mistral_layer.mlp.gate_proj.weight.data)
            boss_layer.mlp.up_proj.weight.data.copy_(mistral_layer.mlp.up_proj.weight.data)
            boss_layer.mlp.down_proj.weight.data.copy_(mistral_layer.mlp.down_proj.weight.data)

            # 复制注意力参数（将 Mistral 的 self_attn 参数复制到 BoSs 的 attn_boss）
            boss_layer.attn_boss.q_proj.weight.data.copy_(mistral_layer.self_attn.q_proj.weight.data)
            boss_layer.attn_boss.k_proj.weight.data.copy_(mistral_layer.self_attn.k_proj.weight.data)
            boss_layer.attn_boss.v_proj.weight.data.copy_(mistral_layer.self_attn.v_proj.weight.data)
            boss_layer.attn_boss.o_proj.weight.data.copy_(mistral_layer.self_attn.o_proj.weight.data)
            # 这里我们暂时保持 gk_proj 的原始初始化状态

            # 复制输入和输出的 RMSNorm 权重
            boss_layer.input_layernorm.weight.data.copy_(mistral_layer.input_layernorm.weight.data)
            boss_layer.post_attention_layernorm.weight.data.copy_(mistral_layer.post_attention_layernorm.weight.data)
        # 复制最终的归一化层参数
        boss.model.norm.weight.data.copy_(mistral.model.norm.weight.data)
        boss.gradient_checkpointing_enable()
        return boss
