from .modeling_hgrn2 import HGRN2ForCausalLM
from .configuration_hgrn2 import HGRN2Config
def get_hgrn2(neox_args):
    # here align the neox_args
    config = HGRN2Config()
    return HGRN2ForCausalLM(config)