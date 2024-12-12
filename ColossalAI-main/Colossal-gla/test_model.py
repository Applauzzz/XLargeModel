# from transformers import AutoTokenizer, AutoModelForMaskedLM, MistralForCausalLM

# model = MistralForCausalLM.from_pretrained("/mnt/nvme_storage/modelbase/mistral")
# config = model.config
# print(config)

# ls /mnt/nvme_storage/miniconda3/envs/CoAI/lib/python3.10/site-packages/torch/include/c10/util/

# /mnt/nvme_storage/miniconda3/envs/CoAI/lib/python3.10/site-packages/torch/include/c10/util/complex.h:8:10: fatal error: thrust/complex.h: No such file or directory
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
