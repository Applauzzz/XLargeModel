import torch
from .configuration_gla import GLAConfig
from .modeling_gla import GLAForCausalLM

import pdb

config = GLAConfig()
model = GLAForCausalLM(config)
input_ids = torch.randint(0, 32000, (4, 1024))

model = model.cuda().to(torch.bfloat16)
input_ids = input_ids.cuda()
outputs = model(input_ids)
pdb.set_trace()