import torch
from configuration_boss import BoSsConfig
from modeling_boss import BoSsForCausalLM

import pdb

config = BoSsConfig()
model = BoSsForCausalLM(config)
input_ids = torch.randint(0, 32000, (1, 128))

model = model.cuda().to(torch.bfloat16)
input_ids = input_ids.cuda()
outputs = model(input_ids)
pdb.set_trace()