import torch
import sys

sys.path.append('../')
from mdm.model.v2 import MDMModel

model = MDMModel.from_pretrained_config()

print(model.state_dict())