from v2 import MDMModel

device = 'cuda'
model = 'robbyant/lingbot-depth-pretrain-vitl-14-v0.5'

model = MDMModel.from_pretrained_config(model).to(device)
print(model.state_dict().keys())
