import numpy as np
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels=num_in, out_channels=num_out, kernel_size=1)

    def forward(self, x):
        adapted_x = self.adapter(x)
        return adapted_x

# depth_raw = np.load("/home/quachmd-admin/Bureau/depth_correction/depth.npy")
# depth_refined = np.load("/home/quachmd-admin/Bureau/depth_correction/lingbot-depth/test/depth_refined.npy")

# print(depth_raw)
# print(depth_refined)

# norm_raw = np.linalg.norm(depth_raw)
# norm_refined = np.linalg.norm(depth_refined)

# cosine_similarity = np.dot(depth_raw , np.transpose(depth_refined)) / (norm_raw * norm_refined)
# print(np.mean((cosine_similarity)))

# model = torch.load('/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/checkpoints/mdm-distill-epoch=06-validation_loss=1.3047.ckpt')
# print(model.state_dict().keys)

# rand = torch.randint(0, 5, (2, ))
# print(len(rand))

adapt = Adapter(num_in=5, num_out=10)
a = np.zeros(shape=(1,5,2,2))
a = torch.tensor(a, dtype=torch.float32)

b = np.zeros(shape=(1,10,2,2))
b = torch.tensor(b, dtype=torch.float32)

c = adapt(a)
print(c.shape)
