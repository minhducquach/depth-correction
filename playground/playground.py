import numpy as np
import torch

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

rand = torch.randint(0, 5, (2, ))
print(len(rand))