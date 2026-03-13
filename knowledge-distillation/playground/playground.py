import numpy as np
import torch
import sys

sys.path.append("../")

from src.models.mdm.model.v2 import MDMModel

depth_raw = np.load("/home/quachmd/Bureau/depth-correction/datasets/darknav/Circular/depth/1739893136_240500000.npy")
depth_refined = np.load("/home/quachmd/Bureau/depth-correction/knowledge-distillation/playground/depth_ref.npy")

print(depth_raw / 1000.0)
print(depth_refined)

# norm_raw = np.linalg.norm(depth_raw)
# norm_refined = np.linalg.norm(depth_refined)

# cosine_similarity = np.dot(depth_raw , np.transpose(depth_refined)) / (norm_raw * norm_refined)
# print(np.mean((cosine_similarity)))

# 1. Initialize an empty student MDMModel (ViT-Tiny)
# model = MDMModel.from_pretrained_config()

# # 2. Load the raw Lightning checkpoint dictionary
# checkpoint = torch.load('/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/checkpoints/mdm-distill-epoch=06-validation_loss=1.3047.ckpt', map_location='cpu')

# # 3. Extract just the "state_dict" containing the weights
# lightning_state_dict = checkpoint["state_dict"]

# # 4. Filter and rename the keys to rip out just the student model
# student_state_dict = {}
# for key, weight in lightning_state_dict.items():
#     if key.startswith("student."):
#         # Remove the "student." prefix so it matches the raw MDMModel
#         clean_key = key.replace("student.", "", 1)
#         student_state_dict[clean_key] = weight

# # 5. Load the clean weights into the MDMModel
# model.load_state_dict(student_state_dict)

# print("Student model successfully loaded!")
# model.eval()

# print(model.state_dict().keys())