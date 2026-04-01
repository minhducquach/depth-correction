import torch
import cv2
import sys
import os
import numpy as np

from models.mdm.model.v2 import MDMModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'

model = MDMModel.from_pretrained_config()
# 2. Load the raw Lightning checkpoint dictionary
checkpoint = torch.load('/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/checkpoints/mdm-distill-epoch=05-validation_loss=0.1856.ckpt')

# 3. Extract just the "state_dict" containing the weights
lightning_state_dict = checkpoint["state_dict"]

# 4. Filter and rename the keys to rip out just the student model
student_state_dict = {}
for key, weight in lightning_state_dict.items():
    if key.startswith("student."):
        # Remove the "student." prefix so it matches the raw MDMModel
        clean_key = key.replace("student.", "", 1)
        student_state_dict[clean_key] = weight

# 5. Load the clean weights into the MDMModel
model.load_state_dict(student_state_dict)
model.to(device)
model.eval()
model.enable_pytorch_native_sdpa()

image = cv2.cvtColor(cv2.imread('/home/quachmd/Bureau/depth-correction/datasets/darknav/Circular/rgb/1739893136_240500000.png'), cv2.COLOR_BGR2RGB)
depth_np = np.load("/home/quachmd/Bureau/depth-correction/datasets/darknav/Circular/depth/1739893136_240500000.npy").astype(np.float32) / 1000.0

image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)[None].to(device)
depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)[None, None].to(device)

def get_num_tokens():
    min_tokens, max_tokens = 1200, 3600
    resolution_level = 0    
    return int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

num_tokens = torch.tensor([get_num_tokens()], dtype=torch.long)

# output_onnx_file = "./model.onnx"
# model.encoder.onnx_compatible_mode = True
# model.encoder.backbone.onnx_compatible_mode = True
# model.encoder.backbone.interpolate_antialias = False

# torch.onnx.export(trt_model = torch_tensorrt.compile(model, 
#     model.float(),
#     (image_tensor, 1200, depth_tensor),
#     output_onnx_file,
#     input_names=["image", "num_tokens", "depth"],
#     output_names=["depth_reg", "mask"],
#     dynamic_axes={
#         "image": {0: "batch_size", 2: "height", 3: "width"},
#         "depth": {0: "batch_size", 2: "height", 3: "width"},
#         "depth_reg": {0: "batch_size", 2: "height", 3: "width"},
#         "mask": {0: "batch_size", 2: "height", 3: "width"}
#     },
#     opset_version=18,
#     dynamo=False
# )
# print("save to onnx file:", output_onnx_file)

# output_engine_file = 'model.trt'

# trt_model = torch_tensorrt.compile(model, 
#     inputs= [torch_tensorrt.Input((1, 3, 224, 224))],
#     enabled_precisions= { torch_tensorrt.dtype.half} # Run with FP16
# )