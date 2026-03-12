import torch
import cv2
import sys
import os
import numpy as np

sys.path.append("../")

from src.models.mdm.model.v2 import MDMModel

# Source - https://stackoverflow.com/a/62556666
# Posted by SHAGUN SHARMA, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-26, License - CC BY-SA 4.0

import torch
torch.cuda.empty_cache()


def depth_to_color_opencv(depth_map, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
    """
    Convert depth map to color visualization using OpenCV colormap.

    Args:
        depth_map (np.ndarray): Depth map (H, W)
        vmin (float): Minimum depth for colormap (auto if None)
        vmax (float): Maximum depth for colormap (auto if None)
        colormap: OpenCV colormap (TURBO, JET, VIRIDIS, etc.)

    Returns:
        np.ndarray: Colored depth map (H, W, 3) in BGR format
    """
    # Handle invalid values
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0

    # Auto-range if not specified
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1

    # Normalize to [0, 255]
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)

    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
# model = MDMModel.from_pretrained().to(device)
model = MDMModel.from_pretrained_config()

# 2. Load the raw Lightning checkpoint dictionary
checkpoint = torch.load('/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/checkpoints/mdm-distill-epoch=24-validation_loss=0.6810.ckpt', map_location='cpu')

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

model = model.to(device)

# print("Student model successfully loaded!")
model.eval()

model2 = MDMModel.from_pretrained().to(device)
model2.eval()

# Load and prepare inputs
# image = cv2.cvtColor(cv2.imread('./color.png'), cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(cv2.imread('/home/quachmd/Bureau/depth-correction/knowledge-distillation/playground/r-1316653580.552634-1318500909.png'), cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
# image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

depth_np = np.load("./depth.npy").astype(np.float32)

# 2. Prepare Depth Tensor (Must be 4D: [Batch, Channel, Height, Width])
# Use [None, None] to add both Batch and Channel dims
# depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)[None, None]
depth = cv2.imread('/home/quachmd/Bureau/depth-correction/knowledge-distillation/playground/d-1316653580.544897-1318140568.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
depth_tensor = torch.tensor(depth, dtype=torch.float32, device=device)[None]

# 3. Prepare Image Tensor (Already correct in your code, but keep for reference)
image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)[None]

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


# Run inference
# starter.record()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        output1 = model.infer(image_tensor, depth_in=depth_tensor)
        output2 = model2.infer(image_tensor, depth_in=depth_tensor)
# ender.record()

# torch.cuda.synchronize()
# currtime = starter.elapsed_time(ender)
# print(currtime)

depth_pred = output1['depth'].squeeze().float().cpu().numpy()
depth_ref = output2['depth'].squeeze().float().cpu().numpy()
# points = output['points']      # 3D point cloud

print(depth_ref.shape, depth_ref.shape)

# Save results
# 1. Save depth maps as numpy arrays
np.save('./depth_refined.npy', depth_pred)
np.save('./depth_ref.npy', depth_ref)

# 2. Save depth visualizations
depth_raw_color = depth_to_color_opencv(depth)
# depth_raw_color = depth_to_color_opencv(depth_np)
depth_pred_color = depth_to_color_opencv(depth_pred)
depth_ref_color = depth_to_color_opencv(depth_ref)
depth_concat = np.concatenate([depth_raw_color, depth_pred_color, depth_ref_color], axis=1)

cv2.imwrite(str('./depth_ref.png'), depth_ref_color)
cv2.imwrite(str('./depth_refined.png'), depth_pred_color)
cv2.imwrite(str('./depth_comparison.png'), depth_concat)