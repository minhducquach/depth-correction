import torch
import cv2
import numpy as np
import sys

sys.path.append("/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/models")

from mdm.model.v2 import MDMModel

# Source - https://stackoverflow.com/a/62556666
# Posted by SHAGUN SHARMA, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-26, License - CC BY-SA 4.0

import torch
torch.cuda.empty_cache()

def extract_weights_from_ckpt(ckpt):
    lightning_state_dict = ckpt["state_dict"]
    state_dict = {}
    for key, weight in lightning_state_dict.items():
        if key.startswith("student."):
            clean_key = key.replace("student.", "", 1)
            state_dict[clean_key] = weight
    return state_dict

CKPT_PATH = "/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/checkpoints/mdm-distill-epoch=23-validation_loss=0.0710.ckpt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(CKPT_PATH, map_location=device)

distilled_model = MDMModel.from_pretrained_config()
distilled_model.load_state_dict(extract_weights_from_ckpt(ckpt))
distilled_model = distilled_model.to(device)
distilled_model.eval()

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
model = MDMModel.from_pretrained('robbyant/lingbot-depth-pretrain-vitl-14-v0.5').to(device)

model.eval()

# Load and prepare inputs
image = cv2.cvtColor(cv2.imread('/home/quachmd/Bureau/depth-correction/playground/color.png'), cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
# image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

depth_np = np.load("/home/quachmd/Bureau/depth-correction/playground/depth.npy").astype(np.float32)

# 2. Prepare Depth Tensor (Must be 4D: [Batch, Channel, Height, Width])
# Use [None, None] to add both Batch and Channel dims
depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)[None, None]

# 3. Prepare Image Tensor (Already correct in your code, but keep for reference)
image_tensor = torch.tensor(image / 255).permute(2, 0, 1)[None]

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


# Run inference
starter.record()
# output = model.infer(image_tensor, depth_in=depth_tensor)
output = distilled_model.infer(image_tensor, depth_in=depth_tensor)
ender.record()

torch.cuda.synchronize()
currtime = starter.elapsed_time(ender)
print(currtime)

depth_pred = output['depth'].squeeze().float().cpu().numpy()
# points = output['points']      # 3D point cloud

# Save results
# 1. Save depth maps as numpy arrays
# np.save('test/depth_refined.npy', depth_pred)

# 2. Save depth visualizations
depth_raw_color = depth_to_color_opencv(depth_np)
depth_pred_color = depth_to_color_opencv(depth_pred)
depth_concat = np.concatenate([depth_raw_color, depth_pred_color], axis=1)

cv2.imwrite(str('/home/quachmd/Bureau/depth-correction/playground/depth_refined_student.png'), depth_pred_color)
cv2.imwrite(str('/home/quachmd/Bureau/depth-correction/playground/depth_comparison_student.png'), depth_concat)