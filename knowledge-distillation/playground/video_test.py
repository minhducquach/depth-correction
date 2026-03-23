import cv2
import numpy as np
import os
import torch

import sys

sys.path.append("../")

from src.data_modules.datasets.darknav_dataset import DarkNav
from src.models.mdm.model.v2 import MDMModel

CKPT_PATH = "/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/checkpoints/mdm-distill-epoch=06-validation_loss=0.2476.ckpt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_weights_from_ckpt(ckpt):
    lightning_state_dict = ckpt["state_dict"]
    state_dict = {}
    for key, weight in lightning_state_dict.items():
        if key.startswith("student."):
            clean_key = key.replace("student.", "", 1)
            state_dict[clean_key] = weight
    return state_dict

def depth_to_color_opencv(depth_map, vmin=0, vmax=50, colormap=cv2.COLORMAP_TURBO):
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

    # print(vmin, vmax)

    # Normalize to [0, 255]
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # print(depth_normalized)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)

    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored

if __name__ == "__main__":
    dataset = DarkNav()
    len_dataset = len(dataset)
    ckpt = torch.load(CKPT_PATH)
    distilled_model = MDMModel.from_pretrained_config()
    distilled_model.load_state_dict(extract_weights_from_ckpt(ckpt))
    distilled_model = distilled_model.to(device)
    distilled_model.eval()
    num_tokens = 1200
    # original_model = MDMModel.from_pretrained().to(device)

    for i in range(len_dataset):
        color, depth = dataset[i]['color'].to(device), torch.as_tensor(dataset[i]['depth'], dtype=torch.float32).to(device)

        with torch.no_grad():
            # pred_o = original_model.infer(image=color, depth_in=depth, num_tokens=num_tokens)['depth'].squeeze().float().cpu().numpy()
            # print(pred_o.shape)  #(1,480,848)
            pred_d = distilled_model.infer(image=color, depth_in=depth, num_tokens=num_tokens)['depth'].squeeze().float().cpu().numpy()
        pred_out = depth_to_color_opencv(pred_d)
        color_np = color.squeeze(0).cpu().float().numpy()
        color_np = color_np.transpose(1,2,0)
        color_np = np.clip(color_np * 255.0, 0, 255).astype(np.uint8)
        # print(color_np.shape)#(3,480,848)
        concat = np.concatenate([color_np, pred_out], axis=1)
        
        cv2.imshow('img', concat)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


