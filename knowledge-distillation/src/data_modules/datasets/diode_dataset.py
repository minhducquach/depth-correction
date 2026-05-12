import torch.utils.data
# from torchvision import transforms

import os.path
import glob
import numpy as np

from collections import namedtuple
# from PIL import Image
import cv2
cv2.setNumThreads(8)
cv2.ocl.setUseOpenCL(False)
from pathlib import Path

# import torchvision.io as io

import matplotlib.pyplot as plt

BASE_DIR = "/home/quachmd/Bureau/depth-correction/datasets/diode-val/val"

def preprocess_input_image(image_path):
    """
    Load and preprocess RGB image.

    Args:
        image_path (str): Path to RGB image
        device (torch.device): Device to load tensor on

    Returns:
        tuple: (numpy_image, tensor_image)
            - numpy_image: RGB numpy array (H, W, 3), uint8
            - tensor_image: RGB tensor (1, 3, H, W), float32, [0,1]
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image and convert BGR to RGB
    image_np = cv2.imread(image_path)
    if image_np is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Convert to tensor and normalize to [0, 1]
    image_tensor = torch.tensor(
        image_np / 255.0,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    return image_tensor

def load_depth_map(depth_path, scale=1000.0):
    """
    Load depth map from PNG file (16-bit) and convert to meters.

    Args:
        depth_path (str): Path to depth image
        scale (float): Scale factor to convert to meters
            - 1000.0 for millimeters
            - 1.0 for meters

    Returns:
        np.ndarray: Depth map in meters (H, W), float32
    """
    if not Path(depth_path).exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    # Read depth map as 16-bit
    # depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth_map = np.load(depth_path)
    if depth_map is None:
        raise ValueError(f"Failed to read depth map: {depth_path}")

    # Convert to meters
    depth_map = depth_map.astype(np.float32) / scale                                                    

    # Replace invalid values with 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    return depth_map

class Diode(torch.utils.data.Dataset):
    def __init__(self, dir=BASE_DIR):
        search_pattern = os.path.join(dir, '**', '*.png')
        self.color_files = sorted(glob.glob(search_pattern, recursive=True))

        self.depth_files = []
        
        for color_path in self.color_files:
            depth_path = color_path.replace('.png', '_depth.npy')
            self.depth_files.append(depth_path)

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        color_path = self.color_files[idx]
        depth_path = self.depth_files[idx]

        color = preprocess_input_image(color_path)
        depth = load_depth_map(depth_path, scale=1.0)

        return {
            'color': color,
            'depth': depth
        }

if __name__ == '__main__':
    dataset = Diode()
    len_dataset = len(dataset)
    color, depth = dataset[0]['color'], dataset[0]['depth']
    print(color.shape, depth.shape)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(color.squeeze().permute(1,2,0))
    plt.subplot(1, 2, 2)
    plt.imshow(depth)
    plt.imshow(depth, vmin=0, vmax=20)  # Clamp visual range to 20 meters
    plt.show()
    print(depth, depth.dtype)
    print(f"Max depth value causing the blue effect: {depth.max()}")