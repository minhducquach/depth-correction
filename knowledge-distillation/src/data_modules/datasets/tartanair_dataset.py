import torch.utils.data
from torchvision import transforms

import os.path
import glob
import numpy as np

from collections import namedtuple
from PIL import Image
import cv2
cv2.setNumThreads(8)
cv2.ocl.setUseOpenCL(False)
from pathlib import Path

import matplotlib.pyplot as plt

MAX_DEPTH = 655.35

BASE_DIR = "/home/quachmd/Bureau/depth-correction/knowledge-distillation/data/tartanair"

def read_decode_depth(depthpath):
    depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)
    # return depth_rgba

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

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

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

    # Read depth map
    depth_map = read_decode_depth(depth_path)
    if depth_map is None:
        raise ValueError(f"Failed to read depth map: {depth_path}")

    # Convert to meters
    depth_map = depth_map.astype(np.float32) / scale
    depth_map = np.clip(depth_map, 0.0, MAX_DEPTH)

    # Replace invalid values with 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    return depth_map

class TartanAir(torch.utils.data.Dataset):
    def __init__(self, dir=BASE_DIR):
        search_pattern = os.path.join(dir, '**', 'image_lcam_front', '*.png')
        self.color_files = sorted(glob.glob(search_pattern, recursive=True))

        self.depth_files = []

        for color_path in self.color_files:
            depth_path = color_path.replace('front.png', 'front_depth.png')
            depth_path = depth_path.replace('/image_lcam_front/', '/depth_lcam_front/')
            self.depth_files.append(depth_path)

    def __len__(self):
        return len(self.color_files)
    
    def __getitem__(self, index):
        color_path = self.color_files[index]
        depth_path = self.depth_files[index]

        color = preprocess_input_image(color_path)
        depth = load_depth_map(depth_path, scale=1.0)

        return {
            'color': color,
            'depth': depth
        }

if __name__ == '__main__':
    dataset = TartanAir()
    len_dataset = len(dataset)
    color, depth = dataset[0]['color'], dataset[0]['depth']
    print(color.shape, depth.shape)
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(color.squeeze().permute(1,2,0))
    # plt.subplot(1, 2, 2)
    # plt.imshow(depth)
    # plt.show()
    # print(depth.dtype)
