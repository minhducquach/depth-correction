import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob

from .utils import faking_helper_functions as helper

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import cv2
from pathlib import Path


IMG_WIDTH = 1280
IMG_HEIGHT = 720
MIN_DEPTH = 0
MAX_DEPTH = 65535

BASE_DIR = "/home/quachmd/Bureau/depth-correction/datasets/faking_depth/icra_dataset/unpaired/"

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

    # Read depth map as 16-bit
    depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth_map is None:
        raise ValueError(f"Failed to read depth map: {depth_path}")

    # Convert to meters
    depth_map = depth_map.astype(np.float32) / scale                                                    

    # Replace invalid values with 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    return depth_map

class FakingDepth(Dataset):
    def __init__(self, dir=BASE_DIR, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
        self.img_width = img_width
        self.img_height = img_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_clip = min_clip
        
        self.depth_paths = sorted(glob.glob(dir + 'transparent/depth/*.bin'))
        self.color_paths = sorted(glob.glob(dir + 'transparent/color/*.png'))
        self.pair_paths = list(zip(self.depth_paths, self.color_paths))
        
    def __len__(self):
        return(len(self.pair_paths))

    def __getitem__(self, index):
    #     depth_path, color_path = self.pair_paths[index]

    #     depth = helper.parse_depth_bin(depth_path, self.img_height, self.img_width, self.min_depth, self.max_depth).astype(np.float32)
    #     color = helper.parse_rgb_img(color_path, self.img_height,self.img_width, False).astype(np.float32)

    #     transform = transforms.Compose([
    #         transforms.ToTensor()
    #     ])

    #     depth = torch.from_numpy(depth).permute(2,0,1).unsqueeze(0)
    #     color = transform(color).unsqueeze(0)

    #     return {
    #         'color': color,
    #         'depth': depth
    #     }

        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

        color = preprocess_input_image(color_path)
        # depth = load_depth_map(depth_path)
        depth = helper.parse_depth_bin(depth_path, self.img_height, self.img_width, self.min_depth, self.max_depth).astype(np.float32).squeeze()

        return {
            'color': color,
            'depth': depth
        }

if __name__ == '__main__':
    dataset = FakingDepth()
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