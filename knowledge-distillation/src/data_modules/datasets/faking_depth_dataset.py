import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob

import utils.faking_helper_functions as helper

import torchvision.transforms as transforms

IMG_WIDTH = 1280
IMG_HEIGHT = 720
MIN_DEPTH = 450
MAX_DEPTH = 2000

BASE_DIR = "/home/quachmd/Bureau/depth-correction/datasets/faking_depth/icra_dataset/unpaired/"

class FakingDepth(Dataset):
    def __init__(self, dir=BASE_DIR, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
        self.img_width = img_width
        self.img_height = img_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_clip = min_clip
        
        self.depth_paths = sorted(glob.glob(dir + 'transparent/depth/*.bin'))
        self.color_paths = sorted(glob.glob(dir + 'transparent/color/*.png'))
        self.pair_paths = list(zip(self.depth_path, self.color_path))
        
    def __len__(self):
        return(len(self.pair_paths))

    def __getitem__(self, index):
        depth_path, color_path = self.pair_paths[index]

        depth = helper.parse_depth_bin(depth_path, self.img_height, self.img_width, self.min_depth, self.max_depth).astype(np.float32)
        color = helper.parse_rgb_img(color_path, self.img_height,self.img_width, False).astype(np.float32)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        depth = torch.from_numpy(depth).unsqueeze(0).unsquueze(0)
        color = transform(color).unsqueeze(0)

        return {
            'color': color,
            'depth': depth
        }