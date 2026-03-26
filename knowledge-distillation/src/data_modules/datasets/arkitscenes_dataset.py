import os
from glob import glob
from typing import Callable, Optional

import cv2
cv2.setNumThreads(8)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
# import torchvision.transforms as transforms

import matplotlib.pyplot as plt


BASE_DIR = "/home/quachmd/Bureau/depth-correction/datasets/arkitscenes/upsampling"

META_DATA_CSV_FILE = 'metadata.csv'
WIDE = 'wide'
LOW_RES = (192, 256)
HIGH_RES = (1440, 1920)
MILLIMETER_TO_METER = 1000

IDENTIFIER = 'identifier'
COLOR_IMG = 'color_img'
HIGH_RES_DEPTH_IMG = 'high_res_depth_img'
LOW_RES_DEPTH_IMG = 'low_res_depth_img'
PREDICTION_DEPTH_IMG = 'prediction_img'
VALID_MASK_IMG = 'valid_mask_img'

def expand_channel_dim(img):
    """
        expand image dimension to add a channel dimension
    """
    return np.expand_dims(img, 0)


def image_hwc_to_chw(img):
    """
        transpose the image from height, width, channel -> channel, height, width
        (pytorch format)
    """
    return img.transpose((2, 0, 1))


def image_chw_to_hwc(img):
    """
        revert image_hwc_to_chw function
    """
    return img.transpose((1, 2, 0))


def batch_to_cuda(batch):
    if torch.cuda.is_available():
        for k in batch:
            if k != 'identifier':
                batch[k] = batch[k].cuda(non_blocking=True)
    return batch

class ARKitScenesDataset(Dataset):
    """`ARKitScenes Dataset class.

    Args:
        root (string): Root directory of dataset where directory
            exists or will be saved to if download is set to True.
        transform (callable, optional): A function that takes in a sample (dict)
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts on the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
            self,
            root: str = BASE_DIR,
            split: str = 'train',
            transform: Optional[Callable] = None,
            upsample_factor: Optional[int] = None,
    ) -> None:

        super(ARKitScenesDataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.upsample_factor = upsample_factor

        if self.upsample_factor is not None and self.upsample_factor not in (2, 4, 8):
            raise ValueError(f'rgb_factor must to be one of (2,4,8) but got {self.upsample_factor}')
        if split == 'train':
            self.split_folder = 'Training'
        elif split == 'val':
            self.split_folder = 'Validation'
        else:
            raise Exception(f'split must to be one of (train, val), got ={split}')
        self.dataset_folder = os.path.join(self.root, self.split_folder)
        if self.upsample_factor is None:
            self.low_res = LOW_RES
            self.high_res = HIGH_RES
        else:
            if self.upsample_factor in (2, 4):
                self.low_res = LOW_RES
                self.high_res = [i * self.upsample_factor for i in LOW_RES]
            elif self.upsample_factor == 8:
                self.high_res = HIGH_RES
                self.low_res = [int(i / self.upsample_factor) for i in HIGH_RES]
            else:
                raise Exception(f'Can\'t load dataset with upsample_factor = {self.upsample_factor}')

        self.samples = []  # videos_id, sample_id, sky_direction
        self.meta_data = pd.read_csv(os.path.join(os.path.dirname(self.dataset_folder), META_DATA_CSV_FILE))
        self.meta_data = self.meta_data[self.meta_data['fold'] == self.split_folder]
        for video_id, sky_direction in zip(self.meta_data['video_id'], self.meta_data['sky_direction']):
            video_folder = os.path.join(self.dataset_folder, str(video_id))
            color_files = glob(os.path.join(video_folder, WIDE, '*.png'))
            self.samples.extend([[str(video_id), str(os.path.basename(file)), sky_direction]
                                 for file in color_files])

    @staticmethod
    def rotate_image(img, direction):
        if direction == 'Up':
            pass
        elif direction == 'Left':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif direction == 'Right':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif direction == 'Down':
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            raise Exception(f'No such direction (={direction}) rotation')
        return img

    @staticmethod
    def load_image(path, shape, is_depth, sky_direction):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.shape[:2] != shape:
            img = cv2.resize(img, shape[::-1], interpolation=cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR)
        img = ARKitScenesDataset.rotate_image(img, sky_direction)
        if is_depth:
            img = expand_channel_dim(np.asarray(img / MILLIMETER_TO_METER, np.float32))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = image_hwc_to_chw(np.asarray(img / 255.0, np.float32))
        return torch.from_numpy(img).unsqueeze(0)
        return img

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (identifier, color, highres_depth, lowres_depth).
        """
        video_id, sample_id, direction = self.samples[index]
        sample = dict()
        sample[IDENTIFIER] = str(sample_id)

        rgb_file = os.path.join(self.dataset_folder, video_id, WIDE, sample_id)
        depth_file = os.path.join(self.dataset_folder, video_id, 'highres_depth', sample_id)
        apple_depth_file = os.path.join(self.dataset_folder, video_id, 'lowres_depth', sample_id)

        sample[COLOR_IMG] = self.load_image(rgb_file, self.high_res, False, direction)
        sample[HIGH_RES_DEPTH_IMG] = self.load_image(depth_file, self.high_res, True, direction)
        sample[LOW_RES_DEPTH_IMG] = self.load_image(apple_depth_file, self.low_res, True, direction)

        if self.transform is not None:
            sample = self.transform(sample)
            
        return {
            'color': sample[COLOR_IMG], 
            'depth': torch.Tensor.numpy(sample[LOW_RES_DEPTH_IMG].squeeze())
        }

    def __len__(self) -> int:
        return len(self.samples)
    
if __name__ == '__main__':
    dataset = ARKitScenesDataset(root=BASE_DIR)
    len_dataset = len(dataset)
    color, depth = dataset[0]['color'], dataset[0]['depth']
    print(color.shape, depth.shape)
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(color.squeeze().permute(1,2,0))
    # plt.subplot(1, 2, 2)
    # plt.imshow(depth)
    # plt.show()
    print(color)