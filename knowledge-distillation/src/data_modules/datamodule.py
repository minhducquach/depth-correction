import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import lightning.pytorch as pl
from .datasets import *
import numpy as np
import albumentations as A


class DatasetWrapper(Dataset):
    def __init__(self, dataset, target_size=(960, 1280), is_train=False):
        self.dataset = dataset
        self.target_size = target_size
        self.is_train = is_train
        
        if self.is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(height=target_size[0], width=target_size[1], scale=(0.9, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
                A.ImageCompression(quality_lower=50, quality_upper=95, p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.ShotNoise(scale_range=(0.1,0.3), p=0.5)
            ], additional_targets={'depth': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1], p=1.0)
            ], additional_targets={'depth': 'mask'})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        color = np.array(sample['color'].squeeze())
        depth = np.array(sample['depth'])

        color = np.transpose(color, (1, 2, 0))
        color = (color * 255.0).astype(np.uint8)

        augmented = self.transform(image=color, depth=depth)
        color, depth = augmented['image'], augmented['depth']

        color = color.astype(np.float32) / 255.0
        color = np.transpose(color, (2, 0, 1))
        sample['color'] = torch.as_tensor(color, dtype=torch.float32)
        sample['depth'] = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0)

        return sample

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=12, target_size=(960, 1280)):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(42)

        d1_raw = ARKitScenesDataset()
        d2_raw = FakingDepth()
        d3_raw = NYUv2()
        d4_raw = TartanAir()
        d5_raw = VirtualKitti()
        d6_raw = DarkNav()

<<<<<<< Updated upstream
        d1_train, d1_val = torch.utils.data.random_split(dataset1, [0.8, 0.2], generator=generator)
        d2_train, d2_val = torch.utils.data.random_split(dataset2, [0.8, 0.2], generator=generator)
        d3_train, d3_val = torch.utils.data.random_split(dataset3, [0.8, 0.2], generator=generator)
        d4_train, d4_val = torch.utils.data.random_split(dataset4, [0.8, 0.2], generator=generator)
        d5_train, d5_val = torch.utils.data.random_split(dataset5, [0.8, 0.2], generator=generator)
        # d6_train, d6_val, d6_test = torch.utils.data.random_split(dataset6, [0.8, 0.1, 0.1], generator=generator)

        self.train_datasets = [d1_train, d2_train, d3_train, d4_train, d5_train]
        self.val_datasets = [d1_val, d2_val, d3_val, d4_val, d5_val]
        self.test_datasets = [dataset6]
=======
        d1_train, d1_val = torch.utils.data.random_split(d1_raw, [0.8, 0.2], generator=generator)
        d2_train, d2_val = torch.utils.data.random_split(d2_raw, [0.8, 0.2], generator=generator)
        d3_train, d3_val = torch.utils.data.random_split(d3_raw, [0.8, 0.2], generator=generator)
        d4_train, d4_val = torch.utils.data.random_split(d4_raw, [0.8, 0.2], generator=generator)
        d5_train, d5_val = torch.utils.data.random_split(d5_raw, [0.8, 0.2], generator=generator)
        # d6_train, d6_val, d6_test = torch.utils.data.random_split(dataset6, [0.8, 0.1, 0.1], generator=generator)

        self.train_datasets = [
            DatasetWrapper(d, target_size=self.target_size, is_train=True)
            for d in [d1_train, d2_train, d3_train, d4_train, d5_train]
        ]
        # self.val_datasets = [d1_val, d2_val, d3_val, d4_val, d5_val]
        self.val_datasets = [
            DatasetWrapper(d, target_size=self.target_size, is_train=False)
            for d in [d1_val, d2_val, d3_val, d4_val, d5_val]
        ]
        self.test_datasets = [DatasetWrapper(d6_raw, target_size=self.target_size, is_train=False)]
>>>>>>> Stashed changes

    def train_dataloader(self):
        combined_dataset = ConcatDataset(self.train_datasets)
        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    
    def val_dataloader(self):
        combined_dataset = ConcatDataset(self.val_datasets)
        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    
    def test_dataloader(self):
        combined_dataset = ConcatDataset(self.test_datasets)
        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)