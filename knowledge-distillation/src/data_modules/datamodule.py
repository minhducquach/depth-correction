import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import lightning.pytorch as pl
from datasets import *
import numpy as np
import albumentations as A

class DatasetWrapper(Dataset):
    def __init__(self, dataset, target_size=(480, 848), is_train=False):
        self.dataset = dataset
        self.target_size = target_size
        self.is_train = is_train
        
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1], p=1.0),
                A.RandomResizedCrop(size=(target_size[0], target_size[1]), scale=(0.9, 1.0), p=1.0),
                A.HorizontalFlip(),
                A.ColorJitter(),
                A.ImageCompression(),
                A.MotionBlur(),
                A.ShotNoise()
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
    def __init__(self, batch_size=16, num_workers=8, target_size=(480, 848)):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

    def _test_len(self):
        d1_raw = ARKitScenesDataset()
        d2_raw = FakingDepth()
        d3_raw = NYUv2()
        d4_raw = TartanAir()
        d5_raw = VirtualKitti()
        d6_raw = DarkNav()

        print("Arkit len:", len(d1_raw))
        print("Faking len:", len(d2_raw))
        print("NYU len:", len(d3_raw))
        print("Tartan len:", len(d4_raw))
        print("KITTI len:", len(d5_raw))
        print("DarkNav len:", len(d6_raw))

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(42)

        d1_raw = ARKitScenesDataset()
        d2_raw = FakingDepth()
        d3_raw = NYUv2()
        d4_raw = TartanAir()
        d5_raw = VirtualKitti()
        d6_raw = DarkNav()

        print("Arkit len:", len(d1_raw))
        print("Faking len:", len(d2_raw))
        print("NYU len:", len(d3_raw))
        print("Tartan len:", len(d4_raw))
        print("KITTI len:", len(d5_raw))
        print("DarkNav len:", len(d6_raw))

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
        # self.test_datasets = [DatasetWrapper(d6_raw, target_size=self.target_size, is_train=False)]
        # self.test_datasets = self.val_datasets
        self.test_datasets = [DatasetWrapper(d3_raw, target_size=self.target_size, is_train=False)]

    def train_dataloader(self):
        combined_dataset = ConcatDataset(self.train_datasets)
        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        combined_dataset = ConcatDataset(self.val_datasets)
        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        combined_dataset = ConcatDataset(self.test_datasets)
        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    

if __name__ == "__main__":
    test = MyDataModule()
    test._test_len()