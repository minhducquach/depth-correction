import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import lightning.pytorch as pl
from .datasets import * # 1. Create the Wrapper Class
class ResizeWrapper(Dataset):
    def __init__(self, dataset, target_size=(480, 640)):
        self.dataset = dataset
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        color = torch.as_tensor(sample['color'], dtype=torch.float32)
        depth = torch.as_tensor(sample['depth'], dtype=torch.float32)
        if color.dim() == 3:
            color = color.unsqueeze(0)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        else:
            depth = depth.unsqueeze(0)

        color_resized = F.interpolate(color, size=self.target_size, mode='bilinear', align_corners=False)
        
        depth_resized = F.interpolate(depth, size=self.target_size, mode='nearest')

        # Remove the fake batch dimension and put them back in the dict
        sample['color'] = color_resized.squeeze(0)
        sample['depth'] = depth_resized.squeeze(0)
        
        return sample

class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(42)

        dataset1 = ResizeWrapper(ARKitScenesDataset(), target_size=(480, 640))
        dataset2 = ResizeWrapper(FakingDepth(), target_size=(480, 640))
        dataset3 = ResizeWrapper(NYUv2(), target_size=(480, 640))
        dataset4 = ResizeWrapper(TartanAir(), target_size=(480, 640))
        dataset5 = ResizeWrapper(VirtualKitti(), target_size=(480, 640))
        dataset6 = ResizeWrapper(DarkNav(), target_size=(480, 640))

        d1_train, d1_val, d1_test = torch.utils.data.random_split(dataset1, [0.8, 0.1, 0.1], generator=generator)
        d2_train, d2_val, d2_test = torch.utils.data.random_split(dataset2, [0.8, 0.1, 0.1], generator=generator)
        d3_train, d3_val, d3_test = torch.utils.data.random_split(dataset3, [0.8, 0.1, 0.1], generator=generator)
        d4_train, d4_val, d4_test = torch.utils.data.random_split(dataset4, [0.8, 0.1, 0.1], generator=generator)
        d5_train, d5_val, d5_test = torch.utils.data.random_split(dataset5, [0.8, 0.1, 0.1], generator=generator)
        # d6_train, d6_val, d6_test = torch.utils.data.random_split(dataset6, [0.8, 0.1, 0.1], generator=generator)

        self.train_datasets = [d1_train, d2_train, d3_train, d4_train, d5_train]
        self.val_datasets = [d1_val, d2_val, d3_val, d4_val, d5_val]
        self.test_datasets = [d1_test, d2_test, d3_test, d4_test, d5_test]

    def train_dataloader(self):
        combined_dataset = ConcatDataset(self.train_datasets)
        return DataLoader(combined_dataset, batch_size=12, shuffle=True, num_workers=16, pin_memory=True)
    
    def val_dataloader(self):
        combined_dataset = ConcatDataset(self.val_datasets)
        return DataLoader(combined_dataset, batch_size=12, shuffle=False, num_workers=16, pin_memory=True)
    
    def test_dataloader(self):
        combined_dataset = ConcatDataset(self.test_datasets)
        return DataLoader(combined_dataset, batch_size=12, shuffle=False, num_workers=16, pin_memory=True)