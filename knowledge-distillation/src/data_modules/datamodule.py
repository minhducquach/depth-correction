import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import os
import pandas as pd

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader

from .datasets import * 

class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(42)

        d1_train, d1_val, d1_test = torch.utils.data.random_split(ARKitScenesDataset(), [0.8, 0.1, 0.1], generator=generator)
        d2_train, d2_val, d2_test = torch.utils.data.random_split(FakingDepth(), [0.8, 0.1, 0.1], generator=generator)
        d3_train, d3_val, d3_test = torch.utils.data.random_split(NYUv2(), [0.8, 0.1, 0.1], generator=generator)
        d4_train, d4_val, d4_test = torch.utils.data.random_split(TartanAir(), [0.8, 0.1, 0.1], generator=generator)
        d5_train, d5_val, d5_test = torch.utils.data.random_split(VirtualKitti(), [0.8, 0.1, 0.1], generator=generator)

        self.train_datasets = [d1_train, d2_train, d3_train, d4_train, d5_train]
        self.val_datasets = [d1_val, d2_val, d3_val, d4_val, d5_val]
        self.test_datasets = [d1_test, d2_test, d3_test, d4_test, d5_test]

    def train_dataloader(self):
        combined_dataset = ConcatDataset(self.train_datasets)
        return DataLoader(combined_dataset, batch_size=32, shuffle=True)
    
    def val_dataloader(self):
        combined_dataset = ConcatDataset(self.val_datasets)
        return DataLoader(combined_dataset, batch_size=32, shuffle=False)
    
    def test_dataloader(self):
        combined_dataset = ConcatDataset(self.test_datasets)
        return DataLoader(combined_dataset, batch_size=32, shuffle=False)