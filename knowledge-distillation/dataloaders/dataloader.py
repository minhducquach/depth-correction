import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self):
        pass

    def __