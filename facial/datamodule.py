from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision import transforms
from typing import *
import torch

from . import transform as CT
from .datasets import FacialKeypointsDataset


def transform_fn(size=224):
    tfm = transforms.Compose([
        CT.Rescale(250),
        CT.RandomCrop(size),
        CT.Normalize(),
        CT.ToTensor()
    ])
    
    return tfm


class FacialKeypointsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 16, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = transform_fn()
        self.valid_transform = transform_fn()
        
        
    def setup(self, stage: Optional[str] = None):
        self.train_path = Path(self.data_dir).joinpath('training')
        self.train_csvpath = Path(self.data_dir).joinpath('training_frames_keypoints.csv')
        
        self.valid_path = Path(self.data_dir).joinpath('test')
        self.valid_csvpath = Path(self.data_dir).joinpath('test_frames_keypoints.csv')


        self.trainset = FacialKeypointsDataset(csv_file=self.train_csvpath, 
                                               root_dir=self.train_path, 
                                               transforms=self.train_transform)
        
        self.validset = FacialKeypointsDataset(csv_file=self.valid_csvpath, 
                                               root_dir=self.valid_path,
                                               transforms=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)