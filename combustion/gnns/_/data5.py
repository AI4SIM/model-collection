import config
import h5py
import json
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch_optimizer as optim
import yaml

class NOGWDDataset(torch.utils.data.Dataset):
    
    x_feat = 191
    y_feat = 126
    shard_len = 2355840

    def __init__(self, root):
        super().__init__()
        self.root = root
        
        # for idx, filename in enumerate(self.raw_filenames):
        #     raw_path = os.path.join(self.raw_dir, filename)
        #     processed_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
        #     if not os.path.isfile(processed_path):
        #         self.process(raw_path)
                
        self.x, self.y = self.load()
        self.compute_stats()
    
    def compute_stats(self):
        with h5py.File(os.path.join(self.root, 'stats.h5'), 'w') as file:
            file['/x_mean'] = np.mean(self.x, axis=0)
            file['/y_mean'] = np.mean(self.y, axis=0)
            file['/x_std'] = np.std(self.x, axis=0)
            file['/y_std'] = np.std(self.y, axis=0)
    
    def load(self):
        x_, y_ = [], []
        
        for filename in self.raw_filenames:
            with h5py.File(os.path.join(self.raw_dir, filename), 'r') as file:
                x = file['/x'][:]
                y = file['/y'][:]
                
            x_.append(np.reshape(x, (191, -1)).T)
            y_.append(np.reshape(y, (126, -1)).T)
            
        x = np.concatenate(x_)
        y = np.concatenate(y_)
        
        return x, y
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_filenames(self):
        with open(os.path.join(self.root, "filenames.yaml"), "r") as stream:
            filenames = yaml.safe_load(stream)
        return filenames

    @property
    def processed_filenames(self):
        return self.raw_filenames

    def __len__(self):
        return len(self.raw_filenames) * self.shard_len

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]


@DATAMODULE_REGISTRY
class NOGWDDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int,):
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()

    def prepare_data(self):
        self.dataset = NOGWDDataset(config.data_path)
        # print(self.dataset[])

    def setup(self, stage):
        
        size = len(self.dataset)
        train_size = int(size * .8)
        test_val_size = int(size * .1)
        
        self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, [train_size, test_val_size, test_val_size])


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
