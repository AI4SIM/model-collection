import config
import h5py
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
        
        for idx, filename in enumerate(self.raw_filenames):
            raw_path = os.path.join(self.raw_dir, filename)
            processed_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
            if not os.path.isfile(processed_path):
                self.process(raw_path)
    
    
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
        file_num = idx // self.shard_len
        i = idx % self.shard_len
        filenames = self.processed_filenames
        path = os.path.join(self.processed_dir, self.processed_filenames[file_num])
        with h5py.File(path, 'r') as file:
            x = torch.tensor(file['/x'][i])
            y = torch.tensor(file['/y'][i])
            
        return x, y

    def process(self, path):
        _, filename = os.path.split(path)
        out_path = os.path.join(self.processed_dir, filename)
        os.makedirs(self.processed_dir, exist_ok=True)
        with h5py.File(path, "r") as file:
            x = file['/x'][:]
            y = file['/y'][:]
            
        x_ = np.reshape(x, (self.x_feat, -1)).T
        y_ = np.reshape(y, (self.y_feat, -1)).T
        
        with h5py.File(out_path, 'w') as file:
            file['/x'] = x_
            file['/y'] = y_


@DATAMODULE_REGISTRY
class NOGWDDataModule(pl.LightningDataModule):
    """
    """

    def __init__(self, batch_size: int, num_workers: int,):
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()

    def prepare_data(self):
        NOGWDDataset(config.data_path)

    def setup(self, stage):
        dataset = NOGWDDataset(config.data_path)
        size = len(dataset)
        train_size = int(size * .8)
        test_val_size = int(size * .1)
        
        self.train, self.val, self.test = torch.utils.data.random_split(dataset, [train_size, test_val_size, test_val_size])


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
