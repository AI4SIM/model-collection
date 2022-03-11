# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import h5py
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
from typing import List, Tuple, Union
import yaml

import config


class NOGWDDataset(torch.utils.data.Dataset):
    """
    Creates the PyTorch Dataset for the Non-Orographic variant of the Gravity Wave Drag (GWD) UC.
    Each raw HDF5 input file contains two datasets—x and y—of 2355840 rows each.
        Input features:
        Output features: .
    Refer to the UC get-started.ipynb notebook for more details on the inputs.
    """
    
    x_feat = 191
    y_feat = 126
    shard_len = 2355840  # FIXME: 'shard_len' is hardcoded, but should be dynamically set.""

    def __init__(self, root: str, mode: Union['train', None]) -> None:
        """
        Creates the Dataset.
        
        Args:
            root (str): Path to the root data folder.
            mode (Union['train', None]): Data processing mode. If 'train', it will compute the stats used by the model in the normalization layer.
        """
        super().__init__()
        print('******************************')
        self.root = root
        self.mode = mode

        self.x, self.y = self.load()
        
        if self.mode == "train":
            self._compute_stats()
    
    def _compute_stats(self) -> None:
        stats = {
            'x_mean': torch.mean(self.x, dim=0, dtype=torch.float32),
            'x_std': torch.std(self.x, dim=0),
            'y_std': torch.std(self.y, dim=0)
        }
        torch.save(stats, os.path.join(self.root, 'stats.pt'))
        
    # TODO: Implement a method to download the data with Climetlab
    def download(self) -> None:
        """
        Downloads the dataset with Climetlab.
        """
        raise NotImplementedError("The 'download' method is not yet available.")
    
    # TODO: Download the data if raw input files are missing in data/raw/
    def load(self) -> Tuple[torch.Tensor]:
        """
        Loads the raw data from HDF5 files into CPU memory, and concatenates the content into a
        single PyTorch Tensor, one for x and one for y.
        
        Returns:
            (torch.Tensor): x input features.
            (torch.Tensor): y output features.
        """
        x_, y_ = [], []
        
        print(self.raw_filenames)
        for filename in self.raw_filenames:
            with h5py.File(os.path.join(self.raw_dir, filename), 'r') as file:
                x = file['/x'][:]
                y = file['/y'][:]
                
                x_.append(np.reshape(x, (191, -1)).T)
                y_.append(np.reshape(y, (126, -1)).T)

        x = torch.tensor(np.concatenate(x_), dtype=torch.float32)
        y = torch.tensor(np.concatenate(y_), dtype=torch.float32)

        return x, y
    
    @property
    def raw_dir(self) -> str:
        """
        Returns the raw data folder.
        
        Returns:
            (str): Raw data folder path.
        """
        return os.path.join(self.root, "raw")

    @property
    def raw_filenames(self) -> List[str]:
        """
        Returns the raw data file names.
        
        Returns:
            (List[str]): Raw data file names list.
        """
        with open(os.path.join(self.root, "filenames-split.yaml"), "r") as stream:
            filenames = yaml.safe_load(stream)
            filenames = filenames[self.mode]
        return filenames

    def __len__(self) -> int:
        """
        Returns the total length of the dataset
        
        Returns:
            (int): Dataset length.
        """
        return len(self.raw_filenames) * self.shard_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Returns the Tensor at the given index.
        
        Returns:
            (tuple of torch.Tensor): the x and y Tensors at the given index.
        """
        return self.x[idx, :], self.y[idx, :]


@DATAMODULE_REGISTRY
class NOGWDDataModule(pl.LightningDataModule):
    """
    Creates a datamodule structure for use in a PyTorch Lightning Trainer. Basically a wrapper around the Dataset. Train, val, test split is given by data/filenames-split.yaml. Training set is randomized.
    """

    def __init__(self, batch_size: int, num_workers: int) -> None:
        """
        Args:
            batch_size (int): Batch size.
            num_workers (int): DataLoader number of workers for loading data.
        """
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self) -> None:
        """
        No used. The download logic is the responsibility for the Dataset.
        """
        pass

    def setup(self, stage: Union['fit', 'test', ]) -> None:
        """
        Args:
            stage (Union['fit', 'test']): Stage for which to setup the DataLoader. If 'fit', it
                will prepare the train and val DataLoaders, otherwise, it will prepare the
                test DataLoader.
        """
        if stage == 'fit':
            self.train = NOGWDDataset(config.data_path, 'train')
            self.val = NOGWDDataset(config.data_path, 'val')
            
        if stage == 'test':
            self.test = NOGWDDataset(config.data_path, 'test')


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the train DataLoader.
        
        Retuns:
            (torch.utils.data.DataLoader): Train DataLoader.
        """
        
        return torch.utils.data.DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=self.num_workers)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the val DataLoader.
        
        Retuns:
            (torch.utils.data.DataLoader): Validation DataLoader.
        """
        
        return torch.utils.data.DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            drop_last=True, 
            num_workers=self.num_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the test DataLoader.
        
        Retuns:
            (torch.utils.data.DataLoader): Test DataLoader.
        """
        
        return torch.utils.data.DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)
