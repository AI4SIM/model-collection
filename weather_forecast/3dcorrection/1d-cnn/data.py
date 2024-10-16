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

import dask
import os
import torch

import climetlab as cml
import os.path as osp
import numpy as np
import pytorch_lightning as pl
import xarray as xr

from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Tuple, Optional

import config
from dataproc import ThreeDCorrectionDataProc 


class ThreeDCorrectionDataset(Dataset):
    """
    Create a PyTorch dataset for the 3DCorrection use-case.
    Based on preprocessed sharded data (from dataproc).
    Load a shard to get a datum (favor neighbour-preserving access over random access).
    """
    
    dask.config.set(scheduler='synchronous')

    def __init__(self, root: str, ds_feat: xr.Dataset, ds_targ: xr.Dataset) -> None:
        """
        Create the dataset from preprocessed/sharded data on disk.
        Args:
            root (str): Path to the preprocessed data folder.
            ds_feat (xr.Dataset): Feature dataset
            ds_targ (xr.Dataset): Target dataset
        """
        super().__init__()
        self.root = root
        self.ds_feat = ds_feat
        self.ds_targ = ds_targ

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Load data from chunks on disk.
        Return x, y tuple.
        Args:
            idx (int): Index of datum to load.
        """
        feat_sample = self.ds_feat.isel({"column": idx})
        targ_sample = self.ds_targ.isel({"column": idx})
        
        features = {k: torch.from_numpy(v.to_numpy()) for k, v in feat_sample.items()}
        targets = {k: torch.from_numpy(v.to_numpy()) for k, v in targ_sample.items()}
        
        return features, targets

    def __len__(self) -> int:
        return self.ds_feat.dims["column"]
    
@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """DataModule for the 3dcorrection dataset."""

    def __init__(self,
                 date: str,
                 timestep: int,
                 patchstep: int,
                 batch_size: int,
                 num_workers: int,
                 splitting_lengths: Optional[list] = None):
        """
        Args:
            data_path (str): Path containing the preprocessed data (by dataproc).
            batch_size (int): Size of the batches produced by the data loaders.
            num_workers (int): Number of workers used.
            splitting_lengths (list): List of lengths of train, val and test dataset.
        """
        super().__init__()

        self.date = date
        self.timestep = timestep
        self.patchstep = patchstep
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_lengths = splitting_lengths

    def prepare_data(self):
        dataproc = ThreeDCorrectionDataProc(config.data_path,
                                            self.date,
                                            self.timestep,
                                            self.patchstep)
        self.feature_ds, self.target_ds = dataproc.process()

    def setup(self, stage: Optional[str] = None):
        self.dataset = ThreeDCorrectionDataset(config.data_path, self.feature_ds, self.target_ds)
        length = len(self.dataset)
        
        if stage == "fit":
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [int(length * split) for split in self.splitting_lengths])
        
        if stage == "test":
            self.test_dataset = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)