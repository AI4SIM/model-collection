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

import os
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional
import numpy as np
import dask.array as da
import torch

import config
from dataproc import ThreeDCorrectionDataproc

# TODO: add specific test set
# TODO: download ECMWF dataset in prepare_data

class ThreeDCorrectionDataset(Dataset):
    """
    Create a PyTorch dataset for the 3DCorrection use-case.
    Based on preprocessed sharded data (from dataproc).
    Load a shard to get a datum (favor neighbour-preserving access over random access).
    """

    def __init__(self, root: str) -> None:
        """
        Create the dataset from preprocessed/sharded data on disk.

        Args:
            data_path (str): Path to the preprocessed data folder.
        """
        super().__init__()
        self.root = root

        # Lazily load and assemble chunks.
        self.x = da.from_npy_stack(osp.join(self.root, 'processed', 'x'))
        self.y = da.from_npy_stack(osp.join(self.root, 'processed', 'y'))
        self.z = da.from_npy_stack(osp.join(self.root, 'processed', 'z'))

        # Load number of data.
        stats = torch.load(osp.join(self.root, "processed", "stats.pt"))
        self.n_data = stats['x_nb'].item()

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Load data from chunks on disk.
        Return x, y tuple.

        Args:
            i (int): Index of datum to load.
        """
        x, y, z = da.compute(self.x[i], self.y[i], self.z[i])

        return torch.Tensor(x), torch.Tensor(y), torch.Tensor(z)

    def __len__(self) -> int:
        return self.n_data
    
@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """DataModule for the 3dcorrection dataset."""

    def __init__(self,
                 timestep: int,
                 patchstep: int,
                 batch_size: int,
                 num_workers: int,
                 splitting_lengths: list):
        """
        Args:
            data_path (str): Path containing the preprocessed data (by dataproc).
            batch_size (int): Size of the batches produced by the data loaders.
            num_workers (int): Number of workers used.
            splitting_lengths (list): List of lengths of train, val and test dataset.
        """
        super().__init__()

        self.timestep = timestep
        self.patchstep = patchstep
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_lengths = splitting_lengths

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dataset = ThreeDCorrectionDataset(config.data_path)
        length = len(self.dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [int(length * split) for split in self.splitting_lengths])

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
