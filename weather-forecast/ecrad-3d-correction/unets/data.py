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

import os.path as osp
from typing import Optional, Tuple

import dask.array as da
import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


class ThreeDCorrectionDataset(Dataset):
    """
    Create a PyTorch dataset for the 3DCorrection use-case.
    Based on preprocessed sharded data (from dataproc).
    Load a shard to get a datum (favor neighbour-preserving access over random access).
    """

    def __init__(self, data_path: str) -> None:
        """
        Create the dataset from preprocessed/sharded data on disk.

        Args:
            data_path (str): Path to the preprocessed data folder, e.g. data/processed/
        """
        super().__init__()

        # Lazily load and assemble chunks.
        self.x = da.from_npy_stack(osp.join(data_path, "x"))
        self.y = da.from_npy_stack(osp.join(data_path, "y"))

        # Load number of data.
        stats = torch.load(osp.join(data_path, "stats.pt"))
        self.n_data = stats["x_nb"].item()

    def __getitem__(self, i: int) -> Tuple[np.ndarray]:
        """
        Load data from chunks on disk.
        Return (x, y) as tensors.

        Args:
            i (int): Index of datum to load.
        """
        x = self.x[i].compute()
        y = self.y[i].compute()
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y

    def __len__(self) -> int:
        return self.n_data


class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """DataModule for the 3dcorrection dataset."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int,
        splitting_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ):
        """
        Args:
            data_path (str): Path containing the preprocessed data (by dataproc).
            batch_size (int): Size of the batches produced by the data loaders.
            num_workers (int): Number of workers used.
            splitting_ratios (tuple): ratios (summing to 1) of (train, val, test) datasets.
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_ratios = splitting_ratios
        self.dataset = None

    def prepare_data(self):
        ThreeDCorrectionDataset(self.data_path)

    def setup(self, stage: Optional[str] = None):

        # Define subsets.
        tr, va, te = self.splitting_ratios
        if (tr + va + te) != 1:
            raise RuntimeError(
                f"The the splitting ratios does not cover the full dataset: {(tr + va + te)} =! 1"
            )
        self.dataset = ThreeDCorrectionDataset(self.data_path)
        length = len(self.dataset)
        idx = list(range(length))
        train_idx = idx[: int(tr * length)]
        val_idx = idx[int(tr * length) : int((tr + va) * length)]
        test_idx = idx[int((tr + va) * length) :]

        # Define samplers.
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)
        self.test_sampler = SubsetRandomSampler(test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
        )
