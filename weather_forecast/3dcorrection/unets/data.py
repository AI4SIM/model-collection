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
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np
import dask.array as da

import config


class ThreeDCorrectionShardedDataset(Dataset):
    """
    Create a PyTorch dataset for the 3DCorrection use-case.
    Based on sharded data.
    """

    def __init__(self,
                 root: str,
                 timestep: int = 500,
                 patchstep: int = 1,
                 force: bool = False) -> None:
        """
        Create the dataset from preprocessed/sharded data on disk.

        Args:
            root (str): Path to the root data folder.
            timestep (int): Increment between two outputs (time increment is 12min so
                step 125 corresponds to an output every 25h).
            patchstep (int): Step of the patchs (16 Earth's regions)
            force (bool): Either to remove the processed files or not.
        """
        super().__init__()

        self.root = root
        self.timestep = timestep
        self.patchstep = patchstep

        # Recover the shard size from the data.
        x = np.load(osp.join(self.root, f'features-{self.timestep}', 'x', "0.npy"))
        self.shard_size = x.shape[0]

        path = self.processed_paths[0]
        if force:
            os.remove(path)

    @property
    def raw_file_names(self) -> List[str]:
        """Return the raw data file names."""
        return [f"data-{self.timestep}.nc"]  # NetCDF file.

    @property
    def processed_file_names(self) -> List[str]:
        """Return the processed data file names."""
        return [f"data-{self.timestep}.pt"]

    def __getitem__(self, i: int) -> Tuple[da.Array]:
        """
        Load data from chunks on disk.

        Args:
            i (int): Index of datum to load.
        """

        step_dir = osp.join(self.root, f'features-{self.timestep}')
        shard_idx = i // self.shard_size
        idx_in_shard = i % self.shard_size
        x = np.load(osp.join(step_dir, 'x', f"{shard_idx}.npy"))
        y = np.load(osp.join(step_dir, 'y', f"{shard_idx}.npy"))
        return (x[idx_in_shard], y[idx_in_shard])

    def __len__(self) -> int:
        return os.listdir(self.root) // 2


@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """DataModule for the 3dcorrection dataset."""

    def __init__(self,
                 timestep: int,
                 batch_size: int,
                 num_workers: int,
                 force: bool = False):

        self.timestep = timestep
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force = force
        super().__init__()

    def prepare_data(self):
        ThreeDCorrectionShardedDataset(config.data_path, self.timestep, self.force)

    def setup(self, stage):
        ds = ThreeDCorrectionShardedDataset(
            config.data_path,
            self.timestep,
            self.force).shuffle()
        self.test_dataset = ds[int(0.9 * ds.len()):]
        self.val_dataset = ds[int(0.8 * ds.len()):int(0.9 * ds.len())]
        self.train_dataset = ds[:int(0.8 * ds.len())]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
