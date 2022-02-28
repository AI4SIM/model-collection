'''
    Licensed under the Apache License, Version 2.0 (the "License");
    * you may not use this file except in compliance with the License.
    * You may obtain a copy of the License at
    *
    *     http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing, software
    * distributed under the License is distributed on an "AS IS" BASIS,
    * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    * See the License for the specific language governing permissions and
    * limitations under the License.
'''

import config as cfg
import h5py
import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
from typing import Union


class RandomCropper3D(object):
    """Randomly crop a sub-block out of a 3D tensor.

    Args:
        out_shape (tuple or int): desired output shape.
    """

    def __init__(self, out_shape: Union[int, tuple]):
        assert isinstance(out_shape, (int, tuple))
        if isinstance(out_shape, int):
            self.out_shape = (out_shape, out_shape, out_shape)
        else:
            assert len(out_shape) == 3
            self.out_shape = out_shape

    def __call__(self, x, y):
        h, w, d = x.shape[0], x.shape[1], x.shape[2]
        block_h, block_w, block_d = self.out_shape
        tx = np.random.randint(0, h - block_h)
        ty = np.random.randint(0, w - block_w)
        tz = np.random.randint(0, d - block_d)
        x_cropped = x[tx:tx+block_h, ty:ty+block_w, tz:tz+block_d]
        y_cropped = y[tx:tx+block_h, ty:ty+block_w, tz:tz+block_d]
        return x_cropped, y_cropped


class CnfCombustionDataset(torch.utils.data.Dataset):

    def __init__(self, root,  y_normalizer: float = None, subblock_shape: Union[int, tuple] = None):
        super().__init__()
        self.root = root
        self.y_normalizer = y_normalizer
        self.random_cropper = RandomCropper3D(subblock_shape) if subblock_shape is not None else None
        for idx, filename in enumerate(self.raw_filenames):
            raw_path = os.path.join(self.raw_dir, filename)
            processed_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
            if not os.path.isfile(processed_path):
                self.process(idx, raw_path)

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_filenames(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_filenames(self):
        return [f"{filename.split('.')[0]}.pt" for filename in self.raw_filenames]

    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.processed_dir, self.processed_filenames[idx]))

    def process(self, idx, path) -> None:
        out_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
        os.makedirs(self.processed_dir, exist_ok=True)
        with h5py.File(path, "r") as file:
            c = torch.tensor(file["/filt_8"][:])
            sigma = torch.tensor(file["/filt_grad_8"][:])  # Or try with surrogate 'grad_filt_8'.
        c = c[1:, 1:, 1:]  # Crop first boundary (equal to the opposite one).
        sigma = sigma[1:, 1:, 1:]
        if self.random_cropper is not None:  # Randomly crop a subblock.
            c, sigma = self.random_cropper(c, sigma)
        if self.y_normalizer is not None:  # Normalize target.
            sigma = sigma / self.y_normalizer
        c = c[None, :]  # Add a dummy axis for channels.
        sigma = sigma[None, :]
        torch.save((c, sigma), out_path)


@DATAMODULE_REGISTRY
class CnfCombustionDataModule(pl.LightningDataModule):
    """
    Providing 3D blocks of the CNF combustion dataset.

    Args:
        splitting_lengths (list): lengths of training, validation and testing sets.
        shuffling: whether to shuffle the trainset.
        subblock_shape (int or tuple): data augmentation by randomly cropping sub-blocks.
    """

    def __init__(self, batch_size: int, num_workers: int, splitting_lengths: list,
            shuffling: bool = False, y_normalizer: float = None, subblock_shape: Union[int, tuple] = None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_lengths = splitting_lengths
        self.shuffling = shuffling
        self.y_normalizer = y_normalizer
        self.subblock_shape = subblock_shape
        super().__init__()

    def prepare_data(self):
        CnfCombustionDataset(cfg.data_path,  self.y_normalizer, self.subblock_shape)

    def setup(self, stage = None):
        dataset = CnfCombustionDataset(cfg.data_path,  self.y_normalizer, self.subblock_shape)
        self.train, self.val, self.test = torch.utils.data.random_split(dataset, self.splitting_lengths)
        if self.shuffling: self.train = self.train.shuffle()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
