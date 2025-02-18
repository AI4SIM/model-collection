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

from os import listdir, makedirs
from os.path import isfile, join
from typing import Optional, Union, Tuple

from h5py import File
from lightning import LightningDataModule
from torch import load, save, tensor
from torch.utils.data import DataLoader, Dataset, random_split

from utils import RandomCropper3D


class CnfCombustionDataset(Dataset):

    def __init__(
        self,
        root: str,
        y_normalizer: float = None,
        subblock_shape: Union[int, tuple] = None,
    ):
        super().__init__()
        self.root = root
        self.y_normalizer = y_normalizer
        self.random_cropper = (
            RandomCropper3D(subblock_shape) if subblock_shape is not None else None
        )
        for i, filename in enumerate(self.raw_filenames):
            raw_path = join(self.raw_dir, filename)
            processed_path = join(self.processed_dir, self.processed_filenames[i])
            if not isfile(processed_path):
                self.process(i, raw_path)

    @property
    def raw_dir(self):
        return join(self.root, "raw")

    @property
    def processed_dir(self):
        return join(self.root, "processed")

    @property
    def raw_filenames(self):
        return listdir(self.raw_dir)

    @property
    def processed_filenames(self):
        return [f"{filename.split('.')[0]}.pt" for filename in self.raw_filenames]

    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self, i):
        return load(join(self.processed_dir, self.processed_filenames[i]))

    def process(self, i, path) -> None:
        makedirs(self.processed_dir, exist_ok=True)
        with File(path, "r") as file:
            c = tensor(file["/filt_8"][:])
            sigma = tensor(
                file["/filt_grad_8"][:]
            )  # Or try with surrogate 'grad_filt_8'.

        # Crop first boundary (equal to the opposite one).
        c = c[1:, 1:, 1:]
        sigma = sigma[1:, 1:, 1:]

        # Randomly crop a subblock.
        if self.random_cropper is not None:
            c, sigma = self.random_cropper(c, sigma)

        if self.y_normalizer is not None:
            sigma = sigma / self.y_normalizer

        # Add a dummy axis for channels.
        c = c[None, :]
        sigma = sigma[None, :]
        save((c, sigma), join(self.processed_dir, self.processed_filenames[i]))


class CnfCombustionDataModule(LightningDataModule):
    """
    Providing 3D blocks of the CNF combustion dataset.

    Args:
        splitting_ratios (list): ratios of the full dataset for training, validation and testing
            sets.
        shuffling: whether to shuffle the trainset.
        subblock_shape (int or tuple): data augmentation by randomly cropping sub-blocks.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        splitting_ratios: Tuple[float, float, float],
        data_path: str,
        shuffling: bool = False,
        y_normalizer: float = None,
        subblock_shape: Union[int, tuple] = None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_ratios = splitting_ratios
        self.data_path = data_path
        self.shuffling = shuffling
        self.y_normalizer = y_normalizer
        self.subblock_shape = subblock_shape
        super().__init__()

    def prepare_data(self):
        """Initialize dataset."""
        CnfCombustionDataset(self.data_path, self.y_normalizer, self.subblock_shape)

    def setup(self, stage: Optional[str] = None):
        """Preprocessing: splitting and shuffling."""
        dataset = CnfCombustionDataset(
            self.data_path, self.y_normalizer, self.subblock_shape
        )

        tr, va, te = self.splitting_ratios
        if (tr + va + te) != 1:
            raise RuntimeError(
                f"The the splitting ratios does not cover the full dataset: {(tr + va + te)} =! 1"
            )
        length = len(dataset)
        idx = list(range(length))
        train_size = len(idx[: int(tr * length)])
        val_size = len(idx[int(tr * length) : int((tr + va) * length)])
        test_size = len(idx[int((tr + va) * length) :])

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        if not (self.val_dataset and self.test_dataset and self.train_dataset):
            raise ValueError(
                "The dataset is too small to be split properly. "
                f"Current length is : {length}."
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
