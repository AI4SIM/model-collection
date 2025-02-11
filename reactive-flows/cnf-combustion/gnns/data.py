"""This module proposes Pytorch style Dataset classes for the gnn use-case"""

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
from typing import List, Tuple

import h5py
import lightning as pl
import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
import yaml
from torch.utils.data import random_split


class CombustionDataset(pyg.data.Dataset):
    """Create graphs usable for GNNs using the standard PyG data structure. Each graph is built with
    x = c and y = sigma, and is a regular cartesian grid with uniform connexions. Each graph is
    serialized on disk using standard PyTorch API.

    Please refer to https://pytorch-geometric.readthedocs.io/en/latest for more information about
    the Dataset creation.
    """

    def __init__(self, root: str, y_normalizer: float = None) -> None:
        """Create the Dataset.

        Args:
            root (str): Path to the root data folder.
            y_normalizer (str): normalizing value
        """
        self.y_normalizer = y_normalizer
        super().__init__(root)

    @property
    def raw_file_names(self) -> List[str]:
        """Return the raw data file names.

        Returns:
            (List[str]): Raw data file names list.
        """
        with open(os.path.join(self.root, "filenames.yaml"), "r") as stream:
            filenames = yaml.safe_load(stream)
        return filenames

    @property
    def processed_file_names(self) -> List[str]:
        """Return the processed data file names.

        Returns:
            (List[str]): Processed data file names list.
        """
        return [f"data-{idx}.pt" for idx in range(self.len())]

    def download(self) -> None:
        """Raise a RunetimeError."""
        raise RuntimeError(
            f"Data not found. Please download the data at {'https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/access/JUDAC/dc4eef36-1929-41f6-9eb9-c11417be1dcf'} "  # noqa: E501 line too long
            f"and move all files in file.tgz/DATA in {self.raw_dir}"
        )

    def get(self, idx: int) -> pyg.data.Data:
        """Return the graph at the given index.

        Returns:
            (pyg.data.Data): Graph at the given index.
        """
        data = torch.load(os.path.join(self.processed_dir, f"data-{idx}.pt"))
        return data

    def len(self) -> int:
        """Return the total length of the dataset

        Returns:
            (int): Dataset length.
        """
        return len(self.raw_file_names)


class R2Dataset(CombustionDataset):
    """Class to process data for the R2 Use case."""

    def __init__(self, root: str, y_normalizer: float = None):
        """Create the Dataset.

        Args:
            root (str): Path to the root data folder.
            y_normalizer (str): normalizing value
        """
        super().__init__(root, y_normalizer)

    def process(self) -> None:
        """
        Create a graph for each volume of data, and saves each graph in a separate file index by
        the order in the raw file names list.
        """
        i = 0
        for raw_path in self.raw_paths:
            with h5py.File(raw_path, "r") as file:
                feat = file["/c_filt"][:]

                sigma = file["/c_grad_filt"][:]
                if self.y_normalizer:
                    sigma /= self.y_normalizer

            x_size, y_size, z_size = feat.shape

            grid_shape = (z_size, y_size, x_size)

            g0 = nx.grid_graph(dim=grid_shape)
            graph = pyg.utils.convert.from_networkx(g0)
            undirected_index = graph.edge_index
            coordinates = list(g0.nodes())
            coordinates.reverse()

            data = pyg.data.Data(
                x=torch.tensor(feat.reshape(-1, 1), dtype=torch.float),
                edge_index=undirected_index.clone().detach().type(torch.LongTensor),
                pos=torch.tensor(np.stack(coordinates)),
                y=torch.tensor(sigma.reshape(-1, 1), dtype=torch.float),
            )

            torch.save(data, os.path.join(self.processed_dir, f"data-{i}.pt"))
            i += 1


class CnfDataset(CombustionDataset):
    """Class to process data for the CNF Use case."""

    def __init__(self, root: str, y_normalizer: float = None):
        """Create the Dataset.

        Args:
            root (str): Path to the root data folder.
            y_normalizer (str): normalizing value.
        """
        super().__init__(root, y_normalizer)

    def process(self) -> None:
        """
        Create a graph for each volume of data, and saves each graph in a separate file index by
        the order in the raw file names list.
        """
        i = 0
        for raw_path in self.raw_paths:
            with h5py.File(raw_path, "r") as file:
                feat = file["/filt_8"][:]

                sigma = file["/filt_grad_8"][:]
                if self.y_normalizer is not None:
                    sigma /= self.y_normalizer

            x_size, y_size, z_size = feat.shape
            grid_shape = (z_size, y_size, x_size)

            g0 = nx.grid_graph(dim=grid_shape)
            graph = pyg.utils.convert.from_networkx(g0)
            undirected_index = graph.edge_index
            coordinates = list(g0.nodes())
            coordinates.reverse()

            data = pyg.data.Data(
                x=torch.tensor(feat.reshape(-1, 1), dtype=torch.float),
                edge_index=undirected_index.type(torch.LongTensor),
                pos=torch.tensor(np.stack(coordinates)),
                y=torch.tensor(sigma.reshape(-1, 1), dtype=torch.float),
            )

            torch.save(data, os.path.join(self.processed_dir, f"data-{i}.pt"))
            i += 1


class LitCombustionDataModule(pl.LightningDataModule):
    """
    Create train, val, and test splits for the CNF Combustion UC—current setup being (111, 8, 8).
    Training set is randomized.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        y_normalizer: float,
        data_path: str,
        splitting_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        source_raw_data_path: str = None,
    ) -> None:
        """Init the LitCombustionDataModule class.

        Args:
            batch_size (int): Batch size.
            num_workers (int): DataLoader number of workers for loading data.
            y_normalizer (float): Normalizing value.
            splitting_ratios (list): ratios of the full dataset for training, validation and testing
                sets.
            data_path (str): path to the data used by the training.
            source_raw_data_path (str): path to raw data that will be symlinked in data_path.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.y_normalizer = y_normalizer
        self.splitting_ratios = splitting_ratios
        self.data_path = data_path
        self.source_raw_data_path = source_raw_data_path

        super().__init__()

        # init the attribute
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataset = None

    def prepare_data(self) -> None:
        """Not used."""
        CombustionDataset(self.data_path, self.y_normalizer)

    def setup(
        self,
        stage: str,
    ) -> None:
        """Create the main Dataset and splits the train, test and validation Datasets from the main
        Dataset. Currently the repartition is respectively, 80%, 10% and 10% from the main Dataset
        size. Creates symbolic links from origin data.

        Args:
            data_path (str): Path of the data.

        Raises:
            ValueError: if the main dataset is too small and leads to have an empty dataset.
        """
        if self.source_raw_data_path:
            LinkRawData(self.source_raw_data_path, self.data_path)

        dataset = R2Dataset(self.data_path, y_normalizer=self.y_normalizer)

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

    def train_dataloader(self) -> pyg.loader.DataLoader:
        """Return the train DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Train DataLoader.
        """
        return pyg.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> pyg.loader.DataLoader:
        """Return the validation DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Validation DataLoader.
        """
        return pyg.loader.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> pyg.loader.DataLoader:
        """Return the test DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Test DataLoader.
        """
        return pyg.loader.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class LinkRawData:
    """Link dataset to the use case."""

    def __init__(self, source_raw_data_path, data_path):
        """Link the source_raw_data_path in the data_path, if it does not already exists."""
        self.source_raw_data_path = source_raw_data_path
        self.local_data_path = data_path
        self.local_raw_data = os.path.join(self.local_data_path, "raw")

        if os.path.exists(self.source_raw_data_path):
            if os.path.exists(self.local_raw_data):
                try:
                    if (
                        len(os.listdir(self.local_raw_data)) == 0
                        or os.readlink(self.local_raw_data) != self.source_raw_data_path
                    ):
                        self.rm_old_dataset()
                        self.symlink_dataset()
                    else:
                        pass
                except OSError:
                    pass
            else:
                self.symlink_dataset()

    def symlink_dataset(self):
        """Create the filenames.yaml file from the content of the source_raw_data_path."""
        filenames = os.listdir(self.source_raw_data_path)
        temp_file_path = os.path.join(self.local_data_path, "filenames.yaml")
        with open(temp_file_path, "w") as file:
            yaml.dump(filenames, file)

        if not os.path.exists(self.local_raw_data):
            os.makedirs(self.local_raw_data, exist_ok=True)

        for filename in filenames:
            os.symlink(
                os.path.join(self.source_raw_data_path, filename),
                os.path.join(self.local_raw_data, filename),
            )

    def rm_old_dataset(self):
        """Clean the local_data_path."""
        for item in ["raw", "filenames.yaml", "processed"]:
            file_location = os.path.join(self.local_data_path, item)
            try:
                os.remove(file_location)
            except IsADirectoryError:
                os.rmdir(file_location)
            else:
                pass
