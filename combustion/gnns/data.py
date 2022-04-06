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

import h5py
import networkx as nx
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch_geometric as pyg
from typing import List
import yaml

import config


class CombustionDataset(pyg.data.Dataset):
    """
    Creates graphs usable for GNNs using the standard PyG data structure. Each graph is built with
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
        """
        Returns the graph at the given index.
        
        Returns:
            (pyg.data.Data): Graph at the given index.
        """
        data = torch.load(os.path.join(self.processed_dir, 'data-{}.pt'.format(idx)))
        return data
    
    def len(self) -> int:
        """
        Returns the total length of the dataset
        
        Returns:
            (int): Dataset length.
        """
        return len(self.raw_file_names)

    

class R2Dataset(CombustionDataset) :
    
    def __init__(self, root: str, y_normalizer: float = None):
        """
        Creates the Dataset.
        
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
            with h5py.File(raw_path, 'r') as file:
                c = file["/c_filt"][:]
                
                sigma = file["/c_grad_filt"][:]
                if self.y_normalizer:
                    sigma /= self.y_normalizer
                
            x_size, y_size, z_size = c.shape

            grid_shape = (z_size, y_size, x_size)

            g0 = nx.grid_graph(dim=grid_shape)
            graph = pyg.utils.convert.from_networkx(g0)
            undirected_index = graph.edge_index
            coordinates = list(g0.nodes())
            coordinates.reverse()

            data = pyg.data.Data(
                x=torch.tensor(feat.reshape(-1, 1), dtype=torch.float),
                edge_index=torch.tensor(undirected_index, dtype=torch.long),
                pos=torch.tensor(np.stack(coordinates)),
                y=torch.tensor(sigma.reshape(-1, 1), dtype=torch.float)
            )

            torch.save(data, os.path.join(self.processed_dir, f'data-{i}.pt'))
            i += 1
            
            
class CnfDataset(CombustionDataset) :
    
    def __init__(self, root: str, y_normalizer: float = None):
        """
        Creates the Dataset.
        
        Args:
            root (str): Path to the root data folder.
            y_normalizer (str): normalizing value
        """
        
        super().__init__(root, y_normalizer)
        
        
    
    def process(self) -> None:
        """
        Creates a graph for each volume of data, and saves each graph in a separate file index by the order in the raw file names list.
        """
        
        i = 0
        for raw_path in self.raw_paths:
            
            with h5py.File(raw_path, 'r') as file:
                c = file["/filt_8"][:]
                
                sigma = file["/filt_grad_8"][:]
                if self.y_normalizer:
                    sigma /= self.y_normalizer
                        
            x_size, y_size, z_size = c.shape
            grid_shape = (z_size, y_size, x_size)
            
            g0 = nx.grid_graph(dim=grid_shape)
            graph = pyg.utils.convert.from_networkx(g0)
            undirected_index = graph.edge_index
            coordinates = list(g0.nodes())
            coordinates.reverse()
                
            data = pyg.data.Data(
                x=torch.tensor(c.reshape(-1,1), dtype=torch.float), 
                edge_index=torch.tensor(undirected_index, dtype=torch.long),
                pos=torch.tensor(np.stack(coordinates)),
                y=torch.tensor(sigma.reshape(-1,1), dtype=torch.float)
            )

            torch.save(data, os.path.join(self.processed_dir, 'data-{}.pt'.format(i)))
            i += 1


@DATAMODULE_REGISTRY
class LitCombustionDataModule(pl.LightningDataModule):
    """
    Creates train, val, and test splits for the CNF Combustion UC—current setup being (111, 8, 8).
    Training set is randomized.
    """
    
    def __init__(self, 
                 batch_size: int, 
                 num_workers: int, 
                 y_normalizer: float) -> None:
        """

        Args:
            batch_size (int): Batch size.
            num_workers (int): DataLoader number of workers for loading data.
            y_normalizer (float): Normalizing value.
            data_path (str): path to raw data.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.y_normalizer = y_normalizer
        self.local_raw_data = os.path.join(config.data_path , 'raw')
        
        super().__init__()

        # init the attribute
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataset = None

    def prepare_data(self) -> None:
        """Not used."""
        CombustionDataset(config.data_path, self.y_normalizer)

    def setup(self, data_path=config.data_path) -> None:
        """Create the main Dataset and splits the train, test and validation Datasets from the main

        Dataset. Currently the repartition is respectively, 80%, 10% and 10% from the main Dataset
        size. Creates symbolic links from origin data.

        Args:
            data_path (str): Path of the data.

        Raises:
            ValueError: if the main dataset is too small and leads to have an empty dataset.
        """

        config.LinkRawData(raw_data_path, data_path)
        
        dataset = R2Dataset(data_path, y_normalizer=self.y_normalizer).shuffle()


        dataset_size = len(dataset)

        self.val_dataset = dataset[int(dataset_size * 0.9):]
        self.test_dataset = dataset[int(dataset_size * 0.8):int(dataset_size * 0.9)]
        self.train_dataset = dataset[:int(dataset_size * 0.8)]

        if not (self.val_dataset and self.test_dataset and self.train_dataset):
            raise ValueError("The dataset is too small to be split properly. "
                             f"Current length is : {dataset_size}.")
        
        

    def train_dataloader(self) -> pyg.loader.DataLoader:
        """Return the train DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Train DataLoader.
        """
        return pyg.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self) -> pyg.loader.DataLoader:
        """Return the validation DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Validation DataLoader.
        """
        return pyg.loader.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self) -> pyg.loader.DataLoader:
        """Return the test DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Test DataLoader.
        """
        return pyg.loader.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
