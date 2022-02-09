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
    Creates graphs usable for GNNs using the standard PyG data structure. Each graph is built with x = c and y = sigma, and is a regular cartesian grid with uniform connexions. Each graph is serialized on disk using standard PyTorch API. Please refer to https://pytorch-geometric.readthedocs.io/en/latest for more information about the Dataset creation.
    """
    
    def __init__(self, root: str, y_normalizer: float = None) -> None:
        """
        Creates the Dataset.
        
        Args:
            root (str): Path to the root data folder.
        """
        self.y_normalizer = y_normalizer
        super().__init__(root)

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the raw data file names.
        
        Returns:
            (List[str]): Raw data file names list.
        """
        with open(os.path.join(self.root, "filenames.yaml"), "r") as stream:
            filenames = yaml.safe_load(stream)
        return filenames

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns the processed data file names.
        
        Returns:
            (List[str]): Processed data file names list.
        """
        return [f"data-{idx}.pt" for idx in range(self.len())]
    
    def download(self) -> None:
        """
        Raises a RunetimeError.
        """
        raise RuntimeError(
            'Data not found. Please download the data at {} and move all files in file.tgz/DATA in {}'.format(
                'https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/access/JUDAC/dc4eef36-1929-41f6-9eb9-c11417be1dcf',
                self.raw_dir))

    def process(self) -> None:
        """
        Creates a graph for each volume of data, and saves each graph in a separate file index by the order in the raw file names list.
        """
        
        i = 0
        for raw_path in self.raw_paths:
            
            with h5py.File(raw_path, 'r') as file:
                c = file["/filt_8"][:]
                
                if self.y_normalizer is not None :
                    sigma = file["/filt_grad_8"][:] / self.y_normalizer
                else :
                    sigma = file["/filt_grad_8"][:]
                
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


@DATAMODULE_REGISTRY
class LitCombustionDataModule(pl.LightningDataModule):
    """
    Creates train, val, and test splits for the CNF Combustion UC—current setup being (111, 8, 8). Training set is randomized.
    """
    
    def __init__(self, batch_size: int, num_workers: int, y_normalizer: float) -> None:
        """
        Args:
            batch_size (int): Batch size.
            num_workers (int): DataLoader number of workers for loading data.
        """
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.y_normalizer = y_normalizer
        super().__init__()
        
    def prepare_data(self) -> None:
        """
        Not used.
        """
        CombustionDataset(config.data_path, self.y_normalizer)
    
    def setup(self, stage: str) -> None:
        """
        Creates the main Dataset and splits the train, val, and test Datasets from the main Dataset. Current split takes first 111 elements in the train.
        
        Args:
            stage (str): Unsed.
        """
        dataset = CombustionDataset(config.data_path, self.y_normalizer).shuffle()
        
        self.test_dataset = dataset[119:]
        self.val_dataset = dataset[111:119]
        self.train_dataset = dataset[:111]
    
    def train_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns the train DataLoader.
        
        Retuns:
            (torch.utils.data.DataLoader): Train DataLoader.
        """
        return pyg.loader.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers)
    
    def val_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns the validation DataLoader.
        
        Retuns:
            (torch.utils.data.DataLoader): Validation DataLoader.
        """
        return pyg.loader.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)
    
    def test_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns the test DataLoader.
        
        Retuns:
            (torch.utils.data.DataLoader): Test DataLoader.
        """
        return pyg.loader.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)