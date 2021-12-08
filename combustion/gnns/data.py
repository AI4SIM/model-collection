import config as cfg
import h5py
import networkx as nx
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch_geometric as pyg
import torch_optimizer as optim
import yaml

class CombustionDataset(pyg.data.Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        with open(os.path.join(self.root, "filenames.yaml"), "r") as stream:
            filenames = yaml.safe_load(stream)
        return filenames

    @property
    def processed_file_names(self):
        return [f"data-{idx}.pt" for idx in range(self.len())]

    def download(self):
        raise RuntimeError(
            'Data not found. Please download the data at {} and move all files in file.tgz/DATA in {}'.format(
                'https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/access/JUDAC/dc4eef36-1929-41f6-9eb9-c11417be1dcf',
                self.raw_dir))

    def process(self):

        i = 0
        for raw_path in self.raw_paths:

            with h5py.File(raw_path, 'r') as file:
                c = file["/filt_8"][:]
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
                y=torch.tensor(sigma.reshape(-1,1), dtype=torch.float))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data-{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data-{}.pt'.format(idx)))
        return data

    def len(self):
        return len(self.raw_file_names)


@DATAMODULE_REGISTRY
class LitCombustionDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int):
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()

    def prepare_data(self):
        CombustionDataset(cfg.data_path)

    def setup(self, stage):
        dataset = CombustionDataset(cfg.data_path).shuffle()
        self.test_dataset = dataset[119:]
        self.val_dataset = dataset[111:119]
        self.train_dataset = dataset[:111]

    def train_dataloader(self):
        return pyg.loader.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return pyg.loader.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return pyg.loader.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
