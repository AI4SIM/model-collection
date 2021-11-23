import config
import h5py
import networkx as nx
import os
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
import torch_optimizer as optim
import yaml

class CombustionDataset(pyg.data.Dataset):
    
    def __init__(self, root, transform=None, pre_transform=None):
        self._grid_shape = None
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        with open(os.path.join(self.root, "filenames.yaml"), "r") as stream:
            filenames = yaml.safe_load(stream)
        return filenames

    @property
    def processed_file_names(self):
        return [f"data-{idx}.pt" for idx in range(self.len())]
    
    @property
    def grid_shape(self):
        if not self._grid_shape:
            with h5py.File(self.raw_paths[0], 'r') as file:
                field_val = file['filt_8'][:]
            x_size, y_size, z_size = field_val.shape[0], field_val.shape[1], field_val.shape[2]
            self._grid_shape = (z_size, y_size, x_size)
        return self._grid_shape

    def download(self):
        raise RuntimeError(
            'Data not found. Please download the data at {} and move all files in file.tgz/DATA in {}'.format(
                'https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/access/JUDAC/dc4eef36-1929-41f6-9eb9-c11417be1dcf',
                self.raw_dir))

    def process(self):
        graph = pyg.utils.convert.from_networkx(
            nx.grid_graph(dim=self.grid_shape)
        )
        undirected_index = graph.edge_index
        
        i = 0
        for raw_path in self.raw_paths:
            
            with h5py.File(raw_path, 'r') as file:
                c = file["/filt_8"][:]
                sigma = file["/filt_grad_8"][:]
                
            data = pyg.data.Data(
                x=torch.tensor(c.reshape(-1,1), dtype=torch.float), 
                edge_index=torch.tensor(undirected_index, dtype=torch.long),
                y=torch.tensor(sigma.reshape(-1,1), dtype=torch.float)
            )

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
    
    
class LitCombustionDataModule(pl.LightningDataModule):
    
    def __init__(self, args):
        self.args = args
        super().__init__()
        
    # TODO: change this so that the dataset is only built and loaded once
    @property
    def grid_shape(self):
        return CombustionDataset(config.data_path).grid_shape
        
    def prepare_data(self):
        CombustionDataset(config.data_path)
    
    def setup(self, stage):
        dataset = CombustionDataset(config.data_path).shuffle()
        
        self.test_dataset = dataset[119:]
        self.val_dataset = dataset[111:119]
        self.train_dataset = dataset[:111]
    
    def train_dataloader(self):
        return pyg.loader.DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return pyg.loader.DataLoader(self.val_dataset, batch_size=args.batch_size)
    
    def test_dataloader(self):
        return pyg.loader.DataLoader(self.test_dataset, batch_size=args.batch_size)