import h5py
import networkx as nx
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_optimizer as optim
import yaml
import netCDF4
from tqdm import tqdm

import config

class ThreeDCorrectionDataset(pyg.data.InMemoryDataset):
    
    def __init__(self, root, step, force=False):
        self.step = step
        
        super().__init__(root)
        
        path = self.processed_paths[0]
        if force:
            os.remove(path)
            
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [f"data-{self.step}.nc"]

    @property
    def processed_file_names(self):
        return [f"data-{self.step}.pt"]
    
    def download(self):
        import climetlab as cml

        cml.settings.set("cache-directory", os.path.join(self.root, "raw"))

        cmlds = cml.load_dataset(
            'maelstrom-radiation', 
            dataset='3dcorrection', 
            raw_inputs=False, 
            timestep=list(range(0, 3501, self.step)), 
            minimal_outputs=False,
            patch=list(range(0, 16, 1)),
            hr_units='K d-1',
        )
        
        array = cmlds.to_xarray()
        array.to_netcdf(self.raw_paths[0])

    def process(self):
        
        def broadcast_features(tensor):
            t = torch.unsqueeze(tensor, -1)
            t = t.repeat((1, 1, 138))
            t = t.moveaxis(1, -1)
            return t

        def pad_tensor(tensor):
            return F.pad(tensor, (0, 0, 1, 1, 0, 0))
        
        
        i = 0
        for raw_path in self.raw_paths:
            
            with netCDF4.Dataset(raw_path, "r", format="NETCDF4") as file:
                sca_inputs = torch.tensor(file['sca_inputs'][:])
                col_inputs = torch.tensor(file['col_inputs'][:])
                hl_inputs = torch.tensor(file['hl_inputs'][:])
                inter_inputs = torch.tensor(file['inter_inputs'][:])

                flux_dn_sw = torch.tensor(file['flux_dn_sw'][:])
                flux_up_sw = torch.tensor(file['flux_up_sw'][:])
                flux_dn_lw = torch.tensor(file['flux_dn_lw'][:])
                flux_up_lw = torch.tensor(file['flux_up_lw'][:])
                
                hr_sw = torch.tensor(file["hr_sw"][:])
                hr_lw = torch.tensor(file["hr_lw"][:])
                
            # inter_inputs_ = pad_tensor(inter_inputs)
            sca_inputs_ = broadcast_features(sca_inputs)

            x = torch.cat([
                hl_inputs,
                inter_inputs_,
                sca_inputs_
            ], dim=-1)

            y = torch.cat([
                torch.unsqueeze(flux_dn_sw, -1),
                torch.unsqueeze(flux_up_sw, -1),
                torch.unsqueeze(flux_dn_lw, -1),
                torch.unsqueeze(flux_up_lw, -1),
            ], dim=-1)
            
            stats_path = os.path.join(self.root, f"stats-{self.step}.pt")
            if not os.path.isfile(stats_path):
                stats = {
                    "x_mean" : torch.mean(x, dim=0),
                    "y_mean" : torch.mean(y, dim=0),
                    "x_std" : torch.std(x, dim=0),
                    "y_std" : torch.std(y, dim=0)
                }
                torch.save(stats, stats_path)

            directed_index = np.array([[*range(1, 138)], [*range(137)]])
            undirected_index = np.hstack((
                directed_index, 
                directed_index[[1, 0], :]
            ))
            undirected_index = torch.tensor(undirected_index, dtype=torch.long)
            
            data_list = []
            
            for idx in tqdm(range(x.shape[0])):
                x_ = torch.squeeze(x[idx, ...])
                y_ = torch.squeeze(y[idx, ...])
                
                edge_attr = torch.squeeze(sca_inputs_[idx, ...])
                
                data = pyg.data.Data(
                    x=x_,
                    edge_attr=edge_attr,
                    edge_index=undirected_index,
                    y=y_,
                )
                
                data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[0])


@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """Creates datasets for the """
    
    def __init__(self, 
                 timestep: int,
                 batch_size: int,
                 num_workers: int,
                 force: bool = False):
        
        self.step = timestep
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force = force
        super().__init__()
        
    def prepare_data(self):
        ThreeDCorrectionDataset(config.data_path, self.step, self.force)
    
    def setup(self, stage):
        dataset = ThreeDCorrectionDataset(config.data_path, self.step, self.force).shuffle()
        length = dataset.len()
        
        self.test_dataset = dataset[int(0.9*length):]
        self.val_dataset = dataset[int(0.8*length):int(0.9*length)]
        self.train_dataset = dataset[:int(0.8*length)]
    
    def train_dataloader(self):
        return pyg.loader.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return pyg.loader.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return pyg.loader.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
