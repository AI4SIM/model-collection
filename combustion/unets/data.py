import config
import h5py
import networkx as nx
import os
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
import torch_optimizer as optim
import yaml

class CombustionDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        
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
        return [f"data-{idx}.pt" for idx in range(len(self))]
    
    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.processed_dir, self.processed_filenames[idx]))
        
    def process(self, idx, path):
        out_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
        with h5py.File(path, "r") as file:
            c = torch.tensor(file["/filt_8"][:])
            sigma = torch.tensor(file["/filt_grad_8"][:])
        torch.save((c, sigma), out_path)
    
    
class LitCombustionDataModule(pl.LightningDataModule):
    
    def __init__(self, args):
        self.args = args
        super().__init__()
        
    def prepare_data(self):
        CombustionDataset(config.data_path)
    
    def setup(self, stage):
        dataset = CombustionDataset(config.data_path)#.shuffle()
        
        self.train, self.val, self.test = torch.utils.data.random_split(dataset, [111, 8, 8])
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=1, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=1)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=1)