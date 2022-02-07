import config
import h5py
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch_optimizer as optim
import yaml

class NOGWDDataset(torch.utils.data.Dataset):
    
    x_shape, y_shape, n = None, None, None

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.shapes
        
        for idx, filename in enumerate(self.raw_filenames):
            raw_path = os.path.join(self.raw_dir, filename)
            processed_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
            if not os.path.isfile(processed_path):
                self.process(idx, raw_path)

#     @property
#     def shapes(self):
#         if not self.x_shape:
#             path = os.path.join(self.raw_dir, self.raw_filenames[0])
#             with h5py.File(path, "r") as file:
#                 x = file['/x'][:]
#                 y = file['/y'][:]

#             x_feat = x.shape[0]
#             y_feat = y.shape[0]
        
#             self.n = x.shape[1] * x.shape[2]
        
#             self.x_shape = (self.n, x_feat)
#             self.y_shape = (self.n, y_feat)
        
#         return self.x_shape, self.y_shape
    
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_filenames(self):
        with open(os.path.join(self.root, "filenames.yaml"), "r") as stream:
            filenames = yaml.safe_load(stream)
        return filenames

    # @property
    # def processed_filenames(self):
    #     return [f"{idx}.pt" for idx in range(len(self))]

    def __len__(self):
        return len(self.raw_filenames * self.n)

    def __getitem__(self, idx):
        file_num = idx // self.n
        i = idx % self.n
        x, y = torch.load(os.path.join(self.processed_dir, f'{file_num}.pt'))
        return x[i, :], y[i, :]

    def process(self, idx, path):
        out_path = os.path.join(self.processed_dir, self.processed_filenames[idx])
        os.makedirs(self.processed_dir, exist_ok=True)
        with h5py.File(path, "r") as file:
            x = file['/x'][:]
            y = file['/y'][:]
            
        x_ = torch.tensor(x).reshape(self.x_shape).T
        y_ = torch.tensor(y).reshape(self.y_shape).T
        
        # data = torch.stack((x_, y_))
        
        shard_size = 128
        
        
        torch.save((x_, y_), out_path)


@DATAMODULE_REGISTRY
class NOGWDDataModule(pl.LightningDataModule):
    """
    """

    def __init__(self, batch_size: int, num_workers: int,):
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()

    def prepare_data(self):
        NOGWDDataset(config.data_path)

    def setup(self, stage):
        dataset = NOGWDDataset(config.data_path)
        size = len(dataset)
        train_size = int(size * .8)
        test_val_size = int(size * .1)
        
        self.train, self.val, self.test = torch.utils.data.random_split(dataset, [train_size, test_val_size, test_val_size])


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
