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

import config
import h5py
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_optimizer as optim
import torchmetrics.functional as F


class NOGWDModule(pl.LightningModule):
    """
    Contains the mutualizable model logic for the NO-GWD UC.
    """
    
    def __init__(self):
        """
        Loads previously computed stats for model-level normalization purposes.
        """
        super().__init__()
        # Init attributes to fake data
        self.x_mean = torch.tensor(0)
        self.x_std = torch.tensor(1)
        self.y_std = torch.tensor(1)

        if os.path.exists(os.path.join(config.data_path, 'stats.pt')):
            stats = torch.load(os.path.join(config.data_path, 'stats.pt'))
            self.x_mean = stats['x_mean']
            self.x_std = stats['x_std']
            self.y_std = stats['y_std'].max()

    def forward(self, x: torch.Tensor):
        """
        Computes the forward pass.
        
        Args:
            (torch.Tensor): x input features.
            
        Returns:
            (torch.Tensor): Resulting model forward pass.
        """
        return self.net(x)

    def _common_step(self, batch: torch.Tensor, batch_idx: int, stage: str):
        
        x, y = batch
        x = (x - self.x_mean.to(self.device)) / self.x_std.to(self.device)
        y = y / self.y_std.to(self.device)
        y_hat = self(x)
        
        loss = F.mean_squared_error(y_hat, y)
        # TODO: Add epsilon in TSS for R2 score computation.
        # Currently returns NaN sometimes.
        r2 = F.r2_score(y_hat, y)
    
        self.log(f"{stage}_loss", loss, on_step=True, prog_bar=True)
        self.log(f"{stage}_r2", r2, on_step=True, prog_bar=True)

        return y_hat, loss, r2
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Computes one training step.
        
        Args:
            batch (torch.Tensor): Batch containing x input and y output features.
            batch_idx (int): Batch index.
            
        Returns:
            (torch.Tensor): Loss.
        """
        _, loss, _ = self._common_step(batch, batch_idx, "train")
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Computes one validation step.
        
        Args:
            batch (torch.Tensor): Batch containing x input and y output features.
            batch_idx (int): Batch index.
        """
        _, loss, _ = self._common_step(batch, batch_idx, "val")
        
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Computes one testing step.
        
        Args:
            batch (torch.Tensor): Batch containing x input and y output features.
            batch_idx (int): Batch index.
        """
        _, loss, _ = self._common_step(batch, batch_idx, "test")
        
    def configure_optimizers(self):
        """
        Sets the model optimizer.
            
        Returns:
            (torch_optimizer.Optimizer): Optimizer
        """
        return optim.AdamP(self.parameters(), lr=self.lr)


@MODEL_REGISTRY
class LitMLP(NOGWDModule):
    """
    Creates a good old MLP.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 lr: float):
        super().__init__()
        
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )

@MODEL_REGISTRY
class LitCNN(NOGWDModule):
    """
    Creates a simple 1D ConvNet working on columnar data. Input layer reshapes the data to broadcast surface parameters at every Atmospheric level.
    """
    
    class Reshape(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x: torch.Tensor):
            t0 = torch.reshape(x[:, :3 * 63], [-1, 3, 63])
            t1 = torch.tile(x[:, 3 * 63:].unsqueeze(2), (1, 1, 63))
            
            return torch.cat((t0, t1), dim=1)
                # torch.transpose(t0, -2, -1),
                # torch.transpose(t1, -2, -1)
            # ), dim=2)
    
    def __init__(self, 
                 in_channels: int, 
                 init_feat: int, 
                 conv_size: int, 
                 pool_size: int, 
                 out_channels: int, 
                 lr: float):
        super().__init__()
        
        self.lr = lr
        self.net = nn.Sequential(
            self.Reshape(),
            nn.Conv1d(in_channels, init_feat * 2, conv_size, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Conv1d(init_feat * 2, init_feat * 4, conv_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Flatten(),
            nn.Linear((init_feat * 4 - conv_size) // 2 * (init_feat * 2), out_channels),
            nn.Linear(out_channels, out_channels)
        )