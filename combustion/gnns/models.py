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

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_optimizer as optim
import torchmetrics.functional as F
from typing import List, Tuple

import plotters

class CombustionModule(pl.LightningModule):
    """
    Contains the basic logic meant for all GNN-like models experiments. Loss is MSE and the metric of interest is R2 determination score.
    """

    def __init__(self):
        super().__init__()
        self.grid_shape = None
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass.
        
        Args:
            (torch.Tensor): Nodes features.
            (torch.Tensor): Connectivity matrix.
            
        Returns:
            (torch.Tensor): Resulting model forward pass.
        """
        
        return self.model(x, edge_index)

    def _common_step(self, 
                     batch: torch.Tensor, 
                     batch_idx: int, 
                     stage: str) -> List[torch.Tensor]:
        
        batch_size = batch.ptr[0]-1
        y_hat = self(batch.x, batch.edge_index)
        loss = F.mean_squared_error(y_hat, batch.y)
        r2 = F.r2_score(y_hat, batch.y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=batch_size)
        self.log(f"{stage}_r2", r2, on_step=True, batch_size=batch_size)

        return y_hat, loss, r2

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Computes one training step.
        
        Args:
            batch (torch.Tensor): Batch containing nodes features and connectivity matrix.
            batch_idx (int): Batch index.
            
        Returns:
            (torch.Tensor): Loss.
        """
        _, loss, _ = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Computes one validation step.
        
        Args:
            batch (torch.Tensor): Batch containing nodes features and connectivity matrix.
            batch_idx (int): Batch index.
        """
        y_hat, _, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor]:
        """
        Computes one testing step. Additionally, also generates outputs to plots for the test Dataset
        
        Args:
            batch (torch.Tensor): Batch containing nodes features and connectivity matrix.
            batch_idx (int): Batch index.
            
        Returns:
            (Tuple[torch.Tensor]): (Ground truth, Predictions)
        """
        y_hat, _, _ = self._common_step(batch, batch_idx, "test")
        pos = np.stack(batch.pos.cpu().numpy())
        x_max = np.max(pos[:, 0:1])
        y_max = np.max(pos[:, 1:2])
        z_max = np.max(pos[:, 2:3])

        if not self.grid_shape:
            self.grid_shape = (x_max + 1, y_max + 1, z_max + 1)
            
        return batch.y, y_hat
        
    def test_epoch_end(self, outputs: list) -> None:
        """
        Gather all the outputs from the test_step to plot the test Dataset
        
        Args:
            outputs (List[Tuple[torch.Tensor]]): all batches containing a pair of (ground truth, prediction)
        """
        ys = list()
        y_hats = list()
        
        for out in outputs:
            ys.append(out[0])
            y_hats.append(out[1])
            
        self.ys = np.asarray([t.cpu().numpy().reshape((-1,) + self.grid_shape) for t in ys])
        self.y_hats = np.asarray([t.cpu().numpy().reshape((-1,) + self.grid_shape) for t in y_hats])
        
        self.plotter = plotters.Plotter(self.model.__class__.__name__, self.grid_shape)
        
        self.plotter.cross_section(self.plotter.zslice, self.ys, self.y_hats)
        self.plotter.dispersion_plot(self.ys, self.y_hats)
        self.plotter.histo(self.ys, self.y_hats)
        self.plotter.histo2d(self.ys, self.y_hats)
        self.plotter.boxplot(self.ys, self.y_hats)
        
    def configure_optimizers(self) -> optim.Optimizer:
        """
        Sets the model optimizer.
            
        Returns:
            (torch_optimizer.Optimizer): Optimizer
        """
        return optim.AdamP(self.parameters(), lr=self.lr)


@MODEL_REGISTRY
class LitGAT(CombustionModule):
    """
    Graph-ATtention net as described in the “Graph Attention Networks” paper.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int,
                 num_layers: int, 
                 dropout: float, 
                 heads: int, 
                 jk: str, 
                 lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.model = pyg.nn.GAT(in_channels=in_channels, 
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                num_layers=num_layers,
                                dropout=dropout,
                                act=nn.SiLU(inplace=True),
                                heads=heads,
                                jk=jk)


@MODEL_REGISTRY
class LitGCN(CombustionModule):
    """
    Classic stack of GCN layers. “Semi-supervised Classification with Graph Convolutional Networks”.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 num_layers: int, 
                 dropout: float, 
                 jk: str, 
                 lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.model = pyg.nn.GCN(in_channels=in_channels, 
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                num_layers=num_layers,
                                dropout=dropout,
                                jk=jk,
                                act=nn.SiLU(inplace=True))


@MODEL_REGISTRY
class LitGraphUNet(CombustionModule):
    """
    Graph-Unet as described in “Graph U-Nets”.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 depth: int, 
                 pool_ratios: float, 
                 lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.model = pyg.nn.GraphUNet(in_channels=in_channels, 
                                      hidden_channels=hidden_channels,
                                      out_channels=out_channels,
                                      depth=depth,
                                      pool_ratios=pool_ratios,
                                      act=nn.SiLU(inplace=True))


@MODEL_REGISTRY
class LitGIN(CombustionModule):
    """
    GNN implementation of “How Powerful are Graph Neural Networks?”.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 num_layers: int, 
                 dropout: float, 
                 lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.model = pyg.nn.GIN(in_channels=in_channels, 
                                      hidden_channels=hidden_channels,
                                      out_channels=out_channels,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      act=nn.SiLU(inplace=True))