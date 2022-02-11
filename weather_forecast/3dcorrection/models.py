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
import os

import config

class ThreeDCorrectionModule(pl.LightningModule):

    def forward(self, x, edge_index):
        return self.net(x, edge_index)

    def weight_histograms_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
    
    def gradient_histograms_adder(self):
        global_step = self.global_step
        if global_step % 50 == 0: # do not make the tb file huge
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
        
    def _normalize(self, batch, batch_size):
        stats = torch.load(os.path.join(config.data_path, f"stats-{self.step}.pt"))
        device = self.device
        x_mean = stats["x_mean"].to(device)
        y_mean = stats["y_mean"].to(device)
        x_std = stats["x_std"].to(device)
        y_std = stats["y_std"].to(device)
        
        num_output_features = batch.y.size()[-1]
        
        x = (batch.x.reshape((batch_size, -1, batch.num_features)) - x_mean) / (x_std + torch.tensor(1.e-8))
        y = (batch.y.reshape((batch_size, -1, num_output_features)) - y_mean) / (y_std + torch.tensor(1.e-8))
        
        x = x.reshape(-1, batch.num_features)
        y = y.reshape(-1, num_output_features)
        
        return x, batch.y, batch.edge_index
    
    def _common_step(self, batch, batch_idx, stage, normalize=False):
        batch_size = batch.ptr.size()[0]-1
        
        if normalize:
            x, y, edge_index = self._normalize(batch, batch_size)
        else:
            x, y, edge_index = batch.x, batch.y, batch.edge_index
        
        y_hat = self(x, edge_index)
        loss = F.mean_squared_error(y_hat, y)
        r2 = F.r2_score(y_hat, y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=batch_size)
        self.log(f"{stage}_r2", r2,  prog_bar=True, on_step=True, batch_size=batch_size)
        
        return y_hat, loss, r2

    def training_step(self, batch, batch_idx):
        _, loss, _ = self._common_step(batch,
                                       batch_idx,
                                       "train",
                                       normalize=self.norm)
        return loss
    
    def on_after_backward(self):
        self.gradient_histograms_adder()
    
    def training_epoch_end(self, outputs):
        self.weight_histograms_adder()

    def validation_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "test")


@MODEL_REGISTRY
class LitGAT(ThreeDCorrectionModule):
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 heads,
                 jk,
                 lr,
                 timestep,
                 norm):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.step = timestep
        self.norm = norm
        self.net = pyg.nn.GAT(in_channels=in_channels, 
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                num_layers=num_layers,
                                dropout=dropout,
                                act=nn.SiLU(inplace=True),
                                heads=heads,
                                jk=jk)
        
    
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)


# @MODEL_REGISTRY
# class LitGCN(ThreeDCorrectionModule):
    
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, jk, lr):
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.lr = lr
#         self.model = pyg.nn.GCN(in_channels=in_channels, 
#                                 hidden_channels=hidden_channels,
#                                 out_channels=out_channels,
#                                 num_layers=num_layers,
#                                 dropout=dropout,
#                                 jk=jk,
#                                 act=nn.SiLU(inplace=True))
        
    
#     def configure_optimizers(self):
#         return optim.AdamP(self.parameters(), lr=self.lr)


# @MODEL_REGISTRY
# class LitGraphUNet(ThreeDCorrectionModule):
    
#     def __init__(self, in_channels, hidden_channels, out_channels, depth, pool_ratios, lr):
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.lr = lr
#         self.model = pyg.nn.GraphUNet(in_channels=in_channels, 
#                                       hidden_channels=hidden_channels,
#                                       out_channels=out_channels,
#                                       depth=depth,
#                                       pool_ratios=pool_ratios,
#                                       act=nn.SiLU(inplace=True))
        
    
#     def configure_optimizers(self):
#         return optim.AdamP(self.parameters(), lr=self.lr)


# @MODEL_REGISTRY
# class LitGIN(ThreeDCorrectionModule):
    
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, lr):
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.lr = lr
#         self.model = pyg.nn.GIN(in_channels=in_channels, 
#                                 hidden_channels=hidden_channels,
#                                 out_channels=out_channels,
#                                 num_layers=num_layers,
#                                 dropout=dropout,
#                                 act=nn.SiLU(inplace=True))
        
    
#     def configure_optimizers(self):
#         return optim.AdamP(self.parameters(), lr=self.lr)