import config
import io
import metrics
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotters
import pytorch_lightning as pl
import torch
import torch_optimizer as optim
import torchmetrics as tm
import torchmetrics.functional as F

from PIL import Image


# this script contains the training logic

class LitCombustionModule(pl.LightningModule):
    
    def __init__(self, model, grid_shape, args):
        super().__init__()
        self.save_hyperparameters()
        
        self._model = model
        self.grid_shape = grid_shape
        self.args = args
        
        
    @property
    def model(self):
        return self._model
        
        
    def forward(self, x, edge_index):
        return self._model(x, edge_index)
    
    
    def _common_step(self, batch, batch_idx, stage: str):
        y_hat = self(batch.x, batch.edge_index)
        loss = F.mean_squared_error(y_hat, batch.y)
        r2 = F.r2_score(y_hat, batch.y)
        
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        self.log(f"{stage}_r2", r2, on_step=True)
        
        return y_hat, loss, r2
    
    
    def training_step(self, batch, batch_idx):
        _, loss, _ = self._common_step(batch, batch_idx, "train")
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "val")
#         fig = metrics.flame_surface(batch_idx, y_hat, batch.y, "val", self.grid_shape)
        
#         path = os.path.join(config.plots_path, f'val_total_flame_surface_{batch_idx}.png')
#         image = Image.open(io.BytesIO(fig.to_image('png')))
#         image.save(path)
        
        # self.logger.experiment.add_image(
        #     f"val_total_flame_surface_{batch_idx}", 
        #     np.array(image), 
        #     global_step=batch_idx, 
        #     dataformats='HWC')
        
        
    def test_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "test")
        plot = metrics.flame_surface(batch_idx, y_hat, batch.y, "test", self.grid_shape)
        path = os.path.join(config.plots_path, f'test_total_flame_surface_{batch_idx}.png')
        image = Image.open(io.BytesIO(plot.to_image('png')))
        image.save(path)
        
        self.logger.experiment.add_image(
            f"test_total_flame_surface_{batch_idx}", 
            np.array(image), 
            global_step=batch_idx, 
            dataformats='HWC')
        
        path = os.path.join(config.plots_path, f"dns_total_flame_surface_{batch_idx}.png")
        slice_ = plotters.sigma_slices(y_hat, batch.y, self.grid_shape, slice_idx=16)
        slice_.write_image(path)
        
    
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.args.lr)