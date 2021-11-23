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
    
    def __init__(self, model, args):
        super().__init__()
        self.save_hyperparameters()
        
        self._model = model
        self.args = args
        print('finished init')
        
        
    @property
    def model(self):
        return self._model
        
        
    def forward(self, x):
        print('stuck forward')
        return self._model(x)
    
    
    def _common_step(self, batch, batch_idx, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = F.mean_squared_error(y_hat, y)
        r2 = F.r2_score(y_hat, y)
        
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        self.log(f"{stage}_r2", r2, on_step=True)
        
        return y_hat, loss, r2
    
    
    def training_step(self, batch, batch_idx):
        print('stuck training')
        _, loss, _ = self._common_step(batch, batch_idx, "train")
        print('done first training')
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        print('starting validation')
        x, y = batch
        y_hat, _, _ = self._common_step(batch, batch_idx, "val")
        fig = metrics.flame_surface(batch_idx, y_hat, y, "val")
        
        path = os.path.join(config.plots_path, f'val_total_flame_surface_{batch_idx}.png')
        image = Image.open(io.BytesIO(fig.to_image('png')))
        image.save(path)
        
        self.logger.experiment.add_image(
            f"val_total_flame_surface_{batch_idx}", 
            np.array(image), 
            global_step=batch_idx, 
            dataformats='HWC')
        
        
    def test_step(self, batch, batch_idx):
        print('starting test')
        x, y = batch
        y_hat, _, _ = self._common_step(batch, batch_idx, "test")
        plot = metrics.flame_surface(batch_idx, y_hat, y, "test")
        image = Image.open(io.BytesIO(plot.to_image('png')))
        
        self.logger.experiment.add_image(
            f"test_total_flame_surface_{batch_idx}", 
            np.array(image), 
            global_step=batch_idx, 
            dataformats='HWC')
        
        path = os.path.join(config.plots_path, f"dns_total_flame_surface_{batch_idx}.png")
        slice_ = plotters.sigma_slices(y_hat, y, slice_idx=16)
        slice_.write_image(path)
        
    
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.args.lr)

