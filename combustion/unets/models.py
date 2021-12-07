import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch.nn as nn
import torch_optimizer as optim
import torchmetrics.functional as F

import unet


class CombustionModule(pl.LightningModule):

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        loss = F.mean_squared_error(y_hat, y)
        r2 = F.r2_score(y_hat, y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=len(x))
        self.log(f"{stage}_r2", r2, on_step=True, batch_size=len(x))

        return y_hat, loss, r2

    # TODO: data augmentation

    def training_step(self, batch, batch_idx):
        _, loss, _ = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "test")
        pos = np.stack(batch.pos.cpu().numpy())
        x_max = np.max(pos[:, 0:1])
        y_max = np.max(pos[:, 1:2])
        z_max = np.max(pos[:, 2:3])
        grid_shape = (x_max + 1, y_max + 1, z_max + 1)
        # TODO: plotter
        # plotters.Plotter(batch.y.cpu().numpy(), y_hat.cpu().numpy(), self.model.__class__.__name__, grid_shape)


@MODEL_REGISTRY
class LitUnet3D(CombustionModule):
    """
    Lit wrapper to compile a 3D U-net.
    """

    def __init__(self, in_channels, out_channels, n_levels, n_features_root, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = unet.UNet3D(
            inp_feat=in_channels,
            out_feat=out_channels,
            n_levels=n_levels,
            n_features_root=n_features_root)

    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)
