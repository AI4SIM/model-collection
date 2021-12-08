import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch.nn as nn
import torch_geometric as pyg
import torch_optimizer as optim
import torchmetrics.functional as F

import plotters


class CombustionModule(pl.LightningModule):

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def _common_step(self, batch, batch_idx, stage):
        y_hat = self(batch.x, batch.edge_index)
        loss = F.mean_squared_error(y_hat, batch.y)
        r2 = F.r2_score(y_hat, batch.y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=len(batch))
        self.log(f"{stage}_r2", r2, on_step=True, batch_size=len(batch))

        return y_hat, loss, r2

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

        plotters.Plotter(batch.y.cpu().numpy(), y_hat.cpu().numpy(), self.model.__class__.__name__, grid_shape)


@MODEL_REGISTRY
class LitGAT(CombustionModule):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads, jk, lr):
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

    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)


@MODEL_REGISTRY
class LitGCN(CombustionModule):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, jk, lr):
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

    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)


@MODEL_REGISTRY
class LitGraphUNet(CombustionModule):

    def __init__(self, in_channels, hidden_channels, out_channels, depth, pool_ratios, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = pyg.nn.GraphUNet(in_channels=in_channels,
                                      hidden_channels=hidden_channels,
                                      out_channels=out_channels,
                                      depth=depth,
                                      pool_ratios=pool_ratios,
                                      act=nn.SiLU(inplace=True))

    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)


@MODEL_REGISTRY
class LitGIN(CombustionModule):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = pyg.nn.GIN(in_channels=in_channels,
                                      hidden_channels=hidden_channels,
                                      out_channels=out_channels,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      act=nn.SiLU(inplace=True))

    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)
