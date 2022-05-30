"""This module proposes Pytorch style LightningModule classes for the 3dcorrection use-case."""
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

import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_optimizer as optim
import torchmetrics.functional as tmf
from typing import List, Tuple

import config

EPS = torch.tensor(1.0e-8)


class ThreeDCorrectionModule(pl.LightningModule):
    """Contain the mutualizable model logic for the 3dcorrection use-case."""

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Connectivity matrix.

        Returns:
            (torch.Tensor): Resulting model forward pass.
        """
        return self.net(x, edge_index)

    def weight_histograms_adder(self) -> None:
        """
        For each variable with requires_grad is True, push the weight histogram
        into the logger every epoch.
        """
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def gradient_histograms_adder(self) -> None:
        """
        For each variable with requires_grad is True, push the gradient histogram
        into the logger (every 50 steps).
        """
        global_step = self.global_step
        if global_step % 50 == 0:  # do not make the tb file huge
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.logger.experiment.add_histogram(
                        f"{name}_grad", param.grad, global_step
                    )

    def _normalize(
        self, batch: torch.Tensor, batch_size: int, path: str
    ) -> Tuple[torch.Tensor]:
        """Load the stats computed when building the Dataset and
        normalize the features and outputs.

        Args:
            batch (torch.Tensor): Structure containing node features, node targets
                and connectivity matrix.
            batch_size (int): Batch size.
            path (str): Path the stat file.

        Returns:
            (Tuple[torch.Tensor]): (node features, node targets, connectivity matrix).
        """
        stats = torch.load(os.path.join(path, f"stats-{self.timestep}.pt"))
        device = self.device
        x_mean = stats["x_mean"].to(device)
        y_mean = stats["y_mean"].to(device)
        x_std = stats["x_std"].to(device)
        y_std = stats["y_std"].to(device)

        num_output_features = batch.y.size()[-1]

        x = (batch.x.reshape((batch_size, -1, batch.num_features)) - x_mean) / (
            x_std + EPS.to(device)
        )
        y = (batch.y.reshape((batch_size, -1, num_output_features)) - y_mean) / (
            y_std + EPS.to(device)
        )

        x = x.reshape(-1, batch.num_features)
        y = y.reshape(-1, num_output_features)

        return x, batch.y, batch.edge_index

    def _common_step(
        self, batch: torch.Tensor, batch_idx: int, stage: str, normalize=False
    ) -> List[torch.Tensor]:
        """Define the common operations performed on data.

        Args:
            batch (torch.Tensor): The output of the DataLoader.
            batch_idx (int): Integer displaying index of this batch.
            stage (str): current running stage (fit/val/test).

        Returns:
            (List[torch.Tensor]): predictions and metrics
        """
        batch_size = batch.ptr.size()[0] - 1

        if normalize:
            x, y, edge_index = self._normalize(batch, batch_size, config.data_path)
        else:
            x, y, edge_index = batch.x, batch.y, batch.edge_index

        y_hat = self(x, edge_index)
        loss = tmf.mean_squared_error(y_hat, y)
        mae = tmf.mean_absolute_error(y_hat, y)

        self.log(
            f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=batch_size
        )
        self.log(
            f"{stage}_mae", mae, prog_bar=True, on_step=True, batch_size=batch_size
        )

        return y_hat, loss, mae

    def training_step(self, batch, batch_idx):
        """Compute one training step.
        Args:
            batch (torch.Tensor): Batch containing nodes features and connectivity matrix.
            batch_idx (int): Integer displaying index of this batch.
        Returns:
            (torch.Tensor): Loss.
        """
        _, loss, _ = self._common_step(batch, batch_idx, "train", normalize=self.norm)
        return loss

    def on_after_backward(self) -> None:
        """Log the gradient after backward pass."""
        self.gradient_histograms_adder()

    def training_epoch_end(self, outputs: list) -> None:
        """Log the weight histograms after each epoch.

        Args:
            outputs (List[Tuple[torch.Tensor]]): all batches containing a pair of
                (ground truth, prediction)
        """
        self.weight_histograms_adder()

    def validation_step(self, batch, batch_idx):
        """Compute one validation step.

        Args:
            batch (torch.Tensor): Batch containing nodes features and connectivity matrix.
            batch_idx (int): Batch index.
        """
        y_hat, _, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Compute one testing step.
        Args:
            batch (torch.Tensor): Batch containing x input and y output features.
            batch_idx (int): Batch index.
        """
        y_hat, _, _ = self._common_step(batch, batch_idx, "test")


@MODEL_REGISTRY
class LitGAT(ThreeDCorrectionModule):
    """Graph-ATtention net as described in the “Graph Attention Networks” paper."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        edge_dim,
        heads,
        jk,
        lr,
        timestep,
        norm,
    ):
        """Init the LitGAT class."""
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.timestep = timestep
        self.norm = norm
        self.net = pyg.nn.GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            act=nn.SiLU(inplace=True),
            heads=heads,
            jk=jk,
        )

    def configure_optimizers(self) -> optim.Optimizer:
        """Set the model optimizer.
        Returns:
            (torch_optimizer.Optimizer): Optimizer
        """
        return optim.AdamP(self.parameters(), lr=self.lr)
