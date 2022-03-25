"""This module proposes Pytorch style LightningModule classes for the gwd use-case."""
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
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_optimizer as optim
import torchmetrics.functional as tmf
from typing import Tuple


class NOGWDModule(pl.LightningModule):
    """Contains the mutualizable model logic for the NO-GWD UC."""

    def __init__(self):
        """Load previously computed stats for model-level normalization purposes."""
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

    def forward(self, features: torch.Tensor):
        """Compute the forward pass.

        Args:
            features (torch.Tensor): input features.

        Returns:
            (torch.Tensor): Resulting model forward pass.
        """
        return self.net(features)

    def _common_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str):
        """Define the common operations performed on data."""
        x_val, y_val = batch
        x_val = (x_val - self.x_mean.to(self.device)) / self.x_std.to(self.device)
        y_val = y_val / self.y_std.to(self.device)
        y_hat = self(x_val)

        loss = tmf.mean_squared_error(y_hat, y_val)
        # TODO: Add epsilon in TSS for R2 score computation.
        # Currently returns NaN sometimes.
        r2 = tmf.r2_score(y_hat, y_val)

        self.log(f"{stage}_loss", loss, on_step=True, prog_bar=True)
        self.log(f"{stage}_r2", r2, on_step=True, prog_bar=True)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Compute one training step.

        Args:
            batch (torch.Tensor): Batch containing x input and y output features.
            batch_idx (int): Batch index.

        Returns:
            (torch.Tensor): Loss.
        """
        loss = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Compute one validation step.

        Args:
            batch (torch.Tensor): Batch containing x input and y output features.
            batch_idx (int): Batch index.
        """
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Compute one testing step.

        Args:
            batch (torch.Tensor): Batch containing x input and y output features.
            batch_idx (int): Batch index.
        """
        self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """Set the model optimizer.

        Returns:
            (torch_optimizer.Optimizer): Optimizer
        """
        return optim.AdamP(self.parameters(), lr=self.lr)


@MODEL_REGISTRY
class LitMLP(NOGWDModule):
    """Create a good old MLP."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 lr: float):
        """Define the network LitMLP architecture."""
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
    """Create a simple 1D ConvNet working on columnar data. Input layer reshapes the data to
    broadcast surface parameters at every Atmospheric level.
    """

    class Reshape(nn.Module):
        """Create a custom Reshape layer."""

        def forward(self, tensor: torch.Tensor):
            """Reshape the input tensor to expose the 5 features of each columnar cell.
            The 3 first ones features are Atmospheric level dependant quantities, while the last 2
            ones are constants.

            Args:
                tensor (torch.Tensor): the raw tensor to be reshaped.

            Returns:
                (torch.Tensor): the reshaped tensor.
            """
            t0 = torch.reshape(tensor[:, :3 * 63], [-1, 3, 63])
            t1 = torch.tile(tensor[:, 3 * 63:].unsqueeze(2), (1, 1, 63))

            return torch.cat((t0, t1), dim=1)

    def __init__(self,
                 in_channels: int,
                 init_feat: int,
                 conv_size: int,
                 pool_size: int,
                 out_channels: int,
                 lr: float):
        """Define the network LitCNN architecture."""
        super().__init__()

        self.lr = lr
        self.net = nn.Sequential(
            self.Reshape(),
            nn.Conv1d(in_channels, init_feat, conv_size, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Conv1d(init_feat, init_feat * 2, conv_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Flatten(),
            nn.Linear(480, out_channels),
            nn.Linear(out_channels, out_channels)
        )
