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

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
from torchmetrics.functional import mean_squared_error
from typing import Tuple, Union
import os.path as osp
from torch_optimizer import AdamP
import numpy as np

from unet import UNet1D

"""
TODO: improve the preprocessing:
- learn the heating rate
- link it in the DAG
"""


class ThreeDCorrectionModule(pl.LightningModule):
    """Create a Lit module for 3dcorrection."""

    def __init__(self, data_path: str, normalize=False):
        """
        Args:
            data_path (str): Path to folder containing the stats.pt.
            normalize (bool): Whether or not to normalize during training.
        """
        super().__init__()
        self.normalize = normalize

        stats = torch.load(osp.join(data_path, "stats.pt"))
        self.x_mean = stats["x_mean"].to(self.device)
        self.y_mean = stats["y_mean"].to(self.device)
        self.x_std = stats["x_std"].to(self.device)
        self.y_std = stats["y_std"].to(self.device)

    def forward(self, x: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            x (ndarray | tensor): Input tensor (numpy or tensor).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x, _ = self.preprocess(x)
        return self.model(x)

    def weight_histograms_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def gradient_histograms_adder(self):
        global_step = self.global_step
        if global_step % 50 == 0:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def preprocess(self,
                   x: torch.Tensor,
                   y: torch.Tensor = None,
                   normalize: bool = False) -> Tuple[torch.Tensor]:
        """
        Apply the preprocessing steps inside the network:
            * Reshaping the data;
            * Normalizing (if normalize is True).
        If y is None, then only process x (e.g. forward mode).
        """

        # Reshaping: (?, spatial, channels) -> (?, channels, spatial)
        x = torch.moveaxis(x, 1, -1)
        if y is not None:
            y = torch.moveaxis(y, 1, -1)

        if normalize:
            eps = torch.tensor(1.e-8)
            x = (x - self.x_mean) / (self.x_std + eps)
            if y is not None:
                y = (y - self.y_mean) / (self.y_std + eps)

        return x, y

    def _common_step(self, batch, stage, normalize=False):
        """Compute the loss, additional metrics, and log them."""
        x, y = batch
        self.preprocess(x, y)

        y_hat = self(x)
        loss = mean_squared_error(y_hat, y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=len(x))
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        _, loss, _ = self._common_step(batch, "train", normalize=self.normalize)
        return loss

    def on_after_backward(self):
        self.gradient_histograms_adder()

    def training_epoch_end(self, outputs):
        self.weight_histograms_adder()

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, "test")


@MODEL_REGISTRY
class LitUnet1D(ThreeDCorrectionModule):
    """Compile a 1D U-Net, which needs a stats.pt (in the folder of data_path)."""

    def __init__(self,
                 data_path: str,
                 normalize: bool,
                 in_channels: int,
                 out_channels: int,
                 n_levels: int,
                 n_features_root: int,
                 lr: float):
        super(LitUnet1D, self).__init__(data_path, normalize)
        self.save_hyperparameters()

        self.lr = lr
        self.model = UNet1D(
            inp_ch=in_channels,
            out_ch=out_channels,
            n_levels=n_levels,
            n_features_root=n_features_root)

    def configure_optimizers(self):
        return AdamP(self.parameters(), lr=self.lr)
