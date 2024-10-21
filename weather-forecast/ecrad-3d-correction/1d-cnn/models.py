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


import torch
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.nn import Module
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from torchmetrics import MeanAbsoluteError

import config
from layers import HRLayer, Normalization


class ThreeDCorrectionModule(pl.LightningModule):
    """Create a Lit module for 3dcorrection."""

    def __init__(
        self,
        flux_loss_weight: float,
        hr_loss_weight: float,
        log_gradients: bool = False,
        log_weights: bool = False,
    ):
        """
        Args:
            data_path (str): Path to folder containing the stats.pt.
        """
        super().__init__()

        self.log_gradients = log_gradients
        self.log_weights = log_weights

        self.flux_loss_weight = torch.tensor(flux_loss_weight)
        self.hr_loss_weight = torch.tensor(hr_loss_weight)
        
        self.mae = MeanAbsoluteError()

        stats = torch.load(osp.join(config.data_path, "processed", "stats.pt"))
        self.x_mean = stats["x_mean"]
        self.x_std = stats["x_std"]

    def forward(self, x):
        return fluxes + hr
        # x = self.normalization(x)
        # x_ = x[..., :-1]
        # x_ = torch.moveaxis(x_, -2, -1)
        # fluxes = self.net(x_)
        # fluxes = torch.moveaxis(fluxes, -2, -1)
        # hr = self.hr_layer([fluxes, x[..., -1]])
        # return fluxes, hr

    def weight_histograms_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def gradient_histograms_adder(self):
        global_step = self.global_step
        if global_step % 50 == 0:  # do not make the tb file huge
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def _common_step(self, batch, batch_idx, stage):
        """Compute the loss, additional metrics, and log them."""

        x, y = batch
        y_flux_hat, y_hr_hat = self(x)

        flux_loss = mean_squared_error(y_flux_hat, y)
        hr_loss = mean_squared_error(y_hr_hat, z)
        loss = self.flux_loss_weight * flux_loss + self.hr_loss_weight * hr_loss

        flux_mae = self.mae(y_flux_hat, y)
        hr_mae = self.mae(y_hr_hat, z)

        kwargs = {
            "prog_bar": True,
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "batch_size": len(x)
        }
        self.log(f"{stage}_loss", loss, **kwargs)
        self.log(f"{stage}_flux_loss", flux_loss, **kwargs)
        self.log(f"{stage}_hr_loss", hr_loss, **kwargs)
        self.log(f"{stage}_flux_mae", flux_mae, **kwargs)
        self.log(f"{stage}_hr_mae", hr_mae, **kwargs)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "train")
        return loss

    def on_after_backward(self):
        if self.log_gradients:
            self.gradient_histograms_adder()

    def training_epoch_end(self, outputs):
        if self.log_weights:
            self.weight_histograms_adder()

    def validation_step(self, batch, batch_idx):
        _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        _ = self._common_step(batch, batch_idx, "test")


@MODEL_REGISTRY
class LitCNN1dAttention(ThreeDCorrectionModule):
    def __init__(self, kwargs):
        pass
    super().__init__()
    self.save_hyperparameters()
    
    self.preprocessing = PreProcessing()
    self.block_cnn_dilation = BlockCNNDilation()
    self.mha = MultiHeadAttention()
    self.block_cnn_regular = BlockCNNRegular()
    self.cnn_1d = CNN1d()