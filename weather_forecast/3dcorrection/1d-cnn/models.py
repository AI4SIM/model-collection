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
from layers import HRLayer, Normalization, PreProcessing
import torch.nn as nn
import numpy as np


class ThreeDCorrectionModule(pl.LightningModule):
    """Create a Lit module for 3dcorrection."""

    def __init__(self,
                 flux_loss_weight: int,
                 hr_loss_weight: int):
        """
        Args:
            data_path (str): Path to folder containing the stats.pt.
        """
        super().__init__()

        self.flux_loss_weight = torch.tensor(flux_loss_weight)
        self.hr_loss_weight = torch.tensor(hr_loss_weight)
        
        self.mae = MeanAbsoluteError()

        # stats = torch.load(osp.join(config.data_path, "processed", "stats.pt"))
        self.x_mean = {
            "sca_inputs": torch.tensor(0),
            "col_inputs": torch.tensor(0),
            "hl_inputs": torch.tensor(0),
            "inter_inputs": torch.tensor(0),
        }
        self.x_std = {
            "sca_inputs": torch.tensor(1),
            "col_inputs": torch.tensor(1),
            "hl_inputs": torch.tensor(1),
            "inter_inputs": torch.tensor(1),
        }

    def forward(self, x):
        x = self.preprocessing(x)
        x = torch.moveaxis(x, -2, -1)
        y = self.cnn_1d(x)
        y = torch.moveaxis(y, -2, -1)
        return y
        # return self.cnn_1d(x)
        # x = self.normalization(x)
        # x_ = x[..., :-1]
        # x_ = torch.moveaxis(x_, -2, -1)
        # fluxes = self.net(x_)
        # fluxes = torch.moveaxis(x, -2, -1)
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
        # y_flux_hat, y_hr_hat = self(x)
        y_flux_hat = self(x)

        batch_y = ([val.cpu().numpy() for k, val in y.items()
                   if k in ['delta_sw_diff', 'delta_sw_add', 'delta_lw_diff', 'delta_lw_add']])
        y_ = torch.tensor(np.array(batch_y))
        y_ = torch.moveaxis(y_, 0, -1)
        y_ = y_.to(self.device)

        flux_loss = mean_squared_error(y_flux_hat, y_)
        # hr_loss = mean_squared_error(y_hr_hat, z)
        loss = self.flux_loss_weight * flux_loss #+ self.hr_loss_weight * hr_loss

        # flux_mae = self.mae(y_flux_hat, y)
        # hr_mae = self.mae(y_hr_hat, z)

        kwargs = {
            "prog_bar": True,
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "batch_size": len(x)
        }
        self.log(f"{stage}_loss", loss, **kwargs)
        self.log(f"{stage}_flux_loss", flux_loss, **kwargs)
        # self.log(f"{stage}_hr_loss", hr_loss, **kwargs)
        # self.log(f"{stage}_flux_mae", flux_mae, **kwargs)
        # self.log(f"{stage}_hr_mae", hr_mae, **kwargs)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "train")
        return loss

    def on_after_backward(self):
        self.gradient_histograms_adder()

    def training_epoch_end(self, outputs):
        self.weight_histograms_adder()

    def validation_step(self, batch, batch_idx):
        _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        _ = self._common_step(batch, batch_idx, "test")


@MODEL_REGISTRY
class LitCNN(ThreeDCorrectionModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: int,
        n_features_root: int,
        norm: bool,
        flux_loss_weight: int,
        hr_loss_weight: int
    ):
        # pass
        print("LITmodel")
        super().__init__(flux_loss_weight, hr_loss_weight)
        # self.save_hyperparameters()

        self.preprocessing = PreProcessing(self.x_mean, self.x_std)
        # self.block_cnn_dilation = BlockCNNDilation()
        # self.mha = MultiHeadAttention()
        # self.block_cnn_regular = BlockCNNRegular()
        # self.cnn_1d = torch.nn.LazyConv1d(
        self.cnn_1d = torch.nn.LazyConv1d(
            out_channels=out_channels,
            kernel_size=5,
            padding='same'
        )
