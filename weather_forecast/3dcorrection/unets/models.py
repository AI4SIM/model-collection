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
from torch.nn import Module
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from torchmetrics import MeanAbsoluteError
import os.path as osp
from torch_optimizer import AdamP

import config
from unet import UNet1D
from layers import HRLayer, Normalization

# TODO: learning rate scheduler (cosine annealing with warm restart)


class ThreeDCorrectionModule(pl.LightningModule):
    """Create a Lit module for 3dcorrection."""

    def __init__(self,
                 norm: bool,
                 flux_loss_weight: float,
                 hr_loss_weight: float):
        """
        Args:
            data_path (str): Path to folder containing the stats.pt.
        """
        super().__init__()

        self.norm = norm
        self.flux_loss_weight = torch.tensor(flux_loss_weight)
        self.hr_loss_weight = torch.tensor(hr_loss_weight)
        
        self.mae = MeanAbsoluteError()

        stats = torch.load(osp.join(config.data_path, "processed", "stats.pt"))
        self.x_mean = stats["x_mean"]
        self.x_std = stats["x_std"]

    def forward(self, x):
        x = self.normalization(x)
        x_ = x[..., :-1]
        x_ = torch.moveaxis(x_, -2, -1)
        fluxes = self.net(x_)
        fluxes = torch.moveaxis(fluxes, -2, -1)
        hr = self.hr_layer([fluxes, x[..., -1]])
        return fluxes, hr

    def weight_histograms_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def gradient_histograms_adder(self):
        global_step = self.global_step
        if global_step % 50 == 0:  # do not make the tb file huge
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def _common_step(self, batch, batch_idx, stage, normalize=False):
        """Compute the loss, additional metrics, and log them."""

        x, y, z = batch
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

        return y_flux_hat, y_hr_hat, flux_mae, hr_mae, loss

    def training_step(self, batch, batch_idx):
        _, _, _, _, loss = self._common_step(batch, batch_idx, "train", self.norm)
        return loss

    def on_after_backward(self):
        self.gradient_histograms_adder()

    def training_epoch_end(self, outputs):
        self.weight_histograms_adder()

    def validation_step(self, batch, batch_idx):
        _, _, _, _, _ = self._common_step(batch, batch_idx, "val", self.norm)

    def test_step(self, batch, batch_idx):
        _, _, _, _, _ = self._common_step(batch, batch_idx, "test", self.norm)


@MODEL_REGISTRY
class LitUnet1D(ThreeDCorrectionModule):
    """Compile a 1D U-Net, which needs a stats.pt (in the folder of data_path)."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_levels: int,
                 n_features_root: int,
                 norm: bool,
                 flux_loss_weight: float,
                 hr_loss_weight: float):
        super(LitUnet1D, self).__init__(norm, flux_loss_weight, hr_loss_weight)
        self.save_hyperparameters()

        self.normalization = Normalization(self.x_mean, self.x_std)
        self.net = UNet1D(
            inp_ch=in_channels,
            out_ch=out_channels,
            n_levels=n_levels,
            n_features_root=n_features_root)
        self.hr_layer = HRLayer()

    # def configure_optimizers(self):
    #     return AdamP(self.parameters(), lr=self.lr)
