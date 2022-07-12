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
from torchmetrics.functional import mean_squared_error
import os.path as osp
from torch_optimizer import AdamP

from unet import UNet1D

# TODO: learn the heating rate
#TODO: learning rate scheduler (cosine annealing with warm restart)


class ThreeDCorrectionModule(pl.LightningModule):
    """Create a Lit module for 3dcorrection."""

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Path to folder containing the stats.pt.
        """
        super().__init__()

        stats = torch.load(osp.join(data_path, "stats.pt"))
        self.x_mean = stats["x_mean"].to(self.device)
        self.x_std = stats["x_std"].to(self.device)
        
        # self.flux_loss_weight = flux_loss_weight
        # self.hr_loss_weight = hr_loss_weight

    def forward(self, x, press):
        fluxes = self.model(x)
        hr = self.hr_layer([fluxes, press])
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

    def _normalize(self, x):
        eps = torch.tensor(1.e-8)
        x = (x - self.x_mean) / (self.x_std + eps)
        return x

    def _common_step(self, batch, stage, normalize=False):
        """Compute the loss, additional metrics, and log them."""
        x, y, z = batch

        if normalize:
            x = self._normalize(x)

        y_flx_hat, y_hr_hat = self(x, press)
        flux_loss = mean_squared_error(y_flx_hat, y)
        hr_loss = mean_squared_error(y_hr_hat, z)
        loss = torch.tesnor(1.e3) * flux_loss + torch.tensor(1.e5) * hr_loss

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=len(x))
        return y_flux_hat, y_hr_hat, loss

    def training_step(self, batch, batch_idx):
        _, _, loss = self._common_step(batch, batch_idx, "train", normalize=self.norm)
        return loss

    def on_after_backward(self):
        self.gradient_histograms_adder()

    def training_epoch_end(self, outputs):
        self.weight_histograms_adder()

    def validation_step(self, batch, batch_idx):
        y_flux_hat, y_hr_hat, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        y_flux_hat, y_hr_hat, _ = self._common_step(batch, batch_idx, "test")


# @MODEL_REGISTRY
class LitUnet1D(ThreeDCorrectionModule):
    """Compile a 1D U-Net, which needs a stats.pt (in the folder of data_path)."""
    
    class HRLayer(Module):
        """
        Layer to calculate heating rates given fluxes and half-level pressures.
        This could be used to deduce the heating rates within the model so that
        the outputs can be constrained by both fluxes and heating rates.
        """
        def __init__(self):
            super().__init__()
            self.g_cp = torch.tensor(24 * 3600 * 9.80665 / 1004)
            
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = []
            hlpress = inputs[1]
            net_press = hlpress[..., 1:, 0] - net_press[..., :-1, 0]
            for i in [0, 2]:
                netflux = inputs[0][..., i]
                flux_diff = netflux[..., 1:] - netflux[..., :-1]
                outputs.append(-self.g_cp * torch.Tensor.divide(flux_diff, net_press))
            return outputs

    def __init__(self,
                 data_path: str,
                 in_channels: int,
                 out_channels: int,
                 n_levels: int,
                 n_features_root: int,
                 lr: float):
        super(LitUnet1D, self).__init__(data_path)
        self.save_hyperparameters()

        self.lr = lr
        self.model = UNet1D(
            inp_ch=in_channels,
            out_ch=out_channels,
            n_levels=n_levels,
            n_features_root=n_features_root)
        self.hr_layer = LitUnet1D.HRLayer()

    def configure_optimizers(self):
        return AdamP(self.parameters(), lr=self.lr)
