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

from lightning import LightningModule
from torch import flatten
from torch_optimizer import AdamP
from torchmetrics.functional import mean_squared_error, r2_score

from unet import UNet3D


class CombustionModule(LightningModule):
    """LightningModule for combustion use-cases (R2-scored)."""

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, stage):
        """Compute the loss, additional metrics, and log them."""
        x, y = batch
        y_hat = self(x)
        loss = mean_squared_error(y_hat, y)
        r2 = r2_score(flatten(y_hat), flatten(y))  # R2 between mesh points.

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=len(x))
        self.log(f"{stage}_r2", r2, on_step=True, batch_size=len(batch))

        return y_hat, loss, r2

    def training_step(self, batch, batch_idx):
        _, loss, _ = self._common_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, "test")


class LitUnet3D(CombustionModule):
    """Lit wrapper to compile a 3D U-net, generic volume shapes."""

    def __init__(self, in_channels, out_channels, n_levels, n_features_root, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = UNet3D(
            inp_ch=in_channels,
            out_ch=out_channels,
            n_levels=n_levels,
            n_features_root=n_features_root)

    def configure_optimizers(self):
        return AdamP(self.parameters(), lr=self.lr)
