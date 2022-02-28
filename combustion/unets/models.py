'''
    Licensed under the Apache License, Version 2.0 (the "License");
    * you may not use this file except in compliance with the License.
    * You may obtain a copy of the License at
    *
    *     http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing, software
    * distributed under the License is distributed on an "AS IS" BASIS,
    * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    * See the License for the specific language governing permissions and
    * limitations under the License.
'''

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
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
        r2 = F.r2_score(torch.flatten(y_hat), torch.flatten(y))  # R2 between mesh points.

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=len(x))
        self.log(f"{stage}_r2", r2, on_step=True, batch_size=len(batch))

        return y_hat, loss, r2

    def training_step(self, batch, batch_idx):
        _, loss, _ = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "test")
        # TODO: plotter


@MODEL_REGISTRY
class LitUnet3D(CombustionModule):
    """
    Lit wrapper to compile a 3D U-net, generic volume shapes.
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
