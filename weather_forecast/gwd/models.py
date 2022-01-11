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

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_optimizer as optim
import torchmetrics.functional as F


class NOGWDModule(pl.LightningModule):

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)

        loss = F.mean_squared_error(y_hat, y)
        r2 = F.r2_score(y_hat, y)

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


@MODEL_REGISTRY
class LitMLP(NOGWDModule):

    def __init__(self, in_channels, hidden_channels, out_channels, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),)

    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.ModuleList([
#             Reshape(),
#             nn.Conv1d(5, 16, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, 3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear((32 - 3) // 2 * 32, 126),
#             nn.Linear(126, 126)
#         ])

#     def forward(self, x):
#         for module in self.model:
#             x = module(x)
#             # print(x.shape)
#         return x

@MODEL_REGISTRY
class LitCNN(NOGWDModule):

    class Reshape(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            t0 = torch.reshape(x[:, :3 * 63], [-1, 3, 63])
            t1 = torch.tile(x[:, 3 * 63:].unsqueeze(2), (1, 1, 63))

            return torch.cat((t0, t1), dim=1)
                # torch.transpose(t0, -2, -1),
                # torch.transpose(t1, -2, -1)
            # ), dim=2)

    def __init__(self, in_channels, init_feat, conv_size, pool_size, out_channels, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = nn.Sequential(
            self.Reshape(),
            nn.Conv1d(5, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear((32 - 3) // 2 * 32, 126),
            nn.Linear(126, 126))

    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)
