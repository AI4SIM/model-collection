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

import lightning as L
import torch
from torch.optim import AdamW
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from typing import Dict


class RadiationCorrectionModel(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        min_lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        fused: bool,
        num_warmup_steps: int,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.fused = fused
        self.num_warmup_steps = num_warmup_steps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def common_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        # x = batch["x"]
        # pressure_hl = batch["pressure_hl"]
        # y_hat = self(x, pressure_hl)
        # y = batch["y"]
        # loss = self.loss(y_hat, y)
        # return {"loss": loss, "y_hat": y_hat, "y": y}
        pass

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.common_step(batch, batch_idx)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.common_step(batch, batch_idx)

    def test_test(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.common_step(batch, batch_idx)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=torch.tensor(self.lr),
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            fused=self.fused,
        )
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.max_steps,
            min_lr=self.min_lr,
        )

        return [optimizer], [scheduler]
