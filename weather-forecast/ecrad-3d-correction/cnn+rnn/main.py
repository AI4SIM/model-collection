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
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from models import CNNModel
from utils import Keys, VarInfo, get_total_features


class RadiationCorrectionModel(L.LightningModule):
    def __init__(
        self,
        flux_weights: float,
        hr_weights: float,
        lr: float,
        min_lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        fused: bool,
        num_warmup_steps: int,
    ):
        super().__init__()

        # self.flux_weights = flux_weights
        # self.hr_weights = hr_weights
        self.lr = lr
        self.min_lr = min_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.fused = fused
        self.num_warmup_steps = num_warmup_steps

        var_info = VarInfo()
        self.num_levels = var_info.hl_variables["temperature_hl"]["shape"][0]
        self.num_features = get_total_features(var_info)

        self.register_buffer("flux_weights", torch.tensor(flux_weights))
        self.register_buffer("hr_weights", torch.tensor(hr_weights))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.rad_model(x)

    def common_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> dict[str, torch.Tensor]:
        batch = batch[0]

        batch["delta_sw_diff"] = batch["sw"][..., 0] - batch["sw"][..., 1]
        batch["delta_sw_add"] = batch["sw"][..., 0] + batch["sw"][..., 1]
        batch["delta_lw_diff"] = batch["lw"][..., 0] - batch["lw"][..., 1]
        batch["delta_lw_add"] = batch["lw"][..., 0] + batch["lw"][..., 1]
        batch.pop("sw")
        batch.pop("lw")

        batch["hr_sw"] = batch["hr_sw"].squeeze(dim=-1)
        batch["hr_lw"] = batch["hr_lw"].squeeze(dim=-1)

        inputs = {
            key: value for key, value in batch.items() if key in Keys().input_keys
        }
        predictions = self(inputs)

        # if self.trainer.global_rank == 0:
        #     print(batch["hr_lw"].dtype)
        #     print(batch["hr_sw"][0, ...])
        #     print(predictions["hr_sw"][0, ...])

        # Check if the model returns sw or lw --> RNN_SW or RNN_LW
        if len(predictions) == 3:
            for key in predictions.keys():
                if "sw" not in key:
                    sw_key = key.replace("lw", "sw")
                    predictions[sw_key] = torch.empty_like(predictions[key])
                if "lw" not in key:
                    lw_key = key.replace("sw", "lw")
                    predictions[lw_key] = torch.empty_like(predictions[key])

        # losses = {
        #     "delta_sw_diff": F.mse_loss(
        #         predictions["delta_sw_diff"], batch["delta_sw_diff"]
        #     ),
        #     "delta_sw_add": F.mse_loss(
        #         predictions["delta_sw_add"], batch["delta_sw_add"]
        #     ),
        #     "delta_lw_diff": F.mse_loss(
        #         predictions["delta_lw_diff"], batch["delta_lw_diff"]
        #     ),
        #     "delta_lw_add": F.mse_loss(
        #         predictions["delta_lw_add"], batch["delta_lw_add"]
        #     ),
        #     "hr_sw": F.mse_loss(predictions["hr_sw"], batch["hr_sw"]),
        #     "hr_lw": F.mse_loss(predictions["hr_lw"], batch["hr_lw"]),
        # }
        losses = {
            k: F.mse_loss(predictions[k], batch[k])
            for k in [
                "delta_sw_diff",
                "delta_sw_add",
                "delta_lw_diff",
                "delta_lw_add",
                "hr_sw",
                "hr_lw",
            ]
        }
        loss_weights = {
            "delta_sw_diff": self.flux_weights,
            "delta_sw_add": self.flux_weights,
            "delta_lw_diff": self.flux_weights,
            "delta_lw_add": self.flux_weights,
            "hr_sw": self.hr_weights,
            "hr_lw": self.hr_weights,
        }

        individual_loss = {
            f"{stage}_{key}": losses[key] * loss_weights[key] for key in losses
        }
        total_loss = torch.sum(
            torch.tensor(
                [value for _, value in individual_loss.items()],
                requires_grad=True,
                device=self.flux_weights.device,
            )
        )

        losses_ = {f"{stage}_total_loss": total_loss, **individual_loss}
        self.log_dict(
            losses_,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            sync_dist=True,
        )

        metrics = {
            f"{stage}_mae_delta_sw_diff": F.l1_loss(
                predictions["delta_sw_diff"], batch["delta_sw_diff"]
            ),
            f"{stage}_mae_delta_sw_add": F.l1_loss(
                predictions["delta_sw_add"], batch["delta_sw_add"]
            ),
            f"{stage}_mae_delta_lw_diff": F.l1_loss(
                predictions["delta_lw_diff"], batch["delta_lw_diff"]
            ),
            f"{stage}_mae_delta_lw_add": F.l1_loss(
                predictions["delta_lw_add"], batch["delta_lw_add"]
            ),
            f"{stage}_mae_hr_sw": F.l1_loss(predictions["hr_sw"], batch["hr_sw"]),
            f"{stage}_mae_hr_lw": F.l1_loss(predictions["hr_lw"], batch["hr_lw"]),
        }
        self.log_dict(
            metrics,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
        )

        return predictions, losses_

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        _, losses = self.common_step(batch, "train")
        loss = losses["train_total_loss"]
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        predictions, _ = self.common_step(batch, "val")

    def test_test(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        predictions, _ = self.common_step(batch, "test")

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.rad_model.parameters(),
            lr=torch.tensor(self.lr),
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            fused=self.fused,
        )
        # scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.num_warmup_steps,
        #     num_training_steps=self.trainer.max_steps,
        #     min_lr=self.min_lr,
        # )

        # lr_scheduler_config = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        #     "frequency": 1,
        # }

        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optimizer


class LitCNNModel(RadiationCorrectionModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        kernel_size: int,
        dilation_rates: range,
        conv_layers: int,
        num_heads: int,
        qkv_bias: bool,
        attention_dropout: float,
        flash_attention: bool,
        colum_padding: tuple[int, ...],
        inter_padding: tuple[int, ...],
        path_to_params: str,
        flux_weights: float,
        hr_weights: float,
        lr: float,
        min_lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        fused: bool,
        num_warmup_steps: int,
    ):
        super().__init__(
            flux_weights,
            hr_weights,
            lr,
            min_lr,
            beta1,
            beta2,
            weight_decay,
            fused,
            num_warmup_steps,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.conv_layers = conv_layers
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.flash_attention = flash_attention
        self.colum_padding = colum_padding
        self.inter_padding = inter_padding
        self.path_to_params = path_to_params

        self.save_hyperparameters()

    def setup(self, stage: str):
        with self.trainer.init_module(empty_init=False):
            self.rad_model = CNNModel(
                self.in_channels,
                self.out_channels,
                self.hidden_size,
                self.kernel_size,
                self.dilation_rates,
                self.conv_layers,
                self.num_heads,
                self.qkv_bias,
                self.attention_dropout,
                self.flash_attention,
                self.colum_padding,
                self.inter_padding,
                self.path_to_params,
            )
