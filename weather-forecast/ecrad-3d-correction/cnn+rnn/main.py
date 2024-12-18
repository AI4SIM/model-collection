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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import defaultdict

from models import CNNModel
from utils import Keys
from plotter import Plotter


class RadiationCorrectionModel(L.LightningModule):
    def __init__(
        self,
        batch_size: int,
        flux_weights: float,
        hr_weights: float,
        initial_lr: float,
        min_lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        fused: bool,
        use_lr_scheduler: bool,
        num_warmup_steps: int,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.fused = fused
        self.use_lr_scheduler = use_lr_scheduler
        self.num_warmup_steps = num_warmup_steps

        self.register_buffer("flux_weights", torch.tensor(flux_weights))
        self.register_buffer("hr_weights", torch.tensor(hr_weights))

        self.outputs = defaultdict(lambda: defaultdict(list))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.rad_model(x)

    def common_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> dict[str, torch.Tensor]:

        if self.trainer.state.fn in ["fit", "validate"]:
            batch = batch[0]

        batch["delta_sw_diff"] = batch["sw"][..., 0] - batch["sw"][..., 1]
        batch["delta_sw_add"] = batch["sw"][..., 0] + batch["sw"][..., 1]
        batch["delta_lw_diff"] = batch["lw"][..., 0] - batch["lw"][..., 1]
        batch["delta_lw_add"] = batch["lw"][..., 0] + batch["lw"][..., 1]

        batch["hr_sw"] = batch["hr_sw"].squeeze(dim=-1)
        batch["hr_lw"] = batch["hr_lw"].squeeze(dim=-1)

        inputs = {
            key: value for key, value in batch.items() if key in Keys().input_keys
        }
        predictions = self(inputs)

        # Check if the model returns sw or lw --> RNN_SW or RNN_LW
        if len(predictions) == 3:
            pred_ = {}
            for key in predictions:
                if "sw" not in key:
                    sw_key = key.replace("lw", "sw")
                    pred_[sw_key] = torch.empty_like(predictions[key])
                if "lw" not in key:
                    lw_key = key.replace("sw", "lw")
                    pred_[lw_key] = torch.empty_like(predictions[key])
            predictions = {**predictions, **pred_}

        losses = {
            f"{stage}_{k}": F.mse_loss(predictions[k], batch[k]) for k in predictions
        }
        loss_weights = {
            "delta_sw_diff": self.flux_weights,
            "delta_sw_add": self.flux_weights,
            "delta_lw_diff": self.flux_weights,
            "delta_lw_add": self.flux_weights,
            "hr_sw": self.hr_weights,
            "hr_lw": self.hr_weights,
        }

        individual_weighted_loss = {
            f"{stage}_loss_{key}": losses[f"{stage}_{key}"] * loss_weights[key]
            for key in loss_weights
        }

        total_loss = 0.0
        for key in individual_weighted_loss:
            total_loss = total_loss + individual_weighted_loss[key]

        losses_ = {f"{stage}_total_loss": total_loss, **losses}

        self.log_dict(
            losses_,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
        )

        if stage == "val":
            metrics = {
                f"{stage}_mae_{k}": F.l1_loss(predictions[k], batch[k])
                for k in predictions
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
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        predictions, _ = self.common_step(batch, "val")
        # batch = batch[0]
        # for f in ["sw", "lw"]:
        #     predictions["dn_" + f] = 0.5 * (
        #         predictions[f"delta_{f}_diff"] + predictions[f"delta_{f}_add"]
        #     )
        #     predictions["up_" + f] = 0.5 * (
        #         predictions[f"delta_{f}_add"] - predictions[f"delta_{f}_diff"]
        #     )
        #     batch["dn_" + f] = batch[f][..., 0]
        #     batch["up_" + f] = batch[f][..., 1]

        # for key in predictions.keys():
        #     self.outputs["preds"][key].append(predictions[key])
        #     self.outputs["targets"][key].append(batch[key])

    # def on_validation_epoch_end(self):
    #     preds = {k: torch.cat(v) for k, v in self.outputs["preds"].items()}
    #     targets = {k: torch.cat(v) for k, v in self.outputs["targets"].items()}

    #     preds = self.all_gather(preds)
    #     targets = self.all_gather(targets)
    #     for key in preds:
    #         preds[key] = (
    #             preds[key].reshape(-1, preds[key].shape[-1]).detach().cpu().numpy()
    #         )
    #         targets[key] = (
    #             targets[key].reshape(-1, targets[key].shape[-1]).detach().cpu().numpy()
    #         )

    #     plotter = Plotter(preds, targets)
    #     for flux in ["sw", "lw"]:
    #         fig_samples = plotter.random_samples(flux=flux, n_samples=4)
    #         fig_mae_bias = plotter.mae_bias_profile(flux=flux)
    #         self.logger.experiment.add_figure(
    #             f"validation_{flux}_random_samples",
    #             fig_samples,
    #             global_step=self.current_epoch,
    #         )
    #         self.logger.experiment.add_figure(
    #             f"validation_{flux}_mae_bias_profiles",
    #             fig_mae_bias,
    #             global_step=self.current_epoch,
    #         )

    #     self.outputs.clear()
    #     preds.clear()
    #     targets.clear()
    #     del plotter

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        predictions, _ = self.common_step(batch, "test")
        for f in ["sw", "lw"]:
            predictions["dn_" + f] = 0.5 * (
                predictions[f"delta_{f}_diff"] + predictions[f"delta_{f}_add"]
            )
            predictions["up_" + f] = 0.5 * (
                predictions[f"delta_{f}_add"] - predictions[f"delta_{f}_diff"]
            )
            batch["dn_" + f] = batch[f][..., 0]
            batch["up_" + f] = batch[f][..., 1]

        for key in predictions.keys():
            self.outputs["preds"][key].append(predictions[key])
            self.outputs["targets"][key].append(batch[key])

    def on_test_epoch_end(self):
        preds = {k: torch.cat(v) for k, v in self.outputs["preds"].items()}
        targets = {k: torch.cat(v) for k, v in self.outputs["targets"].items()}

        preds = self.all_gather(preds)
        targets = self.all_gather(targets)
        for key in preds:
            preds[key] = (
                preds[key].reshape(-1, preds[key].shape[-1]).detach().cpu().numpy()
            )
            targets[key] = (
                targets[key].reshape(-1, targets[key].shape[-1]).detach().cpu().numpy()
            )

        plotter = Plotter(preds, targets)
        for flux in ["sw", "lw"]:
            fig_samples = plotter.random_samples(flux=flux, n_samples=4)
            fig_mae_bias = plotter.mae_bias_profile(flux=flux)
            self.logger.experiment.add_figure(
                f"validation_{flux}_random_samples",
                fig_samples,
                global_step=self.current_epoch,
            )
            self.logger.experiment.add_figure(
                f"validation_{flux}_mae_bias_profiles",
                fig_mae_bias,
                global_step=self.current_epoch,
            )

        self.outputs.clear()
        preds.clear()
        targets.clear()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        true_lr, true_min_lr = tuple(
            [
                init_lr * self.batch_size / 128 * self.trainer.world_size
                for init_lr in [self.initial_lr, self.min_lr]
            ]
        )
        optimizer = AdamW(
            self.rad_model.parameters(),
            lr=torch.tensor(true_lr),
            betas=(self.beta1, self.beta2),
            eps=1.0e-7,
            weight_decay=self.weight_decay,
            fused=self.fused,
        )
        if self.use_lr_scheduler:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.num_warmup_steps,
                T_mult=2,
                eta_min=true_min_lr,
            )

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
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
        batch_size: int,
        initial_lr: float,
        min_lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        fused: bool,
        use_lr_scheduler: bool,
        num_warmup_steps: int,
    ):
        super().__init__(
            batch_size,
            flux_weights,
            hr_weights,
            initial_lr,
            min_lr,
            beta1,
            beta2,
            weight_decay,
            fused,
            use_lr_scheduler,
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
        with self.trainer.init_module(empty_init=True):
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
