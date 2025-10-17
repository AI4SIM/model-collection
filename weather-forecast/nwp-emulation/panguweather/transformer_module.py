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


from functools import cached_property
from typing import Any, List, Type, Union

import numpy as np
import torch
import xarray
from lightning import LightningModule
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.types import (
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
)
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn, optim

from data import PLEVEL_LIST, PLEVEL_VARIABLES, SURFACE_VARIABLES
from metrics import LatitudeWeightedMeanSquaredError
from pangu import PanguModel

PLEVEL_VARIABLE_WEIGHTS = {
    "geopotential": 3.0,
    "specific_humidity": 0.6,
    "temperature": 1.5,
    "u_component_of_wind": 0.77,
    "v_component_of_wind": 0.54,
}
SURFACE_VARIABLE_WEIGHTS = {
    "mean_sea_level_pressure": 1.5,
    "10m_u_component_of_wind": 0.77,
    "10m_v_component_of_wind": 0.66,
    "2m_temperature": 3.0,
}


class PanguModule(LightningModule):
    """Base transformer module to train PanguWeather.
    Args:
        criterion (nn.Module): Loss function. Default is mean absolute error (L1).
        loss_weight (float): Weight of the surface variable loss. Default is 0.25.
        lr (float): Learning rate. Default is 5e-4.
        plevel_variable_weights (tuple): Weight of each plevel variable in the loss.
        Default is (3.0, 0.6, 1.5, 0.77, 0.54) for Z, Q, T, U and V, respectively.
        surface_variable_weights (tuple): Weight of each surface variable in the loss.
        Default is (1.5, 0.77, 0.66, 3.0) for MSLP, U10, V10 and T2M, respectively.
        time_step (int): timeframe of one forecast iteration in hour.
    """

    def __init__(
        self,
        constant_masks: dict,
        latitude: int,
        longitude: int,
        criterion: Type[nn.Module] = nn.L1Loss,
        loss_weight: float = 0.25,
        lr: float = 5e-4,
        weight_decay: float = 3e-6,
        plevel_variable_weights: tuple | None = None,
        surface_variable_weights: tuple | None = None,
        surface_variables: List = SURFACE_VARIABLES,
        plevel_variables: List = PLEVEL_VARIABLES,
        plevels: List = PLEVEL_LIST,
        time_step: int = 6,
        mean_file: str | None = None,
        stddev_file: str | None = None,
        wl_norm: bool = False,
        norm_cst_mask: bool = False,
        use_scheduler: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.criterion = criterion()
        self.loss_weight = loss_weight
        self.lr = lr
        self.weight_decay = weight_decay
        if plevel_variable_weights is None:
            plevel_variable_weights = tuple(
                v for _, v in sorted(PLEVEL_VARIABLE_WEIGHTS.items())
            )
        assert plevel_variable_weights is not None
        self.plevel_variable_weights = plevel_variable_weights
        if surface_variable_weights is None:
            surface_variable_weights = tuple(
                v for _, v in sorted(SURFACE_VARIABLE_WEIGHTS.items())
            )
        assert surface_variable_weights is not None
        self.surface_variable_weights = surface_variable_weights
        self.time_step = time_step
        self.use_scheduler = use_scheduler

        self.plevel_variables = sorted(plevel_variables)
        self.surface_variables = sorted(surface_variables)
        self.plevels = sorted(plevels)
        # Compute RMSE for Z500, T850, T2M and U10
        self.rmse_z500 = LatitudeWeightedMeanSquaredError(squared=False)
        self.rmse_t850 = LatitudeWeightedMeanSquaredError(squared=False)
        self.rmse_t2m = LatitudeWeightedMeanSquaredError(squared=False)
        self.rmse_u10 = LatitudeWeightedMeanSquaredError(squared=False)

        # Load static data
        if mean_file is not None and stddev_file is not None:
            era5_means = xarray.load_dataset(mean_file)
            file_surface_means = np.stack(
                [
                    era5_means[v].to_numpy().astype(np.float32)
                    for v in self.surface_variables
                ]
            )
            surface_means_tensor = torch.from_numpy(file_surface_means)
            self.surface_means: Tensor
            self.register_buffer(
                "surface_means",
                surface_means_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0),
            )
            file_plevel_means = np.stack(
                [
                    era5_means[v].to_numpy().astype(np.float32)
                    for v in self.plevel_variables
                ]
            )
            plevel_means_tensor = torch.from_numpy(file_plevel_means)
            self.plevel_means: Tensor
            self.register_buffer(
                "plevel_means",
                plevel_means_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0),
            )
            era5_stddev = xarray.load_dataset(stddev_file)
            file_surface_stddev = np.stack(
                [
                    era5_stddev[v].to_numpy().astype(np.float32)
                    for v in self.surface_variables
                ]
            )
            surface_stddev_tensor = torch.from_numpy(file_surface_stddev)
            self.surface_stddev: Tensor
            self.register_buffer(
                "surface_stddev",
                surface_stddev_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0),
            )
            file_plevel_stddev = np.stack(
                [
                    era5_stddev[v].to_numpy().astype(np.float32)
                    for v in self.plevel_variables
                ]
            )
            plevel_stddev_tensor = torch.from_numpy(file_plevel_stddev)
            self.plevel_stddev: Tensor
            self.register_buffer(
                "plevel_stddev",
                plevel_stddev_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0),
            )
            self.normalize_data = True
        else:
            self.normalize_data = False

        if norm_cst_mask and constant_masks is not None:
            for cst_data in constant_masks.values():
                cst_data["mask"] = (
                    cst_data["mask"] - cst_data["mask"].mean()
                ) / cst_data["mask"].std()

        self.model = PanguModel(
            constant_masks=constant_masks,
            latitude=latitude,
            longitude=longitude,
            surface_variables=len(surface_variables),
            plevel_variables=len(plevel_variables),
            plevels=len(plevels),
            **kwargs,
        )

    def forward(self, input_plevel: Tensor, input_surface: Tensor) -> dict:
        return self.model(input_plevel, input_surface)

    def normalize(
        self, plevel_data: Tensor, surface_data: Tensor
    ) -> tuple[Tensor, Tensor]:
        plevel_data_normlized = (plevel_data - self.plevel_means) / self.plevel_stddev
        surface_data_normlized = (
            surface_data - self.surface_means
        ) / self.surface_stddev
        return plevel_data_normlized, surface_data_normlized

    def denormalize(
        self, plevel_data: Tensor, surface_data: Tensor
    ) -> tuple[Tensor, Tensor]:
        plevel_data_denormlized = plevel_data * self.plevel_stddev + self.plevel_means
        surface_data_denormlized = (
            surface_data * self.surface_stddev + self.surface_means
        )
        return plevel_data_denormlized, surface_data_denormlized

    def common_step(
        self, batch: dict, stage: str, batch_idx: int | None = None
    ) -> STEP_OUTPUT:
        plevel_data = batch["plevel_data"]
        surface_data = batch["surface_data"]
        plevel_labels = batch["plevel_labels"]
        surface_labels = batch["surface_labels"]

        if self.normalize_data:
            plevel_data, surface_data = self.normalize(plevel_data, surface_data)
            plevel_labels, surface_labels = self.normalize(
                plevel_labels, surface_labels
            )
        plevel_output, surface_output = self(plevel_data, surface_data)
        # plevel data has shape (B, variables, levels, lat, long)
        loss_plevel = torch.sum(
            torch.stack(
                [
                    self.criterion(plevel_output[:, i], plevel_labels[:, i]) * weight
                    for i, weight in enumerate(self.plevel_variable_weights)
                ]
            )
        )
        # surface data has shape (B, variables, lat, long)
        loss_surface = torch.sum(
            torch.stack(
                [
                    self.criterion(surface_output[:, i], surface_labels[:, i]) * weight
                    for i, weight in enumerate(self.surface_variable_weights)
                ]
            )
        )
        loss = loss_plevel + loss_surface * self.loss_weight

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=True,
            batch_size=len(plevel_data),
            sync_dist=True,
        )

        if self.global_rank == 0:
            nn_norm = 0
            for p in self.parameters():
                nn_norm += (p.clone().to("cpu") ** 2).sum()
            self.log("Model norm", nn_norm, prog_bar=True, on_step=True)

        if stage == "val":
            if self.normalize_data:
                plevel_output, surface_output = self.denormalize(
                    plevel_output, surface_output
                )
                plevel_labels, surface_labels = self.denormalize(
                    plevel_labels, surface_labels
                )
            # Compute RMSE for Z500, T850, T2M and U10
            z500_index = (
                self.plevel_variables.index("geopotential"),
                self.plevels.index(500),
            )
            t850_index = (
                self.plevel_variables.index("temperature"),
                self.plevels.index(850),
            )
            t2m_index = self.surface_variables.index("2m_temperature")
            u10_index = self.surface_variables.index("10m_u_component_of_wind")
            self.rmse_z500(
                plevel_output[:, z500_index[0], z500_index[1]],
                plevel_labels[:, z500_index[0], z500_index[1]],
                batch["latitudes"],
            )
            self.rmse_t850(
                plevel_output[:, t850_index[0], t850_index[1]],
                plevel_labels[:, t850_index[0], t850_index[1]],
                batch["latitudes"],
            )
            self.rmse_t2m(
                surface_output[:, t2m_index],
                surface_labels[:, t2m_index],
                batch["latitudes"],
            )
            self.rmse_u10(
                surface_output[:, u10_index],
                surface_labels[:, u10_index],
                batch["latitudes"],
            )
            # Log RMSEs
            self.log(f"{stage}_rmse_z500", self.rmse_z500, on_epoch=True)
            self.log(f"{stage}_rmse_t850", self.rmse_t850, on_epoch=True)
            self.log(f"{stage}_rmse_t2m", self.rmse_t2m, on_epoch=True)
            self.log(f"{stage}_rmse_u10", self.rmse_u10, on_epoch=True)

        if stage == "train":
            esb_norm = {}
            for n, p in self.named_parameters():
                if "earth_specific_bias" in n:
                    esb_norm[n] = torch.linalg.norm(p).item()
            self.log_dict(esb_norm, on_step=True, rank_zero_only=True)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        return self.common_step(batch=batch, stage="train", batch_idx=batch_idx)

    def validation_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT | None:
        return self.common_step(batch=batch, stage="val", batch_idx=batch_idx)

    def test_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT | None:
        plevel_data = batch["plevel_data"]
        surface_data = batch["surface_data"]
        plevel_labels = batch["plevel_labels"]
        surface_labels = batch["surface_labels"]
        if self.normalize_data:
            plevel_data, surface_data = self.normalize(plevel_data, surface_data)
        data_time = batch["data_time"]
        label_time = batch["label_time"]

        # Compute data time step
        l_time = label_time[0].cpu().numpy().astype(np.int64)
        d_time = data_time[0].cpu().numpy().astype(np.int64)
        time_horizon = np.timedelta64(l_time - d_time, "s")
        num_iterations = time_horizon // np.timedelta64(self.time_step, "h")
        try:
            time_horizon % np.timedelta64(self.time_step, "h") == 0
        except ValueError:
            print(
                "The time horizon is not divisible by the model time step, it will amount to: ",
                num_iterations * self.time_step,
            )
        last_plevel_predictions = plevel_data
        last_surface_predictions = surface_data
        for _ in range(num_iterations):
            last_plevel_predictions, last_surface_predictions = self(
                last_plevel_predictions, last_surface_predictions
            )
        if self.normalize_data:
            last_plevel_predictions, last_surface_predictions = self.denormalize(
                last_plevel_predictions, last_surface_predictions
            )
        # Compute RMSE for Z500, T850, T2M and U10
        z500_index = (
            self.plevel_variables.index("geopotential"),
            self.plevels.index(500),
        )
        t850_index = (
            self.plevel_variables.index("temperature"),
            self.plevels.index(850),
        )
        t2m_index = self.surface_variables.index("2m_temperature")
        u10_index = self.surface_variables.index("10m_u_component_of_wind")
        self.rmse_z500(
            last_plevel_predictions[:, z500_index[0], z500_index[1]],
            plevel_labels[:, z500_index[0], z500_index[1]],
            batch["latitudes"],
        )
        self.rmse_t850(
            last_plevel_predictions[:, t850_index[0], t850_index[1]],
            plevel_labels[:, t850_index[0], t850_index[1]],
            batch["latitudes"],
        )
        self.rmse_t2m(
            last_surface_predictions[:, t2m_index],
            surface_labels[:, t2m_index],
            batch["latitudes"],
        )
        self.rmse_u10(
            last_surface_predictions[:, u10_index],
            surface_labels[:, u10_index],
            batch["latitudes"],
        )

        # Log RMSEs
        self.log("test_rmse_z500", self.rmse_z500, on_epoch=True)
        self.log("test_rmse_t850", self.rmse_t850, on_epoch=True)
        self.log("test_rmse_t2m", self.rmse_t2m, on_epoch=True)
        self.log("test_rmse_u10", self.rmse_u10, on_epoch=True)

        return None

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        plevel_data = batch["plevel_data"]
        surface_data = batch["surface_data"]
        if self.normalize_data:
            plevel_data, surface_data = self.normalize(plevel_data, surface_data)
        data_time = batch["data_time"]
        plevel_output, surface_output = self(plevel_data, surface_data)
        if self.normalize_data:
            plevel_output, surface_output = self.denormalize(
                plevel_output, surface_output
            )
        prediction_time = data_time + self.time_step
        return {
            "plevel_predictions": plevel_output,
            "surface_predictions": surface_output,
            "prediction_times": prediction_time,
        }

    def configure_optimizers(self) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        optim_dict: Union[OptimizerConfig, OptimizerLRSchedulerConfig]
        if self.use_scheduler:
            max_epochs = self.trainer.max_epochs

            if max_epochs is None:
                raise ValueError(
                    "self.trainer.max_epochs is not set, \
                    but it's required for CosineAnnealingLR scheduler."
                )
            optim_dict = dict(
                optimizer=optimizer,
                lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, max_epochs
                ),
            )
        else:
            optim_dict = dict(optimizer=optimizer)
        return optim_dict

    def on_before_optimizer_step(self, optimizer: optim.Optimizer) -> None:
        from lightning.pytorch.utilities import grad_norm

        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True, rank_zero_only=True)

    @cached_property
    def mlflow_logger(self) -> Union[MLFlowLogger, None]:
        """Get the MLFlowLogger if it has been set."""
        return next(
            iter([o for o in self.loggers if isinstance(o, MLFlowLogger)]), None
        )
