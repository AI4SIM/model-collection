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

import os
from typing import List, Union

import numpy as np
import torch
import xarray
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

AVAILABLE_ERA5_MASK = (
    "land_sea_mask",
    "soil_type",
    "geopotential_at_surface",
    "low_vegetation_cover",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
)

PLEVEL_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
SURFACE_VARIABLES = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]
PLEVEL_LIST = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


class ERA5DataModule(LightningDataModule):
    """Data module to load ERA5 cartesian data. See `ERA5CartesianZarr` for a
    description of the arguments.
    """

    def __init__(
        self,
        zarr_store: str,
        train_time: Union[slice, List, str] = slice("1979", "2017"),
        val_time: Union[slice, List, str] = slice("2019"),
        test_time: Union[slice, List[str], str] = ["2018", "2020", "2021"],
        train_time_step: int = 6,
        test_time_step: int = 6,
        surface_variables: List[str] = SURFACE_VARIABLES,
        plevel_variables: List[str] = PLEVEL_VARIABLES,
        plevels: List[int] = PLEVEL_LIST,
        constant_mask_list: Union[List[str], None] = [
            "land_sea_mask",
            "soil_type",
            "geopotential_at_surface",
        ],
        chunks: Union[dict, str] = "auto",
        batch_size: int = 16,
        num_workers: int = 16,
    ) -> None:
        super().__init__()
        self.zarr_store = zarr_store
        self.train_time = train_time
        self.val_time = val_time
        self.test_time = test_time
        self.train_time_step = train_time_step
        self.test_time_step = test_time_step
        self.surface_variables = surface_variables
        self.plevel_variables = plevel_variables
        self.plevels = plevels
        self.constant_mask_list = constant_mask_list
        self.chunks = chunks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_latitude = None
        self.data_longitude = None
        self.constant_masks = None

        data = xarray.open_dataset(self.zarr_store, engine="zarr", chunks=chunks)
        self.data_latitude = data.latitude.size
        self.data_longitude = data.longitude.size
        self.constant_masks = load_era5_constant_mask_from_zarr(
            self.zarr_store, mask_list=self.constant_mask_list
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = prepare_zarr_dataset(
                zarr_store=self.zarr_store,
                time=self.train_time,
                time_step=self.train_time_step,
                surface_variables=self.surface_variables,
                plevel_variables=self.plevel_variables,
                plevels=self.plevels,
                chunks=self.chunks,
            )
            self.val_dataset = prepare_zarr_dataset(
                zarr_store=self.zarr_store,
                time=self.val_time,
                time_step=self.train_time_step,
                surface_variables=self.surface_variables,
                plevel_variables=self.plevel_variables,
                plevels=self.plevels,
                chunks=self.chunks,
            )
        if stage == "test":
            self.test_dataset = prepare_zarr_dataset(
                zarr_store=self.zarr_store,
                time=self.test_time,
                time_step=self.test_time_step,
                surface_variables=self.surface_variables,
                plevel_variables=self.plevel_variables,
                plevels=self.plevels,
                chunks=self.chunks,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            multiprocessing_context="spawn",
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            multiprocessing_context="spawn",
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            multiprocessing_context="spawn",
        )


class ERA5CartesianZarr(Dataset):
    """Dataset to load ERA5 data in the zarr format, WeatherBench2 version.
    The zarr file contains all the data points, that can be loaded by chunk.
    The pressure levels have already been filtered to the 13 used by
    PanguWeather. The time is defined hourly.
    13 pressure levels: 50hPa, 100hPa, 150hPa, 200hPa, 250hPa, 300hPa, 400hPa,
    500hPa, 600hPa, 700hPa, 850hPa, 925hPa, and 1000hPa
    five upper-air atmospheric variables: geopotential, specific humidity,
    temperature, ucomponent and v-component of wind speed
    four surface weather variables: 2m temperature, u-component and
    v-component 10m wind speed, and mean sea level pressure

            Args:
                zarr_store (str): file containing pressure level data
                time (slice[str]], optional): time to extract from the
                original dataset. Defaults to slice('1979', '2017') for
                PanguWeather training.
                time_step (int, optional): time step in hours between input
                and label. Defaults to 6.
                surface_variables (list[str], optional): list of surface
                variables to select. Defaults to SURFACE_VARIABLES.
                plevel_variables (list[str], optional): list of plevel
                variables to select. Defaults to PLEVEL_VARIABLES.
                plevels (list[int], optional): list of pressure levels to
                select. Defaults to None, keeping all the 13 pressure levels.
                chunk (_type_, optional):  Chunk of data to load. Defaults to
                {'time': 2}.
    """

    def __init__(
        self,
        zarr_store: str,
        time: slice = slice("1979", "2017"),
        time_step: int = 6,
        surface_variables: List[str] = SURFACE_VARIABLES,
        plevel_variables: List[str] = PLEVEL_VARIABLES,
        plevels: List[int] = PLEVEL_LIST,
        chunks: Union[dict, str] = "auto",
    ) -> None:

        super().__init__()
        data = xarray.open_dataset(zarr_store, engine="zarr", chunks=chunks)
        data = data.sel(level=plevels)
        self.dataset = data.sel(time=time)
        self.time_step = np.timedelta64(time_step, "h")
        self.times = self.dataset.time.to_numpy().astype("datetime64[s]")
        self.data_time_span = self.times[1] - self.times[0]
        assert (
            self.time_step >= self.data_time_span
        ), f"time_step must at least equal to the time step in the data but \
        time_step={time_step} and data time step={self.data_time_span}"
        self.surface_variables = sorted(surface_variables)
        self.plevel_variables = sorted(plevel_variables)
        self.plevels = sorted(plevels)

    def __len__(self):
        offset = self.time_step // self.data_time_span
        return len(self.dataset.time) - offset

    def __getitem__(self, idx: int) -> dict[Tensor]:
        data_time = self.times[idx]
        label_time = data_time + self.time_step
        data_point = self.dataset.sel(time=data_time)
        label_point = self.dataset.sel(time=label_time)
        plevel_data = []
        plevel_labels = []
        surface_data = []
        surface_labels = []
        for plevel_variable in self.plevel_variables:
            plevel_data.append(torch.from_numpy(data_point[plevel_variable].to_numpy()))
            plevel_labels.append(
                torch.from_numpy(label_point[plevel_variable].to_numpy())
            )
        for surface_variable in self.surface_variables:
            surface_data.append(
                torch.from_numpy(data_point[surface_variable].to_numpy())
            )
            surface_labels.append(
                torch.from_numpy(label_point[surface_variable].to_numpy())
            )
        plevel_data = torch.stack(plevel_data)
        plevel_labels = torch.stack(plevel_labels)
        surface_data = torch.stack(surface_data)
        surface_labels = torch.stack(surface_labels)

        return {
            "plevel_data": plevel_data,
            "surface_data": surface_data,
            "data_time": data_time.astype(np.float32),
            "plevel_labels": plevel_labels,
            "surface_labels": surface_labels,
            "label_time": label_time.astype(np.float32),
            "longitudes": data_point.longitude.to_numpy(),
            "latitudes": data_point.latitude.to_numpy(),
        }


class MockERA5DataModule(LightningDataModule):
    """Data module to load ERA5 cartesian data. See `ERA5CartesianZarr` for a
    description of the arguments.
    """

    def __init__(
        self,
        n_samples: int = 100,
        batch_size: int = 16,
        num_workers: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.prepare_data()

    def prepare_data(self) -> None:
        self.train_dataset = MockERA5Cartesian(n_samples=self.n_samples)
        self.val_dataset = MockERA5Cartesian(n_samples=self.n_samples)
        self.test_dataset = MockERA5Cartesian(n_samples=self.n_samples)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MockERA5Cartesian(Dataset):
    """Mock ERA5 dataset for testing purpose.

    Args:
        n_samples (int, optional): number of samples.
    """

    def __init__(
        self,
        n_samples: int = 100,
    ) -> None:

        super().__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[Tensor]:
        return {
            "plevel_data": torch.rand(5, 13, 721, 1440),
            "surface_data": torch.rand(4, 721, 1440),
            "data_time": torch.rand(1),
            "plevel_labels": torch.rand(5, 13, 721, 1440),
            "surface_labels": torch.rand(4, 721, 1440),
            "label_time": torch.rand(1),
            "longitudes": torch.rand(1440),
            "latitudes": torch.rand(721),
        }


def load_era5_constant_mask_from_necdf(
    data_path: str, mask_list: list[str] = None
) -> dict[dict[xarray.Dataset]]:
    """Load constant masks: e. g., topography, land-sea and soil type. Each
    mask is in a dedicated netcdf file.

    Args:
        data_path (str): constant mask file location
        mask_list (list[str], optional): List of the required constant mask.
        Defaults to None.

    Returns:
        dict[xarray.Dataset]: list of constant data
    """
    # Check that constant masks in the list are known
    if mask_list is not None:
        mask_list = check_era5_constant_masks(mask_list=mask_list)
        mask_list.sort()
        constant_masks = dict()
        for mask in mask_list:
            mask_file = os.path.join(data_path, "constant_masks", mask + ".nc")
            mask_data = xarray.load_dataset(mask_file)
            assert (
                len(list(mask_data.keys())) == 1
            ), "Can't handle mask file containing several masks..."
            key = list(mask_data.keys())[0]
            constant_masks[mask] = {
                "longitude": mask_data.longitude,
                "latitude": mask_data.latitude,
                "mask": mask_data[key].to_numpy(),
            }
        return constant_masks
    else:
        return None


def load_era5_constant_mask_from_zarr(
    zarr_store: str, mask_list: list[str] = None
) -> dict[dict[xarray.Dataset]]:
    """Load constant masks: e. g., topography, land-sea and soil type

    Args:
        zarr_store (str): zarr store containing the constant masks
        mask_list (list[str], optional): List of the required constant mask.
        Defaults to None.

    Returns:
        dict[dict[xarray.Dataset]]: list of constant data
    """
    # open zarr storage
    data = xarray.open_zarr(zarr_store)
    # Check that constant masks in the list are known
    if mask_list is not None:
        mask_list = check_era5_constant_masks(mask_list=mask_list)
        mask_list.sort()
        constant_masks = dict()
        for mask in mask_list:
            mask_data = data[mask]
            constant_masks[mask] = {
                "longitude": data.longitude,
                "latitude": data.latitude,
                "mask": mask_data.to_numpy(),
            }

        return constant_masks
    else:
        return None


def check_era5_constant_masks(mask_list: list) -> list:
    """Check that constant masks in the list exist. If not, withdraw from the
    list.

    Args:
        mask_list (list): constant masks required

    Returns:
        list: curated list of constant masks
    """
    curated_list = []
    for mask in mask_list:
        if mask in AVAILABLE_ERA5_MASK:
            curated_list.append(mask)
        else:
            print(
                f"Constant mask {mask} doesn't exist in the data, it will be discarded"
            )
    return curated_list


def prepare_zarr_dataset(
    zarr_store, time, time_step, surface_variables, plevel_variables, plevels, chunks
):
    assert isinstance(
        time, (slice, str, list)
    ), f"time argument type must be in [slice, str, list], but is {type(time)}"
    if isinstance(time, list):
        return ConcatDataset(
            [
                ERA5CartesianZarr(
                    zarr_store=zarr_store,
                    time=t,
                    time_step=time_step,
                    surface_variables=surface_variables,
                    plevel_variables=plevel_variables,
                    plevels=plevels,
                    chunks=chunks,
                )
                for t in time
            ]
        )
    else:
        return ERA5CartesianZarr(
            zarr_store=zarr_store,
            time=time,
            time_step=time_step,
            surface_variables=surface_variables,
            plevel_variables=plevel_variables,
            plevels=plevels,
            chunks=chunks,
        )
