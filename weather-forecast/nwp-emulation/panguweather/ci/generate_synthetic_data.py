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
import os.path as osp
import sys
from datetime import datetime, timedelta
from os.path import exists

import numpy as np
import xarray as xr
import zarr

sys.path.insert(1, "/".join(osp.realpath(__file__).split("/")[0:-2]))

# isort: off
from data import (  # noqa: E402
    PLEVEL_LIST,
    PLEVEL_VARIABLES,
    SURFACE_VARIABLES,
)


def create_mock_era5_zarr(
    output_path: str = "./data",
    n_lat=120,
    n_lon=240,
    n_samples: int = 10,
    start_date: str = "2020-01-01",
    time_step_hours: int = 6,
    include_constant_masks: bool = True,
):
    """
    Create a mock ERA5 dataset and store it in a Zarr file format.

    Args:
        output_path (str): Path where the Zarr store will be created
        n_samples (int): Number of time samples to generate
        n_lat (int): Number of latitude points
        n_lon (int): Number of longitude points
        start_date (str): Starting date in YYYY-MM-DD format
        time_step_hours (int): Time step between samples in hours
        include_constant_masks (bool): Whether to include constant mask variables

    Returns:
        str: Path to the created Zarr store
    """
    if exists(output_path):
        raise Exception(f"Remove manually {output_path}")

    print(f"Creating mock ERA5 dataset with {n_samples} samples...")

    # Create time coordinates
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    time_values = np.array(
        [
            start_datetime + timedelta(hours=i * time_step_hours)
            for i in range(n_samples)
        ],
        dtype="datetime64[ns]",
    )

    # Create lat, lon coordinates
    lat_values = np.linspace(-90, 90, n_lat)
    lon_values = np.linspace(-180, 180, n_lon)

    # Create pressure levels
    plevels = PLEVEL_LIST  # Using the predefined levels from the module

    # Initialize data arrays
    data_vars = {}

    # Create pressure level variables
    for i, var_name in enumerate(PLEVEL_VARIABLES):
        # Extract the ith variable from all samples
        data = np.random.rand(n_samples, len(plevels), n_lat, n_lon).astype(np.float32)
        # Create DataArray
        data_vars[var_name] = xr.DataArray(
            data,
            dims=["time", "level", "latitude", "longitude"],
            coords={
                "time": time_values,
                "level": plevels,
                "latitude": lat_values,
                "longitude": lon_values,
            },
            attrs={"long_name": var_name, "units": "mock_units"},
        )

    # Create surface variables
    for i, var_name in enumerate(SURFACE_VARIABLES):
        # Extract the ith variable from all samples
        data = np.random.rand(n_samples, n_lat, n_lon).astype(np.float32)

        # Create DataArray
        data_vars[var_name] = xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": time_values,
                "latitude": lat_values,
                "longitude": lon_values,
            },
            attrs={"long_name": var_name, "units": "mock_units"},
        )

    # Add constant masks if requested
    if include_constant_masks:
        for mask_name in ["land_sea_mask", "soil_type", "geopotential_at_surface"]:
            if mask_name == "land_sea_mask":
                # Land-sea mask is binary (0 for sea, 1 for land)
                data = np.random.randint(0, 2, (n_lat, n_lon)).astype(np.float32)
            elif mask_name == "soil_type":
                # Soil type typically ranges from 0 to 9
                data = np.random.randint(0, 10, (n_lat, n_lon)).astype(np.float32)
            elif mask_name == "geopotential_at_surface":
                # Surface geopotential typically ranges from ~0 to ~9000 m^2/s^2
                data = np.random.uniform(0, 9000, (n_lat, n_lon)).astype(np.float32)
            else:
                data = np.random.rand(n_lat, n_lon).astype(np.float32)

            # Create DataArray
            data_vars[mask_name] = xr.DataArray(
                data,
                dims=["latitude", "longitude"],
                coords={"latitude": lat_values, "longitude": lon_values},
                attrs={"long_name": mask_name, "units": "mock_units"},
            )

    # Create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={
            "title": "Mock ERA5 Dataset",
            "creation_date": str(datetime.now()),
            "description": "Synthetic ERA5 data created for testing purposes",
        },
    )

    # Set encoding for efficient storage
    encoding = {
        var: {"compressor": zarr.Blosc(cname="zstd", clevel=3)} for var in ds.variables
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to zarr
    print(f"Writing dataset to Zarr store at {output_path}...")
    ds.to_zarr(output_path, encoding=encoding, consolidated=True)

    print(f"Mock ERA5 Zarr dataset created successfully at {output_path}")
    return output_path


if __name__ == "__main__":
    zarr_path = "./data"
    create_mock_era5_zarr(
        output_path=zarr_path,
        n_samples=10,
        start_date="2020-01-01",
        time_step_hours=6,
        include_constant_masks=True,
    )
