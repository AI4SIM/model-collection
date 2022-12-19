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
import sys
import climetlab as cml
import os.path as osp
import xarray as xr

from typing import Dict, Tuple, Union


class ThreeDCorrectionDataProc:
    """
    Download, preprocess and shard the data of the 3DCorrection use-case.
    To be called once before experiments to build the dataset on the filesystem.
    """

    def __init__(self,
                 root: str,
                 date: str = "20200101",
                 timestep: int = 500,
                 patchstep: int = 1) -> None:

        self.raw_path = osp.join(root, "raw")
        self.processed_path = osp.join(root, "processed")
        self.date = date
        self.timestep = timestep
        self.patchstep = patchstep

    def process(self):
        """
        Proceeds with all the following steps:
            * Download the raw data and return a xarray Dataset,
            * Extract the features and targets into different datasets,
            * Modify the target to ease the training.
        """
        print("Processing...", file=sys.stderr)

        xr_dataset = self.download()
        feature_dataset = self.extract_feature(xr_dataset)
        target_dataset = self.extract_target(xr_dataset)
        target_dataset = self.expand_target(target_dataset)

        print("Done!", file=sys.stderr)
        feature_dataset.close()
        target_dataset.close()

        return feature_dataset, target_dataset

    def download(self) -> xr.Dataset:
        """Download the data for 3D correction US and return an xr.Dataset. """
        cml.settings.set("cache-directory", self.raw_path)
        cml_ds = cml.load_dataset(
            "maelstrom-radiation",
            dataset="3dcorrection",
            date=self.date,
            raw_inputs=True,
            minimal_outputs=False,
            all_outputs=True,
            timestep=list(range(0, 3501, self.timestep)),
            patch=list(range(0, 16, self.patchstep)),
            hr_units="k d-1",
        )
        
        xrds = cml_ds.to_xarray()
        xrds.close()
        
        # # Test if directory exists
        # xrds.to_zarr(osp.join(self.processed_path, "3dcorrection.zarr"))
        # zarr_ds = xr.open_zarr(osp.join(self.processed_path, "3dcorrection.zarr"))

        return xrds # zarr_ds

    def extract_feature(self, ds) -> xr.Dataset:
        """Extract and order the features in a new dataset."""
        feature_keys = [
            "skin_temperature",
            "cos_solar_zenith_angle",
            "sw_albedo",
            "sw_albedo_direct",
            "lw_emissivity",
            "solar_irradiance",
            "q",
            "o3_mmr",
            "co2_vmr",
            "n2o_vmr",
            "ch4_vmr",
            "o2_vmr",
            "cfc11_vmr",
            "cfc12_vmr",
            "hcfc22_vmr",
            "ccl4_vmr",
            "cloud_fraction",
            "aerosol_mmr",
            "q_liquid",
            "q_ice",
            "re_liquid",
            "re_ice",
            "temperature_hl",
            "pressure_hl",
            "overlap_param",
        ]
        feature_ds = xr.Dataset()

        for key in feature_keys:
            feature_ds[key] = ds[key]

        return feature_ds

    def extract_target(self, ds) -> xr.Dataset:
        """Extract the targets them in a new dataset."""
        target_keys = [
            "hr_sw",
            "hr_lw",
            "flux_dn_sw",
            "flux_up_sw",
            "flux_dn_lw",
            "flux_up_lw",
        ]
        target_ds = xr.Dataset()

        for key in target_keys:
            target_ds[key] = ds[key]
        
        return target_ds

    def expand_target(self, target_ds: xr.Dataset) -> None:
        """Modify the targets to ease the learning."""
        target_ds["delta_sw_diff"] = target_ds["flux_dn_sw"] - target_ds["flux_up_sw"]
        target_ds["delta_sw_add"] = target_ds["flux_dn_sw"] + target_ds["flux_up_sw"]
        target_ds["delta_lw_diff"] = target_ds["flux_dn_lw"] - target_ds["flux_up_lw"]
        target_ds["delta_lw_add"] = target_ds["flux_dn_lw"] + target_ds["flux_up_lw"]
        return target_ds
