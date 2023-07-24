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

import glob
import os
import sys
import torch
import climetlab as cml
import os.path as osp
import xarray as xr

from typing import Tuple

import keys

import tensorflow
cpu = tensorflow.config.list_physical_devices('CPU')
gpus = tensorflow.config.list_physical_devices('GPU')
try:
    tensorflow.config.set_visible_devices([], 'GPU')
    tensorflow.config.set_visible_devices(cpu, 'CPU')
    # tensorflow.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("Something wrong happened...")


class ThreeDCorrectionDataProc:
    """
    Download, preprocess and shard the data of the 3DCorrection use-case.
    To be called once before experiments to build the dataset on the filesystem.
    """

    def __init__(
        self,
        root: str,
        date: str = "20200101",
        timestep: int = 500,
        patchstep: int = 1,
        subset: str = None,
        norm: bool = False,
    ) -> None:
        self.raw_path = osp.join(root, "raw")
        self.processed_path = osp.join(root, "processed")
        self.date = date
        self.timestep = timestep
        self.patchstep = patchstep
        self.subset = subset
        self.norm = norm

    def process(self) -> Tuple[xr.Dataset]:
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
        self.expand_target(target_dataset)

        print("Done!", file=sys.stderr)
        feature_dataset.close()
        target_dataset.close()

        return feature_dataset, target_dataset

    def download(self) -> xr.Dataset:
        """Download the data for 3D correction US and return an xr.Dataset."""
        cml.settings.set("cache-directory", self.raw_path)

        if self.subset is not None:
            cml_ds = cml.load_dataset(
                "maelstrom-radiation",
                dataset="3dcorrection",
                subset=self.subset,
                raw_inputs=True,
                minimal_outputs=False,
                all_outputs=True,
                heating_rate=True,
                hr_units="k d-1",
            )
        else:
            cml_ds = cml.load_dataset(
                "maelstrom-radiation",
                dataset="3dcorrection",
                date=self.date,
                timestep=list(range(0, 3501, self.timestep)),
                patch=list(range(0, 16, self.patchstep)),
                raw_inputs=True,
                minimal_outputs=False,
                all_outputs=True,
                heating_rate=True,
                hr_units="k d-1",
            )

        xrds = cml_ds.to_xarray()
        xrds.close()

        # The normalisation parameters are only available for the tf-dataset
        # Loading a small TFRecordDataset to call load_norms()
        if self.norm:
            norm_path = osp.join(self.raw_path, "norms")
            cml.settings.set("cache-directory", norm_path)
            dummy_ds = cml.load_dataset(
                "maelstrom-radiation-tf",
                dataset="3dcorrection",
                timestep=[0],
                filenum=[0],
                norm=False,
            )
            self.means, self.stds = dummy_ds.load_norms(dataset=None)

            for key in self.means.keys():
                self.means[key] = torch.from_numpy(self.means[key].numpy())
                self.stds[key] = torch.from_numpy(self.stds[key].numpy())

        return xrds

    def extract_feature(self, ds) -> xr.Dataset:
        """Extract and order the features in a new dataset."""
        feature_ds = xr.Dataset()

        for key in [*keys.SCA, *keys.COL, *keys.HL, *keys.INTER]:
            feature_ds[key] = ds[key]

        return feature_ds

    def extract_target(self, ds) -> xr.Dataset:
        """Extract the targets them in a new dataset."""
        target_ds = xr.Dataset()

        for key in keys.TARGET:
            target_ds[key] = ds[key]

        return target_ds

    def expand_target(self, target_ds: xr.Dataset) -> None:
        """Modify the targets to ease the learning."""
        target_ds["delta_sw_diff"] = target_ds["flux_dn_sw"] - target_ds["flux_up_sw"]
        target_ds["delta_sw_add"] = target_ds["flux_dn_sw"] + target_ds["flux_up_sw"]
        target_ds["delta_lw_diff"] = target_ds["flux_dn_lw"] - target_ds["flux_up_lw"]
        target_ds["delta_lw_add"] = target_ds["flux_dn_lw"] + target_ds["flux_up_lw"]
