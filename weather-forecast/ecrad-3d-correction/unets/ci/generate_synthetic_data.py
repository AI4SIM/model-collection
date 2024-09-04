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

import dask.array as da
import os.path as osp
import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from dataproc import ThreeDCorrectionDataproc # noqa:
import config # noqa:


class ThreeDCorrectionDataprocSyntheticData(ThreeDCorrectionDataproc):
    """
    Generate, preprocess and shard synthetic data of the 3DCorrection use-case.
    To be called once before experiments to build the dataset on the filesystem.
    """

    def __init__(self,
                 root: str,
                 subset: str = None,
                 timestep: int = 1,
                 patchstep: int = 1,
                 num_workers: int = 16) -> None:
        """
        Preprocess and shard data on disk.

        Args:
            root (str): Path to the root data folder.
            subset (str): Which subset to download (e.g. "tier-1"), if None download all the data.
            timestep (int): Increment between two outputs (time increment is 12min so
                step 125 corresponds to an output every 25h).
            patchstep (int): Step of the patchs (16 Earth's regions)
            num_workers (int): Number of workers.
        """
        self.cached_data_path = osp.join(root, 'cached')
        self.raw_data_path = osp.join(root, 'raw')
        self.processed_data_path = osp.join(root, 'processed')

        for path in [self.cached_data_path, self.raw_data_path, self.processed_data_path]:
            os.makedirs(path, exist_ok=True)

        self.subset = subset
        self.timestep = timestep
        self.patchstep = patchstep
        self.num_workers = num_workers

    def download(self) -> xr.DataArray:
        """Create data folder with fake raw data"""
        mu = 0.0
        sigma = 1.0

        sca_inputs = da.from_array(np.random.normal(mu, sigma, (33920, 17)).astype('float32'))
        col_inputs = da.from_array(np.random.normal(mu, sigma, (33920, 137, 27)).astype('float32'))
        hl_inputs = da.from_array(np.random.normal(mu, sigma, (33920, 138, 2)).astype('float32'))
        pressure_hl = da.from_array(np.random.normal(mu, sigma, (33920, 138, 1)).astype('float32'))
        inter_inputs = da.from_array(np.random.normal(mu, sigma, (33920, 136, 1)).astype('float32'))
        lat = da.from_array(np.random.normal(mu, sigma, (33920,)).astype('float32'))
        lon = da.from_array(np.random.normal(mu, sigma, (33920,)).astype('float32'))
        flux_dn_sw = da.from_array(np.random.normal(mu, sigma, (33920, 138)).astype('float32'))
        flux_up_sw = da.from_array(np.random.normal(mu, sigma, (33920, 138)).astype('float32'))
        flux_dn_lw = da.from_array(np.random.normal(mu, sigma, (33920, 138)).astype('float32'))
        flux_up_lw = da.from_array(np.random.normal(mu, sigma, (33920, 138)).astype('float32'))
        hr_sw = da.from_array(np.random.normal(mu, sigma, (33920, 137)).astype('float32'))
        hr_lw = da.from_array(np.random.normal(mu, sigma, (33920, 137)).astype('float32'))

        xrds = xr.Dataset(data_vars=dict(sca_inputs=({"column": 33920, "sca_variables": 17},
                                                     sca_inputs),
                          col_inputs=({"column": 33920, "level": 137, "col_variables": 17},
                                      col_inputs),
                          hl_inputs=({"column": 33920, "half_level": 138, "hl_variable": 2},
                                     hl_inputs),
                          pressure_hl=({"column": 33920, "half_level": 138, "p_variable": 1},
                                       pressure_hl),
                          inter_inputs=({"column": 33920, "level_interface": 136,
                                         "inter_variable": 1}, inter_inputs),
                          lat=({"column": 33920}, lat),
                          lon=({"column": 33920}, lon),
                          flux_dn_sw=({"column": 33920, "half_level": 138}, flux_dn_sw),
                          flux_up_sw=({"column": 33920, "half_level": 138}, flux_up_sw),
                          flux_dn_lw=({"column": 33920, "half_level": 138}, flux_dn_lw),
                          flux_up_lw=({"column": 33920, "half_level": 138}, flux_up_lw),
                          hr_sw=({"column": 33920, "level": 137}, hr_sw),
                          hr_lw=({"column": 33920, "level": 137}, hr_lw)),
                          attrs=dict(description="Weather fake data.")
                          )
        return xrds


if __name__ == '__main__':

    data = ThreeDCorrectionDataprocSyntheticData(config.data_path)
    data.process()
