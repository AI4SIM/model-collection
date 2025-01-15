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

import os.path as osp
import sys

import dask.array as da
import numpy as np
import xarray as xr

sys.path.insert(1, "/".join(osp.realpath(__file__).split("/")[0:-2]))

import config  # noqa:  E402
from dataproc import ThreeDCorrectionDataproc  # noqa:  E402


class ThreeDCorrectionDataprocSyntheticData(ThreeDCorrectionDataproc):
    """
    Generate, preprocess and shard synthetic data of the 3DCorrection use-case.
    To be called once before experiments to build the dataset on the filesystem.
    """

    def download(self) -> xr.DataArray:
        """Create data folder with fake raw data"""
        mu = 0.0
        sigma = 1.0

        sca_inputs = da.from_array(np.random.normal(mu, sigma, (100, 17)).astype('float32'))
        col_inputs = da.from_array(np.random.normal(mu, sigma, (100, 137, 27)).astype('float32'))
        hl_inputs = da.from_array(np.random.normal(mu, sigma, (100, 138, 2)).astype('float32'))
        pressure_hl = da.from_array(np.random.normal(mu, sigma, (100, 138, 1)).astype('float32'))
        inter_inputs = da.from_array(np.random.normal(mu, sigma, (100, 136, 1)).astype('float32'))
        lat = da.from_array(np.random.normal(mu, sigma, (100,)).astype('float32'))
        lon = da.from_array(np.random.normal(mu, sigma, (100,)).astype('float32'))
        flux_dn_sw = da.from_array(np.random.normal(mu, sigma, (100, 138)).astype('float32'))
        flux_up_sw = da.from_array(np.random.normal(mu, sigma, (100, 138)).astype('float32'))
        flux_dn_lw = da.from_array(np.random.normal(mu, sigma, (100, 138)).astype('float32'))
        flux_up_lw = da.from_array(np.random.normal(mu, sigma, (100, 138)).astype('float32'))
        hr_sw = da.from_array(np.random.normal(mu, sigma, (100, 137)).astype('float32'))
        hr_lw = da.from_array(np.random.normal(mu, sigma, (100, 137)).astype('float32'))

        xrds = xr.Dataset(data_vars=dict(sca_inputs=({"column": 100, "sca_variables": 17},
                                                     sca_inputs),
                          col_inputs=({"column": 100, "level": 137, "col_variables": 17},
                                      col_inputs),
                          hl_inputs=({"column": 100, "half_level": 138, "hl_variable": 2},
                                     hl_inputs),
                          pressure_hl=({"column": 100, "half_level": 138, "p_variable": 1},
                                       pressure_hl),
                          inter_inputs=({"column": 100, "level_interface": 136,
                                         "inter_variable": 1}, inter_inputs),
                          lat=({"column": 100}, lat),
                          lon=({"column": 100}, lon),
                          flux_dn_sw=({"column": 100, "half_level": 138}, flux_dn_sw),
                          flux_up_sw=({"column": 100, "half_level": 138}, flux_up_sw),
                          flux_dn_lw=({"column": 100, "half_level": 138}, flux_dn_lw),
                          flux_up_lw=({"column": 100, "half_level": 138}, flux_up_lw),
                          hr_sw=({"column": 100, "level": 137}, hr_sw),
                          hr_lw=({"column": 100, "level": 137}, hr_lw)),
                          attrs=dict(description="Weather fake data.")
                          )
        return xrds


if __name__ == '__main__':

    data = ThreeDCorrectionDataprocSyntheticData(config.data_path, n_shards=1)
    data.process()
