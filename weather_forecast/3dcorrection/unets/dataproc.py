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

import climetlab as cml
import dask
import dask.array as da
import numpy as np
import os
import os.path as osp
import shutil
import torch
from typing import Dict, Tuple, Union
import xarray as xr

import config

class Dataproc:

    def __init__(self,
                 step: int = 500,
                 num_workers: int = 16) -> None:

        self.cached_data_path = osp.join(config.data_path, 'cached')
        self.raw_data_path = osp.join(config.data_path, 'raw')
        self.processed_data_path = osp.join(config.data_path, 'processed')

        for path in [self.cached_data_path,
                     self.raw_data_path,
                     self.processed_data_path]:
            os.makedirs(path, exist_ok=True)

        self.step = step
        self.num_workers = num_workers

    def process(self) -> None:
        """
        Proceeds with all the following steps:
            * Download the raw data and return an Xarray;
            * Select and build features;
            * Reshard and Convert to Numpy;
            * Compute stats multi-threaded fashion.
        """

        xr_array = self.download(self.step)
        x, y = self.build_features(xr_array)
        x_path, y_path = self.reshard_and_convert(x, y)
        self.compute_stats(x_path, y_path)

    def download(self, step) -> xr.DataArray:
        """Download the data for 3D Correction UC and return an xr.Array"""

        cml.settings.set("cache-directory", self.cached_data_path)
        cmlds = cml.load_dataset(
            'maelstrom-radiation',
            dataset='3dcorrection',
            raw_inputs=False,
            timestep=list(range(0, 3501, step)),
            minimal_outputs=False,
            patch=list(range(0, 16, 1)),
            hr_units='K d-1')

        return cmlds.to_xarray()

    def build_features(self, xr_array) -> Tuple[da.Array]:
        """
        Select features from the array provided in input,
        then rechunk on row dimension,
        and finally lazily build (x, y).
        """

        def broadcast_features(arr: da.Array, sz: int):
            """Repeat a scalar in a vector."""
            a = da.repeat(arr, sz, axis=-1)
            return da.moveaxis(a, -2, -1)

        def pad_tensor(arr: da.Array, pads: Tuple):
            """Pads zeros to the vertical axis (n_before, n_after)."""
            return da.pad(arr, ((0, 0), pads, (0, 0)))

        features = [
            'sca_inputs', 'col_inputs', 'hl_inputs', 'inter_inputs',
            'flux_dn_sw', 'flux_up_sw', 'flux_dn_lw', 'flux_up_lw']
        dataset_len = xr_array.dims['column']
        n_shards = 53 * 2 ** 6
        self.shard_size = dataset_len // n_shards

        dask.config.set(scheduler='threads')
        data = {}
        for f in features:
            arr = xr_array[f].data
            arr = da.rechunk(arr, chunks=(self.shard_size, *arr.shape[1:]))
            data.update({f: arr})

        x = da.concatenate([
            data['hl_inputs'],
            pad_tensor(data['inter_inputs'], (1, 1)),
            pad_tensor(data['col_inputs'], (1, 0)),
            broadcast_features(data['sca_inputs'][..., np.newaxis], 138)
        ], axis=-1)

        y = da.concatenate([
            data['flux_dn_sw'][..., np.newaxis],
            data['flux_up_sw'][..., np.newaxis],
            data['flux_dn_lw'][..., np.newaxis],
            data['flux_up_lw'][..., np.newaxis],
        ], axis=-1)

        return x, y

    def purgedirs(self, paths: Union[str, list]) -> Union[str, list]:
        """Removes all content of directories in paths."""

        if isinstance(paths, str):
            paths = [paths]

        for p in paths:
            if osp.isdir(p):
                shutil.rmtree(p)
            os.makedirs(p, exist_ok=True)

        if len(paths) == 1:
            return paths[0]
        return paths

    def reshard_and_convert(self, x: da.Array, y: da.Array) -> Tuple[str]:
        """
        Reshard the arrays by rechunking on memory,
        store the chunks on disk in Numpy file format (.npy)
        """

        x_chunked = da.rechunk(x, chunks=(self.shard_size, *x.shape[1:]))
        y_chunked = da.rechunk(y, chunks=(self.shard_size, *y.shape[1:]))

        out_dir = osp.join(self.processed_data_path, f'features-{self.step}')

        x_path, y_path = self.purgedirs([osp.join(out_dir, 'x'), osp.join(out_dir, 'y')])
        da.to_npy_stack(x_path, x_chunked, axis=0)
        da.to_npy_stack(y_path, y_chunked, axis=0)

        return x_path, y_path

    def compute_stats(self, x_path: str, y_path: str) -> Dict[str, torch.Tensor]:
        """Computes stats: mean and standard deviation for features of x and y."""

        stats = {}
        for a in [da.from_npy_stack(x_path), da.from_npy_stack(y_path)]:
            a_mean = da.mean(a, axis=0)
            a_std = da.std(a, axis=0)

            m = a_mean.compute(num_workers=self.num_workers)
            s = a_std.compute(num_workers=self.num_workers)

            name = a.name.split("/")[-1]
            stats.update({
                f'{name}_mean': torch.tensor(m),
                f'{name}_std': torch.tensor(s)})

        torch.save(stats, osp.join(self.processed_data_path, f"stats-{self.step}.pt"))

        return stats
