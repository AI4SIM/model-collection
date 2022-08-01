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
from dask.config import set
import dask.array as da
import numpy as np
import os
import os.path as osp
import torch
from typing import Dict, Tuple, Union
import xarray as xr


class ThreeDCorrectionDataproc:
    """
    Download, preprocess and shard the data of the 3DCorrection use-case.
    To be called once before experiments to build the dataset on the filesystem.
    """

    def __init__(self,
                 root: str,
                 subset: str = None,
                 timestep: int = 500,
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

    def process(self) -> None:
        """
        Proceeds with all the following steps:
            * Download the raw data and return an Xarray;
            * Select and build features;
            * Reshard and convert to Numpy;
            * Compute stats multi-threaded fashion.
        """

        xr_array = self.download()
        x, y = self.build_features(xr_array)
        x_path, y_path = self.reshard(x, y)
        self.compute_stats(x_path, y_path)

    def download(self) -> xr.DataArray:
        """Download the data for 3D Correction UC and return an xr.Array."""

        cml.settings.set("cache-directory", self.cached_data_path)
        cml_ds = cml.load_dataset(
            'maelstrom-radiation',
            subset=self.subset,
            dataset='3dcorrection',
            raw_inputs=False,
            minimal_outputs=False,
            timestep=list(range(0, 3501, self.timestep)),
            patch=list(range(0, 16, self.patchstep)),
            hr_units='K d-1')

        return cml_ds.to_xarray()

    def build_features(self, xr_array) -> Tuple[da.Array]:
        """
        Build feature vectors (of spatial dimension 138)
            * Select features from the array provided in input;
            * Rechunk on row dimension;
            * Finally lazily build (x, y).
        """

        def broadcast_features(arr: da.Array, sz: int) -> da.Array:
            """Repeat a scalar in a vector."""
            a = da.repeat(arr, sz, axis=-1)
            return da.moveaxis(a, -2, -1)

        def pad_tensor(arr: da.Array, pads: Tuple) -> da.Array:
            """Pads zeros to the vertical axis (n_before, n_after)."""
            return da.pad(arr, ((0, 0), pads, (0, 0)))

        features = [
            'sca_inputs', 'col_inputs', 'hl_inputs', 'inter_inputs',
            'flux_dn_sw', 'flux_up_sw', 'flux_dn_lw', 'flux_up_lw']
        dataset_len = xr_array.dims['column']
        n_shards = 53 * 2 ** 6
        self.shard_size = dataset_len // n_shards

        set(scheduler='threads')
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
                from shutil import rmtree
                rmtree(p)
            os.makedirs(p, exist_ok=True)

        return paths[0] if len(paths) == 1 else paths

    def reshard(self, x: da.Array, y: da.Array) -> Tuple[str]:
        """
        Reshard the arrays by rechunking on memory,
        store the chunks on disk in Numpy file format (.npy)
        """

        # Chunk on the zeroth axis.
        x_chunked = da.rechunk(x, chunks=(self.shard_size, *x.shape[1:]))
        y_chunked = da.rechunk(y, chunks=(self.shard_size, *y.shape[1:]))

        x_path, y_path = self.purgedirs([
            osp.join(self.processed_data_path, 'x'),
            osp.join(self.processed_data_path, 'y')])
        da.to_npy_stack(x_path, x_chunked, axis=0)
        da.to_npy_stack(y_path, y_chunked, axis=0)

        return x_path, y_path

    def compute_stats(self, x_path: str, y_path: str) -> Dict[str, np.array]:
        """Computes stats: mean and standard deviation for features of x and y."""

        stats = {}
        for a in [da.from_npy_stack(x_path), da.from_npy_stack(y_path)]:
            m = da.mean(a, axis=0).compute(num_workers=self.num_workers)
            s = da.std(a, axis=0).compute(num_workers=self.num_workers)

            name = a.name.split("/")[-1]
            stats.update({
                f'{name}_mean': torch.tensor(m),
                f'{name}_std': torch.tensor(s),
                f'{name}_nb': torch.tensor(a.shape[0])})

        torch.save(stats, osp.join(self.processed_data_path, "stats.pt"))


if __name__ == '__main__':
    import config
    dataproc = ThreeDCorrectionDataproc(config.data_path)
    dataproc.process()
