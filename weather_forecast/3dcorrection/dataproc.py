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
import time
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from typing import Dict, List, Tuple, Union
import yaml
import xarray as xr

class Dataproc:
    
    def __init__(self, 
                 root_path: str = "/home/jupyter/data/", 
                 step: int = 500,
                 num_workers: int = 16) -> None:
        
        self.root_path = root_path
        self.step = step
        self.num_workers = num_workers
    
    
    def process(self) -> None:
        """Proceeds with all the following steps: 
            * Download the raw data and return an Xarray
            * Select and build features
            * Reshard and Convert to Numpy
            * Compute stats multi-threaded fashion
            * Build PyG graphs"""
        
        xr_array = self.download(self.step)
        x, y, edge = self.build_feats(xr_array)
        x_path, y_path, edge_path = self.reshard_and_convert(x, y, edge)
        self.compute_stats(x_path, y_path, edge_path)
        self.build_graphs(x_path, y_path, edge_path)

        
    def download(self, step) -> xr.DataArray:
        """Download the data for 3D Correction UC and return an xr.Array"""

        cml.settings.set("cache-directory", osp.join(self.root_path, "cached"))
        cmlds = cml.load_dataset(
            'maelstrom-radiation', 
            dataset='3dcorrection', 
            raw_inputs=False, 
            timestep=list(range(0, 3501, step)), 
            minimal_outputs=False,
            patch=list(range(0, 16, 1)),
            hr_units='K d-1',
        )
        xr_array = cmlds.to_xarray()
        
        return xr_array
    

    def build_feats(self, xr_array) -> Tuple[da.Array]:
        """Select features from the array provided in input, then rechunk on row dimension, and finally lazily build x, y and edge features."""

        def broadcast_features(array: da.Array):
            a = da.repeat(array, 138, axis=-1)
            a = da.moveaxis(a, -2, -1)
            return a

        def pad_tensor(array: da.Array):
            a = da.pad(array, ((0, 0), (1, 1), (0, 0)))
            return a

        features = [
            'sca_inputs',
            'col_inputs',
            'hl_inputs',
            'inter_inputs',
            'flux_dn_sw',
            'flux_up_sw',
            'flux_dn_lw',
            'flux_up_lw',
        ]

        dataset_len = xr_array.dims['column']
        num_shards = 53 * 2 ** 6
        shard_size = dataset_len // num_shards

        dask.config.set(scheduler='threads')

        data = {}
        for feat in features:
            array = xr_array[feat].data
            array = da.rechunk(array, chunks=(shard_size, *array.shape[1:]))
            data.update({feat: array})

        x = da.concatenate([
            data['hl_inputs'],
            pad_tensor(data['inter_inputs']),
            broadcast_features(data['sca_inputs'][..., np.newaxis])
        ], axis=-1)

        y = da.concatenate([
            data['flux_dn_sw'][..., np.newaxis],
            data['flux_up_sw'][..., np.newaxis],
            data['flux_dn_lw'][..., np.newaxis],
            data['flux_up_lw'][..., np.newaxis],
        ], axis=-1)

        edge = data['col_inputs']
        
        return x, y, edge
    

    def purgedirs(self, paths: Union[str, list]) -> Union[str, list]:
        """Removes all content of directories which path is in 'paths'"""

        if isinstance(paths, str):
            paths = [paths]

        for p in paths: 
            if osp.isdir(p):
                shutil.rmtree(p)

            os.makedirs(p, exist_ok=True)

        if len(paths) == 1:
            return paths[0]

        return paths

    
    def reshard_and_convert(self,
                            x: da.Array,
                            y: da.Array,
                            edge: da.Array) -> Tuple[str]:
        """Reshard the arrays by rechunking on memory and storing the chunks on disk in Numpy file format (.npy)"""

        x_ = da.rechunk(x, chunks=(shard_size, *x.shape[1:]))
        y_ = da.rechunk(y, chunks=(shard_size, *y.shape[1:]))
        edge_ = da.rechunk(edge, chunks=(shard_size, *edge.shape[1:]))

        out_dir = osp.join(self.root_path, "processed", f'features-{self.step}')

        x_path, y_path, edge_path = self.purgedirs([
            osp.join(out_dir, 'x'), 
            osp.join(out_dir, 'y'), 
            osp.join(out_dir, 'edge')
        ])

        da.to_npy_stack(x_path, x, axis=0)
        da.to_npy_stack(y_path, y, axis=0)
        da.to_npy_stack(edge_path, edge, axis=0)
        
        return x_path, y_path, edge_path
        
        
    def compute_stats(self,
                      x_path: str,
                      y_path: str,
                      edge_path: str) -> Dict[str, torch.Tensor]:
        """Computes stats: mean and standard deviation for features of x, y, and edge"""
        
        arrays = [
            da.from_npy_stack(x_path),
            da.from_npy_stack(y_path),
            da.from_npy_stack(edge_path)
        ]
        stats = {}
        
        for a in arrays:

            a_mean = da.mean(a, axis=0)
            a_std = da.std(a, axis=0)

            m = a_mean.compute(num_workers=self.num_workers)
            s = a_std.compute(num_workers=self.num_workers)

            name = a.name.split("/")[-1]
            stats.update({
                f'{name}_mean': torch.tensor(m),
                f'{name}_std': torch.tensor(s)
            })
        
        stats_path = osp.join(self.root_path, "processed", f"stats-{self.step}.pt")
        torch.save(stats, stats_path)

        return stats
    
    def _bulk_graphs(self):
    
    def build_graphs(self):
        
        import numpy as np
        
        # Each 'common' step can store a shared property to all steps
        # Start is common here, and we branch on next step
        # More info https://docs.metaflow.org/metaflow/basics#branch
        self.out_dir = osp.join(self.PROCESSED_DATA_PATH, f"graphs-{self.timestep}")
        if osp.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)
        
        # To launch in thread in parallel, just call the next step over an attribute's list
        self.shard = np.arange(self.num_shards)
        self.next(self.build_graphs, foreach="shard")
        
        if mp.cpu_count() < self.num_processes:
            raise ValueError("The number of CPU's specified exceed the amount available.")

        subset_raw_paths = np.array_split(self.raw_paths, self.num_processes)
        pool = mp.Pool(self.num_processes)
        pool.map(self.build_graph, subset_raw_paths)
        pool.close()
        pool.join()
        
        main_dir = osp.join(self.PROCESSED_DATA_PATH, f"features-{self.timestep}")
        
        def load(name: Union['x', 'y', 'edge']) -> torch.Tensor:
            return torch.tensor(
                np.lib.format.open_memmap(
                    mode='r', 
                    dtype=self.dtype, 
                    filename=osp.join(main_dir, name, f"{self.input}.npy"), 
                    shape=getattr(self, f'{name}_shape')))
                
        x, y, edge = load("x"), load("y"), load("edge")
        
        directed_idx = np.array([[*range(1, 138)], [*range(137)]])
        undirected_idx = np.hstack((
            directed_idx, 
            directed_idx[[1, 0], :]
        ))
        undirected_idx = torch.tensor(undirected_idx, dtype=torch.long)
        
        data_list = []
        for idx in range(len(x)):
            
            x_ = torch.squeeze(x[idx, ...])
            y_ = torch.squeeze(y[idx, ...])
            edge_ = torch.squeeze(edge[idx, ...])

            data = pyg.data.Data(x=x_, edge_attr=edge_, edge_index=undirected_idx, y=y_,)
            data_list.append(data)
            
        out_path = osp.join(self.out_dir, f"data-{self.input}.pt")
        torch.save(data_list, out_path)