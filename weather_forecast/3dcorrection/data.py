"""This module proposes Pytorch style Dataset classes for the gnn use-case"""
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
# limitations under the License..

import os.path as osp
from typing import Dict, List, Optional

import dask
import sys
import torch
import climetlab as cml
import numpy as np
import pytorch_lightning as pl
import torch_geometric as pyg
import xarray as xr

from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.nn.functional import pad
from dataproc import ThreeDCorrectionDataProc

import config
import keys


class ThreeDCorrectionDataset(pyg.data.Dataset):
    """
    Create a PyG InMeMoryDataset for the 3DCorrection UC.
    The data are downloaded using the Climetlab library developped by ECMWF.
    Normalization parameters are computed at this point and are used later in the model.
    """

    dask.config.set(scheduler="synchronous")

    profile_per_file = 16960 // 8

    def __init__(
        self,
        root: str,
        ds_feature: xr.Dataset,
        ds_target: xr.Dataset,
        ds_means: xr.Dataset,
        ds_stds: xr.Dataset,
        # length: int
    ) -> None:
        """
        Create the dataset.

        Args:
            root (str): Path to the root data folder.
            ds_feature: xarray Dataset containing the features
            ds_target: xarray Dataset containing the targets
        """

        # self.length = length
        self.root = root
        self.ds_feature = ds_feature
        self.ds_target = ds_target
        self.ds_means = ds_means
        self.ds_stds = ds_stds

        self.ds_means["inter_inputs"] = pad(
            self.ds_means["inter_inputs"], (0, 0, 1, 1, 0, 0), "constant", 0
        )

        self.ds_stds["inter_inputs"] = pad(
            self.ds_stds["inter_inputs"], (0, 0, 1, 1, 0, 0), "constant", 1
        )

        super().__init__(root)

        # If using lazy loading, uncomment the next line
        # self.process_()

    @property
    def processed_file_names(self) -> List[str]:
        return [f"data-{ii}.pt" for ii in range(self.len() // self.profile_per_file)]

    def build_graph(self, idx) -> pyg.data.Data:
        def normalise(array, norm_key):
            return (
                array - torch.squeeze(self.ds_means[norm_key], dim=0)
            ) / torch.squeeze(self.ds_stds[norm_key], dim=0)

        def create_global_features(feat, keys, norm_key=None):
            for i, key in enumerate(keys):
                tmp_ = torch.flatten(feat[key])
                if i == 0:
                    out = tmp_
                else:
                    out = torch.cat((out, tmp_), dim=-1)
            out = torch.unsqueeze(out, dim=0)

            if norm_key is not None:
                out = normalise(out, norm_key)

            return out

        def stack_tensors(feat, keys, norm_key=None):
            for i, key in enumerate(keys):
                tmp_ = torch.unsqueeze(feat[key], dim=-1)
                if i == 0:
                    out = tmp_
                else:
                    if key == "aerosol_mmr":
                        tmp_ = torch.permute(torch.squeeze(tmp_), (1, 0))
                    out = torch.cat((out, tmp_), dim=-1)

            if norm_key is not None:
                out = normalise(out, norm_key)

            return out

        def create_graph(feat, targ):
            # x --> hl_inputs / inter_inputs
            # edge_attr --> col_inputs
            edge_attr = stack_tensors(feat, keys.COL, norm_key="col_inputs")
            hl = stack_tensors(feat, keys.HL, norm_key="hl_inputs")
            inter = stack_tensors(feat, keys.INTER, norm_key="inter_inputs")
            x = torch.cat((hl, inter), dim=-1)
            y = stack_tensors(targ, keys.CUSTOM_TARGET[2:])
            z = stack_tensors(targ, keys.CUSTOM_TARGET[:2])
            global_features = create_global_features(feat, keys.SCA)

            index = torch.tensor(
                [[*torch.arange(137)], [*torch.arange(1, 138)]], dtype=torch.long
            )
            edge_index, edge_attr = pyg.utils.to_undirected(index, edge_attr)

            kwargs = {
                "x": x,
                "y": y,
                "z": z,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "u": global_features,
                "node_features": [*keys.HL, *keys.INTER],
                "edge_features": keys.COL,
                "global_features": keys.SCA,
                "node_targets": keys.CUSTOM_TARGET[2:],
                "edge_targets": keys.CUSTOM_TARGET[0:2],
            }

            graph = pyg.data.Data(**kwargs)

            return graph

        feature_sample = self.ds_feature.isel({"column": idx})
        target_sample = self.ds_target.isel({"column": idx})

        feature = {k: torch.from_numpy(v.to_numpy()) for k, v in feature_sample.items()}
        target = {k: torch.from_numpy(v.to_numpy()) for k, v in target_sample.items()}

        for key in keys.INTER:
            feature[key] = pad(feature[key], (1, 1), "constant", 0)

        g = create_graph(feature, target)

        return g

    # TBM: number of profiles to be stored --> config file ?
    def process(self):
        data_list = []
        for idx in range(self.len()):
            print(f"Index : {idx}", file=sys.stderr)
            single_graph = self.build_graph(idx)
            data_list.append(single_graph)

            if idx != 0 and (idx + 1) % self.profile_per_file == 0:
                file_idx = idx // self.profile_per_file
                batch = pyg.data.Batch.from_data_list(data_list)
                batch.batch = None
                # batch.mini_batch = batch.mini_batch
                print(
                    f"Writing batch of graphs into file {file_idx}...", file=sys.stderr
                )
                torch.save(batch, osp.join(self.processed_dir, f"data-{file_idx}.pt"))
                data_list = []

    def get(self, idx) -> pyg.data.Data:
        file_idx = idx // (16960 // 8)
        # print(f"Reading index {idx} from file {file_idx}")
        batch = torch.load(osp.join(self.processed_dir, f"data-{file_idx}.pt"))
        return batch.get_example(idx % self.profile_per_file)
        # return self.process_(idx)

    # Set the number of profiles manually for testing
    # Should remove it before PR
    def len(self) -> int:
        return self.ds_feature.dims["column"]


@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """
    Create a datamodule structure for use un a PyTorch Lightning Trainer. Basically a wrapper
    around the Dataset. Training set is randomized.
    """

    def __init__(
        self,
        date: str,
        timestep: int,
        patchstep: int,
        subset: str,
        norm: bool,
        batch_size: int,
        num_workers: int,
    ) -> None:
        """
        Init the LitThreeDCorrectionDataModule class.

        Args:
            timestep (int): Increment between two outputs
              (time increment is 12 minutes so step 125 corresponds to an output every 25 hours.).
            patchstep (int): Step of the patchs (16 Earth's regions).
            batch_size (int): Batch size.
            num_workers (int): DataLoader number of workers for loading data.
        """
        self.date = date
        self.timestep = timestep
        self.patchstep = patchstep
        self.subset = subset
        self.norm = norm
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.mean_norm = torch.tensor([0.0])
        self.std_norm = torch.tensor([1.0])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        super().__init__()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Args:
            stage (str): Stage for which to setup the DataLoader. If 'fit', it will prepare the
                train and val DataLoader. If 'test', it will prepare the test DataLoader.
        """
        # NB: Now, we split the dataset.
        # In the future, we will load different dataset depending on the stage.

        if stage == "fit":
            dataproc = ThreeDCorrectionDataProc(
                config.data_path,
                self.date,
                self.timestep,
                self.patchstep,
                self.subset,
                self.norm,
            )
            self.features, self.targets = dataproc.process()
            if self.norm:
                self.mean_norm, self.std_norm = dataproc.means, dataproc.stds

            dataset = ThreeDCorrectionDataset(
                config.data_path,
                self.features,
                self.targets,
                self.mean_norm,
                self.std_norm,
            )
            length = dataset.len()

            self.test_dataset = dataset[int(0.9 * length) :]
            self.val_dataset = dataset[int(0.8 * length) : int(0.9 * length)]
            self.train_dataset = dataset[: int(0.8 * length)]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the train DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Train DataLoader.
        """
        return pyg.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=8,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the val DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Val DataLoader.
        """
        return pyg.loader.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=2,
            pin_memory=False,
            drop_last=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the test DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Test DataLoader.
        """
        return pyg.loader.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
