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

import os
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
from torch.nn.functional import pad
import torch_geometric as pyg
import netCDF4
import climetlab as cml

import config


class ThreeDCorrectionDataset(pyg.data.InMemoryDataset):
    """
    Create a PyG InMeMoryDataset for the 3DCorrection UC.
    The data are downloaded using the Climetlab library developped by ECMWF.
    Normalization parameters are computed at this point and are used later in the model.
    """

    def __init__(
        self, root: str, timestep: int, patchstep: int = 1, force: bool = False
    ) -> None:
        """
        Create the dataset.

        Args:
            root (str): Path to the root data folder.
            timestep (int): Increment between two outputs
              (time increment is 12 minutes so step 125 corresponds to an output every 25 hours.).
            patchstep (int): Step of the patchs (16 Earth's regions).
            force (bool): Either to remove the processed files or not.
        """
        self.timestep = timestep
        self.patchstep = patchstep

        super().__init__(root)

        path = self.processed_paths[0]
        if force:
            os.remove(path)

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        """
        Return the raw data file names.

        Returns:
            (List[str]): Raw data file names list.
        """
        return [f"data-{self.timestep}.nc"]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Return the processed data file names.

        Returns:
            (List[str]): Processed data file names list.
        """
        return [f"data-{self.timestep}.pt"]

    def download(self) -> None:
        """Download the dataset using Climetlab and stores it in a unique NetCDF file."""
        cml.settings.set("cache-directory", os.path.join(self.root, "raw"))

        cmlds = cml.load_dataset(
            "maelstrom-radiation",
            dataset="3dcorrection",
            raw_inputs=False,
            timestep=list(range(0, 3501, self.timestep)),
            minimal_outputs=False,
            patch=list(range(0, 16, self.patchstep)),
            hr_units="K d-1",
        )

        array = cmlds.to_xarray()
        array.to_netcdf(self.raw_paths[0])

    def process(self) -> None:
        """
        Process the data, creates a graph for each sample,
        and saves each graph in a separate file index by the order in the raw file names list.
        """

        def broadcast_features(tensor):
            tensor_ = torch.unsqueeze(tensor, -1)
            tensor_ = tensor_.repeat((1, 1, 138))
            tensor_ = tensor_.moveaxis(1, -1)
            return tensor_

        for raw_path in self.raw_paths:

            with netCDF4.Dataset(raw_path, "r", format="NETCDF4") as file:
                sca_inputs = torch.tensor(file["sca_inputs"][:])
                col_inputs = torch.tensor(file["col_inputs"][:])
                hl_inputs = torch.tensor(file["hl_inputs"][:])
                pressure_hl = torch.tensor(file["pressure_hl"][:])
                inter_inputs = torch.tensor(file["inter_inputs"][:])

                flux_dn_sw = torch.tensor(file["flux_dn_sw"][:])
                flux_up_sw = torch.tensor(file["flux_up_sw"][:])
                flux_dn_lw = torch.tensor(file["flux_dn_lw"][:])
                flux_up_lw = torch.tensor(file["flux_up_lw"][:])

                hr_sw = torch.tensor(file["hr_sw"][:])
                hr_lw = torch.tensor(file["hr_lw"][:])

            feats = torch.cat(
                [
                    broadcast_features(sca_inputs),
                    hl_inputs,
                    pad(inter_inputs, (0, 0, 1, 1, 0, 0)),
                    pressure_hl,
                ],
                dim=-1,
            )

            targets = torch.cat(
                [
                    torch.unsqueeze(flux_dn_lw, -1),
                    torch.unsqueeze(flux_up_lw, -1),
                    torch.unsqueeze(flux_dn_sw, -1),
                    torch.unsqueeze(flux_up_sw, -1),
                    torch.unsqueeze(pad(hr_lw, (1, 0)), -1),
                    torch.unsqueeze(pad(hr_sw, (1, 0)), -1),
                ],
                dim=-1,
            )

            stats_path = os.path.join(self.root, f"stats-{self.timestep}.pt")
            if not os.path.isfile(stats_path):
                stats = {
                    "x_mean": torch.mean(feats, dim=0),
                    "y_mean": torch.mean(targets, dim=0),
                    "x_std": torch.std(feats, dim=0),
                    "y_std": torch.std(targets, dim=0),
                }
                torch.save(stats, stats_path)

            directed_index = np.array([[*range(1, 138)], [*range(137)]])
            undirected_index = np.hstack((directed_index, directed_index[[1, 0], :]))
            undirected_index = torch.tensor(undirected_index, dtype=torch.long)

            data_list = []

            for idx in range(feats.shape[0]):
                feats_ = torch.squeeze(feats[idx, ...])
                targets_ = torch.squeeze(targets[idx, ...])

                edge_attr = torch.squeeze(col_inputs[idx, ...])

                data = pyg.data.Data(
                    x=feats_,
                    edge_attr=edge_attr,
                    edge_index=undirected_index,
                    y=targets_,
                )

                data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[0])


@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """
    Create a datamodule structure for use un a PyTorch Lightning Trainer. Basically a wrapper
    around the Dataset. Training set is randomized.
    """

    def __init__(
        self,
        timestep: int,
        patchstep: int,
        batch_size: int,
        num_workers: int,
        force: bool = False,
    ) -> None:
        """
        Init the LitThreeDCorrectionDataModule class.

        Args:
            timestep (int): Increment between two outputs
              (time increment is 12 minutes so step 125 corresponds to an output every 25 hours.).
            patchstep (int): Step of the patchs (16 Earth's regions).
            batch_size (int): Batch size.
            num_workers (int): DataLoader number of workers for loading data.
            force (bool): Either to remove the processed files or not.
        """
        self.timestep = timestep
        self.patchstep = patchstep
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force = force

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        super().__init__()

    def prepare_data(self) -> None:
        """Not used. The download logic is the responsability for the Dataset."""
        ThreeDCorrectionDataset(
            config.data_path, self.timestep, self.patchstep, self.force
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Args:
            stage (str): Stage for which to setup the DataLoader. If 'fit', it will prepare the
                train and val DataLoader. If 'test', it will prepare the test DataLoader.
        """
        dataset = ThreeDCorrectionDataset(
            config.data_path, self.timestep, self.patchstep, self.force
        ).shuffle()
        length = dataset.len()

        self.test_dataset = dataset[int(0.9 * length):]
        self.val_dataset = dataset[int(0.8 * length):int(0.9 * length)]
        self.train_dataset = dataset[:int(0.8 * length)]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the train DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Train DataLoader.
        """
        return pyg.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the val DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Val DataLoader.
        """
        return pyg.loader.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the test DataLoader.

        Returns:
            (torch.utils.data.DataLoader): Test DataLoader.
        """
        return pyg.loader.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
