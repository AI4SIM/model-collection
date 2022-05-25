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
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from typing import List
import netCDF4

import config


class ThreeDCorrectionDataset(Dataset):
    """
    Create a PyTorch dataset for the 3DCorrection use-case.
    The data are downloaded using the Climetlab library (developped by the ECMWF).
    Normalization parameters are computed at this point and are used as preprocessing in the model.
    """

    def __init__(self, root: str, timestep: int, patchstep: int = 1, force: bool = False) -> None:
        """
        Create the dataset.

        Args:
            root (str): path to the root data folder.
            timestep (int): increment between two outputs (time increment is 12min so
                step 125 corresponds to an output every 25h).
            patchstep (int): step of the patchs (16 Earth's regions)
            force (bool): either to remove the processed files or not.
        """
        super().__init__(root)

        self.timestep = timestep
        self.patchstep = patchstep
        path = self.processed_paths[0]
        if force:
            os.remove(path)

    @property
    def raw_file_names(self) -> List[str]:
        """Return the raw data file names."""
        return [f"data-{self.timestep}.nc"]  # NetCDF file.

    @property
    def processed_file_names(self) -> List[str]:
        """Return the processed data file names."""
        return [f"data-{self.timestep}.pt"]

    def download(self) -> None:
        """Download the dataset using Climetlab and store it in a unique NetCDF file."""
        import climetlab as cml

        cml.settings.set("cache-directory", os.path.join(self.root, "raw"))

        cmlds = cml.load_dataset(
            'maelstrom-radiation',
            dataset='3dcorrection',
            raw_inputs=False,
            timestep=list(range(0, 3501, self.timestep)),
            minimal_outputs=False,
            patch=list(range(0, 16, self.patchstep)),
            hr_units='K d-1')

        # Convert to NetCDF.
        data = cmlds.to_xarray()
        print(data)
        data.to_netcdf(self.raw_paths[0])

        # TODO: split between timestep files

    def process(self) -> None:
        """
        Process the data:
        - create a graph for each sample;
        - save each graph in a separate file indexed by the order in the raw file names list.
        """

        def broadcast_features(t):
            """Broadcast scalar on a column profile."""
            # TODO: do the broadcasting in the network, to reduce mem footprint.
            t = torch.unsqueeze(t, -1)
            t = t.repeat((1, 1, 138))
            t = t.moveaxis(1, -1)
            return t

        for raw_path in self.raw_paths:
            with netCDF4.Dataset(raw_path, "r", format="NETCDF4") as file:

                # Inputs.
                sca_inputs = torch.tensor(file['sca_inputs'][:])
                col_inputs = torch.tensor(file['col_inputs'][:])
                hl_inputs = torch.tensor(file['hl_inputs'][:])
                inter_inputs = torch.tensor(file['inter_inputs'][:])

                # Outputs.
                flux_dn_sw = torch.tensor(file['flux_dn_sw'][:])
                flux_up_sw = torch.tensor(file['flux_up_sw'][:])
                flux_dn_lw = torch.tensor(file['flux_dn_lw'][:])
                flux_up_lw = torch.tensor(file['flux_up_lw'][:])
                hr_sw = torch.tensor(file["hr_sw"][:])
                hr_lw = torch.tensor(file["hr_lw"][:])

            x = torch.cat(
                [
                    hl_inputs,
                    pad(inter_inputs, (0, 0, 1, 1, 0, 0)),  # pad the column to 138.
                    pad(col_inputs, (0, 0, 1, 0, 0, 0)),  # shift column to the bottom to 138.
                    broadcast_features(sca_inputs)  # broadcast scalars.
                ], dim=-1)

            y = torch.cat(
                [  # TODO: why unsqueeze?
                    torch.unsqueeze(flux_dn_sw, -1),
                    torch.unsqueeze(flux_up_sw, -1),
                    torch.unsqueeze(flux_dn_lw, -1),
                    torch.unsqueeze(flux_up_lw, -1),
                    torch.unsqueeze(hr_sw, -1),
                    torch.unsqueeze(hr_lw, -1),
                ], dim=-1)

            stats_path = os.path.join(self.root, f"stats-{self.timestep}.pt")
            if not os.path.isfile(stats_path):
                torch.save(
                    {
                        "x_mean": torch.mean(x, dim=0),
                        "y_mean": torch.mean(y, dim=0),
                        "x_std": torch.std(x, dim=0),
                        "y_std": torch.std(y, dim=0)
                    }, stats_path)

            torch.save((x, y), self.processed_paths[0])


@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """DataModule for the 3dcorrection dataset."""

    def __init__(self,
                 timestep: int,
                 batch_size: int,
                 num_workers: int,
                 force: bool = False):

        self.timestep = timestep
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force = force
        super().__init__()

    def prepare_data(self):
        ThreeDCorrectionDataset(config.data_path, self.timestep, self.force)

    def setup(self, stage):
        dataset = ThreeDCorrectionDataset(config.data_path, self.timestep, self.force).shuffle()
        self.test_dataset = dataset[int(0.9 * dataset.len()):]
        self.val_dataset = dataset[int(0.8 * dataset.len()):int(0.9 * dataset.len())]
        self.train_dataset = dataset[:int(0.8 * dataset.len())]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
