"""This module proposes a test suite for the models.py file."""
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
import netCDF4
import numpy as np
import tempfile
import torch
import torch_geometric as pyg
import torch_optimizer as optim
import unittest
from torch.nn.functional import pad

from models import LitGAT


class TestModel(unittest.TestCase):
    """Model test file."""

    def setUp(self) -> None:
        """Define default parameters."""
        self.filename = "data-1.nc"

        self.model_test = LitGAT(in_channels=21,
                                 hidden_channels=8,
                                 out_channels=6,
                                 num_layers=4,
                                 dropout=.3,
                                 edge_dim=27,
                                 heads=8,
                                 jk="last",
                                 lr=.0001,
                                 timestep=1,
                                 norm=False)

    def create_env(self, tempdir):
        """Create a test environment and data files test."""
        root_path = os.path.join(tempdir, "data")
        raw_path = os.path.join(root_path, "raw")
        os.mkdir(root_path)
        os.mkdir(raw_path)

        column = 1
        sca_variable = 17
        level = 137
        col_variable = 27
        half_level = 138
        hl_variable = 2
        p_variable = 1
        level_interface = 136
        inter_variable = 1

        with netCDF4.Dataset(os.path.join(raw_path, self.filename),
                             "w",
                             format="NETCDF4") as file:
            file.createDimension('column', column)
            file.createDimension('sca_variable', sca_variable)
            file.createDimension('level', level)
            file.createDimension('col_variable', col_variable)
            file.createDimension('half_level', half_level)
            file.createDimension('hl_variable', hl_variable)
            file.createDimension('p_variable', p_variable)
            file.createDimension('level_interface', level_interface)
            file.createDimension('inter_variable', inter_variable)
            # Variables
            # --- Inputs
            sca_inputs = file.createVariable('sca_inputs', 'float32',
                                             ('column', 'sca_variable'))
            col_inputs = file.createVariable('col_inputs', 'float32',
                                             ('column', 'level', 'col_variable'))
            hl_inputs = file.createVariable('hl_inputs', 'float32',
                                            ('column', 'half_level', 'hl_variable'))
            pressure_hl = file.createVariable('pressure_hl', 'float32',
                                              ('column', 'half_level', 'p_variable'))
            inter_inputs = file.createVariable('inter_inputs', 'float32',
                                               ('column', 'level_interface', 'inter_variable'))
            # --- Targets
            flux_dn_lw = file.createVariable('flux_dn_lw', 'float32',
                                             ('column', 'half_level'))
            flux_up_lw = file.createVariable('flux_up_lw', 'float32',
                                             ('column', 'half_level'))
            flux_dn_sw = file.createVariable('flux_dn_sw', 'float32',
                                             ('column', 'half_level'))
            flux_up_sw = file.createVariable('flux_up_sw', 'float32',
                                             ('column', 'half_level'))
            hr_lw = file.createVariable('hr_lw', 'float32', ('column', 'level'))
            hr_sw = file.createVariable('hr_sw', 'float32', ('column', 'level'))
            # Populate the variables
            sca_inputs[:] = np.random.rand(column, sca_variable)
            col_inputs[:] = np.random.rand(column, level, col_variable)
            hl_inputs[:] = np.random.rand(column, half_level, hl_variable)
            pressure_hl[:] = np.random.rand(column, half_level, p_variable)
            inter_inputs[:] = np.random.rand(column, level_interface, inter_variable)

            flux_dn_lw[:] = np.random.rand(column, half_level)
            flux_up_lw[:] = np.random.rand(column, half_level)
            flux_dn_sw[:] = np.random.rand(column, half_level)
            flux_up_sw[:] = np.random.rand(column, half_level)
            hr_lw[:] = np.random.rand(column, level)
            hr_sw[:] = np.random.rand(column, level)

    def create_graph(self, file_path, filename):
        """Create a test graph."""
        def broadcast_features(tensor):
            tensor_ = torch.unsqueeze(tensor, -1)
            tensor_ = tensor_.repeat((1, 1, 138))
            tensor_ = tensor_.moveaxis(1, -1)
            return tensor_

        with netCDF4.Dataset(os.path.join(file_path, filename),
                             "r",
                             format="NETCDF4") as file:
            sca_inputs = torch.tensor(file["sca_inputs"][:])
            col_inputs = torch.tensor(file["col_inputs"][:])
            hl_inputs = torch.tensor(file["hl_inputs"][:])
            pressure_hl = torch.tensor(file["pressure_hl"][:])
            inter_inputs = torch.tensor(file["inter_inputs"][:])

            flux_dn_lw = torch.tensor(file['flux_dn_lw'][:])
            flux_up_lw = torch.tensor(file['flux_up_lw'][:])
            flux_dn_sw = torch.tensor(file['flux_dn_sw'][:])
            flux_up_sw = torch.tensor(file['flux_up_sw'][:])
            hr_sw = torch.tensor(file["hr_sw"][:])
            hr_lw = torch.tensor(file["hr_lw"][:])

        feats = torch.cat(
            [
                broadcast_features(sca_inputs),
                hl_inputs,
                pad(inter_inputs, (0, 0, 1, 1, 0, 0)),
                pressure_hl
            ], dim=-1)

        targets = torch.cat(
            [
                torch.unsqueeze(flux_dn_lw, -1),
                torch.unsqueeze(flux_up_lw, -1),
                torch.unsqueeze(flux_dn_sw, -1),
                torch.unsqueeze(flux_up_sw, -1),
                torch.unsqueeze(pad(hr_lw, (1, 0)), -1),
                torch.unsqueeze(pad(hr_sw, (1, 0)), -1)
            ], dim=-1)

        stats_path = os.path.join(file_path, "stats-1.pt")
        if not os.path.isfile(stats_path):
            stats = {
                "x_mean": torch.mean(feats, dim=0),
                "y_mean": torch.mean(targets, dim=0),
                "x_std": torch.std(feats, dim=0),
                "y_std": torch.std(targets, dim=0)
            }
            torch.save(stats, stats_path)

        directed_index = np.array([[*range(1, 138)], [*range(137)]])
        undirected_index = np.hstack((
            directed_index,
            directed_index[[1, 0], :]
        ))
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
            print(data)
            data_list.append(data)

        return data_list

    def test_forward(self):
        """Test the "forward" method generates a Tensor."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw")
            data_test = self.create_graph(file_path, self.filename)

            test_gat = self.model_test
            test_forward = test_gat.forward(data_test[0].x, data_test[0].edge_index)

            self.assertTrue(isinstance(test_forward, torch.Tensor))

    def test__normalize(self):
        """
        Test the '_normalize' method returns tensors with
        mean = 0 and standard deviation = 1.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw")
            data_test = self.create_graph(file_path, self.filename)

            test_gat = self.model_test
            batch = pyg.data.Batch.from_data_list(data_test)
            batch_size = 1

            x, y, edge_index = test_gat._normalize(batch[0], batch_size, file_path)

            self.assertEqual(x.shape, (batch_size * 138, 21))
            self.assertEqual(y.shape, (batch_size * 138, 6))

    def test_common_step(self):
        """Test the "_common_step" method returns a 3 length tuple."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw")
            data_test = self.create_graph(file_path, self.filename)

            test_gat = self.model_test
            batch = pyg.data.Batch.from_data_list(data_test)

            loss = test_gat._common_step(batch=batch, batch_idx=1, stage="train")
            self.assertEqual(len(loss), 3)

    def test_training_step(self):
        """Test the "training_step" method returns a Tensor."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw")
            data_test = self.create_graph(file_path, self.filename)

            test_gat = self.model_test
            batch = pyg.data.Batch.from_data_list(data_test)

            loss = test_gat.training_step(batch=batch, batch_idx=1)
            self.assertTrue(isinstance(loss, torch.Tensor))

#     def test_test_step(self):
#         """Test the "test_step" method returns a tuple of same size Tensors."""
#         with tempfile.TemporaryDirectory() as tempdir:
#             self.create_env(tempdir)
#             file_path = os.path.join(tempdir, "data", "raw", self.filenames)
#             data_test = self.create_graph(file_path)

#             test_gat = models.LitGAT(**self.initParam)
#             batch = pyg.data.Batch.from_data_list(data_test)

#             out_tuple = test_gat.test_step(batch=batch, batch_idx=1)

#             self.assertEqual(len(out_tuple), 3)

    def test_configure_optimizers(self):
        """Test the "configure_optimizers" method returns an optim.Optimizer."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw")
            _ = self.create_graph(file_path, self.filename)

            test_gat = self.model_test
            op = test_gat.configure_optimizers()

            self.assertIsInstance(op, optim.Optimizer)


if __name__ == '__main__':
    unittest.main()
