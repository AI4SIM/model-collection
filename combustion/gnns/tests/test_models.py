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

import unittest
import os
import tempfile
import h5py
import numpy as np
import yaml
import networkx as nx
import torch
import torch_geometric as pyg
import torch_optimizer as optim

import models


class TestModel(unittest.TestCase):
    """Model test file."""

    def setUp(self) -> None:
        """Define default parameters."""
        self.filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']

        self.initParam = {
            'in_channels': 1,
            'hidden_channels': 32,
            'out_channels': 1,
            'num_layers': 4,
            'dropout': .5,
            'jk': "last",
            'lr': .0001
        }

    def create_env(self, tempdir):
        """Create a test environment and data files test."""
        os.mkdir(os.path.join(tempdir, "data"))
        os.mkdir(os.path.join(tempdir, "data", "raw"))

        for file_h5 in self.filenames:
            with h5py.File(os.path.join(tempdir, "data", "raw", file_h5), 'w') as file:
                file['filt_8'] = np.zeros((10, 10, 10))
                file['filt_grad_8'] = np.zeros((10, 10, 10))
                file['grad_filt_8'] = np.zeros((10, 10, 10))

        temp_file_path = os.path.join(tempdir, 'data', 'filenames.yaml')
        with open(temp_file_path, 'w') as tmpfile:
            _ = yaml.dump(self.filenames, tmpfile)

    def create_graph(self, file_path):
        """Create a test graph."""
        with h5py.File(file_path, 'r') as file:
            col = file["/filt_8"][:]
            sigma = file["/filt_grad_8"][:]

        x_size, y_size, z_size = col.shape
        grid_shape = (z_size, y_size, x_size)

        g0 = nx.grid_graph(dim=grid_shape)
        graph = pyg.utils.convert.from_networkx(g0)
        undirected_index = graph.edge_index
        coordinates = list(g0.nodes())
        coordinates.reverse()

        data = pyg.data.Data(
            x=torch.tensor(col.reshape(-1, 1), dtype=torch.float),
            edge_index=undirected_index.clone().detach().type(torch.LongTensor),
            pos=torch.tensor(np.stack(coordinates)),
            y=torch.tensor(sigma.reshape(-1, 1), dtype=torch.float))

        return data

    def test_forward(self):
        """Test the "forward" method generates a Tensor."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            data_test = self.create_graph(file_path)

            test_gcn = models.LitGCN(**self.initParam)
            test_forward = test_gcn.forward(data_test.x, data_test.edge_index)

            self.assertTrue(isinstance(test_forward, torch.Tensor))

    def test_common_step(self):
        """Test the "_common_step" method returns a 3 length tuple."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            data_test = self.create_graph(file_path)

            test_gcn = models.LitGCN(**self.initParam)
            batch = pyg.data.Batch.from_data_list([data_test, data_test])

            loss = test_gcn._common_step(batch=batch, batch_idx=1, stage="train")

            self.assertEqual(len(loss), 3)

    def test_training_step(self):
        """Test the "training_step" method returns a Tensor."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            data_test = self.create_graph(file_path)

            test_gcn = models.LitGCN(**self.initParam)
            batch = pyg.data.Batch.from_data_list([data_test, data_test])

            loss = test_gcn.training_step(batch=batch, batch_idx=1)
            self.assertTrue(isinstance(loss, torch.Tensor))

    def test_test_step(self):
        """Test the "test_step" method returns a tuple of same size Tensors."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            data_test = self.create_graph(file_path)

            test_gcn = models.LitGCN(**self.initParam)
            batch = pyg.data.Batch.from_data_list([data_test, data_test])

            out_tuple = test_gcn.test_step(batch=batch, batch_idx=1)

            self.assertEqual(len(out_tuple), 2)
            self.assertEqual(out_tuple[0].size(), out_tuple[1].size())

    def test_configure_optimizers(self):
        """Test the "configure_optimizers" method returns an optim.Optimizer."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            _ = self.create_graph(file_path)

            test_gcn = models.LitGCN(**self.initParam)
            op = test_gcn.configure_optimizers()

            self.assertIsInstance(op, optim.Optimizer)


if __name__ == '__main__':
    unittest.main()
