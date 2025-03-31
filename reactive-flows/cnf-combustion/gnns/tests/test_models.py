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
import tempfile
import unittest

import h5py
import numpy as np
import torch
import torch_geometric as pyg
import torch_optimizer as optim
import yaml
from lightning.pytorch.trainer import Trainer

from models import LitGAT, LitGCN, LitGIN, LitGraphUNet
from utils import create_graph_topo


class TestModel(unittest.TestCase):
    """Model test file."""

    def setUp(self) -> None:
        """Define default parameters."""
        self.filenames = ["DNS1_00116000.h5", "DNS1_00117000.h5", "DNS1_00118000.h5"]

        self.init_param = {
            "in_channels": 1,
            "hidden_channels": 32,
            "out_channels": 1,
            "num_layers": 4,
            "dropout": 0.5,
            "jk": "last",
            "lr": 0.0001,
        }

    def create_env(self, tempdir):
        """Create a test environment and data files test."""
        os.mkdir(os.path.join(tempdir, "data"))
        os.mkdir(os.path.join(tempdir, "data", "raw"))

        for file_h5 in self.filenames:
            with h5py.File(os.path.join(tempdir, "data", "raw", file_h5), "w") as file:
                file["filt_8"] = np.zeros((10, 10, 10))
                file["filt_grad_8"] = np.zeros((10, 10, 10))
                file["grad_filt_8"] = np.zeros((10, 10, 10))

        temp_file_path = os.path.join(tempdir, "data", "filenames.yaml")
        with open(temp_file_path, "w") as tmpfile:
            _ = yaml.dump(self.filenames, tmpfile)

    def create_graph(self, file_path):
        """Create a test graph."""
        with h5py.File(file_path, "r") as file:
            col = file["/filt_8"][:]
            sigma = file["/filt_grad_8"][:]

        data = pyg.data.Data(
            x=torch.tensor(col.reshape(-1, 1), dtype=torch.float),
            y=torch.tensor(sigma.reshape(-1, 1), dtype=torch.float),
        )
        graph_topo = create_graph_topo(col.shape)

        return data, graph_topo

    def test_forward(self):
        """Test the "forward" method generates a Tensor."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            data_test, graph_topo = self.create_graph(file_path)
            self.init_param.update({"graph_topology": graph_topo}),

            test_gcn = LitGCN(**self.init_param)
            test_forward = test_gcn.forward(data_test.x, graph_topo.edge_index)

            self.assertTrue(isinstance(test_forward, torch.Tensor))

    def test_common_step(self):
        """Test the "_common_step" method returns a 3 length tuple."""
        gin_init_param = {
            "in_channels": 1,
            "hidden_channels": 32,
            "out_channels": 1,
            "num_layers": 4,
            "dropout": 0.5,
            "lr": 0.0001,
        }

        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            data_test, graph_topo = self.create_graph(file_path)
            gin_init_param.update({"graph_topology": graph_topo}),

            test_gin = LitGIN(**gin_init_param)
            batch = pyg.data.Batch.from_data_list([data_test, data_test])

            loss = test_gin._common_step(batch=batch, batch_idx=1, stage="train")

            self.assertEqual(len(loss), 3)

    def test_training_step(self):
        """Test the "training_step" method returns a Tensor."""
        gat_init_param = {
            "in_channels": 1,
            "hidden_channels": 32,
            "out_channels": 1,
            "num_layers": 4,
            "dropout": 0.5,
            "heads": 8,
            "jk": "last",
            "lr": 0.0001,
        }

        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            data_test, graph_topo = self.create_graph(file_path)
            gat_init_param.update({"graph_topology": graph_topo}),

            test_gat = LitGAT(**gat_init_param)
            batch = pyg.data.Batch.from_data_list([data_test, data_test])

            loss = test_gat.training_step(batch=batch, batch_idx=1)
            self.assertTrue(isinstance(loss, torch.Tensor))

    def test_test_step(self):
        """Test the "test_step" method returns a tuple of same size Tensors."""
        gunet_init_param = {
            "in_channels": 1,
            "hidden_channels": 32,
            "out_channels": 1,
            "depth": 4,
            "pool_ratios": 0.5,
            "lr": 0.0001,
        }

        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            data_test, graph_topo = self.create_graph(file_path)
            gunet_init_param.update({"graph_topology": graph_topo}),

            test_gunet = LitGraphUNet(**gunet_init_param)
            batch = pyg.data.Batch.from_data_list([data_test, data_test])

            out_tuple = test_gunet.test_step(batch=batch, batch_idx=1)

            self.assertEqual(len(out_tuple), 2)
            self.assertEqual(out_tuple[0].size(), out_tuple[1].size())

            test_gunet._trainer = Trainer()
            test_gunet.on_test_epoch_end()

            self.assertTrue(
                os.path.exists(os.path.join(test_gunet.trainer.log_dir, "plots"))
            )

    def test_configure_optimizers(self):
        """Test the "configure_optimizers" method returns an optim.Optimizer."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir, "data", "raw", self.filenames[0])
            _, graph_topo = self.create_graph(file_path)
            self.init_param.update({"graph_topology": graph_topo}),

            test_gcn = LitGCN(**self.init_param)
            op = test_gcn.configure_optimizers()

            self.assertIsInstance(op, optim.Optimizer)


if __name__ == "__main__":
    unittest.main()
