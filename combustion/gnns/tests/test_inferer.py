"""This module proposes a test suite for the inferer module."""
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
from pathlib import Path
import unittest
import h5py
import numpy
import pickle
import torch
import torch.nn as nn
import torch_geometric as pyg

from models import LitGIN
from inferer import Inferer, InferencePthGnn

CURRENT_DIR = Path(__file__).parent.absolute()
TEST_DATA_PATH = os.path.join(CURRENT_DIR, "test_data")


class TestInferer(unittest.TestCase):
    """Inferer test suite."""

    def setUp(self) -> None:
        """Init the inferer object used."""
        self.model_pt = os.path.join(TEST_DATA_PATH, "test_model.pt")
        model = nn.Linear(in_features=5, out_features=2)
        torch.save(model, self.model_pt)
        self.inferer = Inferer(model_path=self.model_pt, data_path="", wkd=TEST_DATA_PATH)

    def test_data_processed_path(self) -> None:
        """Test the data_processed_path property returns the proper path of saved data."""
        self.inferer.data_path = os.path.join(TEST_DATA_PATH, "test_data.origin")
        saved_data = self.inferer.data_processed_path
        self.assertEqual(saved_data, os.path.join(TEST_DATA_PATH, "test_data.data"))

    def test_load_data_not_implemented(self) -> None:
        """Test the load_data method raises a NotImplementedError."""
        self.assertRaises(NotImplementedError,
                          self.inferer.load_data)

    def test_preprocess_not_implemented(self) -> None:
        """Test the preprocess method raises a NotImplementedError."""
        self.assertRaises(NotImplementedError,
                          self.inferer.preprocess)

    def test_predict(self) -> None:
        """Test the predict method returns a torch.Tensor."""
        self.inferer.data = torch.testing.make_tensor((5,), device='cpu', dtype=torch.float32)
        preds = self.inferer.predict()
        self.assertIsInstance(preds, torch.Tensor)

    def tearDown(self) -> None:
        """Clean up the test artifacts."""
        os.remove(self.model_pt)


class TestInferencePthGnn(unittest.TestCase):
    """InferencePthGnn test suite."""

    @classmethod
    def setUpClass(cls) -> None:
        """Init the global data used."""
        cls.infer_data = os.path.join(TEST_DATA_PATH, 'test_infer_data.h5')
        with h5py.File(cls.infer_data, 'w') as file:
            file['/c_filt'] = numpy.random.rand(42, 7, 66)
            file['/c_grad_filt'] = numpy.random.rand(42, 7, 66)
            file['/c_filt_grad'] = numpy.random.rand(42, 7, 66)

    def setUp(self) -> None:
        """Init the inferer object used."""
        self.model_ckpt = os.path.join(TEST_DATA_PATH, "test_model.ckpt")
        self.inferer = InferencePthGnn(model_path=self.model_ckpt,
                                       data_path=self.infer_data,
                                       model_class=LitGIN,
                                       wkd=TEST_DATA_PATH)

    def test_load_data(self) -> None:
        """Test the load_data method returns an array from an hdf5 file."""
        data = self.inferer.load_data()
        self.assertIsInstance(data, numpy.ndarray)
        with h5py.File(self.infer_data, 'r') as file:
            expected_array = file['/c_filt'][:]
        numpy.testing.assert_equal(data, expected_array)

    def test_load_y_dns(self) -> None:
        """Test the load_y_dns method returns an array from an hdf5 file."""
        data = self.inferer.load_y_dns()
        self.assertIsInstance(data, numpy.ndarray)
        with h5py.File(self.infer_data, 'r') as file:
            expected_array = file['/c_grad_filt'][:]
        numpy.testing.assert_equal(data, expected_array)

    def test_load_y_les(self) -> None:
        """Test the load_y_les method returns an array from an hdf5 file."""
        data = self.inferer.load_y_les()
        self.assertIsInstance(data, numpy.ndarray)
        with h5py.File(self.infer_data, 'r') as file:
            expected_array = file['/c_filt_grad'][:]
        numpy.testing.assert_equal(data, expected_array)

    def test_create_graph(self) -> None:
        """Test the _create_graph method creates a data graph from input features."""
        fake_data = torch.testing.make_tensor((2, 3, 4), device="cpu", dtype=torch.float32)
        self.inferer._create_graph(fake_data)
        self.assertIsInstance(self.inferer.data, pyg.data.Data)
        torch.testing.assert_close(fake_data.reshape(-1, 1), self.inferer.data.x)

    def test_create_graph_save(self) -> None:
        """Test the _create_graph method creates a data graph from input features and save it."""
        patch_data_file = os.path.join(TEST_DATA_PATH, 'test_save_file.data')
        self.inferer.data_path = os.path.join(TEST_DATA_PATH, 'test_save_file.h5')
        fake_data = torch.testing.make_tensor((2, 3, 4), device="cpu", dtype=torch.float32)
        self.inferer._create_graph(fake_data, save=True)
        self.assertTrue(Path(patch_data_file).resolve().is_file())
        os.remove(patch_data_file)

    def test_create_graph_existing(self) -> None:
        """Test the _create_graph method creates a data graph from data loaded from a previously
        saved file.
        """
        # create the a false previoulsy saved data file
        saved_data = torch.testing.make_tensor((1, 2, 3), device="cpu", dtype=torch.float32)
        with open(self.inferer.data_processed_path, 'wb') as file:
            pickle.dump(saved_data, file)
        # create graph on same data -> no re-creation, just load the file
        self.inferer._create_graph(saved_data)
        torch.testing.assert_close(saved_data, self.inferer.data)
        os.remove(self.inferer.data_processed_path)

    def test_preprocess(self) -> None:
        """Test the preprocess method set the self.data attribute."""
        self.assertIsNone(self.inferer.data)
        self.inferer.preprocess()
        self.assertIsInstance(self.inferer.data, pyg.data.Data)

    def test_predict(self) -> None:
        """Test the predict method returns a torch.Tensor."""
        self.inferer.preprocess()
        preds = self.inferer.predict()
        self.assertIsInstance(preds, torch.Tensor)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean the test artifacts."""
        os.remove(cls.infer_data)


if __name__ == '__main__':
    unittest.main()
