"""This module provides a unit tests suite for the models.py module."""
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
import shutil
from pathlib import Path
import unittest
import torch
import torch_optimizer as optim

from test_utils import get_filenames, populate_test_data
from data import NOGWDDataset
from models import LitMLP, LitCNN

CURRENT_DIR = Path(__file__).parent.absolute()
TEST_DATA_PATH = os.path.join(CURRENT_DIR, "test_data", "data")
REF_FILENAMES_FILE = os.path.join(CURRENT_DIR, "test_data", "filenames-split.yaml")


class TestLitMLP(unittest.TestCase):
    """Test the NOGWDDataset class in train mode."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment to get the data input files."""
        # get the list of filename that should be created
        filenames = get_filenames(REF_FILENAMES_FILE)
        # create the required files with fake data
        populate_test_data(TEST_DATA_PATH, filenames)
        # put a copy of the 'filenames-split.yaml' file in the expected place
        shutil.copy(REF_FILENAMES_FILE, TEST_DATA_PATH)

    def setUp(self) -> None:
        """Instantiate the LitMLP class and produce input dataset."""
        # Generate data fir test purpose
        data_test = NOGWDDataset(root=TEST_DATA_PATH, mode='train')
        self.data = data_test.load()
        # Instantiate a LitMLP
        self.model_test = LitMLP(in_channels=191,
                                 hidden_channels=10,
                                 out_channels=126,
                                 lr=0.001)

    def test_forward(self):
        """Test the 'forward' method returns a properly formatted tensor."""
        test_forward = self.model_test.forward(self.data[0])
        self.assertIsInstance(test_forward, torch.Tensor)
        # 3 : nb of train .h5 files
        # 36 and 10 : dimension of input data
        # 126 : nb of feature for y
        self.assertEqual(test_forward.size(), torch.Size([3 * 36 * 10, 126]))

    def test_compute_stats(self):
        """Test the '_compute_stats' method returns a scalar tensor as the loss."""
        loss = self.model_test._common_step(self.data, 10, "stage")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.numel(), 1)

    def test_training_step(self):
        """Test the 'training_step' method returns a scalar tensor as the loss."""
        loss = self.model_test.training_step(self.data, 10)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.numel(), 1)

    def test_configure_optimizers(self):
        """Test the 'configure_optimizers' method returns an 'optim.AdamP' object."""
        optimizer = self.model_test.configure_optimizers()
        self.assertIsInstance(optimizer, optim.AdamP)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean the test artifacts."""
        # clean the test data
        shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)


class TestLitCNN(unittest.TestCase):
    """Test the NOGWDDataset class in train mode."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment to get the data input files."""
        # get the list of filename that should be created
        filenames = get_filenames(REF_FILENAMES_FILE)
        # create the required files with fake data
        populate_test_data(TEST_DATA_PATH, filenames)
        # put a copy of the 'filenames-split.yaml' file in the expected place
        shutil.copy(REF_FILENAMES_FILE, TEST_DATA_PATH)

    def setUp(self) -> None:
        """Instantiate the LitMLP class and produce input dataset."""
        # Generate data fir test purpose
        data_test = NOGWDDataset(root=TEST_DATA_PATH, mode='train')
        self.data = data_test.load()
        # Instantiate a LitCNN
        self.model_test = LitCNN(in_channels=5,
                                 init_feat=16,
                                 out_channels=126,
                                 conv_size=1,
                                 pool_size=2,
                                 lr=0.001)

    def test_forward(self):
        """Test the 'forward' method returns a properly formatted tensor."""
        print(self.model_test)
        test_forward = self.model_test.forward(self.data[0])
        self.assertIsInstance(test_forward, torch.Tensor)
        self.assertEqual(test_forward.size(), torch.Size([1080, 126]))

    def test_compute_stats(self):
        """Test the '_compute_stats' method returns a scalar tensor as the loss."""
        loss = self.model_test._common_step(self.data, 10, "stage")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.numel(), 1)

    def test_training_step(self):
        """Test the 'training_step' method returns a scalar tensor as the loss."""
        loss = self.model_test.training_step(self.data, 10)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.numel(), 1)

    def test_configure_optimizers(self):
        """Test the 'configure_optimizers' method returns an 'optim.AdamP' object."""
        optimizer = self.model_test.configure_optimizers()
        self.assertIsInstance(optimizer, optim.AdamP)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean the test artifacts."""
        # clean the test data
        shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
