"""This module provides a unit tests suite for the data.py module."""
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
import unittest
import torch
from pathlib import Path

from data import ThreeDCorrectionDataset, LitThreeDCorrectionDataModule

CURRENT_DIR = Path(__file__).parent.absolute()
TEST_DATA_PATH = os.path.join(CURRENT_DIR, "test_data", "data")


class TestData(unittest.TestCase):
    """Data test suite."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment to test the ThreeDCorrectionDataset class."""
        if not os.path.exists(os.path.join(TEST_DATA_PATH)):
            os.mkdir(os.path.join(TEST_DATA_PATH))

        if not os.path.exists(os.path.join(TEST_DATA_PATH, "raw")):
            os.mkdir(os.path.join(TEST_DATA_PATH, "raw"))

    def setUp(self) -> None:
        """
        Define default parameters and
        instantiate the ThreeDCorrectionDataset class in train mode.
        """
        self.initParam = {'timestep': 3500,
                          'patchstep': 16,
                          'batch_size': 1,
                          'num_workers': 0}
        self.num_files = 2 * (16 // self.initParam["patchstep"]) \
            * (3501 // self.initParam["timestep"] + 1)
        self.data_test = ThreeDCorrectionDataset(root=TEST_DATA_PATH,
                                                 timestep=self.initParam["timestep"],
                                                 patchstep=self.initParam["patchstep"])
        self.data_len = len(self.data_test)
        print(f"DATA LENGTH = {self.data_len}")

    def test_raw_file_names(self):
        """Test raw file name."""
        expected_file = ["data-3500.nc"]
        self.assertEqual(self.data_test.raw_file_names, expected_file)

    def test_processed_file_names(self):
        """Test processed file name."""
        expected_file = ["data-3500.pt"]
        self.assertEqual(self.data_test.processed_file_names, expected_file)

    def test_download(self):
        """Test download raise error."""
        self.data_test.download()

        self.assertEqual(len(next(os.walk(os.path.join(TEST_DATA_PATH, "raw")))[2]) - 1,
                         self.num_files + 1)

    def test_process(self):
        """Test download raise error."""
        self.data_test.process()

        self.assertTrue(os.path.exists(os.path.join(TEST_DATA_PATH, "processed")))
        self.assertTrue(os.path.isfile(os.path.join(TEST_DATA_PATH, 'stats-3500.pt')))

        # insert +2 to have transform and filter files
        self.assertEqual(
            len(os.listdir(os.path.join(TEST_DATA_PATH, "processed"))),
            len(self.data_test.processed_file_names) + 2
        )

    def test_setup(self):
        """Test the "setup" method."""
        dataset_test = LitThreeDCorrectionDataModule(**self.initParam)
        dataset_test.setup(stage=None)

        self.assertEqual(len(dataset_test.train_dataset),
                         int(self.data_len * 0.8))
        self.assertEqual(len(dataset_test.val_dataset),
                         int(self.data_len) - int(self.data_len * 0.9))
        self.assertEqual(len(dataset_test.test_dataset),
                         int(self.data_len * 0.9) - int(self.data_len * 0.8))

    def test_train_dataloader(self):
        """Test the "train_dataloader"."""
        dataset_test = LitThreeDCorrectionDataModule(**self.initParam)
        dataset_test.setup(stage=None)

        test_train_dl = dataset_test.train_dataloader()
        self.assertTrue(isinstance(test_train_dl, torch.utils.data.DataLoader))

    def test_val_dataloader(self):
        """Test the "val_dataloader"."""
        dataset_test = LitThreeDCorrectionDataModule(**self.initParam)
        dataset_test.setup(stage=None)

        test_val_dl = dataset_test.val_dataloader()
        self.assertTrue(isinstance(test_val_dl, torch.utils.data.DataLoader))

    def test_test_dataloader(self):
        """Test the "test_dataloader"."""
        dataset_test = LitThreeDCorrectionDataModule(**self.initParam)
        dataset_test.setup(stage=None)

        test_test_dl = dataset_test.test_dataloader()
        self.assertTrue(isinstance(test_test_dl, torch.utils.data.DataLoader))

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean the test artifacts."""
        # clean the test data
        shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
