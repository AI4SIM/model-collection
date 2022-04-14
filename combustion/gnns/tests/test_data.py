"""This module proposes a test suite for the data module."""
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
import unittest
import h5py
import yaml
import tempfile
import numpy as np
import torch
import warnings

from data import CnfDataset, LitCombustionDataModule


class TestData(unittest.TestCase):
    """Data test suite."""

    def setUp(self) -> None:
        """Define default parameters."""
        self.filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']
        self.initParam = {'batch_size': 1, 'num_workers': 0, 'y_normalizer': 342.553}

    def create_env(self, tempdir):
        """Create a test environment and data test."""
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

    def create_obj_rm_warning(self, path):
        """Instantiante the CombustionDataset object with a warning filtering."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CnfDataset(path)

    def test_raw_file_names(self):
        """Test raw file name."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            data_test = self.create_obj_rm_warning(os.path.join(tempdir, "data"))

            raw_test = data_test.raw_file_names
            self.assertEqual(len(self.filenames), len(raw_test))

    def test_processed_file_names(self):
        """Test processed file name."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)

            data_test = self.create_obj_rm_warning(os.path.join(tempdir, "data"))
            processed_test = data_test.processed_file_names

            self.assertEqual(len(self.filenames), len(processed_test))

    def test_download(self):
        """Test download raise error."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            data_test = self.create_obj_rm_warning(os.path.join(tempdir, "data"))

            with self.assertRaises(RuntimeError) as context:
                _ = data_test.download()
                self.assertTrue('Data not found.' in str(context.exception))

    def test_process(self):
        """Test download raise error."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            data_test = self.create_obj_rm_warning(os.path.join(tempdir, "data"))
            data_test.process()

            self.assertTrue(os.path.exists(os.path.join(tempdir, "data", "processed")))

            # insert +2 to have transform and filter files
            self.assertEqual(
                len(os.listdir(os.path.join(tempdir, "data", "processed"))),
                len(self.filenames) + 2
            )

    def test_get(self):
        """Test download raise error."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            data_test = self.create_obj_rm_warning(os.path.join(tempdir, "data"))
            data_get = data_test.get(2)

            self.assertEqual(len(data_get.x), 10 * 10 * 10)

    def test_setup(self):
        """Test the "setup" method."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            _ = self.create_obj_rm_warning(os.path.join(tempdir, "data"))

            dataset_test = LitCombustionDataModule(**self.initParam)

            with self.assertRaises(ValueError) as context:
                dataset_test.setup(stage=None, data_path=os.path.join(tempdir, "data"))
                self.assertTrue(
                    'The dataset is too small to be split properly.' in str(context.exception)
                )

            self.assertEqual(len(dataset_test.train_dataset),
                             int(len(self.filenames) * 0.8))
            self.assertEqual(len(dataset_test.val_dataset),
                             int(len(self.filenames)) - int(len(self.filenames) * 0.9))
            self.assertEqual(len(dataset_test.test_dataset),
                             int(len(self.filenames) * 0.9) - int(len(self.filenames) * 0.8))

    def test_train_dataloader(self):
        """Test the "train_dataloader"."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            _ = self.create_obj_rm_warning(os.path.join(tempdir, "data"))

            dataset_test = LitCombustionDataModule(**self.initParam)

            with self.assertRaises(ValueError):
                _ = dataset_test.setup(stage=None, data_path=os.path.join(tempdir, "data"))

            test_train_dl = dataset_test.train_dataloader()
            self.assertTrue(isinstance(test_train_dl, torch.utils.data.DataLoader))

    def test_val_dataloader(self):
        """Test the "val_dataloader"."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            _ = self.create_obj_rm_warning(os.path.join(tempdir, "data"))

            dataset_test = LitCombustionDataModule(**self.initParam)

            with self.assertRaises(ValueError):
                _ = dataset_test.setup(stage=None, data_path=os.path.join(tempdir, "data"))

            test_val_dl = dataset_test.val_dataloader()
            self.assertTrue(isinstance(test_val_dl, torch.utils.data.DataLoader))

    def test_test_dataloader(self):
        """Test the "test_dataloader"."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            _ = self.create_obj_rm_warning(os.path.join(tempdir, "data"))

            dataset_test = LitCombustionDataModule(**self.initParam)

            with self.assertRaises(ValueError):
                _ = dataset_test.setup(stage=None, data_path=os.path.join(tempdir, "data"))

            test_test_dl = dataset_test.test_dataloader()
            self.assertTrue(isinstance(test_test_dl, torch.utils.data.DataLoader))


if __name__ == '__main__':
    unittest.main()
