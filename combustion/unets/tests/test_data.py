'''
    Licensed under the Apache License, Version 2.0 (the "License");
    * you may not use this file except in compliance with the License.
    * You may obtain a copy of the License at
    *
    *     http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing, software
    * distributed under the License is distributed on an "AS IS" BASIS,
    * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    * See the License for the specific language governing permissions and
    * limitations under the License.
'''

import unittest
import os

import h5py
import data
import yaml
import tempfile
import numpy as np
import torch

import warnings

SPLITTING_LENGTHS = [111, 8, 8]


class TestData(unittest.TestCase):
    """
    Data test file
    """

    def test_random_cropper(self):
        n, n_ = 64, 32
        x = np.random.rand(n,n,n)
        y = np.copy(x)
        random_cropper = data.RandomCropper3D(n_)
        x_, y_ = random_cropper(x, y)
        self.assertEqual(x_.shape, (n_,n_,n_))
        self.assertEqual(y_.shape, (n_,n_,n_))
        self.assertEqual(x_[0,0,0], y_[0,0,0])

    def setUp(self) -> None:
        """
        Define default parameters.
        """
        self.filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']
        self.initParam = {
            'batch_size': 1,
            'num_workers': 0,
            'y_normalizer': 342.553,
            'splitting_lengths': [111, 8, 8],
            'subblock_shape': [32, 16, 16]}

    def create_env(self, tempdir):
        os.mkdir(os.path.join(tempdir, "data"))
        os.mkdir(os.path.join(tempdir, "data", "raw"))

        for file_h5 in self.filenames:
            with h5py.File(os.path.join(tempdir, "data", "raw", file_h5), 'w') as f:
                f['filt_8']      = np.zeros((10, 10, 10))
                f['filt_grad_8'] = np.zeros((10, 10, 10))
                f['grad_filt_8'] = np.zeros((10, 10, 10))

        temp_file_path = os.path.join(tempdir, 'data', 'filenames.yaml')
        with open(temp_file_path, 'w') as tmpfile:
            yaml.dump(self.filenames, tmpfile)

    def create_obj_rm_warning(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return data.CnfCombustionDataset(path)

    def test_process(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            data_test = self.create_obj_rm_warning(os.path.join(tempdir, "data"))
            data_test.process(0, os.path.join(tempdir, "data", "DNS1_00116000.h5"))
            self.assertTrue(os.path.exists(os.path.join(tempdir, "data", "processed")))

            # insert +2 to have transform and filter files
            self.assertEqual(len(os.listdir(os.path.join(tempdir, "data", "processed"))), len(self.filenames)+2)

    def test_get(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            data_test = self.create_obj_rm_warning(os.path.join(tempdir, "data"))
            data_get = data_test.get(2)
            self.assertEqual(len(data_get.x), 10*10*10)

    def test_setup(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            dataset_test = data.CnfCombustionDataModule(**self.initParam)

            with self.assertRaises(ValueError) as context:
                dataset_test.setup()
                self.assertTrue('The dataset is too small to be split properly.' in str(context.exception))

            self.assertEqual(len(dataset_test.train_dataset),
                             int(len(self.filenames)*0.8))
            self.assertEqual(len(dataset_test.val_dataset),
                             int(len(self.filenames))-int(len(self.filenames)*0.9))
            self.assertEqual(len(dataset_test.test_dataset),
                             int(len(self.filenames)*0.9)-int(len(self.filenames)*0.8))

    def test_train_dataloader(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            dataset_test = data.CnfCombustionDataModule(**self.initParam)

            with self.assertRaises(ValueError) as context:
                dataset_test.setup()

            test_train_dl = dataset_test.train_dataloader()
            self.assertTrue(isinstance(test_train_dl, torch.utils.data.DataLoader))

    def test_val_dataloader(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            dataset_test = data.CnfCombustionDataModule(**self.initParam)

            with self.assertRaises(ValueError) as context:
                dataset_test.setup()

            test_val_dl = dataset_test.train_dataloader()
            self.assertTrue(isinstance(test_val_dl, torch.utils.data.DataLoader))

    def test_test_dataloader(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            dataset_test = data.CnfCombustionDataModule(**self.initParam)

            with self.assertRaises(ValueError) as context:
                dataset_test.setup()

            test_test_dl = dataset_test.train_dataloader()
            self.assertTrue(isinstance(test_test_dl, torch.utils.data.DataLoader))


if __name__ == '__main__':
    unittest.main()