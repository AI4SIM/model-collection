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

from unittest import TestCase, main
from os.path import join, exists
from os import mkdir, listdir
from h5py import File
from data import CnfCombustionDataset, CnfCombustionDataModule
from yaml import dump
from tempfile import mkdtemp
from shutil import rmtree
from numpy import zeros
from torch.utils.data import DataLoader
from warnings import catch_warnings, simplefilter


class TestData(TestCase):

    def setUp(self) -> None:
        self.filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']
        self.data_module_params = {
            'batch_size': 1,
            'num_workers': 0,
            'y_normalizer': 342.553,
            'splitting_lengths': [1, 1, 1],
            'subblock_shape': (32, 16, 16)}

        # Creates a temporary environment.
        self.dir = mkdtemp()
        self.create_env(self.dir)

        # Creates dataset and data module.
        with catch_warnings():
            simplefilter("ignore")
            data_path = join(self.dir, "data")
            self.dataset = CnfCombustionDataset(data_path)
            self.data_module = CnfCombustionDataModule(**self.data_module_params)
            self.data_module.prepare_data(data_path)

    def tearDown(self) -> None:
        rmtree(self.dir)

    def create_env(self, tempdir):
        mkdir(join(tempdir, "data"))
        mkdir(join(tempdir, "data", "raw"))

        for file_h5 in self.filenames:
            with File(join(tempdir, "data", "raw", file_h5), 'w') as f:
                f['filt_8'] = zeros((10, 10, 10))
                f['filt_grad_8'] = zeros((10, 10, 10))
                f['grad_filt_8'] = zeros((10, 10, 10))

        temp_file_path = join(tempdir, 'data', 'filenames.yaml')
        with open(temp_file_path, 'w') as tmpfile:
            dump(self.filenames, tmpfile)

    def test_process(self):
        self.dataset.process(0, join(self.dir, "data", "raw", "DNS1_00116000.h5"))
        self.assertTrue(exists(join(self.dir, "data", "processed")))
        self.assertEqual(
            len(listdir(join(self.dir, "data", "processed"))),
            len(self.filenames))

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_setup(self):
        self.data_module.setup()
        self.assertEqual(len(self.data_module.train_dataset), 1)
        self.assertEqual(len(self.data_module.val_dataset), 1)
        self.assertEqual(len(self.data_module.test_dataset), 1)

    def test_train_dataloader(self):
        self.data_module.setup()
        test_train_dl = self.data_module.train_dataloader()
        self.assertTrue(isinstance(test_train_dl, DataLoader))

    def test_val_dataloader(self):
        self.data_module.setup()
        test_val_dl = self.data_module.train_dataloader()
        self.assertTrue(isinstance(test_val_dl, DataLoader))

    def test_test_dataloader(self):
        self.data_module.setup()
        test_test_dl = self.data_module.train_dataloader()
        self.assertTrue(isinstance(test_test_dl, DataLoader))


if __name__ == '__main__':
    main()
