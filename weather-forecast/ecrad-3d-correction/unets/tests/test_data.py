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
import os
import os.path as osp
from data import ThreeDCorrectionDataset, LitThreeDCorrectionDataModule
from tempfile import mkdtemp
from shutil import rmtree
from torch.utils.data import DataLoader
from warnings import catch_warnings, simplefilter
import numpy as np
import dask.array as da
import torch


class TestData(TestCase):

    def setUp(self) -> None:
        # Creates a temporary environment.
        self.root = mkdtemp()
        self.create_env(self.root)

        # Creates dataset and data module.
        with catch_warnings():
            simplefilter("ignore")
            data_path = osp.join(self.root, "processed")
            self.dataset = ThreeDCorrectionDataset(data_path=data_path)
            self.data_module = LitThreeDCorrectionDataModule(
                data_path=data_path,
                batch_size=1,
                num_workers=0,
                splitting_ratios=(0.5, 0.25, 0.25))
            self.data_module.prepare_data()

    def tearDown(self) -> None:
        rmtree(self.root)

    def create_env(self, root) -> None:
        """Build a data/processed environment with a single input."""
        data_path = osp.join(root, "processed")
        os.makedirs(osp.join(data_path, "x"), exist_ok=True)
        os.makedirs(osp.join(data_path, "y"), exist_ok=True)

        self.n_data = 8
        shard_size = 4
        x = np.random.rand(self.n_data, 4, 138)  # mock inputs.
        y = np.random.rand(self.n_data, 4, 138)  # mock flux outputs.

        # Save specific datum to check.
        self.datum1_x = x[1]
        self.datum1_y = y[1]

        # Save to shards.
        x_chunked = da.from_array(x, chunks=shard_size)
        y_chunked = da.from_array(y, chunks=shard_size)
        da.to_npy_stack(osp.join(data_path, 'x'), x_chunked, axis=0)
        da.to_npy_stack(osp.join(data_path, 'y'), y_chunked, axis=0)

        torch.save({
            'x_mean': torch.tensor(np.random.rand(4, 138)),
            'x_std': torch.tensor(np.random.rand(4, 138)),
            'x_nb': torch.tensor(self.n_data),
            'y_mean': torch.tensor(np.random.rand(4, 138)),
            'y_std': torch.tensor(np.random.rand(4, 138)),
            'y_nb': torch.tensor(self.n_data)
        }, osp.join(data_path, "stats.pt"))

    def test_get(self) -> None:
        x, y = self.dataset[1]
        np.testing.assert_array_equal(x, self.datum1_x)
        np.testing.assert_array_equal(y, self.datum1_y)

    def test_len(self) -> None:
        self.assertEqual(len(self.dataset), self.n_data)

    def test_setup(self) -> None:
        self.data_module.setup()
        self.assertTrue(isinstance(self.data_module.train_dataloader(), DataLoader))
        self.assertTrue(isinstance(self.data_module.val_dataloader(), DataLoader))
        self.assertTrue(isinstance(self.data_module.test_dataloader(), DataLoader))


if __name__ == '__main__':
    main()
