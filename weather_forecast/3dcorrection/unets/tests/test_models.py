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
from numpy import zeros
from torch import Tensor
from tempfile import mkdtemp
from shutil import rmtree
from models import LitUnet1D
from torch_optimizer import Optimizer
import torch
import numpy as np
import os.path as osp


class TestModels(TestCase):

    def setUp(self) -> None:
        self.in_ch = 47
        self.out_ch = 6
        self.height = 138

        # Creates a temporary environment.
        self.root = mkdtemp()
        self.create_env(self.root)

        self.initParam = {
            'data_path': self.root,
            'normalize': True,
            'in_channels': self.in_ch,
            'out_channels': self.out_ch,
            'n_levels': 3,
            'n_features_root': 32,
            'lr': .0001}

    def tearDown(self) -> None:
        rmtree(self.root)

    def create_env(self, root) -> None:
        """Build an environment with stats.pt."""
        torch.save({
            'x_mean': torch.tensor(np.random.rand(self.in_ch, self.height)),
            'x_std': torch.tensor(np.random.rand(self.in_ch, self.height)),
            'x_nb': torch.tensor(10),
            'y_mean': torch.tensor(np.random.rand(self.out_ch, self.height)),
            'y_std': torch.tensor(np.random.rand(self.out_ch, self.height)),
            'y_nb': torch.tensor(10)
        }, osp.join(root, "stats.pt"))

    def test_forward_common_step(self):
        # Fake data, of dim (n_batchs, height, n_channels).
        x = torch.from_numpy(zeros((1, self.height, self.in_ch)))
        y = torch.from_numpy(zeros((1, self.height, self.out_ch)))

        # Forward.
        test_unet = LitUnet1D(**self.initParam)
        y = test_unet.forward(x)
        self.assertTrue(isinstance(y, Tensor))
        self.assertEqual(y.shape, (1, self.out_ch, self.height))

        # Common step.
        loss = test_unet._common_step(batch=(x, y), stage="train")
        self.assertEqual(len(loss), 2)

        # Common step with normalization.
        loss = test_unet._common_step(batch=(x, y), stage="train", normalize=True)
        self.assertEqual(len(loss), 2)

    def test_configure_optimizers(self):
        test_unet = LitUnet1D(**self.initParam)
        op = test_unet.configure_optimizers()
        self.assertTrue(isinstance(op, Optimizer))


if __name__ == '__main__':
    main()
