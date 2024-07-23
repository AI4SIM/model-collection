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
from torch import Tensor, from_numpy
from models import LitUnet3D
from torch_optimizer import Optimizer


class TestModels(TestCase):

    def setUp(self) -> None:
        self.initParam = {
            'in_channels': 1,
            'out_channels': 2,
            'n_levels': 2,
            'n_features_root': 32,
            'lr': .0001}

    def test_forward_common_step(self):
        # Fake data, of dim (n_batchs, n_channels, x, y, z).
        x = from_numpy(zeros((1, self.initParam['in_channels'], 10, 10, 10)))
        y = from_numpy(zeros((1, self.initParam['in_channels'], 10, 10, 10)))

        # Forward.
        test_unet = LitUnet3D(**self.initParam)
        y = test_unet.forward(x)
        self.assertTrue(isinstance(y, Tensor))
        self.assertEqual(y.shape, (1, self.initParam['out_channels'], 10, 10, 10))

        # Common step.
        loss = test_unet._common_step(batch=(x, y), stage="train")
        self.assertEqual(len(loss), 3)

    def test_configure_optimizers(self):
        test_unet = LitUnet3D(**self.initParam)
        op = test_unet.configure_optimizers()
        self.assertTrue(isinstance(op, Optimizer))


if __name__ == '__main__':
    main()
