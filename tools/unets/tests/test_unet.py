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
from numpy.random import rand
from torch import bilinear, from_numpy
from unet import UNet


class TestUnet(TestCase):
    """Testing U-nets."""

    def n_conv(self, n_levels: int, bilinear: bool = False):
        # 2 per DoubleConv + 1 per upsampler.
        return 4 * n_levels + (n_levels - 1 if bilinear else 0)

    def test_3d(self):

        n_levels = 1
        bilinear = True
        net = UNet(
            dim=3,
            inp_ch=1,
            out_ch=1,
            n_levels=n_levels,
            n_features_root=4,
            bilinear=bilinear)

        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2 * n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels - 1)
        self.assertEqual(summary.count("Conv3d"), self.n_conv(n_levels, bilinear))

        n_levels = 5
        bilinear = True
        net = UNet(
            dim=3,
            inp_ch=1,
            out_ch=1,
            n_levels=n_levels,
            n_features_root=4,
            bilinear=bilinear)

        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2 * n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels - 1)
        self.assertEqual(summary.count("Conv3d"), self.n_conv(n_levels, bilinear))

    def test_inference_3d(self):
        net = UNet(dim=3, inp_ch=1, out_ch=1, n_levels=3, n_features_root=4)
        n = 32
        inp = from_numpy(rand(1, 1, n, n, n))
        shp = net(inp).shape
        self.assertEqual(shp, (1, 1, n, n, n))

    def test_1d(self):

        n_levels = 1
        net = UNet(dim=1, inp_ch=1, out_ch=1, n_levels=n_levels)

        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2 * n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels - 1)
        self.assertEqual(summary.count("Conv1d"), self.n_conv(n_levels))

        n_levels = 6
        net = UNet(dim=1, inp_ch=1, out_ch=1, n_levels=n_levels)

        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2 * n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels - 1)
        self.assertEqual(summary.count("Conv1d"), self.n_conv(n_levels))

    def test_inference_1d(self):
        net = UNet(dim=1, inp_ch=1, out_ch=1, n_levels=3)
        inp = from_numpy(rand(1, 1, 16))
        shp = net(inp).shape
        self.assertEqual(shp, (1, 1, 16))


if __name__ == '__main__':
    main()
