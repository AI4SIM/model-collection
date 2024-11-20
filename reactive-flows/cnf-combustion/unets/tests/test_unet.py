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
from torch import from_numpy
from unet import UNet3D, Downsampler, Upsampler


class TestUnet3D(TestCase):
    """Testing 3D U-nets."""

    def n_conv(self, n_levels: int, bilinear: bool = False):
        # 2 per DoubleConv + 1 per upsampler.
        return 4 * n_levels + (n_levels - 1 if bilinear else 0)

    def test_3d(self):

        n_levels = 1
        bilinear = True
        net = UNet3D(
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
        net = UNet3D(
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
        net = UNet3D(inp_ch=1, out_ch=1, n_levels=3, n_features_root=4)
        n = 32
        inp = from_numpy(rand(1, 1, n, n, n))
        shp = tuple(net(inp).shape)
        self.assertEqual(shp, (1, 1, n, n, n))

    def test_downsampler(self):
        sampler = Downsampler(inp_ch=4, out_ch=8).double()
        inp = from_numpy(rand(1, 4, 16, 16, 16))
        shp = tuple(sampler(inp).shape)
        self.assertEqual(shp, (1, 8, 8, 8, 8))

    def test_upsampler(self):
        sampler = Upsampler(inp_ch=8, out_ch=4).double()
        inp = from_numpy(rand(1, 8, 16, 16, 16))
        res = from_numpy(rand(1, 4, 32, 32, 32))
        shp = tuple(sampler(inp, res).shape)
        self.assertEqual(shp, (1, 4, 32, 32, 32))  # last DoubleConv enforces the out_ch.


if __name__ == '__main__':
    main()
