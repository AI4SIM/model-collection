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
from unet import UNet


class TestUnet(TestCase):
    """Testing U-nets."""

    def n_conv(self, n_levels):
        # 2 per DoubleConv + 1 per upsampler.
        return 2 * 2 * n_levels + (n_levels - 1)

    def test_3d(self):

        n_levels = 1
        net = UNet(
            inp_feat=1,
            out_feat=1,
            dim=3,
            n_levels=n_levels,
            n_features_root=4,
            bilinear=True)

        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2 * n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels - 1)
        self.assertEqual(summary.count("Conv3d"), self.n_conv(n_levels))

        n_levels = 5
        net = UNet(
            inp_feat=1,
            out_feat=1,
            dim=3,
            n_levels=n_levels,
            n_features_root=4,
            bilinear=True)

        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2 * n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels - 1)
        self.assertEqual(summary.count("Conv3d"), self.n_conv(n_levels))

    def test_inference(self):
        net = UNet(
            inp_feat=1,
            out_feat=1,
            dim=3,
            n_levels=3,
            n_features_root=4)
        n = 32
        inp = from_numpy(rand(1, 1, n, n, n))
        shp = net(inp).shape
        self.assertEqual(shp, (1, 1, n, n, n))

    def test_1d(self):

        n_levels = 1
        net = UNet(
            inp_feat=1,
            out_feat=1,
            dim=1,
            n_levels=n_levels)

        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2 * n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels - 1)
        self.assertEqual(summary.count("Conv3d"), self.n_conv(n_levels))

        n_levels = 6
        net = UNet(
            inp_feat=1,
            out_feat=1,
            dim=3,
            n_levels=n_levels)

        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2 * n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels - 1)
        self.assertEqual(summary.count("Conv3d"), self.n_conv(n_levels))

    def test_inference(self):
        net = UNet(
            inp_feat=1,
            out_feat=1,
            dim=3,
            n_levels=3)
        n = 32
        inp = from_numpy(rand(1, 1, n))
        shp = net(inp).shape
        self.assertEqual(shp, (1, 1, n))


if __name__ == '__main__':
    main()
