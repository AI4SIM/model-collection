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
import numpy as np
import torch
import unet


class TestUnet(unittest.TestCase):
    """
    Testing 3D isotropic U-Nets.
    """

    def test_architecture(self):
        nb_Conv3D = lambda n_levels: 2*2*n_levels + (n_levels-1)  # 2 per DoubleConv + 1 per upsampler.

        n_levels = 1
        net = unet.UNet3D(inp_feat=1, out_feat=1, n_levels=n_levels, n_features_root=4, bilinear=True)
        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2*n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels-1)
        self.assertEqual(summary.count("Conv3d"), nb_Conv3D(n_levels))

        n_levels = 5
        net = unet.UNet3D(inp_feat=1, out_feat=1, n_levels=n_levels, n_features_root=4, bilinear=True)
        summary = str(net)
        self.assertEqual(summary.count("DoubleConv"), 2*n_levels)
        self.assertEqual(summary.count("Upsampler"), n_levels-1)
        self.assertEqual(summary.count("Conv3d"), nb_Conv3D(n_levels))

    def test_inference(self):
        net = unet.UNet3D(inp_feat=1, out_feat=1, n_levels=3, n_features_root=4)
        n = 32
        inp = torch.from_numpy(np.random.rand(1,1,n,n,n))
        shp = net(inp).shape
        self.assertEqual(shp, (1,1,n,n,n))


if __name__ == '__main__':
    unittest.main()
