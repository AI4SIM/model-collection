"""
Snippet for setting up tests
"""

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
