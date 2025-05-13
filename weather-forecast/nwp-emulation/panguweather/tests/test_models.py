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

import unittest
from unittest import TestCase

import numpy as np
import torch

from panguweather.pangu import (
    CustomPad2d,
    CustomPad3d,
    DownSample,
    PanguModel,
    PatchEmbedding,
    UpSample,
)


class TestPangu(TestCase):

    def setUp(self) -> None:
        self.mock_3d_data = torch.rand(2, 5, 13, 120, 240)
        self.patch_size_3d = (2, 8, 8)

        self.mock_2d_data = torch.rand(2, 7, 120, 240)
        self.patch_size_2d = (8, 8)

        self.mock_2d_data_no_mask = torch.rand(2, 4, 120, 240)
        self.mock_constant_mask = {
            "mask1": {"mask": np.random.rand(120, 240).astype(np.float32)},
            "mask2": {"mask": np.random.rand(120, 240).astype(np.float32)},
            "mask3": {"mask": np.random.rand(120, 240).astype(np.float32)},
        }

        self.mock_token = torch.rand(2, 8 * 181 * 360, 192)
        self.mock_token_upsample = torch.rand(2, 8 * 91 * 180, 384)

    def test_custom_pad_3d(self) -> None:
        pad = CustomPad3d(self.mock_3d_data.shape[-3:], self.patch_size_3d)
        padded_data = pad(self.mock_3d_data)
        self.assertEqual(padded_data.shape, torch.Size((2, 5, 14, 120, 240)))

        with self.assertRaises(AssertionError):
            CustomPad3d(self.mock_2d_data.shape[-2:], self.patch_size_3d)
        with self.assertRaises(AssertionError):
            CustomPad3d(self.mock_3d_data.shape[-3:], self.patch_size_2d)

    def test_custom_pad_2d(self) -> None:
        pad = CustomPad2d(self.mock_2d_data.shape[-2:], self.patch_size_2d)
        padded_data = pad(self.mock_2d_data)
        self.assertEqual(padded_data.shape, torch.Size((2, 7, 120, 240)))

        with self.assertRaises(AssertionError):
            CustomPad2d(self.mock_2d_data.shape[-2:], self.patch_size_3d)
        with self.assertRaises(AssertionError):
            CustomPad2d(self.mock_3d_data.shape[-3:], self.patch_size_2d)

    def test_patch_embedding(self) -> None:
        embed = PatchEmbedding(
            c_dim=32,
            patch_size=(2, 8, 8),
            plevel_size=(5, 13, 120, 240),
            surface_size=(7, 120, 240),
        )
        self.assertEqual(
            embed(self.mock_3d_data, self.mock_2d_data)[0].shape,
            torch.Size([2, 8 * 15 * 30, 32]),
        )

    def test_down_sample(self) -> None:
        down_sample = DownSample((8, 181, 360), 192)
        self.assertEqual(
            down_sample(self.mock_token, (2, 8, 181, 360, 192))[0].shape,
            torch.Size([2, 8 * 91 * 180, 384]),
        )

    def test_up_sample(self) -> None:
        up_sample = UpSample(384, 192)
        self.assertEqual(
            up_sample(self.mock_token_upsample, (2, 8, 181, 360, 384)).shape,
            torch.Size([2, 8 * 181 * 360, 192]),
        )

    def test_pangu(self) -> None:
        with torch.no_grad():
            model = PanguModel(
                latitude=120,
                longitude=240,
                surface_variables=4,
                plevel_variables=5,
                plevels=13,
                plevel_patch_size=(2, 8, 8),
            )
            out_plevel, out_surface = model(
                self.mock_3d_data, self.mock_2d_data_no_mask
            )
        self.assertEqual(out_plevel.shape, self.mock_3d_data.shape)
        self.assertEqual(out_surface.shape, self.mock_2d_data_no_mask.shape)
        with torch.no_grad():
            model = PanguModel(
                latitude=120,
                longitude=240,
                surface_variables=4,
                plevel_variables=5,
                plevels=13,
                plevel_patch_size=(2, 8, 8),
                constant_masks=self.mock_constant_mask,
            )
            out_plevel, out_surface = model(
                self.mock_3d_data, self.mock_2d_data_no_mask
            )
        self.assertEqual(out_plevel.shape, self.mock_3d_data.shape)
        self.assertEqual(out_surface.shape, self.mock_2d_data_no_mask.shape)


if __name__ == "main":
    unittest.main()
