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

import torch

from utils import (
    cat_constant_masks,
    define_3d_earth_position_index,
    define_3d_relative_position_index,
    generate_2d_swin_masks,
    generate_modified_2d_swin_masks,
    slice_deserializer,
)


class TestUtils(TestCase):

    def setUp(self) -> None:
        self.mock_surface_data = torch.rand(2, 4, 721, 1440)
        self.mock_constant_masks = torch.rand(2, 721, 1440)
        self.slice1 = slice("2019-01-01", "2020-03-25")
        self.slice1_str1 = "slice('2019-01-01', '2020-03-25')"
        self.slice1_str2 = "slice('2019-01-01', '2020-03-25', None)"
        self.slice2 = slice("2019-01", "2020-03", "12h")

    # def test_perlin_noise(self):
    #     perlin_noise()

    def test_cat_constant_masks(self) -> None:
        cat_data = cat_constant_masks(self.mock_surface_data, self.mock_constant_masks)
        self.assertEqual(cat_data.shape, torch.Size([2, 6, 721, 1440]))

    def test_2d_attention_masks(self) -> None:
        pad_h = 360
        pad_w = 180
        window_size = (12, 6)
        shift_size = (6, 3)
        num_windows = (pad_h // window_size[0]) * (pad_w // window_size[1])
        self.assertTrue(
            torch.all(
                torch.eq(
                    generate_2d_swin_masks(
                        pad_h, pad_w, window_size, shift_size, num_windows
                    ),
                    generate_modified_2d_swin_masks(
                        pad_h, pad_w, window_size, shift_size, num_windows
                    ),
                )
            )
        )

    def test_define_3d_relative_position_index(self) -> None:
        window_size = (2, 6, 12)
        positional_bias_length = (
            (2 * window_size[0] - 1)
            * (2 * window_size[1] - 1)
            * (2 * window_size[2] - 1)
        )
        index = define_3d_relative_position_index(window_size)
        self.assertEqual(positional_bias_length, len(index.unique()))
        self.assertTrue(
            torch.all(torch.arange(positional_bias_length) == index.unique())
        )
        self.assertEqual(
            (window_size[0] * window_size[1] * window_size[2]) ** 2, len(index)
        )

    def test_define_3d_earth_position_index(self) -> None:
        window_size = (2, 6, 12)
        positional_bias_length = (
            window_size[0] ** 2 * window_size[1] ** 2 * (2 * window_size[2] - 1)
        )
        index = define_3d_earth_position_index(window_size)
        self.assertEqual(positional_bias_length, len(index.unique()))
        self.assertTrue(
            torch.all(torch.arange(positional_bias_length) == index.unique())
        )
        self.assertEqual(
            (window_size[0] * window_size[1] * window_size[2]) ** 2, len(index)
        )

    def test_slice_deserializer(self) -> None:
        self.assertEqual(self.slice1, slice_deserializer(self.slice1_str1))
        self.assertEqual(self.slice1, slice_deserializer(self.slice1_str2))
        self.assertEqual(self.slice2, slice_deserializer(str(self.slice2)))


if __name__ == "main":
    unittest.main()
