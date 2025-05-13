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

from data import check_era5_constant_masks


class TestData(TestCase):

    def setUp(self) -> None:
        self.correct_era5_mask_list = [
            "geopotential_at_surface",
            "soil_type",
            "high_vegetation_cover",
        ]
        self.wrong_era5_mask_list = ["topography", "soil_type", "high_vegetation_cover"]
        self.curated_era5_mask_list = ["soil_type", "high_vegetation_cover"]

    def test_check_era5_constant_masks(self):
        self.assertEqual(
            check_era5_constant_masks(self.correct_era5_mask_list),
            self.correct_era5_mask_list,
        )
        self.assertEqual(
            check_era5_constant_masks(self.wrong_era5_mask_list),
            self.curated_era5_mask_list,
        )


if __name__ == "main":
    unittest.main()
