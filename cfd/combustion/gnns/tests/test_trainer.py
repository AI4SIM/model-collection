"""This module proposes a test suite for the trainer module."""
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
import warnings
import torch

from trainer import Trainer


class TestTrainer(unittest.TestCase):
    """Trainer test suite."""

    def setUp(self) -> None:
        """Define default parameters."""
        self.args_cpu = {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": [0]
        }
        self.args_gpu = {
            "max_epochs": 1,
            "accelerator": "gpu",
            "devices": [0]
        }

    def test_trainer(self) -> None:
        """Test trainer file."""
        if torch.cuda.is_available():
            _ = Trainer(**self.args_gpu)

        # avoids GPU warning when testing CPU usage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            test_trainer_cpu = Trainer(**self.args_cpu)
            self.assertEqual(test_trainer_cpu._devices, None)


if __name__ == '__main__':
    unittest.main()
