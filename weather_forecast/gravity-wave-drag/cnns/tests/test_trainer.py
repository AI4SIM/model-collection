"""This module provides a unit tests suite for the trainer.py module."""
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

import os
import unittest
from unittest import skipIf
from unittest.mock import patch
import torch

import config
from trainer import Trainer


@skipIf(not torch.cuda.is_available(), "Cuda not available.")
class TestTrainerGpu(unittest.TestCase):
    """Test the Trainer class with GPUs."""

    def setUp(self) -> None:
        """Prepare config file parameters-like settings."""
        args_gpu = {"max_epochs": 1,
                    "accelerator": "gpu",
                    "devices": [0]}
        self.test_trainer_gpu = Trainer(**args_gpu)

    def test_trainer_gpu_init(self):
        """Tests the 'devices' mother attribute is properly set if gpu mode is activated."""
        self.assertEqual(self.test_trainer_gpu.devices, [0])

    @patch('pytorch_lightning.Trainer.test')
    def test_trainer_cpu_test(self, mock_test):
        """Tests the 'test' method properly save the result file and model, running on GPUs."""
        # patch the super().test() returned value
        mock_test.return_value = [{'a': 1, 'b': 2}]
        self.test_trainer_gpu.test()
        self.assertTrue(os.path.exists(os.path.join(config.artifacts_path, 'model.pth')))
        self.assertTrue(os.path.exists(os.path.join(config.artifacts_path, 'results.json')))


class TestTrainerCPU(unittest.TestCase):
    """Test the Trainer class with CPUs."""

    def setUp(self) -> None:
        """Prepare config file parameters-like settings."""
        args_cpu = {"max_epochs": 1,
                    "accelerator": "cpu",
                    "devices": [0]}
        self.test_trainer_cpu = Trainer(**args_cpu)

    def test_trainer_cpu_init(self):
        """Tests the 'devices' mother attribute is set to the default value (1) if the cpu mode is
        activated.
        """
        self.assertEqual(self.test_trainer_cpu.devices, 1)

    @patch('pytorch_lightning.Trainer.test')
    def test_trainer_cpu_test(self, mock_test):
        """Tests the 'test' method properly save the result file and model, running on CPUs."""
        # patch the super().test() returned value
        mock_test.return_value = [{'a': 1, 'b': 2}]
        self.test_trainer_cpu.test()
        self.assertTrue(os.path.exists(os.path.join(config.artifacts_path, 'model.pth')))
        self.assertTrue(os.path.exists(os.path.join(config.artifacts_path, 'results.json')))


if __name__ == '__main__':
    unittest.main()
