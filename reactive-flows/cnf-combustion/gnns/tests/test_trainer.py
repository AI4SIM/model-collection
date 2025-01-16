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

import os
import unittest
import warnings

import torch

from trainer import CLITrainer


class TestTrainer(unittest.TestCase):
    """Trainer test suite."""

    def setUp(self) -> None:
        """Define default parameters."""
        self.args_cpu = {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "experiment_dir": "./experiments",
        }
        self.args_gpu = {
            "max_epochs": 1,
            "accelerator": "gpu",
            "devices": [0],
            "experiment_dir": "./experiments",
        }

    def test_trainer(self) -> None:
        """Test trainer file."""
        if torch.cuda.is_available():
            test_trainer_gpu = CLITrainer(**self.args_gpu)
            self.assertEqual(test_trainer_gpu._devices, [0])

        # avoids GPU warning when testing CPU usage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_trainer_cpu = CLITrainer(**self.args_cpu)
            self.assertEqual(test_trainer_cpu._devices, 1)


class TestTrainerPaths(unittest.TestCase):
    """Trainer test suite for paths."""

    def setUp(self) -> None:
        """Define default parameters."""
        args = {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "experiment_dir": "./experiments",
        }
        self.test_trainer = CLITrainer(**args)

    def test_experiment_dir(self):
        """Test if trainer creates the correct experiment dir."""
        self.assertTrue(os.path.exists(self.test_trainer.experiment_dir))

    def test_logs_path(self):
        """Test if trainer creates the correct logs path."""
        self.assertTrue(os.path.exists(self.test_trainer.log_dir))

    def test_artifacts_path(self):
        """Test if trainer creates the correct artifacts path."""
        self.assertTrue(os.path.exists(self.test_trainer.artifacts_path))

    def test_plots_path(self):
        """Test if trainer creates the correct plots path."""
        self.assertTrue(os.path.exists(self.test_trainer.plots_path))


if __name__ == "__main__":
    unittest.main()
