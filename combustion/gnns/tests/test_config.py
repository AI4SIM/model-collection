"""This module provides a test suite for the config.py file."""
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
import os

import config


class TestConfig(unittest.TestCase):
    """Config test file."""

    def test_experiment_path(self):
        """Test if config creates the correct experiment path."""
        self.assertTrue(os.path.exists(config.experiment_path))

    def test_logs_path(self):
        """Test if config creates the correct logs path."""
        self.assertTrue(os.path.exists(config.logs_path))

    def test_artifacts_path(self):
        """Test if config creates the correct artifacts path."""
        self.assertTrue(os.path.exists(config.artifacts_path))

    def test_plots_path(self):
        """Test if config creates the correct plots path."""
        self.assertTrue(os.path.exists(config.plots_path))


if __name__ == '__main__':
    unittest.main()
