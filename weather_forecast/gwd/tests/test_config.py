"""This module provides a unit tests suite for the config.py module."""
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

import config


class TestConfigPath(unittest.TestCase):
    """Test the config.py folders creation."""

    def test_experiment_path(self):
        """Test if config creates the correct experiment paths."""
        self.assertTrue(os.path.exists(config.experiment_path))

    def test_logs_path(self):
        """Test if config creates the correct log paths and file."""
        self.assertTrue(os.path.exists(config.logs_path))
        self.assertTrue(
            os.path.exists(os.path.join(config.logs_path, f'{config._experiment_name}.log')))

    def test_artifacts_path(self):
        """Test if config creates the correct artifacts paths."""
        self.assertTrue(os.path.exists(config.artifacts_path))

    def test_plots_path(self):
        """Test if config creats the correct paths."""
        self.assertTrue(os.path.exists(config.plots_path))


if __name__ == '__main__':
    unittest.main()
