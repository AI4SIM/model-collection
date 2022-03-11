"""This module provides a unit tests suite for the config.py module."""

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
