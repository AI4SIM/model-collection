"""
Test config file
"""

import unittest
import os

import config

class TestConfig(unittest.TestCase):
    """
    Config test file
    """

    def test_experiment_path(self):
        """
        Test if config creats the correct paths
        """
        self.assertTrue(os.path.exists(config.experiment_path))
        
    def test_logs_path(self):
        """
        Test if config creats the correct paths
        """
        self.assertTrue(os.path.exists(config.logs_path))
        
    def test_artifacts_path(self):
        """
        Test if config creats the correct paths
        """
        self.assertTrue(os.path.exists(config.artifacts_path))
        
    def test_plots_path(self):
        """
        Test if config creats the correct paths
        """
        self.assertTrue(os.path.exists(config.plots_path))



if __name__ == '__main__':
    unittest.main()
