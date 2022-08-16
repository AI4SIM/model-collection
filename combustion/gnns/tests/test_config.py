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

import h5py
import os
import tempfile
import unittest
import yaml
import warnings
import numpy as np

import config


class TestConfig(unittest.TestCase):
    """Config test file."""

    def test_experiment_path(self):
        """Test if config creates the correct experiment path."""
        self.assertTrue(os.path.exists(config.experiment_path))
        self.assertTrue(os.getenv("AI4SIM_EXPERIMENT_PATH"), config.experiment_path)

    def test_logs_path(self):
        """Test if config creates the correct logs path."""
        self.assertTrue(os.path.exists(config.logs_path))
        self.assertTrue(os.getenv("AI4SIM_LOGS_PATH"), config.logs_path)
        self.assertTrue(
            os.path.exists(
                os.path.join(config.logs_path, f'{config.experiment_path.split("/")[-1]}.log')
            )
        )

    def test_artifacts_path(self):
        """Test if config creates the correct artifacts path."""
        self.assertTrue(os.path.exists(config.artifacts_path))
        self.assertTrue(os.getenv("AI4SIM_ARTIFACTS_PATH"), config.artifacts_path)

    def test_plots_path(self):
        """Test if config creates the correct plots path."""
        self.assertTrue(os.path.exists(config.plots_path))
        self.assertTrue(os.getenv("AI4SIM_PLOTS_PATH"), config.plots_path)

    def create_env(self, tempdir):
        """Create a test environment and data test."""
        os.mkdir(os.path.join(tempdir, "raw_data"))
        os.mkdir(os.path.join(tempdir, "local_data"))
        os.mkdir(os.path.join(tempdir, "local_data", "raw"))
        os.mkdir(os.path.join(tempdir, "local_data", "processed"))

        self.filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']
        for file_h5 in self.filenames:
            with h5py.File(os.path.join(tempdir, "raw_data", file_h5), 'w') as file:
                file['filt_8'] = np.zeros((10, 10, 10))
                file['filt_grad_8'] = np.zeros((10, 10, 10))
                file['grad_filt_8'] = np.zeros((10, 10, 10))

        temp_file_path = os.path.join(tempdir, 'local_data', 'filenames.yaml')
        with open(temp_file_path, 'w') as tmpfile:
            _ = yaml.dump(self.filenames, tmpfile)

    def create_obj_rm_warning(self, raw_path, local_path):
        """Instantiante the CombustionDataset object with a warning filtering."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return config.LinkRawData(raw_path, local_path)

    def test_linkrawdata(self):
        """Test the LinkRawData class."""
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)

            self.create_obj_rm_warning(
                os.path.join(tempdir, "raw_data"),
                os.path.join(tempdir, "local_data")
            )
            num_files_raw_path = len(os.listdir(os.path.join(tempdir, "raw_data")))
            local_filenames = os.listdir(os.path.join(tempdir, "local_data", "raw"))
            num_files_local_path = len(local_filenames)

            self.assertTrue(num_files_raw_path, num_files_local_path)
            self.assertTrue(self.filenames, local_filenames)


if __name__ == '__main__':
    unittest.main()
