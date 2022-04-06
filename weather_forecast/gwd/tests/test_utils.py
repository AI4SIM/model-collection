"""This module provides utils functions to be used in test modules."""
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
from pathlib import Path
import numpy as np
import h5py
import yaml


def get_filenames(filenames_file):
    """Extract the list of file that are required for test, from the input file.

    Args:
        filenames_file (str): the path of filenames file

    Returns:
        filenames (list): the filenames extracted from the input file.
    """
    with open(filenames_file, 'r') as file:
        yaml_content = yaml.safe_load(file)
        filenames = []
        for files in yaml_content.values():
            filenames.extend(files)
    return filenames


def populate_test_data(root_data_path, filenames) -> None:
    """Create fake random data file.

    Args:
        root_data_path (str): the path of the 'data' folder.
        filenames (list): the list of filenames to be created.
    """
    data_path = os.path.join(root_data_path, "raw")
    Path(data_path).mkdir(parents=True, exist_ok=True)

    for file_h5 in filenames:
        with h5py.File(os.path.join(data_path, file_h5), 'w') as file:
            file['/x'] = np.random.rand(191, 36, 10)
            file['/y'] = np.random.rand(126, 36, 10)
