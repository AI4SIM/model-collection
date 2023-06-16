"""This module configures the experiment environment."""
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
# import logging
import yaml
import randomname
# import shutil


# CAUTION : A refactoring of this file might be requiered for further development
# raw_data_path to be adapted to your local data path.
raw_data_path = "/path/to/your/local/data"
#

root_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(root_path, 'data')

# Create all path for the current experiment
experiments_path = os.path.join(root_path, 'experiments')
os.makedirs(experiments_path, exist_ok=True)
_existing_xps = os.listdir(experiments_path)

# Generate experiment name
_randomize_name = True
while _randomize_name:
    _experiment_name = randomname.get_name()
    if _experiment_name not in _existing_xps:
        break

experiment_path = os.path.join(experiments_path, _experiment_name)

if os.getenv("AI4SIM_EXPERIMENT_PATH") is None:
    os.environ["AI4SIM_EXPERIMENT_PATH"] = experiment_path
else:
    experiment_path = os.getenv("AI4SIM_EXPERIMENT_PATH")

logs_path = os.path.join(experiment_path, 'logs')
artifacts_path = os.path.join(experiment_path, 'artifacts')
plots_path = os.path.join(experiment_path, 'plots')

_paths = [
    experiment_path,
    logs_path,
    artifacts_path,
    plots_path
]
for path in _paths:
    os.makedirs(path, exist_ok=True)


class LinkRawData:
    """Link dataset to the use case."""

    def __init__(self, raw_data_path, data_path):
        """Link the raw_data_path to the data_path, if it does not already exists."""
        self.raw_data_path = raw_data_path
        self.local_data_path = data_path
        self.local_raw_data = os.path.join(self.local_data_path, 'raw')

        if os.path.exists(self.local_raw_data):
            try:
                if len(os.listdir(self.local_raw_data)) == 0 \
                        or os.readlink(self.local_raw_data) != self.raw_data_path:
                    self.rm_old_dataset()
                    self.symlink_dataset()
                else:
                    pass
            except OSError:
                pass
        else:
            self.symlink_dataset()

    def symlink_dataset(self):
        """Create the filenames.yaml file from the content of the raw_data_path."""
        filenames = os.listdir(self.raw_data_path)
        temp_file_path = os.path.join(self.local_data_path, 'filenames.yaml')
        with open(temp_file_path, 'w') as file:
            yaml.dump(filenames, file)

        if not os.path.exists(self.local_raw_data):
            os.makedirs(self.local_raw_data, exist_ok=True)

        for filename in filenames:
            os.symlink(
                os.path.join(self.raw_data_path, filename),
                os.path.join(self.local_raw_data, filename)
            )

    def rm_old_dataset(self):
        """Clean the local_data_path."""
        for item in ['raw', 'filenames.yaml', 'processed']:
            file_location = os.path.join(self.local_data_path, item)
            try:
                os.remove(file_location)
            except IsADirectoryError:
                os.rmdir(file_location)
            else:
                pass
