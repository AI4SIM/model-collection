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
import randomname

root_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(root_path, 'data')

# Create all path for the current experiment
experiments_path = os.path.join(root_path, 'experiments')
existing_xps = os.listdir(experiments_path)

while True:
    name = randomname.get_name()

    if name not in existing_xps:
        break

experiment_path = os.path.join(experiments_path, name)
logs_path = os.path.join(experiment_path, 'logs')
artifacts_path = os.path.join(experiment_path, 'artifacts')
plots_path = os.path.join(experiment_path, 'plots')

paths = [experiment_path, logs_path, artifacts_path, plots_path]
for path in paths:
    os.makedirs(path, exist_ok=True)