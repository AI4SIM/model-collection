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
from os.path import dirname, join, realpath

from randomname import get_name

root_path = dirname(realpath(__file__))

data_path = join(root_path, 'data')

# Create all path for the current experiment
experiments_path = os.path.join(root_path, 'experiments')
os.makedirs(experiments_path, exist_ok=True)
_existing_xps = os.listdir(experiments_path)

# Generate experiment name
_randomize_name = True
while _randomize_name:
    _experiment_name = get_name()
    if _experiment_name not in _existing_xps:
        break

experiment_path = os.path.join(experiments_path, _experiment_name)

if os.getenv("AI4SIM_EXPERIMENT_PATH") is None:
    os.environ["AI4SIM_EXPERIMENT_PATH"] = experiment_path
else:
    experiment_path = os.getenv("AI4SIM_EXPERIMENT_PATH")

logs_path = join(experiment_path, 'logs')
artifacts_path = join(experiment_path, 'artifacts')
plots_path = join(experiment_path, 'plots')

for path in [experiment_path, logs_path, artifacts_path, plots_path]:
    os.makedirs(path, exist_ok=True)
