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
import shutil
from names import get_last_name
from os.path import dirname, join, realpath

name = get_last_name().lower()

root_path = dirname(realpath(__file__))

data_path = join(root_path, 'data')
experiments_path = join(root_path, 'experiments')
experiment_path = join(experiments_path, name)
logs_path = join(experiment_path, 'logs')
artifacts_path = join(experiment_path, 'artifacts')
plots_path = join(experiment_path, 'plots')

for path in [experiment_path, logs_path, artifacts_path, plots_path]:
    os.makedirs(path, exist_ok=True)

if os.getenv("AI4SIM_EXPERIMENT_PATH") is None:
    os.environ["AI4SIM_EXPERIMENT_PATH"] = experiment_path
    os.environ["AI4SIM_LOGS_PATH"] = logs_path
    os.environ["AI4SIM_ARTIFACTS_PATH"] = artifacts_path
    os.environ["AI4SIM_PLOTS_PATH"] = plots_path
elif os.getenv("AI4SIM_EXPERIMENT_PATH") != experiment_path:
    shutil.rmtree(experiment_path)
    experiment_path = os.getenv("AI4SIM_EXPERIMENT_PATH")
    logs_path = os.getenv("AI4SIM_LOGS_PATH")
    artifacts_path = os.getenv("AI4SIM_ARTIFACTS_PATH")
    plots_path = os.getenv("AI4SIM_PLOTS_PATH")