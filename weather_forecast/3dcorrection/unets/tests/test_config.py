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
from unittest import TestCase, main
from os.path import exists

import config


class TestConfig(TestCase):

    def test_paths(self):
        self.assertTrue(exists(config.experiment_path))
        self.assertTrue(os.getenv("AI4SIM_EXPERIMENT_PATH"), config.experiment_path)

        self.assertTrue(exists(config.logs_path))
        self.assertTrue(os.getenv("AI4SIM_LOGS_PATH"), config.logs_path)

        self.assertTrue(exists(config.artifacts_path))
        self.assertTrue(os.getenv("AI4SIM_ARTIFACTS_PATH"), config.artifacts_path)

        self.assertTrue(exists(config.plots_path))
        self.assertTrue(os.getenv("AI4SIM_PLOTS_PATH"), config.plots_path)


if __name__ == '__main__':
    main()
