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

from unittest import TestCase, main
from warnings import catch_warnings, simplefilter
from torch import cuda
from trainer import CLITrainer


class TestTrainer(TestCase):

    def setUp(self) -> None:
        self.args_cpu = {"max_epochs": 1,
                         "accelerator": "cpu",
                         "devices": 1}
        self.args_gpu = {"max_epochs": 1,
                         "accelerator": "gpu",
                         "devices": [0]}

    def test_trainer(self) -> None:
        if cuda.is_available():
            test_trainer_gpu = CLITrainer(**self.args_gpu)
            self.assertEqual(test_trainer_gpu._devices, 0)

        # Avoids GPU warning when testing CPU usage.
        with catch_warnings():
            simplefilter("ignore")
            test_trainer_cpu = CLITrainer(**self.args_cpu)
            self.assertEqual(test_trainer_cpu._devices, 1)


if __name__ == '__main__':
    main()
