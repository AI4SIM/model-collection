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

from lightning.pytorch.cli import LightningCLI
from torch import set_float32_matmul_precision
import torch


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.batch_size", "model.init_args.batch_size", apply_on="instantiate"
        )


def cli_main():
    MyLightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    set_float32_matmul_precision("high")
    cli_main()
