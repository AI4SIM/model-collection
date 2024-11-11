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
from jsonargparse.typing import register_type
from torch import set_float32_matmul_precision
from utils import slice_deserializer


def cli_main():
    LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    set_float32_matmul_precision("high")
    register_type(range, str, slice_deserializer)
    cli_main()
