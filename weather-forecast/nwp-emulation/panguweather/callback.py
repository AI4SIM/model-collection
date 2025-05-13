# Copyright 2024 Eviden.
#
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

import tempfile
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.cli import SaveConfigCallback

from transformer_module import PanguModel


class GlobalSaveConfigCallback(SaveConfigCallback):
    """Work around to save config file in MLFlow.
    See https://github.com/Lightning-AI/pytorch-lightning/issues/16310
    """

    def __init__(
        self,
        parser,
        config,
        config_filename="config.yaml",
        overwrite=False,
        multifile=False,
    ):
        super().__init__(
            parser, config, config_filename, overwrite, multifile, save_to_log_dir=True
        )

    def save_config(self, trainer: Trainer, pl_module: PanguModel, stage: str) -> None:
        if pl_module.mlflow_logger is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / self.config_filename
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                pl_module.mlflow_logger.experiment.log_artifact(
                    local_path=config_path, run_id=pl_module.mlflow_logger.run_id
                )
