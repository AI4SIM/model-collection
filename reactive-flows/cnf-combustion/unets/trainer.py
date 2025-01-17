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
from json import dump
from os.path import join
from typing import Iterable, List, Optional, Union

from lightning import Trainer
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import Logger
from torch import save

import data  # noqa: F401 'data' imported but unused
import models  # noqa: F401 'data' imported but unused


class CLITrainer(Trainer):

    def __init__(
        self,
        accelerator: Union[str, Accelerator, None],
        devices: Union[List[int], str, int, None],
        max_epochs: int,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        fast_dev_run: Union[int, bool] = False,
        callbacks: Union[List[Callback], Callback, None] = None,
    ) -> None:
        """
        Run a training session.
        Args:
            max_epochs (int): Maximum number of epochs if no early stopping logic is implemented.
        """
        self._devices = devices

        super().__init__(
            logger=logger,
            accelerator=accelerator,
            devices=self._devices,
            max_epochs=max_epochs,
        )
        self.artifacts_path = os.path.join(self.log_dir, "artifacts")
        self.plots_path = os.path.join(self.log_dir, "plots")
        if self.is_global_zero:
            os.makedirs(self.artifacts_path, exist_ok=True)
            os.makedirs(self.plots_path, exist_ok=True)

    def test(self, **kwargs):
        """
        Use superclass test results and save raw results as a JSON file.
        Store the model weights for future use in inference mode.
        """
        results = super().test(**kwargs)[0]
        if self.is_global_zero:
            with open(join(self.artifacts_path, "results.json"), "w") as f:
                dump(results, f)
            save(self.model, join(self.artifacts_path, "model.pth"))


def main():
    cli = LightningCLI(trainer_class=CLITrainer, run=False)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
