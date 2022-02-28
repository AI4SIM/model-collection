'''
    Licensed under the Apache License, Version 2.0 (the "License");
    * you may not use this file except in compliance with the License.
    * You may obtain a copy of the License at
    *
    *     http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing, software
    * distributed under the License is distributed on an "AS IS" BASIS,
    * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    * See the License for the specific language governing permissions and
    * limitations under the License.
'''

import config as cfg
import json
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from typing import List, Union

# Needed for CLI (for now).
import data
import models

class Trainer(pl.Trainer):

    def __init__(self,
                 accelerator: Union[str, pl.accelerators.Accelerator, None],
                 devices: Union[List[int], str, int, None],
                 max_epochs: int,
                 fast_dev_run: Union[int, bool] = False,
                 callbacks: Union[List[Callback], Callback, None] = None) -> None:
        """
        Args:
            max_epochs (int): Maximum number of epochs if no early stopping logic is implemented.
        """

        self._accelerator = accelerator
        self._devices     = devices
        self._max_epochs  = max_epochs
        if self._accelerator == 'cpu': self._devices = None

        logger = pl.loggers.TensorBoardLogger(cfg.logs_path, name=None)
        super().__init__(
            default_root_dir=cfg.logs_path,
            logger=logger,
            accelerator=self._accelerator,
            devices=self._devices,
            max_epochs=self._max_epochs,)

    def test(self, **kwargs):
        results = super().test(**kwargs)[0]
        with open(os.path.join(cfg.artifacts_path, "results.json"), "w") as f:
            json.dump(results, f)
        torch.save(self.model, os.path.join(cfg.artifacts_path, 'model.pth'))


def main():
    cli = LightningCLI(trainer_class=Trainer)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    main()
