"""This module proposes a Pytorch Lightning style Trainer class for the gnn use-case."""
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

import json
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from typing import List, Union

import config
import data  # noqa: F401 'data' imported but unused


class Trainer(pl.Trainer):
    """
    Modified PyTorch Lightning Trainer that automatically tests, logs, and writes artifacts by
    the end of training.
    """

    def __init__(self,
                 accelerator: Union[str, pl.accelerators.Accelerator, None],
                 devices: Union[List[int], str, int, None],
                 max_epochs: int,
                 # TODO: delete.
                 # For some reason, those two are mandatory in current version of Lightning.
                 fast_dev_run: Union[int, bool] = False,
                 callbacks: Union[List[Callback], Callback, None] = None) -> None:
        """Init the Trainer.

        Args:
            accelerator (Union[str, pl.accelerators.Accelerator, None]): Type of accelerator to use
                for training.
            devices: (Union[List[int], str, int, None]): Devices explicit names to use for training.
            max_epochs (int): Maximum number of epochs if no early stopping logic is implemented.
        """
        self._accelerator = accelerator
        self._devices = devices
        self._max_epochs = max_epochs

        if self._accelerator == 'cpu':
            self._devices = None

        logger = pl.loggers.TensorBoardLogger(config.logs_path, name=None)
        super().__init__(
            default_root_dir=config.logs_path,
            logger=logger,
            accelerator=self._accelerator,
            devices=self._devices,
            max_epochs=self._max_epochs,
            num_sanity_val_steps=0)

    def save(self, results, path=config.artifacts_path):
        """Save the results of the training and the learned model."""
        result_file = os.path.join(config.artifacts_path, "results.json")
        with open(result_file, "w") as f:
            json.dump(results, f)

        torch.save(self.model, os.path.join(path, 'model.pth'))
        logging.info(f"Torch model saved in {os.path.join(path, 'model.pth')}")

    def test(self, **kwargs) -> None:
        """Use superclass test results, but additionally, saves raw results as a JSON file,
        and stores the model weights for future use in inference mode.

        Returns:
            None
        """
        results = super().test(**kwargs)[0]

        self.save(results)


if __name__ == '__main__':
    cli = LightningCLI(trainer_class=Trainer, run=False)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
