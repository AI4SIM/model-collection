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
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from typing import List, Union

import config
import data  # noqa: F401 'data' imported but unused
import models  # noqa: F401 'data' imported but unused


class Trainer(pl.Trainer):
    """
    Modified PyTorch Lightning Trainer that automatically tests,
    logs, and writes artifacts by the end of training.
    """

    def __init__(self,
                 accelerator: Union[str, pl.accelerators.Accelerator, None],
                 devices: Union[List[int], str, int, None],
                 max_epochs: int,
                 fast_dev_run: Union[int, bool] = False,
                 callbacks: Union[List[Callback], Callback, None] = None) -> None:
        """
        Args:
            accelerator (Union[str, Accelerator, None]): Type of accelerator to use for training.
            devices: (Union[List[int], str, int, None]): Devices to use for training.
            max_epochs (int): Maximum number of epochs if no early stopping logic is implemented.
        """
        self._devices = devices
        if accelerator == 'cpu':
            self._devices = None
        logger = TensorBoardLogger(config.logs_path, name=None)

        super().__init__(
            default_root_dir=config.logs_path,
            logger=logger,
            accelerator=accelerator,
            devices=self._devices,
            max_epochs=max_epochs,
            # for some reason, a forward pass happens in the model before datamodule creation.
            # TODO: learn normalizers (mean, std) in a layer
            num_sanity_val_steps=0)

    def test(self, **kwargs) -> None:
        """
        Uses superclass test results, saves raw results as a JSON file,
        stores the model weights for future use in inference mode.
        """
        results = super().test(**kwargs)[0]
        with open(os.path.join(config.artifacts_path, "results.json"), "w") as f:
            json.dump(results, f)
        torch.save(self.model.net, os.path.join(config.artifacts_path, 'model.pth'))


def main():
    cli = LightningCLI(trainer_class=Trainer)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    main()
