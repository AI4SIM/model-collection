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
from pytorch_lightning.utilities.cli import LightningCLI
import torch

import data
import models


class Trainer(pl.Trainer):

    def __init__(self, accelerator, devices, max_epochs, fast_dev_run=False, callbacks=None):
        logger = pl.loggers.TensorBoardLogger(cfg.logs_path, name=None)
        if accelerator == 'cpu': devices = None

        super().__init__(
            default_root_dir=cfg.logs_path,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,)

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
