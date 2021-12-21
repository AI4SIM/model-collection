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
        
        if accelerator == 'cpu':
            devices = None
        
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