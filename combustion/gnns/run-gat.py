import argparse
import config
import json
import mlflow
import os
import pytorch_lightning as pl
import torch
import torch_geometric as pyg

from data import LitCombustionDataModule
from shell import LitCombustionModule

# this script contains the running logic


def main():
    
    datamodule = LitCombustionDataModule(args)
    
    # TODO define features shape and target shape in a flat config file
    if args.load_model:
        model = torch.load(os.path.join(config.experiments_path, args.load_model, 'artifacts', 'model.pth'))
    else:
        model = pyg.nn.GAT(in_channels=1, 
                           hidden_channels=args.hidden_channels,
                           num_layers=args.num_layers,
                           out_channels=1,
                           dropout=args.dropout,
                           act=torch.nn.SiLU(inplace=True),
                           heads=args.heads,
                           jk=args.jk)
        
    print(f'Training model: {config.name}')

    module = LitCombustionModule(model, datamodule.grid_shape, args)
    
    callbacks = []
    if args.early_stopping:
        callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min'))

    logger = pl.loggers.TensorBoardLogger(config.logs_path, name=None, log_graph=True)
    logger.log_hyperparams({
        'hidden_channels': args.hidden_channels,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    })
    trainer = pl.Trainer(
        default_root_dir=config.logs_path,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=callbacks)
    # mlflow.pytorch.autolog()
    
    # with mlflow.start_run(run_name=name) as run:
    trainer.fit(module, datamodule=datamodule)
    results = trainer.test()[0]
    
    with open(os.path.join(config.artifacts_path, "results.json"), "w") as f:
        json.dump(results, f)
    
    torch.save(module.model, os.path.join(config.artifacts_path, 'model.pth'))

    
    
if __name__ == '__main__':
    
    main()