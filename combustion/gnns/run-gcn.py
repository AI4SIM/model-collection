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
    
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--name', type=str, default=config.name, help='name of the experiment')
    
    parser.add_argument('--load-model', type=str, default=None, help='name of the model to load')
    
    parser.add_argument('--hidden-channels', type=int, default=32, help='hidden channels for the GCN')
    parser.add_argument('--num-layers', type=int, default=8, help='number of layers in the GCN')
    parser.add_argument('--dropout', type=float, default=.3, help='training dropout ratio')
    
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--max-epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=.001, help='learning rate')
    parser.add_argument('--early-stopping', type=bool, default=False, help='early stopping callback')
    
    parser.add_argument('--accelerator', type=str, default="cpu",help='type of hardware accelerator amongst: cpu, gpu, tpu, ipu')
    parser.add_argument('--devices', type=int, nargs='+', default=None, help='list of devices to use for acceleration')
    
    parser.add_argument('--seed', type=int, default=42, help='seed for random number generator')
    # parser.add_argument('--root-path', type=str, default=config.root_path, help=f'root path from where every path is derived')
    args = parser.parse_args()
    
    datamodule = LitCombustionDataModule(args)
    
    # TODO define features shape and target shape in a flat config file
    if args.load_model:
        model = torch.load(os.path.join(config.experiments_path, args.load_model, 'artifacts', 'model.pth'))
    else:
        model = pyg.nn.GCN(in_channels=1, 
                           hidden_channels=args.hidden_channels,
                           num_layers=args.num_layers,
                           out_channels=1,
                           dropout=args.dropout,
                           act=torch.nn.SiLU(inplace=True))
        
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