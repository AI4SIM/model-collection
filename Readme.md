# Model Collection Philosophy

**Important note: at the moment, lots of code is purely being duplicated, awaiting for more UC to really abstract mutualizable data processing logic.**

This project contains a collection of models developed by the Atos AI4sim R&D team and is intended for research purposes.

The current workflow is based entirely on NumPy, PyTorch, PyG and Lightning. 

To take care of the boiler-plate, early stopping, and tensorboard logging, we integrate directly with PyTorch Lightning. For each new UC, we use the same file tree, namely:

* `_/` contains the sandbox environment, including notebooks and dev stuff.
* `configs/` contains the experiments configuration files. An experiment designates a specific training run with a specific model and a specific dataset + split.
* `data/` contains the raw and processed data, including normalization factors and explicit train / val / test split files.
* `experiments/` contains the artifacts, plots, and logs for all runs.
* `config.py` exposes global paths and path specific to current experiment.
* `data.py` deals with dataset and datamodule creation.
* `models.py` deals with model and module creation, including training logic. Declare models to the Lightning Model Registry. Users should only expose arguments that they wish to perform HPO on in their model constructor.
* `plotters.py` takes care of plots generation for the test set.
* `trainer.py` is the main entrypoint responsible for creating a Trainer object, a CLI, and saving artifacts in the experiment directory.