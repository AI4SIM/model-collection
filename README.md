## Philosophy

This project contains a collection of models developed by the Atos AI4sim R&D team and is intended for research purposes. The current workflow is based entirely on NumPy, PyTorch, Dask and Lightning. Domain-specific librairies can be added, like PyTorch Geometric for graph neural networks.

**The repository is organized as a collection of independant model implementation for various use-cases (UC). You will thus find a lot of duplicated code, because the focus has been made on the projects content. Nevertheless the CI/CD and development tools have been mutualized.**

## Collection

The collection of models is developed through partnerships bringing their use-cases.

Currently, the models that have been developed are based on the following use-cases :

- Computational Fluid Dynamics
    - Combustion (with Unets and GNNs), from CERFACS

- Weather Forecast
    - Gravity Wave Drag (with CNNs), from ECMWF
    - 3D Bias Correction (with Unets), from ECMWF

## Projects organization

> In all of the following, an experiment designates a specific training run (hyperparameters) with a specific model (neural architecture) and a specific dataset and splitting strategy.

### Project path tree

All the models are placed in the repository with a path following the rule :

``<domain>``/``<use-case>``/``<NN architecture>``

For example, the code in the path ``cfd/combustion/gnns``, implements some **Graph Neural Network (GNN)** achitectures developed for the **Computational Fluid Dynamics (CFD)**, and applied to a **Combustion** use-case.

### Model project files

To take care of the boiler-plate, early stopping, tensorboard logging, training parallelization etc. we integrate directly with PyTorch Lightning. For each new  model, we use the same code architecture, described below.

All model's implementation should include the following folders and files:

* ``configs/`` contains the experiments configuration YAML files, following the Lightning CLI format.
* ``tests/`` contains unit tests modules.
* ``config.py`` exposes global paths and path specific to the current experiment.
* ``data.py`` contains the dataset and datamodule.
* ``models.py`` contains the model ("module"), including training logic. The architecture can be imported from specific librairies or a local module (e.g. unet.py).
* ``trainer.py`` is the main entrypoint responsible for creating a Trainer object, a CLI, and saving artifacts in the experiment directory.
* ``noxfile.py`` is the Nox build tool configuration file that defines all targets available for the model project.

Optionally it can also include the following folders and files :
* ``data/`` contains the dataset configuartaion files and will contain the `raw` and `processed` data directories, and normalization factors (and optionally explicit train/val/test split sets), once the dataset as been downloaded and processed.
* ``notebooks/`` contains example Jupyter notebooks, for pedagogical purposes.
* ``plotters.py`` takes care of plots generation for the test set.

### Projects architecture

Each model project is made of two pipelines:

* Data pipeline: a **DataModule** wraps a Dataset to provide data (with dataloaders, preprocessing...);
* Model pipeline: a **Module** compiles a neural network architecture with its optimizer.
The Trainer plugs both pipelines when called to run an experiment. It can be called by hand (CLI, with config files) or by another abstraction layer (e.g. hyperparameters optimization, meta learning)0


![Project code architecture](/docs/project_archi.png "Project code architecture")


The Dataset can be implemented with various librairies, following the implicit convention that two folders are handled:

* ``data/raw`` stores the raw data files;
* ``data/processed`` stores the data after preprocessing, used for several experiments.

## Quick Start

Each model can be experimented in an easy and quick way using predifined command exposed through the Nox tool.

### Docker

A docker file is provided to get started. To build the image, run the following command:
::
    docker build -t ai4sim .

To install the corresponding python requirements for a use-case an entrypoint is implemented. It can be selected by one of the folling [combustion_gnns, combustion_unets, wf_gwd]:
::
    docker run -it --rm ai4sim ./script.sh combustion_gnns

To get inside the container:
::
    docker run -it --rm -e bash ai4sim

### Requirements

The following procedures only require [Nox](https://nox.thea.codes/en/stable/) is a python build tool, that allows to define targets (in a similar way that Make does), to simplify command execution in development and CI/CD pipeline. By default, each nox target is executed, in a specific virtualenv that ensure code partitioning and experiments reproducibility.
::
    pip install nox

### Experiment a use case

Several Nox targets allow to handle easily an experimentation of any use case on a demo dataset and configuration.

Choose the _Model Collection_ use case you want to experiment and go in.
::
    cd weather_forcast/gwd
There you can display the list of the available targets with
::
    nox --list

Please note, some of them are experimentation oriented, while other ones are CI/CD oriented.

*Coming soon ...*

You can launch a demo training on the model use case with ``nox -s train``

### Development mode

The nox target are also very useful to launch generic command during development phase.

#### Run unit tests

You can run the whole unit test suite of a use case, using ``pytest``, with ``nox -s tests``.
This target also prints out the coverage report and save a xml version in ``.ci-reports/``.

#### Run linting

You can run the python linting of the code use case, using ``flake8``, with ``nox -s lint``.