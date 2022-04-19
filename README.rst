Philosophy
===============
**Important note: at the moment, lots of code is purely being duplicated, because the focus has been made on the use-case (UC) content. Future works will refactor a bit the repository content to really abstract mutualizable data processing logic.**

This project contains a collection of models developed by the Atos AI4sim R&D team and is intended for research purposes. The current workflow is based entirely on NumPy, PyTorch, PyG and Lightning. 

To take care of the boiler-plate, early stopping, and tensorboard logging, training parallelization, we integrate directly with PyTorch Lightning. For each new UC, we use the same file structure. In all the following data, an experiment designates a specific training run with a specific model and a specific dataset + split.

* `configs/` contains the experiments configuration files, following the Lightning CLI format. 
* `data/` contains the `raw` and `processed` data directories, and normalization factors and explicit train/val/test split sets.
* `tests/` contains unit tests modules.
* `notebooks/` contains example Jupyter notebooks, to illustrate the use case code usage.
* `config.py` exposes global paths and path specific to the current experiment.
* `data.py` deals with dataset and datamodule creation.
* `models.py` deals with model and module creation, including training logic.
* `plotters.py` takes care of plots generation for the test set.
* `trainer.py` is the main entrypoint responsible for creating a Trainer object, a CLI, and saving artifacts in the experiment directory.
* `noxfile.py` is the Nox build tool configuration file that defines all targets available for the UC.

Collections
===============
Collections are developed through partnerships with the ECMWF, the CERFACS, and INRIA.

* Combustion

    - CNF for Combustion and Flame
    - R2 and R3 are simulations of Aachen's flame, with different resolution
    
* Weather Forecast

    - Gravity Wave Drag
    - 3D Bias Correction
    
* WMED, Atmosphere-Oceanic Coupling

Quick Start
===============
Each UC can be experimented in an easy and quick way using predifined command exposed through the Nox tool.

Docker
-----------------
A docker file is provided to get started. To build the image, run the following command:
::
    docker build -t ai4sim .

To install the corresponding python requirements for a use-case an entrypoint is implemented. It can be selected by one of the folling [combustion_gnns, combustion_unets, wf_gwd]:
::
    docker run -it --rm ai4sim ./script.sh combustion_gnns

Requirements
-----------------
The following procedures only require [Nox](https://nox.thea.codes/en/stable/) is a python build tool, that allows to define targets (in a similar way that Make does), to simplify command execution in development and CI/CD pipeline. By default, each nox target is executed, in a specific virtualenv that ensure code partitioning and experiments reproducibility.
::
    pip install nox

Experiment a use case
-----------------
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

Development mode
-----------------
The nox target are also very useful to launch generic command during development phase.

Run unit tests
~~~~~~~~~~~~~~~~~~~~~~
You can run the whole unit test suite of a use case, using ``pytest``, with ``nox -s tests``.
This target also prints out the coverage report and save a xml version in ``.ci-reports/``.

Run linting
~~~~~~~~~~~~~~~~~~~~~~
You can run the python linting of the code use case, using ``flake8``, with ``nox -s lint``.