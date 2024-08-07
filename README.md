# AI4SIM Model Collection

## Table of contents


1. [Philosophy](#philosophy)
2. [Collection](#collection)
3. [Project organization](#project-organization)
    1. [Project path tree](#project-path-tree)
    2. [Model project files](#model-project-files)
    2. [Projects architecture](#projects-architecture)
4. [Experiment a model project](#experiment-a-model-project)
    1. [Setting-up the environment](#setting-up-the-environment)
        1. [Docker container](#docker-container)
        2. [Virtual environment](#virtual-environment)
    2. [Prepare the dataset](#prepare-the-dataset)
    3. [Launch the training](#launch-the-training)
        1. [Docker container](#docker-container)
        2. [Virtual environment](#virtual-environment)

## Philosophy

This project contains a collection of models developed by the Atos AI4sim R&D team and is intended for research purposes. The current workflow is based entirely on NumPy, PyTorch, Dask and Lightning. Domain-specific librairies can be added, like PyTorch Geometric for graph neural networks.

**The repository is organized as a collection of independant model implementation for various use-cases (UC). You will thus find a lot of duplicated code, because the focus has been made on the projects content. Nevertheless the CI/CD and development tools have been mutualized.**

This project is licensed under the [APACHE 2.0 license.](http://www.apache.org/licenses/LICENSE-2.0)

## Collection

The collection of models is developed through partnerships bringing their use-cases.

Currently, the models that have been developed are based on the following use-cases :

- Computational Fluid Dynamics
    - Combustion (with Unets and GNNs), from CERFACS

- Weather Forecast
    - Gravity Wave Drag (with CNNs), from ECMWF
    - 3D Bias Correction (with Unets), from ECMWF

## Project organization

> In all of the following, an experiment designates a specific training run (hyperparameters) with a specific model (neural architecture) and a specific dataset and splitting strategy.

### Project path tree

All the models are placed in the repository with a path following the rule :

``<domain>/<use-case>/<NN architecture>``

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
* ``env.yaml`` lists the environment requirements, like the Ubuntu docker base image or the Python version.
* ``requirements.txt`` contains all the python dependencies of the project, generated using the ``pip freeze`` command.
* ``ci/`` contains all the additional files required to run the functional tests. 

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

## Experiment a model project

### Setting-up the environment

Each model can be experimented using a python environment dedicated to the project, either in a docker container or on bare metal using a virtual environment.

#### Docker container

The AI4SIM github CI/CD publishes in the [github registry](https://github.com/AI4SIM/model-collection/pkgs/container/model-collection) a docker image dedicated to each model project proposed in the **Model collection** repository. Each image is built on an public **Ubuntu** base image (e.g. *nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04*). All the model project's requirements have been installed and the model project code has been added in ``/home/ai4sim/<domain>/<use-case>/<NN architecture>``.

Each image can be identified using its docker tag, ``<domain>-<use-case>-<NN architecture>``, that is automatically built from the model project path ``<domain>/<use-case>/<NN architecture>``. For example, you can pull the docker images for the ``weather-forecast/ecrad-3d-correction/unets`` model project using :

```
>> docker pull ghcr.io/ai4sim/model-collection:weather-forecast-ecrad-3d-correction-unets
```

If you want to experiment the model in a different environment you can build your own docker image following the instruction descibed in [Contribute](#contribute).

#### Virtual environment

To experiment a model on bare metal, we suggest to build a python virtual environment that will embed all the python requirements of the targeted model.

You can use a manual installation :

```
cd <domain>/<use-case>/<NN architecture>
python3 -m venv your-venv
source your-venv/bin/activate
pip install -U $(grep pip== ./requirements.txt)
pip install -r ./requirements.txt
```

or, alternatively, you can use the *Nox* target ``dev_dependencies`` (cf [Nox targets]()) :

```
cd <domain>/<use-case>/<NN architecture>
python3 -m pip install $(grep nox== ./requirements.txt)
nox -s dev_dependencies
source dev_dependencies/bin/activate
```

### Prepare the dataset

You can find, in all model project directories, a ``README.md`` file that describes how to download and prepare the data required to launch a training. Please refer to the corresponding README file :

- Computational Fluid Dynamics
    - Combustion
        - [Unets](cfd/combustion/unets/README.md)
        - [GNNs](cfd/combustion/gnns/README.md)

- Weather Forecast
    - Gravity Wave Drag 
        - [CNNs](weather-forecast/ecrad-3d-correction/unets/README.md)
    - 3D Bias Correction
        - [Unets](weather-forecast/gravity-wave-drag/cnns/README.md)

### Launch the training

#### Docker container

Using the docker image described at [Setting-up the environment](#setting-up-the-environment), you can launch the training of the model with :

```
cd <domain>/<use-case>/<NN architecture>
podman run \
    -v ./data:/home/ai4sim/<domain>/<use-case>/<NN architecture>/data \
    -w /home/ai4sim/<domain>/<use-case>/<NN architecture> \
    ghcr.io/ai4sim/model-collection:<domain>-<use-case>-<NN architecture> \
    python3 trainer.py --config configs/<training-config>.yaml
```

#### Virtual environment

Using the python virtual environment built as descirbed in [Setting-up the environment](#setting-up-the-environment), you can launch the training of the model with :
```
cd <domain>/<use-case>/<NN architecture>
python3 trainer.py --config configs/<training-config>.yaml
```

## Contribute

Contribution to existing models or proposition of new projects are welcome. Please, develop in your own branch, following the below recommendations, and open a pull request to merge your branch on the **main** one. 

### The *Nox* tool

The development mode for all model projects is based on a tool, called *[Nox](https://nox.thea.codes/en/stable/)*.

*Nox* is a python build tool, that allows to define targets (in a similar way that Make does), to simplify command execution in development and CI/CD pipelines. By default, each *Nox* session (equivalent to a make target) is executed in a specific virtualenv that ensure code partitioning and experiments reproducibility.

#### Installation

*Nox* can be installed using *pip*.

```
pip install nox
```

To install the version of *Nox* used in a existing model project please refer to the ``requirements.txt`` file of the project.

```
pip install $(grep nox== requirements.txt)
```

#### Usage

In each model project directory the ``noxfile.py`` file implements all the *Nox* sessions available (note most of them are actually imported from a global file in ``tools/nox/nox_ref_file.py``).

In the model project you can display the list of the available targets with
```
nox --list
```

The main sessions you will use for development will be :
- ``dev_dependencies``
- ``tests``
- ``lint``

Note these seesions are used to run the CI/CD and build the docker image of a model project. So using them and making sure they succeed are major steps on the road of proposing your contribution.

#### Create the proper python development environment

The nox target are also very useful to launch generic command during development phase.

##### Run unit tests

You can run the whole unit test suite of a use case, using ``pytest``, with ``nox -s tests``.
This target also prints out the coverage report and save a xml version in ``.ci-reports/``.

##### Run linting

You can run the python linting of the code use case, using ``flake8``, with ``nox -s lint``.


### Existing model project

Please note, the maintenance policy of the repository is to consider a model project freezed from the moment when it is intergrated. It means you can open an issue, but the only major ones (bugs breaking the functionality or leading the model to be scientifically unrelevant) will be treated, but no upgrade nor refactoring of the existing model code will be adressed.

If you want to propose a correction to an exixting model, or a new model implementation of an existing model project, please follow the recommendations described in the [Setting-up the environment](#setting-up-the-environment) section to develop in the proper Docker container. If you prefer to develop in a python virtual environment, please use the *Nox* session ``dev_dependencies`` that will create the proper virtualenv to develop.



### New model project

To start a new model project from scratch, we recommend to adopt the following procedure to develop your contribution. Of course the existing model project are good sources of inspiration of what you will have to do.

#### Define the environment
Choose Ubuntu image
Choose the python version

##### Create the env.yaml file
##### Build the docker image

#### Initiate noxfile.py
#### Initiate requirements.txt file
pip, nox, torch version


### 