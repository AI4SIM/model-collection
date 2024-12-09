# AI4SIM Model Collection

## Table of Contents

1. [Philosophy](#philosophy)
2. [Collection](#collection)
3. [Project Organization](#project-organization)
    1. [Project Path Tree](#project-path-tree)
    2. [Model Project Files](#model-project-files)
    3. [Project Architecture](#project-architecture)
4. [Experiment a Model Project](#experiment-a-model-project)
    1. [Setting Up the Environment](#setting-up-the-environment)
    2. [Prepare the Dataset](#prepare-the-dataset)
    3. [Launch the Training](#launch-the-training)
5. [Contribute](#contribute)
    1. [The Nox Tool](#the-nox-tool)
    2. [Existing Model Project](#existing-model-project)
    3. [New Model Project](#new-model-project)
    4. [Pull Request](#pull-request)

## Philosophy

This project contains a collection of models developed by the Atos AI4SIM R&D team and is intended for research purposes. The current workflow is based entirely on NumPy, PyTorch, Dask, and Lightning. Domain-specific libraries can be added, like PyTorch Geometric for graph neural networks.

**The repository is organized as a collection of independent model implementations for various use-cases. You will thus find a lot of duplicated code because the focus has been made on the model projects' content. Nevertheless, the CI/CD and development tools have been mutualized.**

This project is licensed under the [APACHE 2.0 license.](http://www.apache.org/licenses/LICENSE-2.0)

## Collection

The collection of models is developed through partnerships bringing their use-cases.

Currently, the models that have been developed are based on the following use-cases:

- Computational Fluid Dynamics
    - Combustion (with UNets and GNNs), from CERFACS

- Weather Forecast
    - Gravity Wave Drag (with CNNs), from ECMWF
    - 3D cloud radiative effect correction (with UNets), from ECMWF

## Project Organization

> In all of the following, an experiment designates a specific training run (hyperparameters) with a specific model (neural architecture) and a specific dataset and splitting strategy.

### Project Path Tree

All the models are placed in the repository with a path following the rule:

``<domain>/<use-case>/<NN architecture>``

For example, the code in the path ``reactive-flows/cnf-combustion/gnns`` implements some **Graph Neural Network (GNN)** architectures developed for **Computational Fluid Dynamics (CFD)** and applied to a **Combustion** use-case.

### Model Project Files

To take care of the boilerplate, early stopping, tensorboard logging, training parallelization, etc., we integrate them directly with PyTorch Lightning. For each new model, we use the same code architecture, described below.

All model implementations should include the following folders and files:

* ``configs/`` contains the experiments configuration YAML files, following the Lightning CLI format.
* ``tests/`` contains unit test modules.
* ``config.py`` exposes global paths and paths specific to the current experiment.
* ``data.py`` contains the dataset and datamodule.
* ``models.py`` contains the model ("module"), including training logic. The architecture can be imported from specific libraries or a local module (e.g. unet.py).
* ``trainer.py`` is the main entry point responsible for creating a Trainer object, a CLI, and saving artifacts in the experiment directory.
* ``noxfile.py`` is the Nox build tool configuration file that defines all targets available for the model project.
* ``env.yaml`` lists the environment requirements, like the Ubuntu docker base image or the Python version.
* ``requirements.txt`` contains all the Python dependencies of the project, generated using the ``pip freeze`` command.
* ``ci/`` contains all the additional files required to run the functional tests.

Optionally, it can also include the following folders and files:
* ``data/`` contains the dataset configuration files and will contain the `raw` and `processed` data directories, and normalization factors (and optionally explicit train/val/test split sets), once the dataset has been downloaded and processed.
* ``notebooks/`` contains example Jupyter notebooks, for pedagogical purposes.
* ``plotters.py`` takes care of plot generation for the test set.

### Project Architecture

Each model project is made of two Lightning pipelines:

* Data pipeline: a **DataModule** wraps a Dataset to provide data (with dataloaders, preprocessing...);
* Model pipeline: a **Module** compiles a neural network architecture with its optimizer.

The Trainer plugs both pipelines when called to run an experiment. It can be called by hand (CLI, with config files) or by another abstraction layer (e.g. hyperparameters optimization, meta-learning).

![Project code architecture](/docs/project_archi.png "Project code architecture")

The Dataset can be implemented with various libraries, following the implicit convention that two folders are handled:

* ``data/raw`` stores the raw data files;
* ``data/processed`` stores the data after preprocessing, used for several experiments.

## Experiment a Model Project

### Setting Up the Environment

Each model can be experimented using a python environment dedicated to the project, either in a docker container or on bare metal using a virtual environment.

#### Docker Container

The AI4SIM GitHub CI/CD publishes in the [GitHub registry](https://github.com/AI4SIM/model-collection/pkgs/container/model-collection) a docker image dedicated to each model project proposed in the **Model collection** repository. Each image is built on a public **Ubuntu** base image (e.g. *nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04*). The model project code has been added in ``/home/ai4sim/<domain>/<use-case>/<NN architecture>`` and its requirements have been installed in the container.

Each image can be identified using its docker tag, ``<domain>-<use-case>-<NN architecture>``, that is automatically built from the model project path ``<domain>/<use-case>/<NN architecture>``. For example, you can pull the docker images for the ``weather-forecast/ecrad-3d-correction/unets`` model project using:

```bash
docker pull ghcr.io/ai4sim/model-collection:weather-forecast-ecrad-3d-correction-unets
```

If you want to experiment with the model in a different environment you can build your own docker image following the instructions described in [Contribute](#contribute).

#### Virtual Environment

To experiment with a model on bare metal, we suggest building a python virtual environment that will embed all the python requirements of the targeted model.

You can use a manual installation:

```bash
cd <domain>/<use-case>/<NN architecture>
python3 -m venv your-venv
source your-venv/bin/activate
pip install -U $(grep pip== ./requirements.txt)
pip install -r ./requirements.txt
```

or, alternatively, you can use the *Nox* target ``dev_dependencies`` (cf [Nox targets]()):

```bash
cd <domain>/<use-case>/<NN architecture>
python3 -m pip install $(grep nox== ./requirements.txt)
nox -s dev_dependencies
source dev_dependencies/bin/activate
```

### Prepare the Dataset

You can find, in all model project directories, a ``README.md`` file that describes how to download and prepare the data required to launch a training. Please refer to the corresponding README file:

- Computational Fluid Dynamics
    - Combustion
        - [Unets](reactive-flows/cnf-combustion/unets/README.md)
        - [GNNs](reactive-flows/cnf-combustion/gnns/README.md)

- Weather Forecast
    - Gravity Wave Drag 
        - [CNNs](weather-forecast/ecrad-3d-correction/unets/README.md)
    - 3D Bias Correction
        - [Unets](weather-forecast/gravity-wave-drag/cnns/README.md)

### Launch the Training

#### Docker Container

Using the docker image described at [Setting-up the environment](#setting-up-the-environment), you can launch the training of the model with:

```bash
cd <domain>/<use-case>/<NN architecture>
podman run \
    -v ./data:/home/ai4sim/<domain>/<use-case>/<NN architecture>/data \
    -w /home/ai4sim/<domain>/<use-case>/<NN architecture> \
    ghcr.io/ai4sim/model-collection:<domain>-<use-case>-<NN architecture> \
    python3 trainer.py --config configs/<training-config>.yaml
```

#### Virtual Environment

Using the python virtual environment built as described in [Setting-up the environment](#setting-up-the-environment), you can launch the training of the model with:

```bash
cd <domain>/<use-case>/<NN architecture>
python3 trainer.py --config configs/<training-config>.yaml
```

## Contribute

Contribution to existing models or proposition of new projects are welcome. Please, develop in your own branch, following the recommendations below, and open a pull request to merge your branch to the **main** one. 

### The *Nox* Tool

The development mode for all model projects is based on a tool, called *[Nox](https://nox.thea.codes/en/stable/)*.

*Nox* is a python build tool, that allows defining targets (in a similar way that Make does), simplifying command execution in development and CI/CD pipelines. By default, each *Nox* session (equivalent to a make target) is executed in a specific virtualenv that ensure code partitioning and experiments reproducibility.

#### Installation

*Nox* can be installed using *pip*.

```bash
pip install nox
```

To install the version of *Nox* used in an existing model project please refer to the ``requirements.txt`` file of the project.

```bash
pip install $(grep nox== requirements.txt)
```

#### Usage

In each model project directory the ``noxfile.py`` file implements all the *Nox* sessions available (note most of them are actually imported from a global file in ``tools/nox/nox_ref_file.py``).

In the model project you can display the list of the available targets with:

```bash
nox --list
```

The main sessions you will use for development will be:
- ``dev_dependencies``
- ``tests``
- ``lint``

Note these sessions are used to run the CI/CD and build the docker image of a model project. So using them and making sure they succeed are major steps on the road of proposing your contribution.

#### Create the Proper Python Development Environment

The nox targets are also very useful to launch generic commands during the development phase.

##### Run Unit Tests

You can run the whole unit test suite of a model project, using ``pytest``, with ``nox -s tests``.
This target also prints out the coverage report and saves an XML version in ``.ci-reports/``.

##### Run Linting

You can run the python linting of the code model project, using ``flake8``, with ``nox -s lint``.

### Existing Model Project

Please note, the maintenance policy of the repository is to consider a model project frozen from the moment when it is integrated. It means you can open an issue, but only the major ones (bugs breaking the functionality or leading the model to be scientifically irrelevant) will be treated, no upgrade nor refactoring of the existing model code will be addressed.

If you want to propose a correction to an existing model, or a new model implementation of an existing model project, please follow the recommendations described in the [Setting-up the environment](#setting-up-the-environment) section to develop in the proper Docker container. If you prefer to develop in a python virtual environment, please use the *Nox* session ``dev_dependencies`` (as described in [Create the proper python development environment](#create-the-proper-python-development-environment)) that will create the proper virtualenv to develop in.

### New Model Project

To start a new model project from scratch, we recommend adopting the following procedure to develop your contribution. The existing model projects are good sources of inspiration for what you will have to do.

You can find in the [Model project files](#model-project-files) section the list of files and directories a model project should include.

#### Setup the Environment

The first step is to initiate the required files that will permit to set up the development environment.

As a reminder of the [Model project files](#model-project-files) section, the required files to define the environment are:

* ``env.yaml`` lists the environment requirements, like the Ubuntu docker base image or the Python version.
* ``noxfile.py`` is the Nox build tool configuration file that defines all targets available for the model project.
* ``requirements.txt`` contains all the python dependencies of the project, generated using the ``pip freeze`` command.

##### Create the *env.yaml* File

This file will actually be used later by the GitHub Actions workflow in charge of building the docker image of the model project, but we recommend creating and filling it now to fix:
- the **Ubuntu image** you will work on,
- the **Python version** you want to use.

The format of this file is the following:

```yaml
# This file defines the global environment dependencies for this model project.
# It is read by the CI/CD workflows in charge of the tests (ai4sim-ci-cd.yaml)
# and the build of the Docker image (images-ci-cd.yaml) to set the environment.
python_version: 3.8
ubuntu_base_image:
  name: nvidia/cuda
  tag: 11.7.1-cudnn8-runtime-ubuntu20.04
```

Note, the **Ubuntu base image** should be available on the *docker.io* registry, and the **Python version** chosen must be available in the apt repositories of the chosen **Ubuntu base image**.

> If you have chosen an **Ubuntu base image** with a pre-installed set of python libraries you plan to use (e.g. an nvidia pytorch image), You must report in the *env.yaml* file the python version already available in the image, to prevent a new install of a different python version that will make the pre-installed libraries unavailable.

##### Initiate *noxfile.py*

The ``noxfile.py`` file implements all the *Nox* sessions available for the model project. This file should, at least, import from the global file in ``tools/nox/nox_ref_file.py``, all the generic sessions. The minimal content of this file is then:

```python
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

"""
This module simply loads all the nox targets defined in the reference noxfile
and makes them available for the model project.
This file can be enriched by model project specific targets.
"""

import os
import sys
import inspect

# Insert the tools/nox folder to the python path to fetch the nox_ref_file.py content
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
common_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
build_ref_dir = os.path.join(common_dir, "tools", "nox")
sys.path.insert(0, build_ref_dir)

# Fetch the nox_ref_file.py content
from nox_ref_file import *

# Insert below the model project specific targets

```

If needed, you can add some sessions, specific to the model project, at the end of the file.

This file must be initiated now because the *Nox* sessions will be used to install the python dependencies (using the ``requirements.txt`` file we will see just below) either in the development docker image, or in the ``dev_dependencies`` virtual environment.

##### Initiate *requirements.txt*

Like the two first files, the ``requirements.txt`` is used to set up the development environment. It should include, at least, the *pip*, *nox*, *torch* and *lightning* version you plan to use:

```python
pip==24.0
nox==2024.4.15
torch==1.13.1+cu117
pytorch-lightning==1.5.7
```

Note this is not the final version of this file, because you will update it at the end of the process (see [Finalize the requirements.txt](#finalize-the-requirementstxt)), just before pushing your development branch to create a pull request. As a consequence, you can add other requirements right now, but any python library you will install, using ``pip install``, during your development process, will be included in the final version of the file.

##### Build the Development Environment

You are now ready to set up your development environment, either as a *Docker container* or a Python *virtual environment*.

###### Docker Container

To build your development docker image, use the generic ``Dockerfile`` in the ``docker`` folder.

The 4 following parameters are mandatory and must be passed to the Dockerfile, using the ``--build-arg`` option. For the 3 last ones, please use the values from the env.yaml file you previously created:
- MODEL_PROJECT_PATH: the path ``<domain>/<use-case>/<NN architecture>`` you are working on.
- UBUNTU_IMAGE_NAME: the base Ubuntu docker image name indicated in the ``env.yaml`` file.
- UBUNTU_IMAGE_TAG: the base Ubuntu docker image tag indicated in the ``env.yaml`` file.
- PYTHON_VERS: the Python version indicated in the ``env.yaml`` file.

To build the Docker image from the repository root directory:

```bash
docker build \
    -t model-collection:<domain>-<use-case>-<NN architecture> \
    -f ./docker/Dockerfile . \
    --build-arg MODEL_PROJECT_PATH=<domain>/<use-case>/<NN architecture> \
    --build-arg UBUNTU_IMAGE_NAME=nvidia/cuda \
    --build-arg UBUNTU_IMAGE_TAG=11.7.1-cudnn8-runtime-ubuntu20.04 \
    --build-arg PYTHON_VERS=3.8
```

Finally, start the container, binding your source code path to ease the development:

```bash
docker run -ti \
    -v <domain>/<use-case>/<NN architecture>:/home/ai4sim/<domain>/<use-case>/<NN architecture> \
    model-collection:<domain>-<use-case>-<NN architecture> \
    bash
```

In the container, you can directly use the ``pip`` and ``python`` command of the system to respectively install the dependencies and execute your code.

###### Virtual Environment

To build your virtual environment, please be sure to have the Python version indicated in our env.yaml file installed on your system, and the *Nox* tool available as described [previously](#installation).

Then it is recommended to use the *Nox* session ``dev_dependencies``:

```bash
nox -s dev_dependencies
source dev_dependencies/bin/activate
```

#### Model Development

During your development phase, it is highly recommended to integrate the code quality standards that have been adopted for this repository. The following points should be addressed and will be automatically checked by the CI/CD workflows when you push your code:

- Linting
- Unit tests
- Functional tests 

##### Linting

Linting is a static code analysis in charge of checking the code formatting to avoid some basic bugs and ensure homogeneity of the code base with respect to the main Python checkstyle rules. We use [flake8](https://flake8.pycqa.org/en/latest/) and some plugins to rate the code.

To evaluate your checkstyle code quality run:

```bash
nox -s lint -v
```

The rating score must be 10/10, so all checkstyle violations returned by the above command should be fixed.

Nevertheless, if after all your best effort, you think a checkstyle violation cannot/should not be fixed, you can deactivate the lint checker on a given line of code with the [noqa](https://flake8.pycqa.org/en/3.1.1/user/ignoring-errors.html#in-line-ignoring-errors) directive:

```python
example = lambda: 'example'  # noqa: E731
```

##### Unit Tests

The new code should be provided with as many unit tests as possible. Because the main part of the code is based on *lightning* and *torch*, it is not interesting to test all classes and methods inherited from these libraries, but there is still some part of the code interesting to be tested.

Note, there is no hard coverage requirement to push and merge some code on the repository, but the best effort should be done on unit tests development, particularly for the functions you developed from scratch (e.g. utils functions).

The unit test framework used is *pytest* and you can run your test suite using the *Nox* session ``tests``:

```bash
nox -s tests -v
```

##### Functional Tests

Each model project should embed functional tests in the ``ci`` folder, that ensure the training code is purely functional, i.e. without any precision or performance consideration.

The principle of this functional test is to launch a training on a dummy dataset. To be sure to detect a possible regression due to changes in the code, and not on the data source side, the dummy dataset must be generated synthetically. 

The ``ci`` folder must contain: 
- a ``generate_synthetic_data.py`` file in charge of the synthetic dataset generation based on the real data format. This script must take care of not erasing real data already present in the ``data`` folder. We suggest raising an exception with an explicit message in case of the presence of existing data in the targeted path. Future work should address this issue by making the data path configurable.
- a ``configs`` folder with the lightning formatted configuration file of the test training,
- a ``run.sh`` script file in charge of running the different training tests.

Optionally it can also include:
- a ``requirements_data.txt`` listing the dependencies related to the synthetic data generation.

The *Nox* session ``train_test`` allows running the functional tests:

```bash
nox -s train_test -v
```

By default, the synthetic data are kept in the ``data`` folder, after the tests have run. To clean the ``data`` folder at the end of the tests, use the ``clean_data`` option:

```bash
nox -s train_test -v -- clean_data
```

### Pull Request

Now, your developments are done, the last step is to open a *Pull Request* to propose your contribution.

#### Document the Model Project

In each model project, a ``README.md`` file should:

- introduce the scientific challenge addressed,
- explain how to get the dataset,
- briefly describe the model input and output variables and the NN architecture implemented.

#### Finalize the *requirements.txt*

Before creating the *Pull Request* itself, you must generate the final *requirements.txt* file to freeze all the python dependencies of your model project.

In your development environment use:

```bash
pip freeze > requirements.txt
```

Then commit this last change.

#### Add the Model Project to the CI/CD Workflows

The CI/CD workflows are defined by the GitHub Actions Yaml files in the ``workflows`` folder. There are 2 different workflows:
- the *code quality* CI/CD workflow: this workflow (see ``workflows/ai4sim-ci-cd.yaml``) is in charge of running tests and lint for all model projects, to detect possible regressions. It is triggered by any change done on any branch and on any pull requests.
- the *docker image builder* CI/CD workflow: this workflow (see ``workflows/images-ci-cd.yaml``) is in charge of building and publishing the docker images for all model projects. It is triggered by any change done on the *main* branch and on any pull requests targeting the *main* branch.

The CI/CD workflows run the same actions for all model projects listed in the *jobs/strategy/matrix/model-project* section of the Yaml files.

To finalize your contribution you then have to add the path of your model project in all the CI/CD workflow files:

```yaml
...
jobs:
    ...
    strategy:
      matrix:
        model-project:
          - reactive-flows/cnf-combustion/gnns
          - reactive-flows/cnf-combustion/unets
          - weather-forecast/gravity-wave-drag/cnns
          - weather-forecast/ecrad-3d-correction/unets
          - <domain>/<use-case>/<NN architecture>       <-- Your new model project path
...
```

Then, commit these last changes and push your branch to GitHub. It should trigger the CI/CD workflow related to the *code quality*. If all the CI/CD runs pass, you are now able to open a *Pull Request*.

#### Open the *Pull Request*

Open a *Pull Request* to merge your development branch to the *main* branch, with a description of your contribution as much as possible complete.
 
The review process will then be taken in charge by the AI4SIM team. The *Pull Request* will be merged by the AI4SIM team once the following criteria are met:
- at least 2 AI4SIM members have validated the PR,
- all CI/CD workflows pass.

Congratulations, you have reached the end of the process ! Thank you a lot for your contribution !!!