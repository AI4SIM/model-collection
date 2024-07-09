"""
This module proposes the list of generic Nox build targets for the different use cases.
This file is generic and aims at:
    - providing the common targets that can be used in all use cases,
    - defining a model of the different targets each use case should propose.
"""

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

import os
import nox

REPORTS_DIR = ".ci-reports/"
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAKE8_CFG = os.path.join(ROOT_PATH, 'tools', 'flake8', 'flake8.cfg')

# The list of the default targets executed with the simple command "nox".
nox.options.sessions = ["lint", "tests"]


def _wheel_version(wheel: str, req_file: str = 'requirements.txt') -> str:
    """Extract the version of a wheel from a requirement.txt, if it is present.

    Args:
        req_file (str): path of the input requirement.txt file.

    Returns:
        (str): the pip version extracted from the requirement.txt file if torch is present,
            an empty string otherwise.
    """
    version = ''
    with open(req_file, encoding='utf-8') as file:
        for line in file.readlines():
            if f"{wheel}==" in line:
                version = line.rstrip()
    return version


def _torch_version(req_file: str = 'requirements.txt') -> str:
    """Extract the torch version and its cuda support from a requirement.txt, if it is present.

    Args:
        req_file (str): path of the input requirement.txt file.

    Returns:
        (str): the torch version extracted from the requirement.txt file if torch is present,
            an empty string otherwise.
    """
    version = ''
    cuda = ''
    version = _wheel_version("torch", req_file).split('==')[1]
    if "+" not in version:
        cuda = "cpu"
    else:
        version, cuda = version.split('+')
    return version, cuda


@nox.session
def dev_dependencies(session):
    """Target to install all requirements of the use-case code."""
    torch_vers, cuda_vers = _torch_version()
    req_file = "requirements.txt"

    additional_url = ''
    extra_url = ''
    if torch_vers:
        # Set the url of precompiled torch version depending on CUDA version
        if cuda_vers != "cpu":
            extra_url = f"https://download.pytorch.org/whl/{cuda_vers}"

        # Set the url of precompiled torch-geometric dependencies depending on torch version
        additional_url = f"https://data.pyg.org/whl/torch-{torch_vers}+{cuda_vers}.html"

    session.run("python3", "-m", "pip", "install",
                _wheel_version("pip", req_file),
                _wheel_version("wheel", req_file),
                _wheel_version("setuptools", req_file))
    session.run("python3", "-m", "pip", "install",
                "-r", req_file,
                "-f", additional_url,
                "--extra-index-url", extra_url)
    if "purge" in session.posargs:
        session.run("python3", "-m", "pip", "cache", "purge")


@nox.session
def tests(session):
    """Target to run unit tests on the code with pytest, and generate coverage report."""
    dev_dependencies(session)
    session.run("python3", "-m", "pip", "install", "pytest-cov")
    session.run("python3", "-m", "pytest", "--cache-clear", "--cov=./", "-v")
    session.notify("coverage_report")


@nox.session
def coverage_report(session):
    """Target to generate coverage report from test results."""
    session.run("python3", "-m", "pip", "install", "coverage")
    session.run("coverage", "xml", "-o", f"{REPORTS_DIR}/pycoverage.xml")


@nox.session
def lint(session):
    """Target to lint the code with flake8."""
    session.run("python3", "-m", "pip", "install",
                "flake8",
                "flake8-docstrings",
                "flake8-use-fstring",
                "flake8-variables-names",
                "pep8-naming")
    session.run("flake8", "--config", FLAKE8_CFG)


@nox.session
def docs(session):
    """Target to build the documentation (not yet implemented)."""
    raise NotImplementedError("This target is not yet implemented.")


@nox.session
def download_data(session):
    """Target to download the data required to train the use-case models (not yet implemented)."""
    raise NotImplementedError("This target is not yet implemented.")


@nox.session
def train(session):
    """Target to launch a basic training of the use-case (not yet implemented)."""
    raise NotImplementedError("This target is not yet implemented.")
