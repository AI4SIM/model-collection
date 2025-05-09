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
import sys
import re
import glob
import nox

REPORTS_DIR = ".ci-reports/"
COV_FT_FILE = ".coverage-ft"
COV_UT_FILE = ".coverage-ut"
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAKE8_CFG = os.path.join(ROOT_PATH, 'tools', 'flake8', 'flake8.cfg')

# The list of the default targets executed with the simple command "nox".
nox.options.sessions = ["lint", "tests"]

# Check the required python version is available
current_python = f'{sys.version_info.major}.{sys.version_info.minor}'
with open("env.yaml", 'r') as env_file:
    python_req = None
    for line in env_file.readlines():
        if 'python_version:' in line:
            match = re.search(r'python_version: ([0-9\.]*)', line)
            python_req = match.group(1)
    if not python_req:
        raise RuntimeError("Required python version not found in the env.yaml file.")
    elif current_python != python_req:
        raise RuntimeError("Current python version does not match the required version: "
                           f"{current_python} != {python_req}.")


def _wheel_version(wheel: str, req_file: str = 'requirements.txt') -> str:
    """Extract the version of a wheel from a requirement.txt, if it is present.

    Args:
        req_file (str): path of the input requirement.txt file.

    Returns:
        (str): the pip version extracted from the requirement.txt file if torch is present,
            an empty string otherwise.
    """
    version = wheel
    with open(req_file, encoding='utf-8') as file:
        for line in file.readlines():
            if f"{wheel}==" in line:
                version = line.rstrip()
                break
    return version


def _torch_version(req_file: str = 'requirements.txt') -> str:
    """Extract the torch version and its cuda support from a requirement.txt, if it is present.

    Args:
        req_file (str): path of the input requirement.txt file.

    Returns:
        (str): the torch version extracted from the requirement.txt file if torch is present,
            an empty string otherwise.
    """
    cuda = 'cpu'
    version = _wheel_version("torch", req_file).split('==')[1]
    if "+" in version:
        version, cuda = version.split('+')
    return version, cuda

@nox.session
def base_dependencies(session):
    """Target to install the basics python requirements."""
    req_file = "requirements.txt"
    session.run("python3", "-m", "pip", "install",
                _wheel_version("pip", req_file),
                _wheel_version("wheel", req_file),
                _wheel_version("setuptools", req_file))

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

    # Install base python dependencies
    base_dependencies(session)

    # Install use-case python dependencies
    session.run("python3", "-m", "pip", "install",
                "-r", req_file,
                "-f", additional_url,
                "--extra-index-url", extra_url)
    if "purge" in session.posargs:
        session.run("python3", "-m", "pip", "cache", "purge")


@nox.session
def tests(session):
    """Target to run unit tests on the code with pytest, and generate coverage report."""
    # Install use-case python dependencies
    dev_dependencies(session)
    session.run("python3", "-m", "pip", "install", "pytest-cov")
    session.run('rm','-rf', COV_UT_FILE, external=True)
    session.run("python3", "-m", "pytest", "--cache-clear", "--cov=./", "-v")
    session.run('mv','.coverage', COV_UT_FILE, external=True)
    session.notify("coverage_report", ['data-file', f'{COV_UT_FILE}'])


def coverage_install(session):
    session.run("python3", "-m", "pip", "install", "coverage")


@nox.session
def coverage_report(session):
    """Target to generate coverage report from test results."""
    coverage_install(session)

    cov_file=".coverage"
    if "combine" in session.posargs:
        session.run("coverage", "combine", "--keep", "--append", *glob.glob(".coverage-*"))
    elif "data-file" in session.posargs:
        # set the coverage input coverage datafile name, if provided in the command line with the "data-file" key word
        # ex: nox -s coverage_report -- data-file .coverage-out
        cov_file = session.posargs[session.posargs.index("data-file") + 1]

    session.run("coverage", "xml", f"--data-file={cov_file}", "-o", f"{REPORTS_DIR}/py{cov_file.lstrip('.')}.xml")
    session.run("coverage", "report", "-m", f"--data-file={cov_file}")


@nox.session
def lint(session):
    """Target to lint the code with flake8."""
    # Install base python dependencies
    base_dependencies(session)
    session.run("python3", "-m", "pip", "install",
                "flake8",
                "flake8-docstrings",
                "flake8-use-fstring",
                "flake8-variables-names",
                "pep8-naming")
    session.run("flake8", "--config", FLAKE8_CFG)


@nox.session
def import_sort(session):
    """Target to sort automatically the import in the code with isort."""
    # Install base python dependencies
    base_dependencies(session)
    session.run("python3", "-m", "pip", "install", "isort")
    if "check-only" in session.posargs:
        session.run("isort", "--profile", "black", "--check-only", ".")
    else:
        session.run("isort", "--profile", "black", ".")


@nox.session
def black(session):
    """Target to reformat automatically the code with black."""
    # Install base python dependencies
    base_dependencies(session)
    session.run("python3", "-m", "pip", "install", "black")
    if "check-only" in session.posargs:
        session.run("black", "--check", ".")
    else:
        session.run("black", ".")


@nox.session
def docs(session):
    """Target to build the documentation (not yet implemented)."""
    raise NotImplementedError("This target is not yet implemented.")


@nox.session
def generate_synthetic_data(session):
    """Target to generate synthetic data related to the use case."""
    # Install python dependencies
    dev_dependencies(session)

    # Install requirement related to synthetic data generation
    req_file = "ci/requirements_data.txt"
    if os.path.isfile(req_file):
        session.run("python3", "-m", "pip", "install","-r", req_file)

    # Generate the data
    session.run("python3", "ci/generate_synthetic_data.py")
    

@nox.session
def download_data(session):
    """Target to download the data required to train the use-case models (not yet implemented).""" 
    raise NotImplementedError("This target is not yet implemented.")


@nox.session
def train_test(session):
    """Target to launch a basic training of the use-case (not yet implemented)."""
    # Generate the synthetic dataset required for the functional tests
    generate_synthetic_data(session)

    # Run te functional tests with coverage
    coverage_install(session)
    session.run('rm','-rf', COV_FT_FILE, external=True)
    session.run('bash', './ci/run.sh', '--runner', f'coverage run --append --data-file={COV_FT_FILE}', external=True)
    if "clean_data" in session.posargs:
        session.run('rm','-rf', './data')
    session.notify("coverage_report", ['data-file', f'{COV_FT_FILE}'])
