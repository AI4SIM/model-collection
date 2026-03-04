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
import glob
import nox
from configparser import ConfigParser

try:
    import uv
except ImportError:
    raise ImportError("uv not found. Please install it in your environment, with 'pip install uv'")

REPORTS_DIR = ".ci-reports/"
COV_FT_FILE = ".coverage-ft"
COV_UT_FILE = ".coverage-ut"
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAKE8_CFG = os.path.join(ROOT_PATH, 'tools', 'flake8', 'flake8.cfg')

# The list of the default targets executed with the simple command "nox".
nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv"
PYPROJECT = nox.project.load_toml("pyproject.toml")
PYTHON_VERSIONS = nox.project.python_versions(PYPROJECT)

def _wheel_version(wheel: str, req_file: str = 'requirements.txt') -> str:
    """Extract the version of a wheel from a requirement.txt, if it is present.

    Args:
        req_file (str): path of the input requirement.txt file.

    Returns:
        (str): the pip version extracted from the requirement.txt file if torch is present,
            an empty string otherwise.
    """
    version = wheel
    for lib in PYPROJECT["project"]["dependencies"]:
        if f"{wheel}==" in lib:
            version = lib.rstrip()
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
    version = _wheel_version("torch").split('==')[1]
    if "+" in version:
        version, cuda = version.split('+')
    return version, cuda


def _build_mypy_config():
    """Build the mypy configuration file by combining common and model specific configs."""
    mypy_common_cfg = os.path.join(ROOT_PATH, 'tools', 'mypy', 'mypy.ini')
    mypy_config = ConfigParser()
    mypy_config.read([mypy_common_cfg, "mypy.ini"])
    with open("mypy_full.ini", "w") as mypy_file:
        # Write the full mypy configuration
        mypy_config.write(mypy_file)


@nox.session(python=PYTHON_VERSIONS)
def dev_dependencies(session):
    """Target to install all requirements of the use-case code."""
    torch_vers, cuda_vers = _torch_version()

    additional_url = ''
    extra_url = ''
    if torch_vers:
        # Set the url of precompiled torch version depending on CUDA version
        if cuda_vers != "cpu":
            extra_url = f"https://download.pytorch.org/whl/{cuda_vers}"

        # Set the url of precompiled torch-geometric dependencies depending on torch version
        additional_url = f"https://data.pyg.org/whl/torch-{torch_vers}+{cuda_vers}.html"

    # Install use-case python dependencies
    session.run("uv", "sync", "--active",
                "-f", additional_url,
                "--extra-index-url", extra_url)
    if "purge" in session.posargs:
        session.run("uv", "cache", "clean")


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Target to run unit tests on the code with pytest, and generate coverage report."""
    session.run('rm','-rf', COV_UT_FILE, external=True)
    session.run("uv", "run", "--active", "--group", "tests", "-m", "pytest", "--cache-clear", "--cov=./", "-v")
    session.run('mv','.coverage', COV_UT_FILE, external=True)
    session.notify("coverage_report", ['data-file', f'{COV_UT_FILE}'])


@nox.session(python=PYTHON_VERSIONS)
def coverage_report(session):
    """Target to generate coverage report from test results."""
    uv_run = ["uv", "run", "--active", "--only-group", "coverage"]
    cov_file=".coverage"
    if "combine" in session.posargs:
        session.run(*uv_run, "coverage", "combine", "--keep", "--append", *glob.glob(".coverage-*"))
    elif "data-file" in session.posargs:
        # set the coverage input coverage datafile name, if provided in the command line with the "data-file" key word
        # ex: nox -s coverage_report -- data-file .coverage-out
        cov_file = session.posargs[session.posargs.index("data-file") + 1]

    session.run(*uv_run, "coverage", "xml", f"--data-file={cov_file}", "-o", f"{REPORTS_DIR}/py{cov_file.lstrip('.')}.xml")
    session.run(*uv_run, "coverage", "report", "-m", f"--data-file={cov_file}")


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    """Target to lint the code with flake8."""
    uv_run = ["uv", "run", "--active", "--only-group", "lint"]
    session.run(*uv_run, "flake8", "--config", FLAKE8_CFG)


@nox.session(python=PYTHON_VERSIONS)
def import_sort(session):
    """Target to sort automatically the import in the code with isort."""
    uv_run = ["uv", "run", "--active", "--only-group", "isort"]
    if "check-only" in session.posargs:
        session.run(*uv_run, "isort", "--profile", "black", "--check-only", ".")
    else:
        session.run(*uv_run, "isort", "--profile", "black", ".")


@nox.session(python=PYTHON_VERSIONS)
def black(session):
    """Target to reformat automatically the code with black."""
    uv_run = ["uv", "run", "--active", "--only-group", "black"]
    if "check-only" in session.posargs:
        session.run(*uv_run, "black", "--check", ".")
    else:
        session.run(*uv_run, "black", ".")


@nox.session(python=PYTHON_VERSIONS)
def mypy(session):
    """Target to check type hint with mypy."""
    # Install base python dependencies
    _build_mypy_config()
    uv_run = ["uv", "run", "--active", "--group", "mypy"]
    try:
        session.run(*uv_run, "mypy", "--config-file", "mypy_full.ini", "--explicit-package-bases", ".")
    finally:
        session.run("rm", "mypy_full.ini", external=True)


@nox.session(python=PYTHON_VERSIONS)
def docs(session):
    """Target to build the documentation (not yet implemented)."""
    raise NotImplementedError("This target is not yet implemented.")


@nox.session(python=PYTHON_VERSIONS)
def generate_synthetic_data(session):
    """Target to generate synthetic data related to the use case."""
    # Generate the data
    session.run("uv", "run", "--active", "--group", "data", "ci/generate_synthetic_data.py")
    

@nox.session(python=PYTHON_VERSIONS)
def download_data(session):
    """Target to download the data required to train the use-case models (not yet implemented).""" 
    raise NotImplementedError("This target is not yet implemented.")


@nox.session(python=PYTHON_VERSIONS)
def train_test(session):
    """Target to launch a basic training of the use-case (not yet implemented)."""
    # Generate the synthetic dataset required for the functional tests
    generate_synthetic_data(session)

    # Run the functional tests with coverage
    session.run('rm','-rf', COV_FT_FILE, external=True)
    session.run('bash', './ci/run.sh', '--runner', f'uv run --group coverage --active coverage run --append --data-file={COV_FT_FILE}', external=True)
    if "clean_data" in session.posargs:
        session.run('rm','-rf', './data')
    session.notify("coverage_report", ['data-file', f'{COV_FT_FILE}'])
