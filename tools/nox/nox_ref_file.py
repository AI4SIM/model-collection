"""
This module proposes the list of generic Nox build targets for the different use cases.
This file is generic and aims at:
    - providing the common targets that can be used in all use cases,
    - defining a model of the different targets each use case should propose.
"""
from pathlib import Path
import nox

REPORTS_DIR = ".ci-reports/"

# The list of the default targets executed with the simple command "nox".
nox.options.sessions = ["lint", "test"]


def _create_file(file_path: str) -> Path:
    """Create the report directory if it does not exists.

    Args:
        file_path (str): the path of the file to be created.

    Returns:
        (pathlib.Path): the created file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _torch_version(req_file: str = 'requirements.txt') -> str:
    """Extract the torch version from a requirement.txt, if it is present.

    Args:
        req_file (str): path of the input requirement.txt file.

    Returns:
        (str): the torch version extracted from the requirement.txt file if torch is present,
            an empty string otherwise.
    """
    version = ''
    with open(req_file, encoding='utf-8') as file:
        for line in file.readlines():
            if "torch==" in line:
                version = line.rstrip().split('==')[1]
    return version


@nox.session
def dev_dependencies(session):
    """Target to install all requirements of the use-case code."""
    torch_vers = _torch_version()
    additional_url = ''
    if torch_vers:
        additional_url = f"https://data.pyg.org/whl/torch-{torch_vers}+cpu.html"
    session.install("--upgrade", "pip")
    session.install("-r", "requirements.txt",
                    "-f", additional_url)
    # FIXME: workaround because of incompatibility between setuptools >= 60 and torch < 1.11
    # cf. https://github.com/pytorch/pytorch/pull/69904#issuecomment-1024338333
    session.install("setuptools==59.5.0")


@nox.session
def tests(session):
    """Target to run unit tests on the code with pytest, and generate coverage report."""
    dev_dependencies(session)
    session.install("pytest-cov")
    session.run("python", "-m", "pytest", "--cache-clear", "--cov=./")
    session.notify("coverage_report")


@nox.session
def coverage_report(session):
    """Target to generate coverage report from test results."""
    session.install("coverage")
    session.run("coverage", "xml", "-o", f"{REPORTS_DIR}/pycoverage.xml")


@nox.session
def lint(session):
    """Target to lint the code with flake8."""
    session.install("flake8")
    session.run("flake8", "--exclude", ".nox,tests/", "--exit-zero")


@nox.session
def docs(session):
    """Target to build the documentation."""
    raise NotImplementedError("This target is not yet implemented.")


@nox.session
def download_data(session):
    """Target to download the data required to train the use-case models."""
    raise NotImplementedError("This target is not yet implemented.")


@nox.session
def train(session):
    """Target to launch a basic training of the use-case."""
    raise NotImplementedError("This target is not yet implemented.")
