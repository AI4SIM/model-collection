"""
This module proposes the list of Nox build targets for the gwf use case.
"""
import nox

REPORTS_DIR = ".ci-reports"

# The list of the default target executed with the simple command "nox".
nox.options.sessions = ["lint", "test"]


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
    torch_vers= _torch_version()
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
def test(session, ):
    """Target to run unit tests on the code with pytest, and generate coverage report."""
    dev_dependencies(session)
    session.install("pytest")
    session.install("coverage")
    # FIXME: activate unit tests when test were written
    session.run("coverage", "run", "-m", "pytest")
    session.notify("coverage_report")


@nox.session
def coverage_report(session):
    """Target to generate coverage report from test reports."""
    session.install("coverage")
    session.run("coverage", "report")
    session.run("coverage", "xml", "-o", f"{REPORTS_DIR}/pycoverage-gwd.xml")


@nox.session
def lint(session):
    """Target to lint the code with flake8."""
    session.install("flake8")
    session.run("flake8", "--exclude", ".nox,tests/", "--exit-zero")


@nox.session
def doc(session):
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
