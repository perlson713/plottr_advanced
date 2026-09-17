"""Tests for how the version is determined.

This runs at import, so a failure here means plottr does not start at all.
The fork is updated on the measurement PC by replacing the `plottr` folder
inside site-packages, and that folder has no project next to it::

    versioningit.errors.NoConfigFileError: No pyproject.toml or
    versioningit.toml file in ...\\site-packages
"""

import plottr
from plottr import _version


def test_a_version_is_reported():
    assert isinstance(plottr.__version__, str) and plottr.__version__


def test_falls_back_when_there_is_no_project_next_to_the_package(monkeypatch):
    """The copied-into-site-packages case."""
    def noProject():
        raise RuntimeError('No pyproject.toml or versioningit.toml file')

    monkeypatch.setattr(_version, '_from_checkout', noProject)
    assert _version._get_version() == _version._from_metadata()


def test_never_raises_even_with_nothing_to_read(monkeypatch):
    def nothing():
        raise RuntimeError('nothing here')

    monkeypatch.setattr(_version, '_from_checkout', nothing)
    monkeypatch.setattr(_version, '_from_metadata', nothing)
    assert _version._get_version() == _version.UNKNOWN_VERSION
