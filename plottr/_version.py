"""The version of plottr, wherever it happens to be installed from.

There are three ways this package gets used, and the version has to come from
a different place in each:

* copied into ``site-packages`` (which is how this fork is updated on the
  measurement PC: replace the folder) -- there is no project next to it, so the
  version comes from the installed distribution's metadata;
* a source checkout with git -- ``versioningit`` reads the tags and gives the
  exact commit being run, which is what a developer wants to see in the log;
* a copy with neither -- there is nothing to read, and that is not a reason to
  refuse to start.

Getting this wrong is not a cosmetic problem: it runs at import, so plottr does
not start at all.  ``plottr-monitr`` used to end in ``NoConfigFileError: No
pyproject.toml or versioningit.toml file in ...site-packages``.
"""

from pathlib import Path

#: Reported when the version cannot be determined at all.
UNKNOWN_VERSION = '0+unknown'


def _from_checkout() -> str:
    """Version of the checkout this file is part of, from its git tags."""
    import versioningit

    import plottr

    path = Path(plottr.__file__).parent
    return str(versioningit.get_version(project_dir=path.parent))


def _from_metadata() -> str:
    """Version recorded when the package was installed."""
    from importlib.metadata import version

    return str(version('plottr'))


def _get_version() -> str:
    for source in (_from_checkout, _from_metadata):
        try:
            found = source()
        except Exception:  # noqa: BLE001 -- any failure just means "try the next"
            continue
        if found:
            return found
    return UNKNOWN_VERSION


__version__ = _get_version()
