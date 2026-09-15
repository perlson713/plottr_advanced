"""A node that fits resonator traces and adds the fitted curve to the data.

The point of this node is to be able to look at a measurement that was taken
*before* the measurement script started saving fit curves: give it a dataset
that contains a complex trace against frequency, and it adds the fitted model
as a new dependent, on the same axes, so both can be plotted on top of each
other.

This module contains:

* :class:`.ResonatorFit` -- the node.
* :class:`.ResonatorFitWidget` -- its node widget.

The physics lives in ``resonator_fit.py`` of the measurement repository
(``qcodes_measurement``), not here.  That module is the one that is validated
against synthetic traces, so this node imports it rather than growing a second
implementation that could silently drift.  It is found

1. through the ``RESONATOR_FIT_PATH`` environment variable (a directory),
2. as a plain ``import resonator_fit`` (i.e. anywhere on ``PYTHONPATH``),
3. next to the plottr checkout, in ``../qcodes_measurement``.

If it cannot be found, the node passes the data through untouched and says so
in its widget; nothing else in plottr depends on it.
"""

import os
import sys
import time
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from plottr import QtWidgets, Signal, Slot
from ..data.datadict import DataDictBase
from ..gui.widgets import FormLayoutWrapper
from .node import Node, NodeWidget, updateOption

__all__ = ['ResonatorFit', 'ResonatorFitWidget', 'resonator_fit_module']


#: cached result of :func:`resonator_fit_module`
_MODULE: Optional[ModuleType] = None
_MODULE_ERROR: str = ''


def resonator_fit_module() -> Tuple[Optional[ModuleType], str]:
    """Import the measurement repository's ``dataset_refit`` module.

    That module does the dataset-shaped part of the work (split into one trace
    per power, fit, put the curves back), and calls ``resonator_fit.py`` for the
    physics.  The same module is what ``refit_dataset.py`` uses to write a
    fitted copy of a dataset offline, so the viewer and the offline tool cannot
    drift apart.

    :return: the module (or ``None``) and a message describing where it came
        from, or why it could not be imported.
    """
    global _MODULE, _MODULE_ERROR
    if _MODULE is not None:
        return _MODULE, _MODULE_ERROR

    candidates: List[Path] = []
    from_env = os.environ.get('RESONATOR_FIT_PATH')
    if from_env:
        candidates.append(Path(from_env).expanduser())
    # ../qcodes_measurement, relative to the plottr checkout.  This one only
    # works from a source checkout: installed into site-packages, its parent is
    # site-packages, and the environment variable is the way in.
    candidates.append(Path(__file__).resolve().parents[2].parent / 'qcodes_measurement')

    found: List[Path] = []
    for path in candidates:
        if (path / 'dataset_refit.py').is_file():
            found.append(path)
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

    try:
        _MODULE = import_module('dataset_refit')
        _MODULE_ERROR = f"using {getattr(_MODULE, '__file__', '?')}"
    except Exception as exc:
        _MODULE = None
        _MODULE_ERROR = _whyNotImported(exc, candidates, found, from_env)
    return _MODULE, _MODULE_ERROR


def _whyNotImported(exc: BaseException, candidates: List[Path],
                    found: List[Path], from_env: Optional[str]) -> str:
    """Explain a failed import in terms the operator can act on.

    The two failures look identical in the traceback but need opposite fixes,
    and saying only "set RESONATOR_FIT_PATH" sends the operator round in
    circles when the path was right all along:

    * the folder is wrong, or has no ``dataset_refit.py`` (an old checkout) --
      a path problem;
    * the file is there but importing it raises, almost always because the
      Python that runs plottr does not have the fit's dependencies (numpy,
      scipy, resonator-tools) -- a package problem in *that* interpreter,
      which may not be the one the measurement scripts run under.
    """
    detail = f'{type(exc).__name__}: {exc}'

    if found:
        missing = getattr(exc, 'name', None) if isinstance(
            exc, ImportError) else None
        advice = (
            f'Install it into the Python that runs plottr ({sys.executable}), '
            f'e.g. `pip install {"resonator-tools" if missing == "resonator_tools" else missing}`.'
            if missing else
            'Run it by hand to see the full traceback: '
            f'`{sys.executable} -c "import dataset_refit"` '
            f'with RESONATOR_FIT_PATH set.')
        return (f'Found {found[0] / "dataset_refit.py"}, but importing it '
                f'failed ({detail}).  This is not a path problem.  {advice}')

    def describe(path: Path) -> str:
        if not path.is_dir():
            return f'{path} (no such folder)'
        return f'{path} (folder exists, but no dataset_refit.py in it)'

    looked = '; '.join(describe(path) for path in candidates) or 'nowhere'
    return (
        f'dataset_refit.py was not found ({detail}).  Looked in: {looked}.  '
        'Set the RESONATOR_FIT_PATH environment variable to the folder that '
        'contains dataset_refit.py (the qcodes_measurement checkout) and '
        'restart plottr.'
        + ('' if from_env else
           '  RESONATOR_FIT_PATH is not set in this process.'))


def _fittable_dependents(data: Optional[DataDictBase]) -> List[str]:
    """Complex dependents that can be fitted; ``[]`` without the fit module."""
    module, _ = resonator_fit_module()
    if module is None or data is None:
        return []
    names: List[str] = module.fittable_dependents(data)
    return names


class _ResonatorFitOptionsWidget(FormLayoutWrapper):
    """Form with the settings of the fit."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(
            parent=parent,
            elements=[
                ('Fit and overlay', QtWidgets.QCheckBox()),
                ('Trace', QtWidgets.QComboBox()),
                ('Resonator type', QtWidgets.QComboBox()),
                ('Nonlinearity', QtWidgets.QCheckBox(
                    'slow: seconds to a minute per trace')),
                ('Sweep direction', QtWidgets.QComboBox()),
                ('Status', QtWidgets.QLabel('')),
            ],
        )
        self.enabled = self.elements['Fit and overlay']
        self.dependent = self.elements['Trace']
        self.portType = self.elements['Resonator type']
        self.nonlinear = self.elements['Nonlinearity']
        self.sweepBranch = self.elements['Sweep direction']
        self.status = self.elements['Status']

        self.dependent.addItem('(automatic)', '')
        self.portType.addItem('reflection (circulator)', 'reflection')
        self.portType.addItem('notch (hanger)', 'notch')
        self.sweepBranch.addItem('low to high', 'up')
        self.sweepBranch.addItem('high to low', 'down')
        self.nonlinear.setToolTip(
            'Fit the kinetic-inductance nonlinearity of Swenson et al. '
            '(arXiv:1305.4281).  The fit runs in the GUI thread, so the window '
            'does not respond while it works: measured 5-25 s per trace.')
        self.status.setWordWrap(True)


class ResonatorFitWidget(NodeWidget):
    """Node widget for :class:`.ResonatorFit`."""

    def __init__(self, node: Optional[Node] = None):
        super().__init__(embedWidgetClass=_ResonatorFitOptionsWidget, node=node)
        self.widget: _ResonatorFitOptionsWidget
        assert self.widget is not None

        self.optSetters = {
            'enabled': self.widget.enabled.setChecked,
            'dependent': self.setDependent,
            'portType': self.setPortType,
            'nonlinear': self.widget.nonlinear.setChecked,
            'sweepBranch': self.setSweepBranch,
        }
        self.optGetters = {
            'enabled': self.widget.enabled.isChecked,
            'dependent': self.getDependent,
            'portType': lambda: str(self.widget.portType.currentData()),
            'nonlinear': self.widget.nonlinear.isChecked,
            'sweepBranch': lambda: str(self.widget.sweepBranch.currentData()),
        }

        self.widget.enabled.toggled.connect(lambda: self.signalOption('enabled'))
        self.widget.dependent.currentIndexChanged.connect(
            lambda: self.signalOption('dependent'))
        self.widget.portType.currentIndexChanged.connect(
            lambda: self.signalOption('portType'))
        self.widget.nonlinear.toggled.connect(lambda: self.signalOption('nonlinear'))
        self.widget.sweepBranch.currentIndexChanged.connect(
            lambda: self.signalOption('sweepBranch'))

        if node is not None:
            node.fittableDependentsChanged.connect(self.setDependentOptions)
            node.fitStatusChanged.connect(self.widget.status.setText)

    def getDependent(self) -> str:
        return str(self.widget.dependent.currentData() or '')

    def setDependent(self, value: str) -> None:
        index = self.widget.dependent.findData(value)
        if index < 0:
            self.widget.dependent.addItem(value, value)
            index = self.widget.dependent.findData(value)
        self.widget.dependent.setCurrentIndex(index)

    def setPortType(self, value: str) -> None:
        index = self.widget.portType.findData(value)
        if index >= 0:
            self.widget.portType.setCurrentIndex(index)

    def setSweepBranch(self, value: str) -> None:
        index = self.widget.sweepBranch.findData(value)
        if index >= 0:
            self.widget.sweepBranch.setCurrentIndex(index)

    @Slot(list)
    def setDependentOptions(self, names: List[str]) -> None:
        """Repopulate the trace selector when the dataset changes."""
        current = self.getDependent()
        combo = self.widget.dependent
        combo.blockSignals(True)
        combo.clear()
        combo.addItem('(automatic)', '')
        for name in names:
            combo.addItem(name, name)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)


class ResonatorFit(Node):
    """Fit each frequency trace in the data and add the fitted curve.

    With ``enabled`` set, the node adds up to three dependents per fitted
    trace ``<dep>``:

    ``<dep>_fit``
        the fitted model, in the same (raw) scale as the data, so it can be
        plotted straight on top of ``<dep>``.
    ``<dep>_cor``, ``<dep>_cor_fit``
        the calibrated trace and the model in the normalised scale the fit
        works in -- easier to read, because the cable delay is divided out.

    Fit parameters (``Ql``, ``Qi``, ``fr``, ``a``, ...) are added as well, on
    the remaining axes (e.g. power), whenever there is at least one.  They are
    prefixed with ``<dep>_fit_`` so they never collide with values that the
    measurement script already saved.

    Options are:

    :enabled: ``bool`` -- do the fitting at all.  Off by default, because
        fitting a whole power sweep takes a moment.
    :dependent: ``str`` -- which trace to fit; ``''`` picks the first complex
        dependent.
    :portType: ``str`` -- ``'reflection'`` (resonator behind a circulator) or
        ``'notch'`` (side-coupled to a feedline).
    :nonlinear: ``bool`` -- include the kinetic-inductance nonlinearity.
        Off by default: it costs seconds per trace.
    :sweepBranch: ``str`` -- ``'up'`` or ``'down'``; which branch of a
        bifurcated response the sweep traces out.
    """

    nodeName = 'ResonatorFit'
    useUi = True
    uiClass: Optional[Type[NodeWidget]] = ResonatorFitWidget
    uiVisibleByDefault = False

    #: emitted with a human-readable status of the last fit
    fitStatusChanged = Signal(str)

    #: emitted with the complex traces of the current dataset that can be fitted
    fittableDependentsChanged = Signal(list)

    def __init__(self, name: str) -> None:
        self._enabled = False
        self._dependent = ''
        self._portType = 'reflection'
        self._nonlinear = False
        self._sweepBranch = 'up'
        self._cache: Optional[Tuple[Any, DataDictBase]] = None
        self._candidates: List[str] = []
        super().__init__(name)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    @updateOption('enabled')
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    @property
    def dependent(self) -> str:
        return self._dependent

    @dependent.setter
    @updateOption('dependent')
    def dependent(self, value: str) -> None:
        self._dependent = str(value or '')

    @property
    def portType(self) -> str:
        return self._portType

    @portType.setter
    @updateOption('portType')
    def portType(self, value: str) -> None:
        self._portType = str(value)

    @property
    def nonlinear(self) -> bool:
        return self._nonlinear

    @nonlinear.setter
    @updateOption('nonlinear')
    def nonlinear(self, value: bool) -> None:
        self._nonlinear = bool(value)

    @property
    def sweepBranch(self) -> str:
        return self._sweepBranch

    @sweepBranch.setter
    @updateOption('sweepBranch')
    def sweepBranch(self, value: str) -> None:
        self._sweepBranch = str(value)

    # -- the work ---------------------------------------------------------- #

    def _fingerprint(self, data: DataDictBase, dependent: str) -> Any:
        """Cheap identity of (data, options), so we do not refit on every tick."""
        values = data.data_vals(dependent)
        return (
            dependent, values.shape, str(data.structure(include_meta=False)),
            float(np.nansum(np.abs(values))),
            self._portType, self._nonlinear, self._sweepBranch,
        )

    def process(self, dataIn: Optional[DataDictBase] = None) \
            -> Optional[Dict[str, Optional[DataDictBase]]]:
        processed = super().process(dataIn=dataIn)
        if processed is None:
            return None
        data = processed['dataOut']
        candidates = _fittable_dependents(data)
        if candidates != self._candidates:
            self._candidates = candidates
            self.fittableDependentsChanged.emit(candidates)
        if data is None or not self._enabled:
            return dict(dataOut=data)

        module, message = resonator_fit_module()
        if module is None:
            self.fitStatusChanged.emit(message)
            self.node_logger.warning(message)
            return dict(dataOut=data)

        dependent = self._dependent or module.default_dependent(candidates)
        if dependent not in candidates:
            self.fitStatusChanged.emit(
                f"'{dependent}' is not a complex trace in this dataset."
                if dependent else 'no complex trace to fit in this dataset.')
            return dict(dataOut=data)

        if self._cache is not None and self._cache[0] == self._fingerprint(data, dependent):
            return dict(dataOut=self._cache[1])

        started = time.monotonic()
        try:
            out, status = self._fit(module, data, dependent)
        except Exception as exc:
            status = f'{type(exc).__name__}: {exc}'
            self.node_logger.exception('resonator fit failed')
            self.fitStatusChanged.emit(f'fit failed -- {status}')
            return dict(dataOut=data)

        self._cache = (self._fingerprint(data, dependent), out)
        self.fitStatusChanged.emit(f'{status} ({time.monotonic() - started:.1f} s)')
        return dict(dataOut=out)

    def _fit(self, module: ModuleType, data: DataDictBase, dependent: str) \
            -> Tuple[DataDictBase, str]:
        """Fit every frequency trace in ``data`` and return a copy with the curves.

        All of this lives in the measurement repository's ``dataset_refit``
        module, so that the offline tool (``refit_dataset.py``) and this node
        produce exactly the same columns.
        """
        out, summary = module.fit_datadict(
            data,
            port_type=self._portType,
            nonlinear=self._nonlinear,
            sweep_branch=self._sweepBranch,
            dependent=dependent or None,
        )
        return out, summary
