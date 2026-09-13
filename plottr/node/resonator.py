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
from ..data.datadict import DataDict, DataDictBase
from ..gui.widgets import FormLayoutWrapper
from .node import Node, NodeWidget, updateOption

__all__ = ['ResonatorFit', 'ResonatorFitWidget', 'resonator_fit_module']


#: cached result of :func:`resonator_fit_module`
_MODULE: Optional[ModuleType] = None
_MODULE_ERROR: str = ''


def resonator_fit_module() -> Tuple[Optional[ModuleType], str]:
    """Import the measurement repository's ``resonator_fit`` module.

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
    # ../qcodes_measurement, relative to the plottr checkout
    candidates.append(Path(__file__).resolve().parents[2].parent / 'qcodes_measurement')

    for path in candidates:
        if (path / 'resonator_fit.py').is_file() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        _MODULE = import_module('resonator_fit')
        _MODULE_ERROR = f"using {getattr(_MODULE, '__file__', '?')}"
    except Exception as exc:
        _MODULE = None
        _MODULE_ERROR = (
            f"resonator_fit.py could not be imported ({type(exc).__name__}: {exc}). "
            "Point the RESONATOR_FIT_PATH environment variable at the folder "
            "that contains it."
        )
    return _MODULE, _MODULE_ERROR


def _frequency_axis(data: DataDictBase, dependent: str) -> Optional[str]:
    """The axis of ``dependent`` that looks like a frequency axis."""
    axes = data.axes(dependent)
    for name in axes:
        if 'freq' in name.lower():
            return name
    # 掃引軸の名前が違う場合の保険: 値の種類が一番多い軸を周波数軸とみなす。
    best, best_count = None, 0
    for name in axes:
        try:
            count = int(np.unique(data.data_vals(name)).size)
        except Exception:
            continue
        if count > best_count:
            best, best_count = name, count
    return best


def _fittable_dependents(data: Optional[DataDictBase]) -> List[str]:
    """Complex dependents that have at least one axis -- what we can fit."""
    if data is None:
        return []
    names = []
    for name in data.dependents():
        if '_fit' in name:
            continue  # フィット曲線そのものは対象にしない
        try:
            values = data.data_vals(name)
        except Exception:
            continue
        if np.iscomplexobj(values) and len(data.axes(name)) > 0:
            names.append(name)
    return names


def _default_dependent(candidates: List[str]) -> str:
    """Which trace to fit when the user has not picked one.

    Prefer the measured trace over an already calibrated copy of it: the fit
    does its own calibration, and the raw trace is what a dataset always has.
    """
    for name in candidates:
        if not name.endswith(('_cor', '_corrected', '_normalized', '_normalised')):
            return name
    return candidates[0] if candidates else ''


def _free_name(data: DataDictBase, name: str) -> str:
    """A field name that is not taken yet, so we never clobber saved data."""
    if name not in data:
        return name
    return f'{name}_refit'


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

        dependent = self._dependent or _default_dependent(candidates)
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
        """Fit every frequency trace in ``data`` and return a copy with the curves."""
        frequency = _frequency_axis(data, dependent)
        if frequency is None:
            return data, 'no frequency axis found.'

        # ddh5 から読んだ直後のデータは「1 レコード = 1 パワー、値は配列」の形なので、
        # 軸（power）の要素数と従属変数の要素数が違う。`expand()` で 1 点 1 行に
        # 展開してから扱う（下流のノードもこの形を前提にしている）。
        if isinstance(data, DataDict) and not data.is_expanded():
            if not data.is_expandable():
                return data, 'the dataset cannot be expanded into records.'
            data = data.expand()

        other_axes = [a for a in data.axes(dependent) if a != frequency]
        original = np.asarray(data.data_vals(dependent))
        values = original.reshape(-1).astype(complex)
        freqs = np.asarray(data.data_vals(frequency), dtype=float).reshape(-1)
        if freqs.size != values.size:
            return data, 'the trace and the frequency axis have different lengths.'

        # 周波数以外の軸の値の組で 1 本のトレースになる（パワー掃引ならパワーごと）。
        if other_axes:
            keys = np.stack([np.asarray(data.data_vals(a)).reshape(-1)
                             for a in other_axes], axis=-1)
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            groups = [np.flatnonzero(inverse == i) for i in range(unique.shape[0])]
        else:
            groups = [np.arange(values.size)]

        nan = np.nan + 1j * np.nan
        model_raw = np.full(values.size, nan, dtype=complex)
        corrected = np.full(values.size, nan, dtype=complex)
        model_cor = np.full(values.size, nan, dtype=complex)
        # フィットパラメータは、測定スクリプトの保存の仕方に合わせて、そのトレースの
        # 全点に同じ値を入れる（plottr の DataDict は列の長さを揃える必要がある）。
        parameters = {name: np.full(values.size, np.nan) for name in _PARAMETERS}
        n_ok = 0

        for index in groups:
            finite = np.isfinite(freqs[index]) & np.isfinite(values[index])
            index = index[finite]
            if index.size < 8:
                continue
            index = index[np.argsort(freqs[index])]
            f, z = freqs[index], values[index]

            analysis = module.analyse_trace(
                f, z, port_type=self._portType,
                nonlinear=self._nonlinear, sweep_branch=self._sweepBranch,
            )
            fit = analysis.fit
            if not fit.ok:
                continue
            for name in _PARAMETERS:
                parameters[name][index] = _as_float(getattr(fit, name, float('nan')))
            if fit.model is None:
                continue

            n_ok += 1
            port = getattr(fit, 'port', None)
            model = np.asarray(fit.model, dtype=complex)
            normalised = getattr(port, 'z_data', None) if port is not None else None
            if normalised is not None:
                normalised = np.asarray(normalised, dtype=complex)
                if normalised.shape == model.shape:
                    corrected[index] = normalised
                    model_cor[index] = model
            environment = module.calibration_from_port(port) if port is not None else None
            if environment is not None:
                environment = np.asarray(environment, dtype=complex)
                if environment.shape == model.shape:
                    model_raw[index] = environment * model

        out = data.copy()
        unit = data.get(dependent, {}).get('unit', '')
        axes = list(data.axes(dependent))
        shape = original.shape
        # 保存済みの列と名前がぶつかったら `_refit` を付ける。測定時のフィットを
        # 黙って上書きしない。
        fit_name = _free_name(data, f'{dependent}_fit')
        out[fit_name] = dict(values=model_raw.reshape(shape), axes=axes, unit=unit)
        out[_free_name(data, f'{dependent}_cor')] = dict(
            values=corrected.reshape(shape), axes=axes, unit='')
        out[_free_name(data, f'{dependent}_cor_fit')] = dict(
            values=model_cor.reshape(shape), axes=axes, unit='')

        # パラメータは周波数以外の軸に載せる（トレース 1 本なら軸が無いので付けない）。
        if other_axes:
            for name, column in parameters.items():
                if not np.any(np.isfinite(column)):
                    continue
                out[_free_name(data, f'{fit_name}_{name}')] = dict(
                    values=column.reshape(shape), axes=list(other_axes),
                    unit='Hz' if name.startswith('fr') else '')

        out.validate()
        status = (f'{n_ok}/{len(groups)} traces fitted, '
                  f"{'nonlinear' if self._nonlinear else 'linear'}, "
                  f'{self._portType}.')
        return out, status


#: fit results that are worth carrying into the dataset
_PARAMETERS = ('fr', 'fr_err', 'Ql', 'Ql_err', 'Qi', 'Qi_err', 'Qc',
               'absQc', 'absQc_err', 'phi0', 'a', 'a_err', 'bifurcated',
               'snr', 'chi_square')


def _as_float(value: Any) -> float:
    """Fit results can be ``None`` or a bool; the dataset wants a float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')
