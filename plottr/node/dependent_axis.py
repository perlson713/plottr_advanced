"""A node that turns a dependent into the x axis.

plottr plots dependents against *axes*.  A power sweep of a resonator saves
the photon number and the quality factors all as dependents of ``power``, so
"Qi against the photon number" -- the plot that shows two-level-system loss --
cannot be selected in the GUI: neither of the two is an axis.

This node makes one of them one.  Pick a dependent (by default it looks for
``photon``); every other dependent that lives on the same axes is re-attached
to it, and everything else is dropped, because it cannot be expressed against
the new axis.  Downstream, ``Data selection`` then offers ``Qi``, ``Ql``,
``Qc`` and the rest against the photon number, error bars included: the
``<dep>_err`` columns travel with their partners and keep being resolved by
``errorBarDataName()``.

The photon number spans decades, and plottr's plot widgets have no logarithmic
axis, so the node offers to take ``log10`` of the abscissa.  That is on by
default -- a linear axis from 1 to a million is not a plot anybody can read.

This module contains:

* :class:`.DependentAsAxis` -- the node.
* :class:`.DependentAsAxisWidget` -- its node widget.

There is no physics here: the node only re-arranges columns that the
measurement script already wrote.  In particular it does not compute the
photon number -- that needs the fit and the line attenuation, and it belongs
in the measurement repository, which is where it is.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from plottr import QtWidgets, Signal, Slot
from ..data.datadict import DataDict, DataDictBase
from ..gui.widgets import FormLayoutWrapper
from .node import Node, NodeWidget, updateOption

__all__ = ['DependentAsAxis', 'DependentAsAxisWidget']


#: Tried in this order when the abscissa is left on '(automatic)'.
PREFERRED = ('photon',)


def _candidates(data: Optional[DataDictBase]) -> List[str]:
    """Dependents that could serve as an axis.

    A dependent qualifies when at least one *other* dependent shares its axes;
    on its own it would produce a dataset with nothing to plot.
    """
    if data is None:
        return []
    try:
        dependents = data.dependents()
    except Exception:
        return []

    by_axes: Dict[Tuple[str, ...], List[str]] = {}
    for name in dependents:
        by_axes.setdefault(tuple(data.axes(name)), []).append(name)
    return [name for name in dependents
            if len(by_axes[tuple(data.axes(name))]) > 1]


def _pick(data: Optional[DataDictBase], wanted: str) -> str:
    """Resolve the abscissa, honouring '(automatic)'."""
    names = _candidates(data)
    if wanted:
        return wanted if wanted in names else ''
    for preferred in PREFERRED:
        if preferred in names:
            return preferred
    return ''


class _DependentAsAxisOptionsWidget(FormLayoutWrapper):
    """Form with the settings of the node."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(
            parent=parent,
            elements=[
                ('Use a dependent as x axis', QtWidgets.QCheckBox()),
                ('x axis', QtWidgets.QComboBox()),
                ('Logarithmic', QtWidgets.QCheckBox('plot log10(x) instead of x')),
                ('Status', QtWidgets.QLabel('')),
            ],
        )
        self.enabled = self.elements['Use a dependent as x axis']
        self.abscissa = self.elements['x axis']
        self.logAbscissa = self.elements['Logarithmic']
        self.status = self.elements['Status']

        self.abscissa.addItem('(automatic)', '')
        self.logAbscissa.setToolTip(
            "plottr's plot widgets have no logarithmic axis, so the node takes "
            'the logarithm itself.  The photon number spans decades, so this is '
            'on by default.')
        self.status.setWordWrap(True)


class DependentAsAxisWidget(NodeWidget):
    """Node widget for :class:`.DependentAsAxis`."""

    def __init__(self, node: Optional[Node] = None):
        super().__init__(embedWidgetClass=_DependentAsAxisOptionsWidget, node=node)
        self.widget: _DependentAsAxisOptionsWidget
        assert self.widget is not None

        self.optSetters = {
            'enabled': self.widget.enabled.setChecked,
            'abscissa': self.setAbscissa,
            'logAbscissa': self.widget.logAbscissa.setChecked,
        }
        self.optGetters = {
            'enabled': self.widget.enabled.isChecked,
            'abscissa': self.getAbscissa,
            'logAbscissa': self.widget.logAbscissa.isChecked,
        }

        self.widget.enabled.toggled.connect(lambda: self.signalOption('enabled'))
        self.widget.abscissa.currentIndexChanged.connect(
            lambda: self.signalOption('abscissa'))
        self.widget.logAbscissa.toggled.connect(
            lambda: self.signalOption('logAbscissa'))

        if node is not None:
            node.candidatesChanged.connect(self.setAbscissaOptions)
            node.statusChanged.connect(self.widget.status.setText)

    def getAbscissa(self) -> str:
        return str(self.widget.abscissa.currentData() or '')

    def setAbscissa(self, value: str) -> None:
        index = self.widget.abscissa.findData(value)
        if index < 0:
            self.widget.abscissa.addItem(value, value)
            index = self.widget.abscissa.findData(value)
        self.widget.abscissa.setCurrentIndex(index)

    @Slot(list)
    def setAbscissaOptions(self, names: List[str]) -> None:
        """Repopulate the selector when the dataset changes."""
        current = self.getAbscissa()
        combo = self.widget.abscissa
        combo.blockSignals(True)
        combo.clear()
        combo.addItem('(automatic)', '')
        for name in names:
            combo.addItem(name, name)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)


class DependentAsAxis(Node):
    """Re-attach dependents to one of their siblings, used as the x axis.

    With ``enabled`` off the data passes through untouched, which is the
    default: the node changes what the dataset *is*, so it should only do that
    when it was asked to.

    :Options:
        - ``enabled``: do the swap at all.
        - ``abscissa``: the dependent to use as the x axis.  Empty means
          automatic, which looks for ``photon``.
        - ``logAbscissa``: emit ``log10`` of the abscissa instead of the
          abscissa itself.
    """

    nodeName = 'DependentAsAxis'
    useUi = True
    uiClass: Optional[Type[NodeWidget]] = DependentAsAxisWidget

    #: names that could be used as the x axis, for the widget
    candidatesChanged = Signal(list)
    #: what the node did, or why it did nothing
    statusChanged = Signal(str)

    def __init__(self, name: str) -> None:
        self._enabled = False
        self._abscissa = ''
        self._logAbscissa = True
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
    def abscissa(self) -> str:
        return self._abscissa

    @abscissa.setter
    @updateOption('abscissa')
    def abscissa(self, value: str) -> None:
        self._abscissa = str(value or '')

    @property
    def logAbscissa(self) -> bool:
        return self._logAbscissa

    @logAbscissa.setter
    @updateOption('logAbscissa')
    def logAbscissa(self, value: bool) -> None:
        self._logAbscissa = bool(value)

    def process(self, dataIn: Optional[DataDictBase] = None) \
            -> Optional[Dict[str, Optional[DataDictBase]]]:
        if dataIn is None:
            return None

        candidates = _candidates(dataIn)
        if candidates != self._candidates:
            self._candidates = candidates
            self.candidatesChanged.emit(candidates)

        if not self._enabled:
            self.statusChanged.emit('')
            return dict(dataOut=dataIn)

        abscissa = _pick(dataIn, self._abscissa)
        if not abscissa:
            self.statusChanged.emit(
                'No dependent can be used as an axis here.  The node needs two '
                'or more dependents that share the same axes -- for example '
                '`photon` and `Qi`, both against `power`.')
            return dict(dataOut=dataIn)

        try:
            dataOut, message = self._swap(dataIn, abscissa)
        except Exception as exc:  # noqa: BLE001 -- never take the viewer down
            self.statusChanged.emit(f'{type(exc).__name__}: {exc}')
            return dict(dataOut=dataIn)

        self.statusChanged.emit(message)
        return dict(dataOut=dataOut if dataOut is not None else dataIn)

    def _swap(self, data: DataDictBase, abscissa: str) \
            -> Tuple[Optional[DataDictBase], str]:
        """Build a dataset with ``abscissa`` as its only axis."""
        axes = list(data.axes(abscissa))
        partners = [name for name in data.dependents()
                    if name != abscissa and list(data.axes(name)) == axes]

        # One row per point of the underlying axes.  `expand()` puts the
        # dataset in one-record-per-row form whether it came from a ddh5
        # (where a record is a whole trace) or not; the per-power values are
        # then simply repeated, so keep the first row of each group.
        work = data.expand()
        if axes:
            keys = np.stack(
                [np.asarray(work.data_vals(a)).flatten() for a in axes], axis=-1)
            _, index = np.unique(keys, axis=0, return_index=True)
            index = np.sort(index)
        else:
            index = np.arange(np.asarray(work.data_vals(abscissa)).size)

        x = np.asarray(work.data_vals(abscissa), dtype=float).flatten()[index]

        name = abscissa
        unit = data.get(abscissa, {}).get('unit', '')
        label = data.label(abscissa) or abscissa
        if self._logAbscissa:
            with np.errstate(divide='ignore', invalid='ignore'):
                x = np.log10(x)
            name = f'log10_{abscissa}'
            label = f'log10({label})'
            unit = ''

        # A point with no abscissa cannot be placed.  Points whose *dependents*
        # are NaN are kept: a power where the fit failed is a real gap, and
        # hiding it would misrepresent the sweep.
        good = np.isfinite(x)
        dropped = int((~good).sum())
        if not good.any():
            reason = (f'`{abscissa}` is NaN at every point.'
                      if dropped == x.size else '')
            if abscissa == 'photon':
                reason += (' The measurement script can only compute the photon '
                           'number when `line_attenuation_db` is set in the '
                           'setup file; without it the column is all NaN.')
            return None, ('Nothing to plot: ' + reason).strip()

        order = np.argsort(x[good], kind='stable')

        out = DataDict()
        out[name] = dict(values=x[good][order], axes=[], unit=unit, label=label)
        for partner in partners:
            values = np.asarray(
                work.data_vals(partner), dtype=float).flatten()[index]
            out[partner] = dict(
                values=values[good][order],
                axes=[name],
                unit=data.get(partner, {}).get('unit', ''),
                label=data.get(partner, {}).get('label', ''),
            )
        out.validate()

        note = f'{len(partners)} dependents against `{name}` ({good.sum()} points)'
        if dropped:
            note += f'; {dropped} dropped where `{abscissa}` was not finite'
        return out, note
