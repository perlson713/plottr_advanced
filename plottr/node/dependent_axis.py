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

The photon number spans decades, so the axis wants to be logarithmic.  The
plot's own ``Scale`` does that properly (ticks at 10^4, 10^5, ...), so this
node leaves the values alone by default.  ``Logarithmic`` here is the older
way: it emits ``log10`` of the abscissa, which reads 4, 5, 6 on a linear axis.
Use one or the other -- with both, the logarithm is taken twice.

This module contains:

* :class:`.DependentAsAxis` -- the node.
* :class:`.DependentAsAxisWidget` -- its node widget.

The node also takes the line attenuation.  ``line_attenuation_db`` is typed in
by hand -- it cannot be read off a trace -- so it is the one number that is
regularly forgotten or left over from another fridge wiring, and without it the
measurement script writes ``photon`` as NaN.  Nothing has to be measured again:
the photon number follows from the saved ``fr``, ``Qc``, ``Qi`` and the drive
power, so entering the attenuation here recomputes the column.

There is still no physics here.  The recomputation is
``dataset_refit.recompute_photons`` in the measurement repository (which calls
``resonator_tools`` for the formula), the same function ``refit_dataset.py``
uses offline; this node only passes the number the operator typed.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from plottr import QtWidgets, Signal, Slot
from ..data.datadict import DataDict, DataDictBase
from ..gui.widgets import FormLayoutWrapper
from .node import Node, NodeWidget, updateOption
from .resonator import resonator_fit_module

__all__ = ['DependentAsAxis', 'DependentAsAxisWidget',
           'swapDependentToAxis']


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
                ('Line attenuation (dB)', QtWidgets.QLineEdit()),
                ('Resonator type', QtWidgets.QComboBox()),
                ('Status', QtWidgets.QLabel('')),
            ],
        )
        self.enabled = self.elements['Use a dependent as x axis']
        self.abscissa = self.elements['x axis']
        self.logAbscissa = self.elements['Logarithmic']
        self.attenuation = self.elements['Line attenuation (dB)']
        self.portType = self.elements['Resonator type']
        self.status = self.elements['Status']

        self.portType.addItem('reflection (circulator)', 'reflection')
        self.portType.addItem('notch (hanger)', 'notch')
        self.portType.setToolTip(
            'Which resonator this was, for the recomputation above.  It is not '
            'cosmetic: from resonator-tools 2.2.0 the notch photon number '
            'carries half the coefficient of the reflection one, so the wrong '
            'choice is wrong by a factor of two.')

        self.abscissa.addItem('(automatic)', '')
        self.attenuation.setPlaceholderText('as measured')
        self.attenuation.setToolTip(
            'Total attenuation between the VNA source and the device, in dB.  '
            'Leave it empty to keep the photon number as it was measured; type '
            'a number to recompute it, which is what to do when '
            'line_attenuation_db was forgotten or wrong.  Nothing needs to be '
            'measured again: the photon number follows from the saved fr, Qc, '
            'Qi and the drive power.')
        self.logAbscissa.setToolTip(
            'Emit log10 of the abscissa instead of the abscissa itself, which '
            'puts a decade at every unit of a linear axis.  Prefer the plot '
            "toolbar's Scale -> x axis -> Log: it keeps the values and labels "
            'the decades.  Using both takes the logarithm twice.')
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
            'attenuation': self.widget.attenuation.setText,
            'portType': self.setPortType,
        }
        self.optGetters = {
            'enabled': self.widget.enabled.isChecked,
            'abscissa': self.getAbscissa,
            'logAbscissa': self.widget.logAbscissa.isChecked,
            'attenuation': self.widget.attenuation.text,
            'portType': lambda: str(self.widget.portType.currentData()),
        }

        self.widget.enabled.toggled.connect(lambda: self.signalOption('enabled'))
        self.widget.abscissa.currentIndexChanged.connect(
            lambda: self.signalOption('abscissa'))
        self.widget.logAbscissa.toggled.connect(
            lambda: self.signalOption('logAbscissa'))
        # editingFinished, not textChanged: recomputing on every keystroke would
        # re-run the whole flowchart for '7', '70', '70.'.
        self.widget.attenuation.editingFinished.connect(
            lambda: self.signalOption('attenuation'))
        self.widget.portType.currentIndexChanged.connect(
            lambda: self.signalOption('portType'))

        if node is not None:
            node.candidatesChanged.connect(self.setAbscissaOptions)
            node.statusChanged.connect(self.widget.status.setText)

    def setPortType(self, value: str) -> None:
        index = self.widget.portType.findData(value)
        if index >= 0:
            self.widget.portType.setCurrentIndex(index)

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

    Recomputing the photon number from a re-entered line attenuation is
    independent of the swap: with ``enabled`` off and an attenuation typed in,
    the dataset keeps its shape and only ``photon`` changes.

    :Options:
        - ``enabled``: do the swap at all.
        - ``abscissa``: the dependent to use as the x axis.  Empty means
          automatic, which looks for ``photon``.
        - ``logAbscissa``: emit ``log10`` of the abscissa instead of the
          abscissa itself.  Off by default: the plot's ``Scale`` makes a
          proper log axis out of the values themselves.
        - ``attenuation``: line attenuation in dB, as text.  Empty leaves the
          photon number as it was measured.
        - ``portType``: ``'reflection'`` or ``'notch'``, for that
          recomputation.  It changes the answer by a factor of two.
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
        self._logAbscissa = False
        self._attenuation = ''
        self._portType = 'reflection'
        #: line attenuation actually applied on the last pass, for the label
        self._appliedAttenuation: Optional[float] = None
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

    @property
    def attenuation(self) -> str:
        return self._attenuation

    @attenuation.setter
    @updateOption('attenuation')
    def attenuation(self, value: str) -> None:
        self._attenuation = str('' if value is None else value).strip()

    @property
    def portType(self) -> str:
        return self._portType

    @portType.setter
    @updateOption('portType')
    def portType(self, value: str) -> None:
        self._portType = str(value or 'reflection')

    def process(self, dataIn: Optional[DataDictBase] = None) \
            -> Optional[Dict[str, Optional[DataDictBase]]]:
        if dataIn is None:
            return None

        data, notes = self._recomputePhotons(dataIn)

        candidates = _candidates(data)
        if candidates != self._candidates:
            self._candidates = candidates
            self.candidatesChanged.emit(candidates)

        if not self._enabled:
            self.statusChanged.emit('; '.join(notes))
            return dict(dataOut=data)

        abscissa = _pick(data, self._abscissa)
        if not abscissa:
            notes.append(
                'No dependent can be used as an axis here.  The node needs two '
                'or more dependents that share the same axes -- for example '
                '`photon` and `Qi`, both against `power`.')
            self.statusChanged.emit('; '.join(notes))
            return dict(dataOut=data)

        try:
            dataOut, message = self._swap(data, abscissa)
        except Exception as exc:  # noqa: BLE001 -- never take the viewer down
            notes.append(f'{type(exc).__name__}: {exc}')
            self.statusChanged.emit('; '.join(notes))
            return dict(dataOut=data)

        notes.append(message)
        self.statusChanged.emit('; '.join(note for note in notes if note))
        return dict(dataOut=dataOut if dataOut is not None else data)

    def _recomputePhotons(self, data: DataDictBase) \
            -> Tuple[DataDictBase, List[str]]:
        """Redo the photon number with the attenuation that was typed in.

        Returns the data (unchanged when the field is empty or the
        recomputation is not possible) and the notes to show in the status
        line.  It never raises: a bad number should say so, not take the
        viewer down.
        """
        self._appliedAttenuation = None
        text = self._attenuation
        if not text:
            return data, []
        try:
            attenuation = float(text)
        except ValueError:
            return data, [f'Line attenuation: `{text}` is not a number in dB.']

        module, source = resonator_fit_module()
        if module is None:
            return data, [f'Line attenuation: {source}']
        if not hasattr(module, 'recompute_photons'):
            # An older checkout of the measurement repository next to a newer
            # plottr.  Say which one is behind rather than raising AttributeError.
            return data, [
                'Line attenuation: this version of dataset_refit.py cannot '
                f'recompute the photon number ({source}).  Update the '
                'measurement repository (qcodes_measurement).']
        try:
            out, message = module.recompute_photons(
                data, attenuation, port_type=self._portType)
        except Exception as exc:  # noqa: BLE001 -- never take the viewer down
            return data, [f'Line attenuation: {type(exc).__name__}: {exc}']
        self._appliedAttenuation = attenuation
        return out, [message]

    def _swap(self, data: DataDictBase, abscissa: str) \
            -> Tuple[Optional[DataDictBase], str]:
        """Build a dataset with ``abscissa`` as its only axis."""
        return swapDependentToAxis(
            data, abscissa, log=self._logAbscissa,
            note=(None if self._appliedAttenuation is None
                  else f'{self._appliedAttenuation:g} dB'))


def swapDependentToAxis(data: DataDictBase, abscissa: str, log: bool = True,
                        note: Optional[str] = None) \
        -> Tuple[Optional[DataDictBase], str]:
    """Build a dataset with ``abscissa`` (a dependent) as its only axis.

    Free function rather than a method because the same transformation has to
    be applied to datasets that are joined in later (``dataset_join``): a file
    added to a plot whose x axis is ``log10_photon`` has to arrive in that same
    shape, and doing it twice would be two things to keep in step.

    :param abscissa: the dependent to turn into the axis.
    :param log: emit ``log10`` of it, under the name ``log10_<abscissa>``.
    :param note: appended to the axis label in parentheses-free form, e.g. the
        line attenuation the photon number was computed with.
    :return: the new dataset (``None`` if nothing could be plotted) and a
        one-line description of what happened.
    """
    axes = list(data.axes(abscissa))
    partners = [name for name in data.dependents()
                if name != abscissa and list(data.axes(name)) == axes]

    # One row per point of the underlying axes.  `expand()` puts the dataset in
    # one-record-per-row form whether it came from a ddh5 (where a record is a
    # whole trace) or not; the per-power values are then simply repeated, so
    # keep the first row of each group.
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
    if note:
        # Say it on the axis, not only in the status line.  Ten dB moves
        # log10(photon) by exactly one, and the plot rescales itself, so the
        # curve looks unchanged and only the tick labels move -- which reads as
        # "typing the attenuation did nothing".  It also keeps the assumption
        # with the figure when it is saved or shown to someone.
        label = f'{label} @ {note}'
    if log:
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.log10(x)
        name = f'log10_{abscissa}'
        label = f'log10({label})'
        unit = ''

    # A point with no abscissa cannot be placed.  Points whose *dependents* are
    # NaN are kept: a power where the fit failed is a real gap, and hiding it
    # would misrepresent the sweep.
    good = np.isfinite(x)
    dropped = int((~good).sum())
    if not good.any():
        reason = (f'`{abscissa}` is NaN at every point.'
                  if dropped == x.size else '')
        if abscissa == 'photon':
            reason += (' The measurement script can only compute the photon '
                       'number when `line_attenuation_db` is set (in the '
                       'measurement config, or in the setup file); without '
                       'it the column is all NaN.  Type the attenuation in '
                       'dB into this node to recompute it -- the '
                       'measurement does not have to be repeated.')
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
    # Carry the dataset's meta over: `title` is what the plot puts above the
    # figure and what names the dataset when several are compared, and a
    # rebuilt dataset that drops it loses both.
    for key, value in data.meta_items():
        out.add_meta(key, value)
    out.validate()

    summary = f'{len(partners)} dependents against `{name}` ({good.sum()} points)'
    if dropped:
        summary += f'; {dropped} dropped where `{abscissa}` was not finite'
    return out, summary
