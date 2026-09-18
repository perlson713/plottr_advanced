"""A node that draws other datasets into the same plot.

Comparing measurements -- the same resonator after a change, several resonators
from one chip, the same device in two cooldowns -- means one plot with one
curve per dataset.  plottr shows one dataset at a time, so this node loads
further ddh5 files and merges them into the one being viewed.

The comparison this exists for is `Qi` against the photon number, and the
photon number is a *dependent* of the power, not an axis: it has exactly one
value per measured power.  So the node builds the x axis itself.  When it
reads a dataset it takes the compared column (`Qi`) together with that
dataset's `photon` column -- the same number of points, point for point -- and
writes a new column whose x values are the photon numbers instead of the
powers.  Every dataset is put through this, the one already open included, so
they all end up on one x axis and one plot.  `X axis` says which column to use;
emptying it keeps the power (or whatever the dataset's own axis is).

Nothing is interpolated or resampled.  Every dataset keeps its own rows: the
added dataset's dependents become ``<name> [<label>]`` and the rows of the
others are NaN, so a point that was never measured is a gap rather than an
invented value.

Because the x axis is built here, the ``Photon axis`` node upstream is not
needed for this plot -- and if it is switched on anyway, its axis is recognised
by name (``photon``, or ``log10_photon``) and used, so the two do not fight.

Only the columns being compared are brought over.  A power sweep saves some
thirty of them, and joining three datasets whole turns the `Data selection`
list into a hundred entries to hunt through for the one curve wanted.  The
``columns`` option says which to keep (``Qi`` by default, the quality factor
these comparisons are usually about); error bars follow their partner without
being asked for.  Emptying it brings everything, for the rare comparison of
something else.

What this node does *not* do is run the rest of the flowchart over the added
files: a file is taken as it was saved.  A dataset with no fit columns will
therefore show no fit curve -- fit it first with ``refit_dataset.py``.

This module contains:

* :class:`.JoinDatasets` -- the node.
* :class:`.JoinDatasetsWidget` -- its node widget.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from plottr import QtWidgets, Signal, Slot
from ..data.datadict import DataDict, DataDictBase
from ..data.datadict_storage import datadict_from_hdf5
from .dependent_axis import swapDependentToAxis
from .node import Node, NodeWidget, updateOption

__all__ = ['DEFAULT_ABSCISSA', 'DEFAULT_COLUMNS', 'JoinDatasets',
           'JoinDatasetsWidget', 'abscissaValues',
           'datasetLabel', 'datasetStamp', 'joinAxes', 'joinDatasets',
           'loadDataset',
           'parseColumns', 'seriesName', 'uniqueLabels', 'wantedColumns']

#: ddh5 group the measurement scripts write into.
GROUPNAME = 'data'

#: Columns brought over unless told otherwise.  Comparing datasets is about one
#: quantity at a time, and for a power sweep that is the internal quality
#: factor; everything else would only lengthen the selection list.
DEFAULT_COLUMNS = 'Qi'

#: Column used as the x axis unless told otherwise.  `Qi` against the photon
#: number is what these comparisons are; the photon number is a dependent of
#: the power with one value per power, so it can stand in for it point for
#: point.  Emptying the field keeps the dataset's own axis.
DEFAULT_ABSCISSA = 'photon'

#: An axis that a dependent was turned into by ``Photon axis``.
_LOG_AXIS = re.compile(r'^log10_(?P<name>.+)$')

#: ``<name>_err`` is resolved as the error bar of ``<name>``, so a label has to
#: go *before* that suffix to keep the pair together.
_ERROR_SUFFIX = '_err'


def datasetLabel(path: str | Path) -> str:
    """A short name for a dataset, for the legend.

    The folder of a measurement is named ``<timestamp>_<id>-<name>``, and the
    part that tells datasets apart is the name the operator gave it (the
    resonator, for our measurements).  Fall back to the folder name.
    """
    folder = Path(path)
    if folder.suffix:  # a file: the folder above it names the measurement
        folder = folder.parent
    name = folder.name
    if '-' in name:
        tail = name.rsplit('-', 1)[-1].strip()
        if tail:
            return tail
    return name


#: The timestamp a measurement folder starts with: 2026-09-18T120001_<id>-<name>.
_STAMP = re.compile(r'^(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})(?P<minute>\d{2})')


def datasetStamp(path: str | Path) -> str:
    """When a dataset was measured, short enough for a legend.

    Used to tell apart datasets that carry the same name -- the same resonator
    measured twice is exactly the comparison this node is for.
    """
    folder = Path(path)
    if folder.suffix:
        folder = folder.parent
    found = _STAMP.match(folder.name)
    if not found:
        return ''
    return f"{found['date'][5:]} {found['hour']}:{found['minute']}"


def uniqueLabels(paths: Sequence[str | Path], fallback: str = 'dataset') \
        -> List[str]:
    """One distinct label per dataset.

    Labels end up in column names, so two datasets sharing one would land in
    the same column: the second would overwrite the first, and a comparison of
    four datasets would quietly draw two curves.  Where names collide, the
    measurement time tells them apart; where even that is the same, a number
    does.
    """
    names = [datasetLabel(path) or fallback for path in paths]
    repeated = {name for name in names if names.count(name) > 1}

    labels: List[str] = []
    for path, name in zip(paths, names):
        if name in repeated:
            stamp = datasetStamp(path)
            name = f'{name} {stamp}'.strip() if stamp else name
        unique, number = name, 2
        while unique in labels:
            unique = f'{name} #{number}'
            number += 1
        labels.append(unique)
    return labels


def seriesName(name: str, label: str) -> str:
    """Name of ``name`` once it belongs to the dataset called ``label``."""
    if not label:
        return name
    if name.endswith(_ERROR_SUFFIX):
        return f'{name[:-len(_ERROR_SUFFIX)]} [{label}]{_ERROR_SUFFIX}'
    return f'{name} [{label}]'


def parseColumns(text: str) -> List[str]:
    """Column names out of what was typed: commas or spaces, either way."""
    return [name for name in re.split(r'[,\s]+', text or '') if name]


def wantedColumns(data: DataDictBase, columns: Sequence[str]) -> List[str]:
    """Which of this dataset's dependents to bring over.

    An empty ``columns`` means all of them.  A named column brings its error
    bars (``<name>_err``) along without being asked: they are drawn as part of
    the curve, and leaving them behind would silently drop them.
    """
    dependents = list(data.dependents())
    if not columns:
        return dependents

    keep: List[str] = []
    for name in columns:
        for candidate in (name, f'{name}{_ERROR_SUFFIX}'):
            if candidate in dependents and candidate not in keep:
                keep.append(candidate)
    return keep


def loadDataset(path: str | Path, groupname: str = GROUPNAME) -> DataDictBase:
    """Read a ddh5 file (or the folder holding one)."""
    file = Path(path)
    if file.is_dir():
        file = file / 'data.ddh5'
    return datadict_from_hdf5(str(file), groupname=groupname)


def reshapeLike(data: DataDictBase, axes: Sequence[str]) \
        -> Tuple[Optional[DataDictBase], str]:
    """Put ``data`` in the shape of a dataset with these axes.

    Only one case needs work: an axis built out of a dependent by
    ``Photon axis`` (``log10_photon``, or ``photon`` itself).  Everything else
    is aligned by axis name, which needs no transformation.
    """
    if len(axes) == 1 and axes[0] not in data.axes():
        wanted = axes[0]
        match = _LOG_AXIS.match(wanted)
        abscissa = match.group('name') if match else wanted
        if abscissa in data.dependents():
            return swapDependentToAxis(data, abscissa, log=match is not None)
        return None, (f'no `{abscissa}` in this dataset, so it cannot be put '
                      f'on a `{wanted}` axis')

    missing = [axis for axis in axes if axis not in data.axes()]
    if missing:
        return None, f'this dataset has no {", ".join(missing)} axis'
    return data, ''


def joinAxes(data: DataDictBase, columns: Sequence[str]) -> List[str]:
    """The axes the compared columns live on.

    `Qi` lives on `power`; the trace it was fitted from lives on `frequency`
    *and* `power`.  Joining on the dataset's full set of axes would put `Qi` on
    the frequency grid too -- one value repeated at every frequency -- and a
    six-resonator comparison of 5001-point sweeps becomes a million rows of
    which fifty are the actual points.  So the axes come from the columns being
    compared, not from the dataset.

    With no columns named (bring everything), there is no single answer -- the
    dataset holds quantities on different axes -- and the full set is used.
    """
    wanted = wantedColumns(data, columns)
    if not columns or not wanted:
        return list(data.axes())
    return list(data.axes(wanted[0]))


def _rowsOn(data: DataDictBase, axes: Sequence[str]) -> np.ndarray:
    """Indices of one row per point of ``axes``, in the order they appear."""
    size = np.asarray(data.data_vals(list(data.axes())[0])).size
    if set(axes) == set(data.axes()):
        return np.arange(size)
    keys = np.stack([np.asarray(data.data_vals(a)).flatten() for a in axes],
                    axis=-1)
    _, index = np.unique(keys, axis=0, return_index=True)
    return np.sort(index)


def abscissaValues(data: DataDictBase, axes: Sequence[str], wanted: str) \
        -> Tuple[Optional[np.ndarray], str]:
    """The x values ``wanted`` gives this dataset, one per row.

    The photon number has one value per measured power -- the same points as
    `Qi`, point for point -- so it can stand in for the power as the x column.
    That is all this does: read the column and hand back its values.

    It is also accepted as an axis, because the ``Photon axis`` node upstream
    may already have made it one; ``log10_photon`` means that node was asked
    for the logarithm, and then the same is done here so the datasets agree.

    :return: the values, or ``None`` and why not.
    """
    if not wanted:
        return None, ''

    match = _LOG_AXIS.match(wanted)
    base = match.group('name') if match else wanted

    known = list(data.axes()) + list(data.dependents())
    if wanted in known:            # already in the shape asked for
        source, takeLog = wanted, False
    elif base in known:            # the plain column; take log10 if asked
        source, takeLog = base, match is not None
    else:
        return None, f'no `{base}` column'

    if source in data.dependents() and set(data.axes(source)) != set(axes):
        return None, (f'`{source}` is on {", ".join(data.axes(source))}, '
                      f'not on {", ".join(axes)}')

    values = np.asarray(data.data_vals(source), dtype=float).flatten()
    if not np.any(np.isfinite(values)):
        # The usual cause: `line_attenuation_db` was never set, so the
        # measurement wrote `photon` as NaN for every power.
        return None, f'`{source}` is NaN everywhere'

    if takeLog:
        with np.errstate(divide='ignore', invalid='ignore'):
            values = np.log10(values)
    return values, ''


def _axisInfo(data: DataDictBase, name: str) -> Tuple[str, str]:
    """``(unit, label)`` of a column, empty where the dataset says nothing."""
    entry = data.get(name, {}) or {}
    return entry.get('unit', ''), entry.get('label', '')


def joinDatasets(parts: Sequence[Tuple[str, DataDictBase]],
                 columns: Sequence[str] = (),
                 abscissa: str = '') \
        -> Tuple[Optional[DataDictBase], str]:
    """Merge datasets into one, on a shared x axis.

    :param parts: ``(label, dataset)`` pairs.  A label of ``''`` leaves that
        dataset's names alone -- used for the dataset already being viewed when
        it is the only one, so that nothing changes until a file is added.
    :param columns: dependents to keep, empty for all.  Applied to every
        dataset, the one being viewed included: the joined dataset is the
        comparison, and a comparison of one quantity has no use for the other
        twenty-nine.
    :param abscissa: column to use as the x axis, ``photon`` for the plot this
        is for.  Each dataset's own values are taken, so the curves share an
        axis without anything being interpolated.  Empty, or not available in
        every dataset, falls back to the axis the datasets already share.
    :return: the merged dataset and a one-line summary.
    """
    parts = [(label, _expanded(data)) for label, data in parts
             if data is not None]
    if not parts:
        return None, 'nothing to join'
    if len(parts) == 1:
        return parts[0][1], ''

    notes: List[str] = []

    # Each dataset's own rows: the points of the axes its compared columns live
    # on.  This is what the x column has to line up with, and what gets
    # concatenated.
    ownAxes = [joinAxes(data, columns) for _, data in parts]
    rows = [_rowsOn(data, axes) for (_, data), axes in zip(parts, ownAxes)]

    # The x column.  Built out of a dependent (`photon`) where every dataset
    # has one; otherwise the axis the datasets already share.
    xColumns: List[Tuple[str, str, str, List[np.ndarray]]] = []
    if abscissa:
        chunks: List[np.ndarray] = []
        for number, ((label, data), axes) in enumerate(zip(parts, ownAxes)):
            values, why = abscissaValues(data, axes, abscissa)
            if values is None:
                notes.append(f'{label}: {why}')
                chunks = []
                break
            picked = values[rows[number]]
            # Points in x order, so the line does not double back when the
            # dataset was measured from the top power down.
            order = np.argsort(picked, kind='stable')
            rows[number] = rows[number][order]
            chunks.append(picked[order])
        if chunks:
            unit, label = _axisInfo(parts[0][1], abscissa)
            xColumns.append((abscissa, unit, label, chunks))

    if not xColumns:
        if abscissa:
            notes.append(f'falling back to {", ".join(ownAxes[0])}')
        axes = ownAxes[0]
        for (label, data), _ in zip(parts[1:], ownAxes[1:]):
            # The order does not have to match: a ddh5 read back can list the
            # axes the other way round, and each dataset's own columns stay
            # aligned with each other whatever the order is.
            if not set(axes) <= set(data.axes()):
                return None, (f'`{label}` has axes {", ".join(data.axes())}, '
                              f'not {", ".join(axes)}')
        rows = [_rowsOn(data, axes) for _, data in parts]
        ownAxes = [list(axes) for _ in parts]
        for name in axes:
            unit, label = _axisInfo(parts[0][1], name)
            xColumns.append((name, unit, label, [
                np.asarray(data.data_vals(name)).flatten()[index]
                for (_, data), index in zip(parts, rows)]))

    axisNames = [name for name, _, _, _ in xColumns]
    lengths = [int(index.size) for index in rows]
    total = int(sum(lengths))
    offsets = np.cumsum([0] + lengths)

    out = DataDict()
    for name, unit, label, chunks in xColumns:
        out[name] = dict(values=np.concatenate(chunks), axes=[],
                         unit=unit, label=label)

    dropped: List[str] = []
    for number, ((label, data), index) in enumerate(zip(parts, rows)):
        start, stop = int(offsets[number]), int(offsets[number + 1])
        rowCount = np.asarray(data.data_vals(list(data.axes())[0])).size
        for dependent in wantedColumns(data, columns):
            values = np.asarray(data.data_vals(dependent)).flatten()
            onOtherAxes = (columns
                           and set(data.axes(dependent)) != set(ownAxes[number]))
            if onOtherAxes or values.size != rowCount:
                # A trace, when quality factors are being compared; or a column
                # that does not line up with the dataset's rows at all.
                if dependent not in dropped:
                    dropped.append(dependent)
                continue
            values = values[index]
            column = np.full(total, np.nan, dtype=_joinDtype(values))
            column[start:stop] = values
            out[seriesName(dependent, label)] = dict(
                values=column, axes=list(axisNames),
                unit=data.get(dependent, {}).get('unit', ''),
                label=data.get(dependent, {}).get('label', ''),
            )

    for key, value in parts[0][1].meta_items():
        out.add_meta(key, value)

    if not out.dependents():
        return None, ('none of the datasets has a column called '
                      + ', '.join(f'`{name}`' for name in columns))

    out.validate()
    summary = (f'{len(parts)} datasets joined on `{", ".join(axisNames)}`: '
               + ', '.join(f'{label or "this one"} ({n} points)'
                           for (label, _), n in zip(parts, lengths)))
    if columns:
        summary += '; columns: ' + ', '.join(sorted(
            {name.rsplit(' [', 1)[0] for name in out.dependents()}))
    if dropped:
        summary += ('; not on ' + ', '.join(axisNames) + ': '
                    + ', '.join(sorted(dropped)))
    if notes:
        summary += '; ' + '; '.join(notes)
    return out, summary


def _thisPath(data: DataDictBase) -> str:
    """File the window was opened on, as the loader recorded it.

    The loader puts it in the ``title`` meta field; datasets that arrive some
    other way have none, and are named by the fallback.
    """
    try:
        title = data.meta_val('title')
    except Exception:  # noqa: BLE001 -- absent, or a meta store without it
        title = None
    return str(title) if title else ''


def _expanded(data: DataDictBase) -> DataDictBase:
    """One row per point, so that every column has the same length.

    A ddh5 read straight off disk holds one record per sweep -- a whole trace
    in one row, next to the single Q of that power.  Rows are what is being
    concatenated here, so the columns have to be in step first; without this,
    the per-power columns are simply the wrong length and get dropped.
    """
    try:
        if data.is_expanded():
            return data
        if data.is_expandable():
            return data.expand()
    except Exception:  # noqa: BLE001 -- shapes this cannot expand are used as-is
        pass
    return data


def _joinDtype(values: np.ndarray) -> Any:
    """dtype able to hold both these values and NaN."""
    if np.iscomplexobj(values):
        return complex
    return float


class _JoinOptionsWidget(QtWidgets.QWidget):
    """List of added files, with buttons to add and remove."""

    #: emitted with the new list of files whenever it changes
    filesChanged = Signal(list)

    #: emitted with the new column list whenever it changes
    columnsChanged = Signal(str)

    #: emitted with the new x axis column whenever it changes
    abscissaChanged = Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setToolTip(
            'Datasets drawn together with the one this window was opened on.')
        self.list.setMaximumHeight(110)

        self.addButton = QtWidgets.QPushButton('Add...')
        self.removeButton = QtWidgets.QPushButton('Remove')
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)

        self.columns = QtWidgets.QLineEdit(DEFAULT_COLUMNS)
        self.columns.setToolTip(
            'Which columns to compare, separated by commas.  A power sweep has '
            'some thirty of them and joining several datasets whole makes the '
            'selection list unusable, so only these are brought over -- from '
            'every dataset, including the one already open.  Error bars follow '
            'their column.  Leave it empty to bring everything.')

        self.abscissa = QtWidgets.QLineEdit(DEFAULT_ABSCISSA)
        self.abscissa.setToolTip(
            'Column put on the x axis.  The photon number is saved with one '
            'value per measured power -- the same points as Qi -- so each '
            'dataset\'s own photon numbers replace its powers when it is read, '
            'and the curves land on one axis without anything being '
            'interpolated.  Leave it empty to keep the power (or whatever the '
            'dataset\'s own axis is).')

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.addButton)
        buttons.addWidget(self.removeButton)
        buttons.addStretch()

        columnRow = QtWidgets.QHBoxLayout()
        columnRow.addWidget(QtWidgets.QLabel('Compare'))
        columnRow.addWidget(self.columns)

        abscissaRow = QtWidgets.QHBoxLayout()
        abscissaRow.addWidget(QtWidgets.QLabel('X axis'))
        abscissaRow.addWidget(self.abscissa)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.list)
        layout.addLayout(buttons)
        layout.addLayout(columnRow)
        layout.addLayout(abscissaRow)
        layout.addWidget(self.status)

        self.addButton.clicked.connect(self._add)
        self.removeButton.clicked.connect(self._remove)
        # editingFinished, not textChanged: nothing should be re-read while a
        # name is half typed.
        self.columns.editingFinished.connect(
            lambda: self.columnsChanged.emit(self.columns.text()))
        self.abscissa.editingFinished.connect(
            lambda: self.abscissaChanged.emit(self.abscissa.text()))

    def files(self) -> List[str]:
        return [self.list.item(i).data(0x0100)  # Qt.UserRole
                for i in range(self.list.count())]

    def setFiles(self, files: Sequence[str]) -> None:
        self.list.clear()
        for path in files:
            item = QtWidgets.QListWidgetItem(
                f'{datasetLabel(path)}   —   {path}')
            item.setData(0x0100, str(path))
            item.setToolTip(str(path))
            self.list.addItem(item)

    @Slot()
    def _add(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Add datasets to this plot', '',
            'plottr data (*.ddh5);;all files (*)')
        if not paths:
            return
        self.setFiles(self.files() + [p for p in paths if p not in self.files()])
        self.filesChanged.emit(self.files())

    @Slot()
    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.filesChanged.emit(self.files())


class JoinDatasetsWidget(NodeWidget):
    """Node widget for :class:`.JoinDatasets`."""

    def __init__(self, node: Optional[Node] = None):
        super().__init__(embedWidgetClass=_JoinOptionsWidget, node=node)
        self.widget: _JoinOptionsWidget
        assert self.widget is not None

        self.optSetters = {'files': self.widget.setFiles,
                           'columns': self.widget.columns.setText,
                           'abscissa': self.widget.abscissa.setText}
        self.optGetters = {'files': self.widget.files,
                           'columns': self.widget.columns.text,
                           'abscissa': self.widget.abscissa.text}

        self.widget.filesChanged.connect(lambda: self.signalOption('files'))
        self.widget.columnsChanged.connect(lambda: self.signalOption('columns'))
        self.widget.abscissaChanged.connect(
            lambda: self.signalOption('abscissa'))

        if node is not None:
            node.statusChanged.connect(self.widget.status.setText)


class JoinDatasets(Node):
    """Draw other saved datasets in the same plot.

    With no files added the data passes through untouched, names included, so
    a plot looks exactly as it did before a file was added.

    :Options:
        - ``files``: paths of the ddh5 files to draw as well.
        - ``columns``: which dependents to compare, comma separated.  Empty
          brings every column of every dataset.
        - ``abscissa``: column put on the x axis (``photon``).  Every dataset
          contributes its own values, so nothing is interpolated.  Empty keeps
          the datasets' own axis.
    """

    nodeName = 'JoinDatasets'
    useUi = True
    uiClass: Optional[Type[NodeWidget]] = JoinDatasetsWidget

    #: what the node did, or why it did nothing
    statusChanged = Signal(str)

    def __init__(self, name: str) -> None:
        self._files: List[str] = []
        self._columns: str = DEFAULT_COLUMNS
        self._abscissa: str = DEFAULT_ABSCISSA
        super().__init__(name)

    @property
    def files(self) -> List[str]:
        return list(self._files)

    @files.setter
    @updateOption('files')
    def files(self, value: Sequence[str]) -> None:
        self._files = [str(path) for path in (value or [])]

    @property
    def columns(self) -> str:
        return self._columns

    @columns.setter
    @updateOption('columns')
    def columns(self, value: str) -> None:
        self._columns = str('' if value is None else value)

    @property
    def abscissa(self) -> str:
        return self._abscissa

    @abscissa.setter
    @updateOption('abscissa')
    def abscissa(self, value: str) -> None:
        self._abscissa = str('' if value is None else value).strip()

    def process(self, dataIn: Optional[DataDictBase] = None) \
            -> Optional[Dict[str, Optional[DataDictBase]]]:
        if dataIn is None:
            return None
        if not self._files:
            self.statusChanged.emit('')
            return dict(dataOut=dataIn)

        axes = list(dataIn.axes())
        notes: List[str] = []

        # All the labels at once: they have to be distinct, and that can only
        # be decided by looking at them together.
        labels = uniqueLabels([_thisPath(dataIn)] + list(self._files),
                              fallback='this one')
        parts: List[Tuple[str, DataDictBase]] = [(labels[0], dataIn)]

        for path, label in zip(self._files, labels[1:]):
            try:
                extra = loadDataset(path)
            except Exception as exc:  # noqa: BLE001 -- never take the viewer down
                notes.append(f'{Path(path).name}: {type(exc).__name__}: {exc}')
                continue
            if self._abscissa:
                # The x axis is built in `joinDatasets`, out of each dataset's
                # own column, so nothing has to match beforehand.
                parts.append((label, extra))
                continue
            shaped, why = reshapeLike(extra, axes)
            if shaped is None:
                notes.append(f'{label}: {why}')
                continue
            parts.append((label, shaped))

        if len(parts) < 2:
            self.statusChanged.emit('; '.join(notes) or 'nothing added')
            return dict(dataOut=dataIn)

        try:
            joined, summary = joinDatasets(parts, parseColumns(self._columns),
                                           self._abscissa)
        except Exception as exc:  # noqa: BLE001 -- never take the viewer down
            self.statusChanged.emit(f'{type(exc).__name__}: {exc}')
            return dict(dataOut=dataIn)

        if joined is None:
            self.statusChanged.emit('; '.join([summary] + notes))
            return dict(dataOut=dataIn)

        self.statusChanged.emit('; '.join([summary] + notes))
        return dict(dataOut=joined)
