"""A node that draws other datasets into the same plot.

Comparing measurements -- the same resonator after a change, several resonators
from one chip, the same device in two cooldowns -- means one plot with one
curve per dataset.  plottr shows one dataset at a time, so this node loads
further ddh5 files and merges them into the one being viewed.

How the merge works: every dataset keeps its own rows.  The added dataset's
dependents become ``<name> [<label>]`` and the rows of the others are filled
with NaN, so a point that was never measured is a gap rather than an invented
value.  Nothing is interpolated or resampled.  The axis is shared by name, so
the curves land on one pair of axes and the legend tells them apart.

The node sits *after* ``Photon axis``, which means the plot's x axis may be
something that node built (``log10_photon``).  A file loaded here is put in the
same shape, by name: an axis called ``log10_<dep>`` means "swap ``<dep>`` in
and take log10 of it".  That is why the transformation lives in
:func:`.dependent_axis.swapDependentToAxis` -- one implementation, used from
both places.

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

__all__ = ['DEFAULT_COLUMNS', 'JoinDatasets', 'JoinDatasetsWidget',
           'datasetLabel', 'joinDatasets', 'loadDataset', 'parseColumns',
           'seriesName', 'wantedColumns']

#: ddh5 group the measurement scripts write into.
GROUPNAME = 'data'

#: Columns brought over unless told otherwise.  Comparing datasets is about one
#: quantity at a time, and for a power sweep that is the internal quality
#: factor; everything else would only lengthen the selection list.
DEFAULT_COLUMNS = 'Qi'

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


def joinDatasets(parts: Sequence[Tuple[str, DataDictBase]],
                 columns: Sequence[str] = ()) \
        -> Tuple[Optional[DataDictBase], str]:
    """Merge datasets that share their axes into one.

    :param parts: ``(label, dataset)`` pairs.  A label of ``''`` leaves that
        dataset's names alone -- used for the dataset already being viewed when
        it is the only one, so that nothing changes until a file is added.
    :param columns: dependents to keep, empty for all.  Applied to every
        dataset, the one being viewed included: the joined dataset is the
        comparison, and a comparison of one quantity has no use for the other
        twenty-nine.
    :return: the merged dataset and a one-line summary.
    """
    parts = [(label, _expanded(data)) for label, data in parts
             if data is not None]
    if not parts:
        return None, 'nothing to join'
    if len(parts) == 1:
        return parts[0][1], ''

    axes = list(parts[0][1].axes())
    for label, data in parts[1:]:
        # The order does not have to match: a ddh5 read back can list the axes
        # the other way round, and each dataset's own columns stay aligned with
        # each other whatever the order is.
        if set(data.axes()) != set(axes):
            return None, (f'`{label}` has axes {", ".join(data.axes())}, '
                          f'not {", ".join(axes)}')

    # Every dataset keeps its own rows; the others are NaN there.
    lengths = [np.asarray(data.data_vals(axes[0])).size for _, data in parts]
    total = int(sum(lengths))
    offsets = np.cumsum([0] + lengths)

    out = DataDict()
    for axis in axes:
        values = np.concatenate(
            [np.asarray(data.data_vals(axis)).flatten() for _, data in parts])
        first = parts[0][1]
        out[axis] = dict(values=values, axes=[],
                         unit=first.get(axis, {}).get('unit', ''),
                         label=first.get(axis, {}).get('label', ''))

    for index, (label, data) in enumerate(parts):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        for dependent in wantedColumns(data, columns):
            values = np.asarray(data.data_vals(dependent)).flatten()
            if values.size != stop - start:
                continue  # not on these axes; it cannot be placed
            column = np.full(total, np.nan, dtype=_joinDtype(values))
            column[start:stop] = values
            out[seriesName(dependent, label)] = dict(
                values=column, axes=list(axes),
                unit=data.get(dependent, {}).get('unit', ''),
                label=data.get(dependent, {}).get('label', ''),
            )

    for key, value in parts[0][1].meta_items():
        out.add_meta(key, value)

    if not out.dependents():
        return None, ('none of the datasets has a column called '
                      + ', '.join(f'`{name}`' for name in columns))

    out.validate()
    summary = (f'{len(parts)} datasets joined: '
               + ', '.join(f'{label or "this one"} ({n} points)'
                           for (label, _), n in zip(parts, lengths)))
    if columns:
        summary += '; columns: ' + ', '.join(sorted(
            {name.rsplit(' [', 1)[0] for name in out.dependents()}))
    return out, summary


def _thisLabel(data: DataDictBase) -> str:
    """Name for the dataset the window was opened on.

    The loader records the file it read as the ``title`` meta field; datasets
    that arrive some other way have no such field, and get a neutral name.
    """
    try:
        title = data.meta_val('title')
    except Exception:  # noqa: BLE001 -- absent, or a meta store without it
        title = None
    return datasetLabel(str(title)) if title else 'this one'


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

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.addButton)
        buttons.addWidget(self.removeButton)
        buttons.addStretch()

        columnRow = QtWidgets.QHBoxLayout()
        columnRow.addWidget(QtWidgets.QLabel('Compare'))
        columnRow.addWidget(self.columns)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.list)
        layout.addLayout(buttons)
        layout.addLayout(columnRow)
        layout.addWidget(self.status)

        self.addButton.clicked.connect(self._add)
        self.removeButton.clicked.connect(self._remove)
        # editingFinished, not textChanged: nothing should be re-read while a
        # name is half typed.
        self.columns.editingFinished.connect(
            lambda: self.columnsChanged.emit(self.columns.text()))

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
                           'columns': self.widget.columns.setText}
        self.optGetters = {'files': self.widget.files,
                           'columns': self.widget.columns.text}

        self.widget.filesChanged.connect(lambda: self.signalOption('files'))
        self.widget.columnsChanged.connect(lambda: self.signalOption('columns'))

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
    """

    nodeName = 'JoinDatasets'
    useUi = True
    uiClass: Optional[Type[NodeWidget]] = JoinDatasetsWidget

    #: what the node did, or why it did nothing
    statusChanged = Signal(str)

    def __init__(self, name: str) -> None:
        self._files: List[str] = []
        self._columns: str = DEFAULT_COLUMNS
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

    def process(self, dataIn: Optional[DataDictBase] = None) \
            -> Optional[Dict[str, Optional[DataDictBase]]]:
        if dataIn is None:
            return None
        if not self._files:
            self.statusChanged.emit('')
            return dict(dataOut=dataIn)

        axes = list(dataIn.axes())
        parts: List[Tuple[str, DataDictBase]] = [(_thisLabel(dataIn), dataIn)]
        notes: List[str] = []

        for path in self._files:
            try:
                extra = loadDataset(path)
            except Exception as exc:  # noqa: BLE001 -- never take the viewer down
                notes.append(f'{Path(path).name}: {type(exc).__name__}: {exc}')
                continue
            shaped, why = reshapeLike(extra, axes)
            if shaped is None:
                notes.append(f'{datasetLabel(path)}: {why}')
                continue
            parts.append((datasetLabel(path), shaped))

        if len(parts) < 2:
            self.statusChanged.emit('; '.join(notes) or 'nothing added')
            return dict(dataOut=dataIn)

        try:
            joined, summary = joinDatasets(parts, parseColumns(self._columns))
        except Exception as exc:  # noqa: BLE001 -- never take the viewer down
            self.statusChanged.emit(f'{type(exc).__name__}: {exc}')
            return dict(dataOut=dataIn)

        if joined is None:
            self.statusChanged.emit('; '.join([summary] + notes))
            return dict(dataOut=dataIn)

        self.statusChanged.emit('; '.join([summary] + notes))
        return dict(dataOut=joined)
