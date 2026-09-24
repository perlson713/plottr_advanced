"""``plottr.plot.mpl.autoplot`` -- This module contains the tools for automatic plotting with matplotlib.
"""

import json
import logging
from collections import OrderedDict
from typing import (Dict, List, Sequence, Tuple, Union, Optional, Any,
                    Type, cast)
from types import TracebackType

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable

from plottr import QtWidgets, QtCore, QtGui, Signal, Slot
from plottr.data.datadict import DataDictBase
from plottr.icons import (get_singleTracePlotIcon, get_multiTracePlotIcon, get_imagePlotIcon,
                          get_colormeshPlotIcon, get_scatterPlot2dIcon)
from plottr.gui.tools import dpiScalingFactor
from .appearance import (COLORMAPS, EXPORT_FORMATS, FIT_COLOR_AUTO,
                         LEGEND_AUTO, LEGEND_LOCATIONS, LEGEND_NONE,
                         LEGEND_OUTSIDE, LINE_STYLES, MARKERS, FigureExport,
                         PlotAxes, PlotColors, PlotLabels, PlotStyle,
                         TraceStyle, appearanceToDict, applyAppearance,
                         parseNumber, renderableText)
from .plotting import PlotType, colorplot2d
from .widgets import MPLPlotWidget
from ..base import AutoFigureMaker as BaseFM, PlotDataType, \
    PlotItem, ComplexRepresentation, determinePlotDataType, PlotWidgetContainer, \
    ERROR_BAR_AUTO, ERROR_BAR_NONE, errorBarData, errorBarDataNames, \
    plottableDependents, fitSourceName, sortFitsLast

logger = logging.getLogger(__name__)


#: The fit curve is drawn with this much more z than the data it belongs to,
#: so that it sits on top of the markers instead of under them.
FIT_ZORDER_BOOST = 0.5

#: Axis scales the toolbar offers.
AXIS_SCALES = (('Linear', 'linear'), ('Log', 'log'))


def _hasPositiveValues(values: Any) -> bool:
    """Whether a log axis can show this data at all."""
    array = np.asanyarray(values)
    if not np.issubdtype(array.dtype, np.number):
        return False
    with np.errstate(invalid='ignore'):
        return bool(np.any(np.isfinite(array) & (array > 0)))


class FigureMaker(BaseFM):
    """Matplotlib implementation for :class:`.AutoFigureMaker`.
    Implements plotting routines for data with 1 or 2 dependents, as well as generation
    and formatting of subplots.

    The class tries to lay out the subplots to be generated on a grid that's as close as possible to square.
    The allocation of plot to subplots depends on the type of plot we're making, and the type of data.
    Subplots may contain either one 2d plot (image, 2d scatter, etc) or multiple 1d plots.
    """

    def __init__(self, fig: Figure) -> None:
        super().__init__()
        self.fig = fig

        #: what kind of plot we're making. needs to be set before adding data.
        #: Incompatibility with the data provided will result in failure.
        self.plotType = PlotType.empty

        #: point size, line widths and fit color
        self.style = PlotStyle()

        #: title, axis labels and legend
        self.labels = PlotLabels()

        #: what the axis labels would say if nothing were typed.  Read back by
        #: the toolbar, to show as the placeholder of the empty fields.
        self.automaticLabels: Dict[str, str] = {}

        #: where the axes start and stop
        self.axisLimits = PlotAxes()

        #: colormap and color range of a 2-D plot
        self.colors = PlotColors()

        #: scale of the x and y axes ('linear' or 'log')
        self.xScale = 'linear'
        self.yScale = 'linear'

        #: color assigned to each measured trace, so its fit can reuse it.
        #: Keyed by (subplot, half of a complex trace, name).
        self._traceColors: Dict[Tuple[int, int, str], str] = {}

    # re-implementing to get correct type annotation.
    def __enter__(self) -> "FigureMaker":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.fig.clear()
        return super().__exit__(exc_type, exc_value, traceback)

    # inherited methods
    def addData(self, *data: Union[np.ndarray, np.ma.MaskedArray],
                join: Optional[int] = None,
                labels: Optional[List[str]] = None,
                plotDataType: PlotDataType = PlotDataType.unknown,
                **plotOptions: Any) -> int:

        if self.plotType == PlotType.multitraces and join is None:
            join = self.previousPlotId()
        return super().addData(*data, join=join, labels=labels,
                               plotDataType=plotDataType, **plotOptions)

    def makeSubPlots(self, nSubPlots: int) -> List[Axes]:
        """Create subplots (`Axes`). They are arranged on a grid that's close to square.

        :param nSubPlots: number of subplots to make
        :return: list of matplotlib axes.
        """
        if nSubPlots > 0:
            nrows = int(nSubPlots ** .5 + .5)
            ncols = int(np.ceil(nSubPlots / nrows))
            gs = GridSpec(nrows, ncols, self.fig)
            axes = [self.fig.add_subplot(gs[i]) for i in range(nSubPlots)]
        else:
            axes = []
        return axes

    def formatSubPlot(self, subPlotId: int) -> None:
        """Format a subplot. Parses the plot items that go into that subplot,
        and attaches axis labels and legend handles.

        :param subPlotId: ID of the subplot.
        """
        labels = self.subPlotLabels(subPlotId)
        axes = self.subPlots[subPlotId].axes

        if isinstance(axes, list) and len(axes) > 0:
            if len(labels) > 0 and len(set(labels[0])) == 1:
                axes[0].set_xlabel(labels[0][0])
            if len(labels) > 1 and len(set(labels[1])) == 1:
                axes[0].set_ylabel(labels[1][0])

        if isinstance(axes, list) and len(axes) > 1:
            if len(labels) > 2 and len(set(labels[2])) == 1:
                axes[1].set_ylabel(labels[2][0])

        # What the operator typed wins over what the columns say, one field at
        # a time: an empty field stays automatic.
        if isinstance(axes, list) and len(axes) > 0:
            self.automaticLabels.setdefault('x', axes[0].get_xlabel())
            self.automaticLabels.setdefault('y', axes[0].get_ylabel())
            if self.labels.xLabel:
                axes[0].set_xlabel(renderableText(self.labels.xLabel))
            if self.labels.yLabel:
                axes[0].set_ylabel(renderableText(self.labels.yLabel))

        if isinstance(axes, list) and len(axes) > 0:
            needed = len(labels) == 2 and len(set(labels[1])) > 1
            self.applyLegend(axes[0], needed)

        if isinstance(axes, list):
            for ax in axes:
                self.applyAxisScales(subPlotId, ax)
            if len(axes) > 0:
                # Only the first panel: a split Re/Im figure has two y axes
                # with different data, and one pair of numbers cannot mean
                # both.  Last, so that the scales do not move them again.
                self.axisLimits.applyTo(axes[0])
        return None

    def applyLegend(self, ax: Axes, needed: bool) -> None:
        """Draw the legend the way the toolbar asks for it.

        ``needed`` is the old rule: a legend where several traces share the
        panel, and none where the y label already says what the single trace
        is.  That is what ``Automatic`` keeps doing; any other choice is the
        operator's and is followed whether or not the legend is needed.

        Entries are renamed here rather than at the point of drawing, because
        the name to rename is the one matplotlib ended up with -- a complex
        trace split into Re and Im is two entries.
        """
        keywords = self.labels.legendKeywords()
        if keywords is None:
            return
        if not needed and self.labels.legendLocation == LEGEND_AUTO:
            return

        handles, autoLabels = ax.get_legend_handles_labels()
        if not handles:
            return
        ax.legend(handles, [self.labels.nameFor(name) for name in autoLabels],
                  **keywords)

    def applyAxisScales(self, subPlotId: int, ax: Axes) -> None:
        """Put the chosen scales on one set of axes.

        Only 1d plots: on an image the axes are the pixel grid, where a log
        scale means nothing.  A log scale is also skipped where the data has no
        positive values at all -- matplotlib would happily draw an empty plot,
        which looks like a bug rather than like a wrong choice of scale.
        """
        items = [self.plotItems[i] for i in self.plotIdsInSubPlot(subPlotId)]
        lines = [item for item in items if len(item.data) == 2]
        if not lines:
            return

        for scale, index, setter in ((self.xScale, 0, ax.set_xscale),
                                     (self.yScale, 1, ax.set_yscale)):
            if scale != 'log':
                setter('linear')
                continue
            if any(_hasPositiveValues(item.data[index]) for item in lines):
                setter('log')
            else:
                setter('linear')

    def plot(self, plotItem: PlotItem) -> Optional[Union[ScalarMappable, List[ScalarMappable]]]:
        """Plots data in a PlotItem.

        :param plotItem: the item to plot.
        :return: matplotlib Artist(s), or ``None`` if nothing was plotted.
        """
        if self.plotType in [PlotType.singletraces, PlotType.multitraces]:
            return self.plotLine(plotItem)
        elif self.plotType in [PlotType.image, PlotType.scatter2d, PlotType.colormesh]:
            return self.plotImage(plotItem)
        else:
            return None

    # methods specific to this class
    def plotLine(self, plotItem: PlotItem) -> Optional[List[ScalarMappable]]:
        axes = self.subPlots[plotItem.subPlot].axes
        assert isinstance(axes, list) and len(axes) > 0
        assert len(plotItem.data) == 2
        lbl = plotItem.labels[-1] if isinstance(plotItem.labels, list) and len(plotItem.labels) > 0 else ''
        x, y = plotItem.data
        assert plotItem.plotOptions is not None
        plotOptions = plotItem.plotOptions.copy()
        yerr = plotOptions.pop('_errorBarData', None)

        # Private keys, set by _plotData and by the complex-data split; they
        # describe the trace rather than how to draw it, so matplotlib never
        # sees them.
        name = plotOptions.pop('_name', '')
        fitOf = plotOptions.pop('_fitOf', None)
        part = int(plotOptions.pop('_part', 0))
        key = (plotItem.subPlot, part, fitOf or name)

        if fitOf is None:
            style = dict(self.style.dataOptions(lbl))
        else:
            style = dict(self.style.fitOptions())
            if 'color' not in style:
                # Same color as the data it fits, so the pair reads as one
                # thing.  The data is always plotted first (sortFitsLast).
                color = self._traceColors.get(key)
                if color is not None:
                    style['color'] = color
            # ... and on top of the markers, not under them.
            style['zorder'] = 2 + FIT_ZORDER_BOOST
        style.update(plotOptions)  # an explicit option always wins

        line = axes[0].plot(x, y, label=lbl, **style)
        if fitOf is None and name and len(line) > 0:
            self._traceColors[key] = line[0].get_color()
        if yerr is not None:
            axes[0].errorbar(x, y, yerr=yerr, fmt='none',
                             ecolor=line[0].get_color(), capsize=2)
        return line

    def plotImage(self, plotItem: PlotItem) -> Optional[ScalarMappable]:
        assert len(plotItem.data) == 3
        x, y, z = plotItem.data
        axes = self.subPlots[plotItem.subPlot].axes
        assert isinstance(axes, list) and len(axes) > 0
        im = colorplot2d(axes[0], x, y, z, plotType=self.plotType,
                         **self.colors.keywords())
        if im is None:
            return None
        cb = self.fig.colorbar(im, ax=axes[0], shrink=0.75, pad=0.02)
        lbl = plotItem.labels[-1] if isinstance(plotItem.labels, list) and len(plotItem.labels) > 0 else ''
        self.automaticLabels.setdefault('colorbar', lbl)
        cb.set_label(renderableText(self.labels.colorbarLabel or lbl))
        return im


class PlotStyleWidget(QtWidgets.QWidget):
    """Form for point size, line widths and fit color.

    It edits a :class:`PlotStyle` in place and says so with :attr:`changed`;
    the plot widget redraws on that.
    """

    #: emitted after any of the values changed
    changed = Signal()

    #: fit colors offered by name, on top of "same as the data"
    COLORS = (
        ('Same as data', FIT_COLOR_AUTO),
        ('Black', 'k'),
        ('Red', '#d62728'),
        ('Blue', '#1f77b4'),
        ('Orange', '#ff7f0e'),
        ('Green', '#2ca02c'),
        ('Grey', '#7f7f7f'),
    )

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._style: Optional[PlotStyle] = None

        self.pointSize = QtWidgets.QDoubleSpinBox()
        self.pointSize.setRange(0.0, 30.0)
        self.pointSize.setSingleStep(0.5)
        self.pointSize.setToolTip('Size of the data points.  0 hides them.')

        self.lineWidth = QtWidgets.QDoubleSpinBox()
        self.lineWidth.setRange(0.0, 10.0)
        self.lineWidth.setSingleStep(0.25)
        self.lineWidth.setToolTip(
            'Thickness of the line through the data points.  0 leaves the '
            'points on their own.')

        self.fitLineWidth = QtWidgets.QDoubleSpinBox()
        self.fitLineWidth.setRange(0.0, 10.0)
        self.fitLineWidth.setSingleStep(0.25)
        self.fitLineWidth.setToolTip('Thickness of the fit curves.')

        self.fitColor = QtWidgets.QComboBox()
        for name, value in self.COLORS:
            self.fitColor.addItem(name, value)
        self.fitColor.addItem('Custom...', None)
        self.fitColor.setToolTip(
            'Color of the fit curves.  By default each fit takes the color of '
            'the trace it belongs to, so the pair reads as one thing.')

        self.perTraceButton = QtWidgets.QPushButton('Per trace...')
        self.perTraceButton.setToolTip(
            'Color, point shape and line style of one trace at a time.  Which '
            'of six resonators is the orange one is otherwise decided by the '
            'order they were added in.')

        form = QtWidgets.QFormLayout(self)
        form.addRow('Point size', self.pointSize)
        form.addRow('Line width', self.lineWidth)
        form.addRow('Fit width', self.fitLineWidth)
        form.addRow('Fit color', self.fitColor)
        form.addRow('', self.perTraceButton)

        self.pointSize.valueChanged.connect(self._apply)
        self.lineWidth.valueChanged.connect(self._apply)
        self.fitLineWidth.valueChanged.connect(self._apply)
        self.fitColor.currentIndexChanged.connect(self._colorChosen)

    def setStyle(self, style: 'PlotStyle') -> None:
        """Show the values of ``style``, and edit that object from now on."""
        self._style = None  # do not write back while filling the form in
        self.pointSize.setValue(style.markerSize)
        self.lineWidth.setValue(style.lineWidth)
        self.fitLineWidth.setValue(style.fitLineWidth)
        self._showColor(style.fitColor)
        self._style = style

    def _showColor(self, color: str) -> None:
        index = self.fitColor.findData(color)
        if index < 0:
            # A custom color: keep one entry for it rather than growing the
            # list every time the operator picks another one.
            index = self.fitColor.findText('Custom color')
            if index < 0:
                self.fitColor.insertItem(0, 'Custom color', color)
                index = 0
            else:
                self.fitColor.setItemData(index, color)
        self.fitColor.setCurrentIndex(index)

    @Slot()
    def _colorChosen(self) -> None:
        if self._style is None:
            return
        value = self.fitColor.currentData()
        if value is None:  # the "Custom..." entry
            current = QtGui.QColor(self._style.fitColor or '#000000')
            chosen = QtWidgets.QColorDialog.getColor(
                current, self, 'Fit curve color')
            if not chosen.isValid():
                self._showColor(self._style.fitColor)
                return
            value = chosen.name()
            self._showColor(value)
        self._style.fitColor = str(value)
        self.changed.emit()

    @Slot()
    def _apply(self) -> None:
        if self._style is None:
            return
        self._style.markerSize = self.pointSize.value()
        self._style.lineWidth = self.lineWidth.value()
        self._style.fitLineWidth = self.fitLineWidth.value()
        self.changed.emit()


class PlotLabelsWidget(QtWidgets.QWidget):
    """Form for the title, the axis labels and the legend.

    It edits a :class:`PlotLabels` in place and says so with :attr:`changed`;
    the plot widget redraws on that.  Every field is "empty means automatic",
    and each shows the automatic value as its placeholder, so it is clear what
    is being replaced before anything is typed.
    """

    #: emitted after any of the values changed
    changed = Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._labels: Optional[PlotLabels] = None
        self._filling = False

        self.title = QtWidgets.QLineEdit()
        self.title.setToolTip(
            'Title above the figure.  Empty keeps the one the dataset carries '
            '(the file it was loaded from).')
        self.showTitle = QtWidgets.QCheckBox('Show')
        self.showTitle.setToolTip(
            'The automatic title is the full path of the file, which is more '
            'than a figure for a talk wants.')

        self.xLabel = QtWidgets.QLineEdit()
        self.xLabel.setToolTip('Empty takes the label from the column.')
        self.yLabel = QtWidgets.QLineEdit()
        self.yLabel.setToolTip('Empty takes the label from the column.')

        self.legendLocation = QtWidgets.QComboBox()
        for name, value in LEGEND_LOCATIONS:
            self.legendLocation.addItem(name, value)
        self.legendLocation.setToolTip(
            '`Automatic` puts a legend where there is more than one trace and '
            'leaves it off otherwise.  `Outside, right` hides no data, which '
            'is what six curves on one plot usually need.')

        self.entries = QtWidgets.QTableWidget(0, 2)
        self.entries.setHorizontalHeaderLabels(['Trace', 'Show as'])
        self.entries.verticalHeader().setVisible(False)
        self.entries.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.entries.setToolTip(
            'What each trace is called in the legend.  Leave a cell empty to '
            'keep the name the column gives it.')
        self.entries.setMinimumWidth(320)
        self.entries.setMaximumHeight(200)
        header = self.entries.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        titleRow = QtWidgets.QHBoxLayout()
        titleRow.addWidget(self.title)
        titleRow.addWidget(self.showTitle)

        form = QtWidgets.QFormLayout(self)
        form.addRow('Title', titleRow)
        form.addRow('X label', self.xLabel)
        form.addRow('Y label', self.yLabel)
        form.addRow('Legend', self.legendLocation)
        form.addRow('Entries', self.entries)

        # editingFinished, not textChanged: the figure should not be redrawn
        # once per keystroke.
        for edit in (self.title, self.xLabel, self.yLabel):
            edit.editingFinished.connect(self._apply)
        self.showTitle.toggled.connect(self._apply)
        self.legendLocation.currentIndexChanged.connect(self._apply)
        self.entries.itemChanged.connect(self._entryChanged)

    def setLabels(self, labels: 'PlotLabels') -> None:
        """Show the values of ``labels``, and edit that object from now on."""
        self._labels = None  # do not write back while filling the form in
        self.title.setText(labels.title)
        self.showTitle.setChecked(labels.showTitle)
        self.xLabel.setText(labels.xLabel)
        self.yLabel.setText(labels.yLabel)
        index = self.legendLocation.findData(labels.legendLocation)
        if index >= 0:
            self.legendLocation.setCurrentIndex(index)
        self._labels = labels

    def setAutomatic(self, title: str = '', xLabel: str = '',
                     yLabel: str = '') -> None:
        """Show what each field would say if it were left empty."""
        self.title.setPlaceholderText(title)
        self.xLabel.setPlaceholderText(xLabel)
        self.yLabel.setPlaceholderText(yLabel)

    def setEntries(self, entries: Sequence[str]) -> None:
        """List the traces that are currently in the plot.

        Called after every redraw, so it must not look like the operator typed
        something: the table is rebuilt with signals held.
        """
        current = [self.entries.item(row, 0).text()
                   for row in range(self.entries.rowCount())
                   if self.entries.item(row, 0) is not None]
        if list(entries) == current:
            return

        self._filling = True
        try:
            self.entries.setRowCount(len(entries))
            for row, name in enumerate(entries):
                first = QtWidgets.QTableWidgetItem(name)
                first.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.entries.setItem(row, 0, first)
                shown = ''
                if self._labels is not None:
                    shown = self._labels.legendNames.get(name, '')
                self.entries.setItem(row, 1, QtWidgets.QTableWidgetItem(shown))
        finally:
            self._filling = False

    @Slot(QtWidgets.QTableWidgetItem)
    def _entryChanged(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._filling or self._labels is None or item.column() != 1:
            return
        key = self.entries.item(item.row(), 0)
        if key is None:
            return
        text = item.text().strip()
        if text:
            self._labels.legendNames[key.text()] = text
        else:
            self._labels.legendNames.pop(key.text(), None)
        self.changed.emit()

    @Slot()
    def _apply(self) -> None:
        if self._labels is None:
            return
        self._labels.title = self.title.text()
        self._labels.showTitle = self.showTitle.isChecked()
        self._labels.xLabel = self.xLabel.text()
        self._labels.yLabel = self.yLabel.text()
        self._labels.legendLocation = str(self.legendLocation.currentData())
        self.changed.emit()


class TraceStyleDialog(QtWidgets.QDialog):
    """Color, point shape and line style, one trace at a time.

    A dialog rather than another popup in the toolbar: this is a table with a
    row per curve and three editors in each, and a comparison of six
    resonators fills it.  Non-modal, so the figure can be watched while it is
    being changed.
    """

    #: emitted after any of the values changed
    changed = Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle('Trace appearance')
        self._style: Optional[PlotStyle] = None
        self._filling = False
        self._names: List[str] = []

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ['Trace', 'Color', 'Points', 'Line'])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeToContents)

        self.resetButton = QtWidgets.QPushButton('Back to automatic')
        self.resetButton.setToolTip(
            'Give every trace back the color, points and line the overall '
            'style decides.')
        closeButton = QtWidgets.QPushButton('Close')

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.resetButton)
        buttons.addStretch()
        buttons.addWidget(closeButton)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        self.resize(520, 320)

        self.resetButton.clicked.connect(self._reset)
        closeButton.clicked.connect(self.close)

    def setStyle(self, style: PlotStyle) -> None:
        """Edit this style object from now on."""
        self._style = style

    def setTraces(self, names: Sequence[str]) -> None:
        """Rebuild the table for the traces that are in the plot now."""
        if list(names) == self._names and self.table.rowCount() == len(names):
            return
        self._names = list(names)

        self._filling = True
        try:
            self.table.setRowCount(len(self._names))
            for row, name in enumerate(self._names):
                label = QtWidgets.QTableWidgetItem(name)
                label.setFlags(QtCore.Qt.ItemIsEnabled
                               | QtCore.Qt.ItemIsSelectable)
                self.table.setItem(row, 0, label)
                self.table.setCellWidget(row, 1, self._colorButton(name))
                self.table.setCellWidget(
                    row, 2, self._choiceBox(name, 'marker', MARKERS))
                self.table.setCellWidget(
                    row, 3, self._choiceBox(name, 'lineStyle', LINE_STYLES))
        finally:
            self._filling = False

    def _traceStyle(self, name: str) -> TraceStyle:
        assert self._style is not None
        return self._style.traceStyle(name)

    def _colorButton(self, name: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton()
        button.setToolTip('Color of this trace.  Its fit follows it.')
        self._showColor(button, self._traceStyle(name).color)
        button.clicked.connect(lambda: self._pickColor(name, button))
        return button

    def _showColor(self, button: QtWidgets.QPushButton, color: str) -> None:
        if color:
            button.setText(color)
            button.setStyleSheet(
                f'background: {color}; color: {_readableOn(color)};')
        else:
            button.setText('Automatic')
            button.setStyleSheet('')

    def _pickColor(self, name: str, button: QtWidgets.QPushButton) -> None:
        current = self._traceStyle(name).color
        chosen = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(current or '#1f77b4'), self, f'Color of {name}',
            QtWidgets.QColorDialog.ShowAlphaChannel)
        if not chosen.isValid():
            # Cancel goes back to the cycle, which is the only way to undo a
            # color once one has been picked.
            self._traceStyle(name).color = ''
        else:
            self._traceStyle(name).color = chosen.name()
        self._showColor(button, self._traceStyle(name).color)
        self.changed.emit()

    def _choiceBox(self, name: str, field: str,
                   choices: Sequence[Tuple[str, str]]) -> QtWidgets.QComboBox:
        box = QtWidgets.QComboBox()
        for text, value in choices:
            box.addItem(text, value)
        index = box.findData(getattr(self._traceStyle(name), field))
        box.setCurrentIndex(max(0, index))
        box.currentIndexChanged.connect(
            lambda: self._choiceMade(name, field, box))
        return box

    def _choiceMade(self, name: str, field: str,
                    box: QtWidgets.QComboBox) -> None:
        if self._filling or self._style is None:
            return
        setattr(self._traceStyle(name), field, str(box.currentData()))
        self.changed.emit()

    @Slot()
    def _reset(self) -> None:
        if self._style is None:
            return
        self._style.traces = {}
        names, self._names = self._names, []
        self.setTraces(names)
        self.changed.emit()


def _readableOn(color: str) -> str:
    """Black or white, whichever can be read on this background."""
    shade = QtGui.QColor(color)
    if not shade.isValid():
        return '#000000'
    # Rec. 601 luma, which is what "is this light or dark" means to an eye.
    luma = (0.299 * shade.red() + 0.587 * shade.green()
            + 0.114 * shade.blue())
    return '#000000' if luma > 150 else '#ffffff'


class PlotAxesWidget(QtWidgets.QWidget):
    """Form for the axis scales and where the axes start and stop.

    The limits are empty by default and empty means automatic, the same rule
    as everywhere else here.  `From view` fills them in from the plot as it is
    now, which is what makes the zoom button useful for a figure that has to
    be made again: get the view right by eye, then keep it as numbers.
    """

    #: emitted with the new (x, y) scales
    scalesChanged = Signal(str, str)
    #: emitted after a limit changed
    limitsChanged = Signal()
    #: emitted when the limits should be taken from the plot as it is now
    takeFromView = Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._axes: Optional[PlotAxes] = None

        self.xScaleBox = QtWidgets.QComboBox()
        self.yScaleBox = QtWidgets.QComboBox()
        for box in (self.xScaleBox, self.yScaleBox):
            for name, value in AXIS_SCALES:
                box.addItem(name, value)
            box.currentIndexChanged.connect(self._emitScales)
        self.xScaleBox.setToolTip(
            'Scale of the x axis.  A log scale is ignored where the data has '
            'no positive values.')
        self.yScaleBox.setToolTip('Scale of the y axis.')

        self.limits: Dict[str, QtWidgets.QLineEdit] = {}
        for name in ('xMin', 'xMax', 'yMin', 'yMax'):
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText('auto')
            edit.setMaximumWidth(120)
            edit.editingFinished.connect(self._applyLimits)
            self.limits[name] = edit

        self.fromViewButton = QtWidgets.QPushButton('From view')
        self.fromViewButton.setToolTip(
            'Take the range the plot is showing right now, so that the same '
            'figure can be made again.')
        self.autoButton = QtWidgets.QPushButton('Automatic')
        self.autoButton.setToolTip('Let the data decide the range again.')

        xRow = QtWidgets.QHBoxLayout()
        xRow.addWidget(self.limits['xMin'])
        xRow.addWidget(QtWidgets.QLabel('to'))
        xRow.addWidget(self.limits['xMax'])

        yRow = QtWidgets.QHBoxLayout()
        yRow.addWidget(self.limits['yMin'])
        yRow.addWidget(QtWidgets.QLabel('to'))
        yRow.addWidget(self.limits['yMax'])

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.fromViewButton)
        buttons.addWidget(self.autoButton)
        buttons.addStretch()

        form = QtWidgets.QFormLayout(self)
        form.addRow('x scale', self.xScaleBox)
        form.addRow('y scale', self.yScaleBox)
        form.addRow('x range', xRow)
        form.addRow('y range', yRow)
        form.addRow('', buttons)

        self.fromViewButton.clicked.connect(self.takeFromView)
        self.autoButton.clicked.connect(self._clearLimits)

    def setAxes(self, axes: PlotAxes) -> None:
        """Edit this limits object from now on, and show what it holds."""
        self._axes = None
        for name, text in axes.texts().items():
            self.limits[name].setText(text)
        self._axes = axes

    def showLimits(self) -> None:
        """Write the limits back into the form (after `From view`)."""
        if self._axes is None:
            return
        axes, self._axes = self._axes, None
        for name, text in axes.texts().items():
            self.limits[name].setText(text)
        self._axes = axes

    def setScales(self, xScale: str, yScale: str) -> None:
        """Show these scales without emitting anything."""
        for box, value in ((self.xScaleBox, xScale), (self.yScaleBox, yScale)):
            index = box.findData(value)
            if index >= 0:
                box.blockSignals(True)
                box.setCurrentIndex(index)
                box.blockSignals(False)

    @Slot()
    def _emitScales(self) -> None:
        self.scalesChanged.emit(str(self.xScaleBox.currentData()),
                                str(self.yScaleBox.currentData()))

    @Slot()
    def _applyLimits(self) -> None:
        if self._axes is None:
            return
        before = self._axes.toDict()
        for name, edit in self.limits.items():
            setattr(self._axes, name, parseNumber(edit.text()))
        if self._axes.toDict() != before:
            self.limitsChanged.emit()

    @Slot()
    def _clearLimits(self) -> None:
        if self._axes is None:
            return
        for edit in self.limits.values():
            edit.clear()
        self._applyLimits()


class PlotColorsWidget(QtWidgets.QWidget):
    """Form for the color scale of a 2-D plot.

    A colormap is not decoration: a rainbow invents edges that are not in the
    data, and a range set by one outlier of a sweep flattens everything else.
    """

    #: emitted after any of the values changed
    changed = Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._colors: Optional[PlotColors] = None
        self._labels: Optional[PlotLabels] = None

        self.colormap = QtWidgets.QComboBox()
        for name, value in COLORMAPS:
            self.colormap.addItem(name, value)
        self.colormap.setToolTip(
            '`Default` is the one from plottr\'s own matplotlib settings.  '
            'The first group is perceptually uniform: equal steps in the data '
            'look like equal steps in color, which a rainbow map does not do.')

        self.vMin = QtWidgets.QLineEdit()
        self.vMax = QtWidgets.QLineEdit()
        for edit in (self.vMin, self.vMax):
            edit.setPlaceholderText('auto')
            edit.setMaximumWidth(120)
            edit.editingFinished.connect(self._apply)

        self.colorbarLabel = QtWidgets.QLineEdit()
        self.colorbarLabel.setToolTip(
            'Label beside the colorbar.  Empty takes the column\'s.')
        self.colorbarLabel.editingFinished.connect(self._apply)

        rangeRow = QtWidgets.QHBoxLayout()
        rangeRow.addWidget(self.vMin)
        rangeRow.addWidget(QtWidgets.QLabel('to'))
        rangeRow.addWidget(self.vMax)

        form = QtWidgets.QFormLayout(self)
        form.addRow('Colormap', self.colormap)
        form.addRow('Color range', rangeRow)
        form.addRow('Colorbar label', self.colorbarLabel)

        self.colormap.currentIndexChanged.connect(self._apply)

    def setColors(self, colors: PlotColors, labels: PlotLabels) -> None:
        """Edit these objects from now on, and show what they hold."""
        self._colors, self._labels = None, None
        index = self.colormap.findData(colors.colormap)
        if index >= 0:
            self.colormap.setCurrentIndex(index)
        texts = colors.texts()
        self.vMin.setText(texts['vMin'])
        self.vMax.setText(texts['vMax'])
        self.colorbarLabel.setText(labels.colorbarLabel)
        self._colors, self._labels = colors, labels

    def setAutomatic(self, colorbarLabel: str = '') -> None:
        """Show what the empty colorbar field would say."""
        self.colorbarLabel.setPlaceholderText(colorbarLabel)

    @Slot()
    def _apply(self) -> None:
        if self._colors is None or self._labels is None:
            return
        before = (self._colors.toDict(), self._labels.colorbarLabel)
        self._colors.colormap = str(self.colormap.currentData())
        self._colors.vMin = parseNumber(self.vMin.text())
        self._colors.vMax = parseNumber(self.vMax.text())
        self._labels.colorbarLabel = self.colorbarLabel.text()
        if (self._colors.toDict(), self._labels.colorbarLabel) != before:
            self.changed.emit()


def exportFigure(fig: Figure, path: str, settings: FigureExport,
                 applyFontSize: Optional[Any] = None) -> None:
    """Write ``fig`` out at the size and resolution that were asked for.

    The figure is resized, saved and put back.  ``forward=False`` keeps the
    resize away from the Qt widget: the window should not jump about because
    a file was written.

    ``applyFontSize`` is ``MPLPlot.applyFontSize``, which sizes the text from
    the size of the figure.  Without it a plot exported at 86 mm from a
    maximised window comes out with text made for a 40 cm figure, shrunk --
    the whole reason for having this dialog.
    """
    before = fig.get_size_inches()
    try:
        fig.set_size_inches(*settings.inches(), forward=False)
        if applyFontSize is not None:
            applyFontSize(rescaleExisting=True)
        fig.savefig(path, **settings.saveKeywords())
    finally:
        fig.set_size_inches(*before, forward=False)
        if applyFontSize is not None:
            applyFontSize(rescaleExisting=True)
        fig.canvas.draw_idle()


class ExportDialog(QtWidgets.QDialog):
    """Size, resolution and format for writing the figure to a file.

    The save button of the matplotlib toolbar writes the figure at whatever
    size the window happens to have, so the same plot saved from a maximised
    window and from a small one come out with text of quite different relative
    size.  A figure for a paper has a width -- one column, two columns -- and
    its text has to be readable at that width.
    """

    def __init__(self, settings: FigureExport,
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle('Export figure')
        self.settings = settings

        self.preset = QtWidgets.QComboBox()
        self.preset.addItem('Custom', None)
        for name, width, height in FigureExport.PRESETS:
            self.preset.addItem(name, (width, height))

        # Not `self.width` / `self.height`: those are QWidget's own methods,
        # and Qt calls them.  Shadowing them makes the dialog raise
        # "QDoubleSpinBox object is not callable" as it lays itself out.
        self.widthBox = QtWidgets.QDoubleSpinBox()
        self.heightBox = QtWidgets.QDoubleSpinBox()
        for box in (self.widthBox, self.heightBox):
            box.setRange(10.0, 1000.0)
            box.setSuffix(' mm')
            box.setDecimals(1)
        self.widthBox.setValue(settings.width)
        self.heightBox.setValue(settings.height)

        self.dpiBox = QtWidgets.QSpinBox()
        self.dpiBox.setRange(50, 1200)
        self.dpiBox.setValue(settings.dpi)
        self.dpiBox.setToolTip(
            'Resolution of the raster formats.  300 is what journals ask for; '
            'it does nothing for PDF or SVG, which have no pixels.')

        self.formatBox = QtWidgets.QComboBox()
        for name, value in EXPORT_FORMATS:
            self.formatBox.addItem(name, value)
        index = self.formatBox.findData(settings.format)
        self.formatBox.setCurrentIndex(max(0, index))

        self.transparent = QtWidgets.QCheckBox('Transparent background')
        self.transparent.setChecked(settings.transparent)
        self.transparent.setToolTip('For a slide that is not white.')

        self.tight = QtWidgets.QCheckBox('Crop to the drawing')
        self.tight.setChecked(settings.tight)
        self.tight.setToolTip(
            'Cut the empty margin off.  Leave it on unless several figures '
            'have to line up, which needs them all the same size.')

        buttons = QtWidgets.QDialogButtonBox()
        self.saveButton = buttons.addButton('Save...',
                                            QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)

        form = QtWidgets.QFormLayout()
        form.addRow('Size', self.preset)
        form.addRow('Width', self.widthBox)
        form.addRow('Height', self.heightBox)
        form.addRow('Format', self.formatBox)
        form.addRow('Resolution', self.dpiBox)
        form.addRow('', self.transparent)
        form.addRow('', self.tight)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.preset.currentIndexChanged.connect(self._presetChosen)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    @Slot()
    def _presetChosen(self) -> None:
        size = self.preset.currentData()
        if size is None:
            return
        self.widthBox.setValue(size[0])
        self.heightBox.setValue(size[1])

    def accept(self) -> None:
        self.apply()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export figure', f'figure.{self.settings.format}',
            f'{self.settings.format.upper()} '
            f'(*.{self.settings.format});;all files (*)')
        if not path:
            return           # cancelled at the file dialog: stay open
        self.path = path
        super().accept()

    def apply(self) -> None:
        """Read the form into the settings object."""
        self.settings.width = self.widthBox.value()
        self.settings.height = self.heightBox.value()
        self.settings.dpi = self.dpiBox.value()
        self.settings.format = str(self.formatBox.currentData())
        self.settings.transparent = self.transparent.isChecked()
        self.settings.tight = self.tight.isChecked()


# A toolbar for setting options on the MPL autoplot
class AutoPlotToolBar(QtWidgets.QToolBar):
    """
    A toolbar that allows the user to configure AutoPlot.

    Currently, the user can select between the plots that are possible, given
    the data that AutoPlot has.
    """

    #: signal emitted when the plot type has been changed
    plotTypeSelected = Signal(PlotType)

    #: signal emitted when the complex data option has been changed
    complexRepresentationSelected = Signal(ComplexRepresentation)

    #: signal emitted when error-bar visibility has been changed
    errorBarsSelected = Signal(bool)

    #: signal emitted when an error-bar source has been selected
    errorBarSourceSelected = Signal(str, str)

    #: signal emitted when point size, line width or fit color have changed
    plotStyleChanged = Signal()

    #: signal emitted when an axis scale has been changed (x scale, y scale)
    axisScaleSelected = Signal(str, str)

    #: signal emitted when the title, an axis label or the legend changed
    plotLabelsChanged = Signal()

    #: signal emitted when an axis limit has been changed
    axisLimitsChanged = Signal()

    #: signal emitted when the limits should be read off the plot as it is
    axisLimitsFromView = Signal()

    #: signal emitted when the colormap or the color range changed
    plotColorsChanged = Signal()

    #: signal emitted when the figure should be written to a file
    exportRequested = Signal()

    #: signal emitted when an appearance should be saved / loaded / reset
    appearanceSaveRequested = Signal()
    appearanceLoadRequested = Signal()
    appearanceResetRequested = Signal()

    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        """Constructor for :class:`AutoPlotToolBar`"""

        super().__init__(name, parent=parent)

        self.plotasMultiTraces = self.addAction(get_multiTracePlotIcon(),
                                                'Multiple traces')
        self.plotasMultiTraces.setCheckable(True)
        self.plotasMultiTraces.triggered.connect(
            lambda: self.selectPlotType(PlotType.multitraces))

        self.plotasSingleTraces = self.addAction(get_singleTracePlotIcon(),
                                                 'Individual traces')
        self.plotasSingleTraces.setCheckable(True)
        self.plotasSingleTraces.triggered.connect(
            lambda: self.selectPlotType(PlotType.singletraces))

        self.addSeparator()

        self.plotasImage = self.addAction(get_imagePlotIcon(),
                                          'Image')
        self.plotasImage.setCheckable(True)
        self.plotasImage.triggered.connect(
            lambda: self.selectPlotType(PlotType.image))

        self.plotasMesh = self.addAction(get_colormeshPlotIcon(),
                                         'Color mesh')
        self.plotasMesh.setCheckable(True)
        self.plotasMesh.triggered.connect(
            lambda: self.selectPlotType(PlotType.colormesh))

        self.plotasScatter2d = self.addAction(get_scatterPlot2dIcon(),
                                              'Scatter 2D')
        self.plotasScatter2d.setCheckable(True)
        self.plotasScatter2d.triggered.connect(
            lambda: self.selectPlotType(PlotType.scatter2d))

        # How complex data is shown.  Behind one button rather than as seven
        # more entries in the row: they are mutually exclusive, so the button's
        # own text can say which one is on, and the row was wide enough to push
        # everything after it off the screen on a laptop.
        self.addSeparator()

        self.complexMenu = QtWidgets.QMenu(parent=self)
        self.complexGroup = QtGui.QActionGroup(self)
        self.complexGroup.setExclusive(True)

        def complexAction(text: str, representation: ComplexRepresentation,
                          tip: str = '') -> Any:
            action = self.complexMenu.addAction(text)
            action.setCheckable(True)
            if tip:
                action.setToolTip(tip)
            action.setData(text)
            self.complexGroup.addAction(action)
            action.triggered.connect(
                lambda: self.selectComplexType(representation))
            return action

        self.plotReal = complexAction(
            'Real', ComplexRepresentation.real, 'The real part alone.')
        self.plotReIm = complexAction(
            'Re/Im', ComplexRepresentation.realAndImag,
            'Real and imaginary parts in one panel.')
        self.plotReImSep = complexAction(
            'Split Re/Im', ComplexRepresentation.realAndImagSeparate,
            'Real and imaginary parts in a panel each.')
        self.plotMag = complexAction(
            'Mag', ComplexRepresentation.mag, 'Magnitude only, in one panel.')
        self.plotPhase = complexAction(
            'Phase', ComplexRepresentation.phase, 'Phase only, in one panel.')
        self.plotMagPhase = complexAction(
            'Mag/Phase', ComplexRepresentation.magAndPhase,
            'Magnitude and phase in a panel each.')
        self.plotComplexPlane = complexAction(
            'Complex plane', ComplexRepresentation.complexPlane,
            'The trace in the complex plane (the resonance circle).')

        self.complexButton = QtWidgets.QToolButton()
        self.complexButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.complexButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.complexButton.setMenu(self.complexMenu)
        self.complexButton.setToolTip('How complex data is shown.')
        self.addWidget(self.complexButton)
        self.complexButtonAction = self.actions()[-1]

        self.addSeparator()

        self.showErrorBars = self.addAction('Error bars')
        self.showErrorBars.setCheckable(True)
        self.showErrorBars.setChecked(True)
        self.showErrorBars.triggered.connect(
            lambda: self.errorBarsSelected.emit(self.showErrorBars.isChecked()))

        self.errorBarMenu = QtWidgets.QMenu(parent=self)
        self.errorBarButton = QtWidgets.QToolButton()
        self.errorBarButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.errorBarButton.setText('Error source')
        self.errorBarButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.errorBarButton.setMenu(self.errorBarMenu)
        self.addWidget(self.errorBarButton)
        self._errorBarMenuRefs: List[Any] = []

        # Point size, line widths and fit color.  These live behind one button
        # rather than as four more widgets in the row: the toolbar is already
        # long, and these are set once and then left alone.
        self.addSeparator()
        self.styleWidget = PlotStyleWidget(self)
        self.styleWidget.changed.connect(self.plotStyleChanged)
        styleMenu = QtWidgets.QMenu(parent=self)
        styleAction = QtWidgets.QWidgetAction(styleMenu)
        styleAction.setDefaultWidget(self.styleWidget)
        styleMenu.addAction(styleAction)
        self.styleButton = QtWidgets.QToolButton()
        self.styleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.styleButton.setText('Style')
        self.styleButton.setToolTip(
            'Size of the data points, thickness of the lines, and the color of '
            'the fit curves.')
        self.styleButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.styleButton.setMenu(styleMenu)
        self.addWidget(self.styleButton)
        self._styleMenu = styleMenu

        # Title, axis labels and legend.  Same reasoning as the style button:
        # set once per figure, then left alone.
        self.labelsWidget = PlotLabelsWidget(self)
        self.labelsWidget.changed.connect(self.plotLabelsChanged)
        labelsMenu = QtWidgets.QMenu(parent=self)
        labelsAction = QtWidgets.QWidgetAction(labelsMenu)
        labelsAction.setDefaultWidget(self.labelsWidget)
        labelsMenu.addAction(labelsAction)
        self.labelsButton = QtWidgets.QToolButton()
        self.labelsButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.labelsButton.setText('Labels')
        self.labelsButton.setToolTip(
            'Title, axis labels, and where the legend goes and what it calls '
            'each trace.')
        self.labelsButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.labelsButton.setMenu(labelsMenu)
        self.addWidget(self.labelsButton)
        self._labelsMenu = labelsMenu

        # Scales and ranges.  Qi against the photon number is read on a log x
        # axis; so is anything spanning decades.
        self.axesWidget = PlotAxesWidget(self)
        self.axesWidget.scalesChanged.connect(self.axisScaleSelected)
        self.axesWidget.limitsChanged.connect(self.axisLimitsChanged)
        self.axesWidget.takeFromView.connect(self.axisLimitsFromView)
        # The tests and the older code reach for these directly.
        self.xScaleBox = self.axesWidget.xScaleBox
        self.yScaleBox = self.axesWidget.yScaleBox
        self.scaleButton = self._popupButton(
            'Axes', self.axesWidget,
            'Linear or logarithmic axes, and where they start and stop.')

        # The color scale of a 2-D plot.  Only useful when there is one, so it
        # is hidden for line plots rather than sitting there greyed out.
        self.colorsWidget = PlotColorsWidget(self)
        self.colorsWidget.changed.connect(self.plotColorsChanged)
        self.colorsButton = self._popupButton(
            'Colors', self.colorsWidget,
            'Colormap, color range and colorbar label of a 2-D plot.')
        self.colorsButtonAction = self.actions()[-1]
        self.colorsButtonAction.setVisible(False)

        # Writing the figure out at a size that is not the window's.
        self.exportButton = QtWidgets.QToolButton()
        self.exportButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.exportButton.setText('Export...')
        self.exportButton.setToolTip(
            'Write the figure to a file at a size and resolution of your own '
            'choosing, rather than at the size of the window.')
        self.exportButton.clicked.connect(self.exportRequested)
        self.addWidget(self.exportButton)

        # Keeping a look, so that the next dataset does not have to be typed
        # into ten fields again.
        presetMenu = QtWidgets.QMenu(parent=self)
        presetMenu.addAction('Save appearance...',
                             self.appearanceSaveRequested.emit)
        presetMenu.addAction('Load appearance...',
                             self.appearanceLoadRequested.emit)
        presetMenu.addSeparator()
        presetMenu.addAction('Back to automatic',
                             self.appearanceResetRequested.emit)
        self.presetButton = QtWidgets.QToolButton()
        self.presetButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.presetButton.setText('Appearance')
        self.presetButton.setToolTip(
            'Save everything this toolbar sets to a file, and put it back on '
            'another dataset.')
        self.presetButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.presetButton.setMenu(presetMenu)
        self.addWidget(self.presetButton)
        self._presetMenu = presetMenu

        self.plotTypeActions = OrderedDict({
            PlotType.multitraces: self.plotasMultiTraces,
            PlotType.singletraces: self.plotasSingleTraces,
            PlotType.image: self.plotasImage,
            PlotType.colormesh: self.plotasMesh,
            PlotType.scatter2d: self.plotasScatter2d,
        })

        self.ComplexActions = OrderedDict({
            ComplexRepresentation.real: self.plotReal,
            ComplexRepresentation.realAndImag: self.plotReIm,
            ComplexRepresentation.realAndImagSeparate: self.plotReImSep,
            ComplexRepresentation.mag: self.plotMag,
            ComplexRepresentation.phase: self.plotPhase,
            ComplexRepresentation.magAndPhase: self.plotMagPhase,
            ComplexRepresentation.complexPlane: self.plotComplexPlane,
        })

        self._currentPlotType = PlotType.empty
        self._currentlyAllowedPlotTypes: Tuple[PlotType, ...] = ()

        self._currentComplex = ComplexRepresentation.realAndImag
        self.ComplexActions[self._currentComplex].setChecked(True)
        self._currentlyAllowedComplexTypes: Tuple[ComplexRepresentation, ...] = ()
        self._showComplexChoice()

    def _showComplexChoice(self) -> None:
        """Write the current representation on the button.

        The row no longer shows seven buttons with one highlighted, so the
        button has to say which one is on.
        """
        action = self.ComplexActions.get(self._currentComplex)
        name = str(action.data()) if action is not None else ''
        self.complexButton.setText(f'Complex: {name}' if name else 'Complex')

    def _popupButton(self, text: str, widget: QtWidgets.QWidget,
                     tip: str) -> QtWidgets.QToolButton:
        """A toolbar button that drops down a form.

        The forms are set once per figure and then left alone, so they belong
        behind a button rather than as ten more widgets in a row that is
        already long enough to push things off a laptop screen.
        """
        menu = QtWidgets.QMenu(parent=self)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)
        button = QtWidgets.QToolButton()
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        button.setText(text)
        button.setToolTip(tip)
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        button.setMenu(menu)
        self.addWidget(button)
        # The menu is owned by the button, but Qt does not keep the reference.
        setattr(self, f'_{text.lower().rstrip(".")}Menu', menu)
        return button

    def setAxisScales(self, xScale: str, yScale: str) -> None:
        """Show these scales without emitting anything."""
        self.axesWidget.setScales(xScale, yScale)

    def setAxisLimits(self, axes: 'PlotAxes') -> None:
        """Hand the limits object the toolbar edits in place."""
        self.axesWidget.setAxes(axes)

    def showAxisLimits(self) -> None:
        """Write the limits back into the form, after they changed elsewhere."""
        self.axesWidget.showLimits()

    def setPlotColors(self, colors: 'PlotColors', labels: 'PlotLabels') -> None:
        """Hand the color objects the toolbar edits in place."""
        self.colorsWidget.setColors(colors, labels)

    def setColorsAvailable(self, available: bool) -> None:
        """Show the color button only where there is a color scale."""
        self.colorsButtonAction.setVisible(available)

    def setPlotStyle(self, style: 'PlotStyle') -> None:
        """Hand the style object the toolbar edits in place."""
        self.styleWidget.setStyle(style)

    def setPlotLabels(self, labels: 'PlotLabels') -> None:
        """Hand the labels object the toolbar edits in place."""
        self.labelsWidget.setLabels(labels)

    def setLegendEntries(self, entries: Sequence[str]) -> None:
        """List the traces that are in the plot now, so they can be renamed."""
        self.labelsWidget.setEntries(entries)

    def setAutomaticLabels(self, title: str = '', xLabel: str = '',
                           yLabel: str = '') -> None:
        """Show what the empty fields would say."""
        self.labelsWidget.setAutomatic(title, xLabel, yLabel)

    def setErrorBarOptions(
        self, data: Optional[DataDictBase], sources: Optional[Dict[str, str]] = None
    ) -> None:
        """Populate the error-bar source menu from the current dataset."""
        self.errorBarMenu.clear()
        self._errorBarMenuRefs = []
        self.errorBarButton.setEnabled(data is not None)
        if data is None:
            return

        if sources is None:
            sources = {}

        dependents = plottableDependents(data, sources)
        if len(dependents) == 0:
            noData = self.errorBarMenu.addAction('No plottable dependents')
            noData.setEnabled(False)
            return

        for dependent in dependents:
            current = sources.get(dependent, ERROR_BAR_AUTO)
            depMenu = QtWidgets.QMenu(dependent, self.errorBarMenu)
            self.errorBarMenu.addMenu(depMenu)
            group = QtWidgets.QActionGroup(depMenu)
            group.setExclusive(True)
            self._errorBarMenuRefs.extend([depMenu, group])

            for label, source in [('Auto', ERROR_BAR_AUTO), ('None', ERROR_BAR_NONE)]:
                action = depMenu.addAction(label)
                action.setCheckable(True)
                action.setChecked(current == source)
                group.addAction(action)
                action.triggered.connect(
                    lambda _checked=False, dep=dependent, src=source:
                        self.errorBarSourceSelected.emit(dep, src)
                )

            candidates = errorBarDataNames(data, dependent)
            if len(candidates) > 0:
                depMenu.addSeparator()

            for candidate in candidates:
                action = depMenu.addAction(candidate)
                action.setCheckable(True)
                action.setChecked(current == candidate)
                group.addAction(action)
                action.triggered.connect(
                    lambda _checked=False, dep=dependent, src=candidate:
                        self.errorBarSourceSelected.emit(dep, src)
                )

    def selectPlotType(self, plotType: PlotType) -> None:
        """makes sure that the selected `plotType` is active (checked), all
        others are not active.

        This method should be used to catch a trigger from the UI.

        If the active plot type has been changed by using this method,
        we emit `plotTypeSelected`.

        :param plotType: type of plot
        """

        # deselect all other types
        for k, v in self.plotTypeActions.items():
            if k is not plotType and v is not None:
                v.setChecked(False)

        # don't want un-toggling - can only be done by selecting another type
        self.plotTypeActions[plotType].setChecked(True)

        if plotType is not self._currentPlotType:
            self._currentPlotType = plotType
            self.plotTypeSelected.emit(plotType)

    def setAllowedPlotTypes(self, *args: PlotType) -> None:
        """Disable all plot type choices that are not allowed.
        If the current selection is now disabled, instead select the first
        enabled one.

        :param args: which types of plots can be selected.
        """

        if args == self._currentlyAllowedPlotTypes:
            return

        for k, v in self.plotTypeActions.items():
            if k not in args:
                v.setChecked(False)
                v.setEnabled(False)
            else:
                v.setEnabled(True)

        if self._currentPlotType not in args:
            self._currentPlotType = PlotType.empty
            for k, v in self.plotTypeActions.items():
                if k in args:
                    v.setChecked(True)
                    self._currentPlotType = k
                    break

            self.plotTypeSelected.emit(self._currentPlotType)

        self._currentlyAllowedPlotTypes = args

    def selectComplexType(self, comp: ComplexRepresentation) -> None:
        """makes sure that the selected `comp` is active (checked), all
        others are not active.
        This method should be used to catch a trigger from the UI.
        If the active plot type has been changed by using this method,
        we emit `complexPolarSelected`.
        """
        # deselect all other types
        for k, v in self.ComplexActions.items():
            if k is not comp and v is not None:
                v.setChecked(False)

        # don't want un-toggling - can only be done by selecting another type
        self.ComplexActions[comp].setChecked(True)
        self._showComplexChoice()

        if comp is not self._currentComplex:
            self._currentComplex = comp
            self.complexRepresentationSelected.emit(self._currentComplex)

    def setAllowedComplexTypes(self, *complexOptions: ComplexRepresentation) -> None:
        """Disable all complex representation choices that are not allowed.
        If the current selection is now disabled, instead select the first
        enabled one.
        """

        if complexOptions == self._currentlyAllowedComplexTypes:
            return

        for k, v in self.ComplexActions.items():
            if k not in complexOptions:
                v.setChecked(False)
                v.setEnabled(False)
            else:
                v.setEnabled(True)

        if self._currentComplex not in complexOptions:
            self._currentComplex = ComplexRepresentation.realAndImag
            for k, v in self.ComplexActions.items():
                if k in complexOptions:
                    v.setChecked(True)
                    self._currentComplex = k
                    break

            self.complexRepresentationSelected.emit(self._currentComplex)

        # Only one choice (the data is real): the button says nothing useful.
        self.complexButtonAction.setVisible(len(complexOptions) > 1)
        self._showComplexChoice()
        self._currentlyAllowedComplexTypes = complexOptions


class AutoPlot(MPLPlotWidget):
    """A widget for plotting with matplotlib.

    When data is set using :meth:`setData` the class will automatically try
    to determine what good plot options are from the structure of the data.

    User options (for different types of plots, styling, etc) are
    presented through a toolbar.
    """

    def __init__(self, parent: Optional[PlotWidgetContainer] = None):
        super().__init__(parent=parent)

        self.plotDataType = PlotDataType.unknown
        self.plotType = PlotType.empty

        # The default complex behavior is set here.
        self.complexRepresentation = ComplexRepresentation.realAndImag
        self.showErrorBars = True
        self.errorBarSources: Dict[str, str] = {}
        #: point size, line widths and fit color; the toolbar edits this
        self.plotStyle = PlotStyle()
        #: title, axis labels and legend; the toolbar edits this
        self.plotLabels = PlotLabels()
        #: where the axes start and stop; the toolbar edits this
        self.plotAxes = PlotAxes()
        #: colormap and color range of a 2-D plot
        self.plotColors = PlotColors()
        #: size, resolution and format the figure is written out with
        self.figureExport = FigureExport()
        #: the dialog for per-trace color / points / line, made on first use
        self.traceStyleDialog: Optional[TraceStyleDialog] = None
        #: scale of the x and y axes ('linear' or 'log')
        self.xScale = 'linear'
        self.yScale = 'linear'

        # A toolbar for configuring the plot
        self.plotOptionsToolBar = AutoPlotToolBar('Plot options', self)
        layout = cast(QtWidgets.QVBoxLayout, self.layout())
        layout.insertWidget(1, self.plotOptionsToolBar)

        self.plotOptionsToolBar.plotTypeSelected.connect(
            self._plotTypeFromToolBar
        )
        self.plotOptionsToolBar.complexRepresentationSelected.connect(
            self._complexPreferenceFromToolBar
        )
        self.plotOptionsToolBar.errorBarsSelected.connect(
            self._errorBarsPreferenceFromToolBar
        )
        self.plotOptionsToolBar.errorBarSourceSelected.connect(
            self._errorBarSourceFromToolBar
        )
        self.plotOptionsToolBar.plotStyleChanged.connect(
            self._plotStyleFromToolBar
        )
        self.plotOptionsToolBar.setPlotStyle(self.plotStyle)
        self.plotOptionsToolBar.plotLabelsChanged.connect(
            self._plotLabelsFromToolBar
        )
        self.plotOptionsToolBar.setPlotLabels(self.plotLabels)
        self.plotOptionsToolBar.axisScaleSelected.connect(
            self._axisScalesFromToolBar
        )
        self.plotOptionsToolBar.setAxisScales(self.xScale, self.yScale)
        self.plotOptionsToolBar.axisLimitsChanged.connect(self._plotData)
        self.plotOptionsToolBar.axisLimitsFromView.connect(
            self._limitsFromView
        )
        self.plotOptionsToolBar.setAxisLimits(self.plotAxes)
        self.plotOptionsToolBar.plotColorsChanged.connect(self._plotData)
        self.plotOptionsToolBar.setPlotColors(self.plotColors, self.plotLabels)
        self.plotOptionsToolBar.exportRequested.connect(self._exportFigure)
        self.plotOptionsToolBar.appearanceSaveRequested.connect(
            self._saveAppearance
        )
        self.plotOptionsToolBar.appearanceLoadRequested.connect(
            self._loadAppearance
        )
        self.plotOptionsToolBar.appearanceResetRequested.connect(
            self._resetAppearance
        )
        self.plotOptionsToolBar.styleWidget.perTraceButton.clicked.connect(
            self._showTraceStyles
        )

        scaling = dpiScalingFactor(self)
        iconSize = int(36 + 8*(scaling - 1))
        self.plotOptionsToolBar.setIconSize(QtCore.QSize(iconSize, iconSize))
        self.setMinimumSize(int(640*scaling), int(480*scaling))

    def updatePlot(self) -> None:
        self.plot.draw()
        QtCore.QCoreApplication.processEvents()

    def setData(self, data: Optional[DataDictBase]) -> None:
        """Analyses data and determines whether/what to plot.

        :param data: input data
        """
        super().setData(data)
        self.plotDataType = determinePlotDataType(data)
        self._processPlotTypeOptions()
        self._processComplexTypeOptions()
        self._processErrorBarOptions()
        self._plotData()

    def _processPlotTypeOptions(self) -> None:
        """Given the current data type, figure out what the plot options are."""
        if self.plotDataType == PlotDataType.grid2d:
            self.plotOptionsToolBar.setAllowedPlotTypes(
                PlotType.image, PlotType.colormesh, PlotType.scatter2d
            )

        elif self.plotDataType == PlotDataType.scatter2d:
            self.plotOptionsToolBar.setAllowedPlotTypes(
                PlotType.scatter2d,
            )

        elif self.plotDataType in [PlotDataType.scatter1d,
                                   PlotDataType.line1d]:
            self.plotOptionsToolBar.setAllowedPlotTypes(
                PlotType.multitraces, PlotType.singletraces,
            )

        else:
            self.plotOptionsToolBar.setAllowedPlotTypes()

    def _processComplexTypeOptions(self) -> None:
        """Given data is complex or not, define what complex options to be selected."""
        if self.data is not None:
            if self.dataIsComplex():
                self.plotOptionsToolBar.setAllowedComplexTypes(
                    ComplexRepresentation.real,
                    ComplexRepresentation.realAndImag,
                    ComplexRepresentation.realAndImagSeparate,
                    ComplexRepresentation.mag,
                    ComplexRepresentation.phase,
                    ComplexRepresentation.magAndPhase,
                    ComplexRepresentation.complexPlane,
                )
            else:
                self.plotOptionsToolBar.setAllowedComplexTypes(
                    ComplexRepresentation.real
                )

    @Slot(PlotType)
    def _plotTypeFromToolBar(self, plotType: PlotType) -> None:
        if plotType is not self.plotType:
            self.plotType = plotType
            self._plotData()

    @Slot(ComplexRepresentation)
    def _complexPreferenceFromToolBar(self, complexRepresentation: ComplexRepresentation) -> None:
        if complexRepresentation is not self.complexRepresentation:
            self.complexRepresentation = complexRepresentation
            self._plotData()

    @Slot(bool)
    def _errorBarsPreferenceFromToolBar(self, showErrorBars: bool) -> None:
        if showErrorBars is not self.showErrorBars:
            self.showErrorBars = showErrorBars
            self._plotData()

    @Slot(str, str)
    def _axisScalesFromToolBar(self, xScale: str, yScale: str) -> None:
        if (xScale, yScale) != (self.xScale, self.yScale):
            self.xScale, self.yScale = xScale, yScale
            self._plotData()

    def _clearPlot(self) -> None:
        """Empty the figure.

        Leaving the previous figure up when there is nothing to draw is worse
        than an empty panel: it shows numbers from a dataset that is no longer
        selected, and every change looks as though it did nothing.
        """
        if self.plot.fig.axes:
            self.plot.clearFig()
            self.updatePlot()

    @Slot()
    def _plotStyleFromToolBar(self) -> None:
        """Redraw with the point size / line widths / fit color from the toolbar."""
        self._plotData()

    @Slot()
    def _plotLabelsFromToolBar(self) -> None:
        """Redraw with the title / axis labels / legend from the toolbar."""
        self._plotData()

    @Slot(str, str)
    def _errorBarSourceFromToolBar(self, dependent: str, source: str) -> None:
        if self.errorBarSources.get(dependent, ERROR_BAR_AUTO) != source:
            self.errorBarSources[dependent] = source
            self._processErrorBarOptions()
            self._plotData()

    def _processErrorBarOptions(self) -> None:
        if self.data is None:
            self.errorBarSources = {}
            self.plotOptionsToolBar.setErrorBarOptions(None, self.errorBarSources)
            return

        self.errorBarSources = {
            name: source
            for name, source in self.errorBarSources.items()
            if name in self.data
        }
        self.plotOptionsToolBar.setErrorBarOptions(self.data, self.errorBarSources)

    def _plotData(self) -> None:
        """Plot the data using previously determined data and plot types."""

        if self.plotDataType is PlotDataType.unknown:
            logger.debug("No plottable data.")
            self._clearPlot()
            return
        if self.plotType is PlotType.empty:
            logger.debug("No plot routine determined.")
            self._clearPlot()
            return

        assert self.data is not None

        # Set the font size for the size the canvas has now, before any artist
        # is made: they are created at whatever `rcParams` says at that moment.
        self.plot.applyFontSize(rescaleExisting=False)

        with FigureMaker(self.plot.fig) as fm:
            fm.plotType = self.plotType
            fm.style = self.plotStyle
            fm.labels = self.plotLabels
            fm.axisLimits = self.plotAxes
            fm.colors = self.plotColors
            fm.xScale, fm.yScale = self.xScale, self.yScale
            if not self.dataIsComplex():
                fm.complexRepresentation = ComplexRepresentation.real
            else:
                fm.complexRepresentation = self.complexRepresentation

            indeps = self.data.axes()
            # Fit curves last: they take their color from the data they belong
            # to, and they are drawn on top of it.
            for dn in sortFitsLast(
                    self.data, plottableDependents(self.data, self.errorBarSources)):
                dvals = self.data.data_vals(dn)
                source = self.errorBarSources.get(dn, ERROR_BAR_AUTO)
                yerr = errorBarData(self.data, dn, source) if self.showErrorBars else None
                kw: Dict[str, Any] = {'_name': dn}
                fitOf = fitSourceName(self.data, dn)
                if fitOf is not None:
                    kw['_fitOf'] = fitOf
                if yerr is not None:
                    kw['_errorBarData'] = yerr
                plotId = fm.addData(
                    *[np.asanyarray(self.data.data_vals(n)) for n in indeps] + [dvals],
                    labels=[str(self.data.label(n)) for n in indeps] + [str(self.data.label(dn))],
                    plotDataType=self.plotDataType,
                    **kw)

        self.setMeta(self.data)
        self._applyTitle()
        self._reportPlotLabels(fm.automaticLabels)
        self.updatePlot()

    def _applyTitle(self) -> None:
        """Put the operator's title on the figure, or take it off.

        ``setMeta`` has just written the dataset's own title (the path of the
        file), which is what the empty field means.
        """
        if not self.plotLabels.showTitle:
            self.plot.setFigureTitle('')
        elif self.plotLabels.title:
            self.plot.setFigureTitle(renderableText(self.plotLabels.title))

    def _automaticTitle(self) -> str:
        if self.data is not None and self.data.has_meta('title'):
            return str(self.data.meta_val('title'))
        return ''

    def _reportPlotLabels(self, automatic: Optional[Dict[str, str]] = None) \
            -> None:
        """Tell the toolbar what the figure ended up with.

        The legend entries can only be known after the figure is drawn (a
        complex trace shown as Re and Im is two of them), and the axis labels
        are what the empty fields stand for.
        """
        automatic = automatic or {}
        axes = self.plot.fig.axes
        entries: List[str] = []
        if axes:
            _, entries = axes[0].get_legend_handles_labels()
        self.plotOptionsToolBar.setAutomaticLabels(
            self._automaticTitle(),
            automatic.get('x', ''), automatic.get('y', ''))
        self.plotOptionsToolBar.setLegendEntries(entries)
        self.plotOptionsToolBar.colorsWidget.setAutomatic(
            automatic.get('colorbar', ''))
        self.plotOptionsToolBar.setColorsAvailable(
            self.plotType in (PlotType.image, PlotType.colormesh,
                              PlotType.scatter2d))
        if self.traceStyleDialog is not None:
            self.traceStyleDialog.setTraces(entries)

    # -- the things the toolbar asks for -----------------------------------

    @Slot()
    def _limitsFromView(self) -> None:
        """Keep the range the plot is showing, as numbers.

        What makes the zoom button useful for a figure that has to be made
        again: get the view right by eye, then press this.
        """
        axes = self.plot.fig.axes
        if not axes:
            return
        self.plotAxes.setFromAxes(axes[0])
        self.plotOptionsToolBar.showAxisLimits()
        self._plotData()

    @Slot()
    def _showTraceStyles(self) -> None:
        """Open the per-trace color / points / line table."""
        if self.traceStyleDialog is None:
            self.traceStyleDialog = TraceStyleDialog(self)
            self.traceStyleDialog.setStyle(self.plotStyle)
            self.traceStyleDialog.changed.connect(self._plotData)
        axes = self.plot.fig.axes
        entries: List[str] = []
        if axes:
            _, entries = axes[0].get_legend_handles_labels()
        self.traceStyleDialog.setTraces(entries)
        self.traceStyleDialog.show()
        self.traceStyleDialog.raise_()

    @Slot()
    def _exportFigure(self) -> None:
        """Write the figure to a file at a size of the operator's choosing."""
        dialog = ExportDialog(self.figureExport, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            exportFigure(self.plot.fig, dialog.path, self.figureExport,
                         self.plot.applyFontSize)
        except Exception as exc:  # noqa: BLE001 -- never take the viewer down
            QtWidgets.QMessageBox.warning(
                self, 'Export failed', f'{type(exc).__name__}: {exc}')

    @Slot()
    def _saveAppearance(self) -> None:
        """Write everything the toolbar sets to a file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save the appearance of this figure', 'appearance.json',
            'plottr appearance (*.json);;all files (*)')
        if not path:
            return
        values = appearanceToDict(self.plotStyle, self.plotLabels,
                                  self.plotAxes, self.plotColors,
                                  self.figureExport, self.xScale, self.yScale)
        try:
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(values, file, indent=2, ensure_ascii=False)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, 'Could not save', f'{type(exc).__name__}: {exc}')

    @Slot()
    def _loadAppearance(self) -> None:
        """Put a saved appearance onto this figure."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load an appearance', '',
            'plottr appearance (*.json);;all files (*)')
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as file:
                values = json.load(file)
            self.applyAppearanceValues(values)
        except Exception as exc:  # noqa: BLE001 -- a bad file is not a crash
            QtWidgets.QMessageBox.warning(
                self, 'Could not load', f'{type(exc).__name__}: {exc}')

    def applyAppearanceValues(self, values: Dict[str, Any]) -> None:
        """Apply a saved appearance and redraw.

        Separate from the file dialog so that it can be tested, and so that
        anything else that has a saved look can use it.
        """
        self.xScale, self.yScale = applyAppearance(
            values, self.plotStyle, self.plotLabels, self.plotAxes,
            self.plotColors, self.figureExport)
        self._showAppearance()
        self._plotData()

    @Slot()
    def _resetAppearance(self) -> None:
        """Back to what the data says, every field at once."""
        self.plotStyle = PlotStyle()
        self.plotLabels = PlotLabels()
        self.plotAxes = PlotAxes()
        self.plotColors = PlotColors()
        self.xScale = self.yScale = 'linear'
        if self.traceStyleDialog is not None:
            self.traceStyleDialog.setStyle(self.plotStyle)
        self._showAppearance()
        self._plotData()

    def _showAppearance(self) -> None:
        """Point every form at the current objects and show their values."""
        bar = self.plotOptionsToolBar
        bar.setPlotStyle(self.plotStyle)
        bar.setPlotLabels(self.plotLabels)
        bar.setAxisLimits(self.plotAxes)
        bar.setPlotColors(self.plotColors, self.plotLabels)
        bar.setAxisScales(self.xScale, self.yScale)
