"""``plottr.plot.mpl.autoplot`` -- This module contains the tools for automatic plotting with matplotlib.
"""

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

#: ``fitColor`` value meaning "whatever color the data got".
FIT_COLOR_AUTO = ''

#: Axis scales the toolbar offers.
AXIS_SCALES = (('Linear', 'linear'), ('Log', 'log'))

#: ``legendLocation`` values that are not matplotlib ``loc`` strings.
LEGEND_AUTO = ''        #: a legend only where one is needed (the old behaviour)
LEGEND_NONE = 'none'    #: never
LEGEND_OUTSIDE = 'outside'  #: beside the axes, where it hides no data

#: Where the legend can go.  Everything but the first three is matplotlib's own
#: ``loc``, spelled the way matplotlib spells it.
LEGEND_LOCATIONS = (
    ('Automatic', LEGEND_AUTO),
    ('Hidden', LEGEND_NONE),
    ('Outside, right', LEGEND_OUTSIDE),
    ('Best', 'best'),
    ('Upper right', 'upper right'),
    ('Upper left', 'upper left'),
    ('Lower left', 'lower left'),
    ('Lower right', 'lower right'),
    ('Center left', 'center left'),
    ('Center right', 'center right'),
    ('Upper center', 'upper center'),
    ('Lower center', 'lower center'),
    ('Center', 'center'),
)


def _hasPositiveValues(values: Any) -> bool:
    """Whether a log axis can show this data at all."""
    array = np.asanyarray(values)
    if not np.issubdtype(array.dtype, np.number):
        return False
    with np.errstate(invalid='ignore'):
        return bool(np.any(np.isfinite(array) & (array > 0)))


def renderableText(text: str) -> str:
    """``text``, with maths matplotlib cannot parse turned into plain text.

    Labels are worth typing maths into -- `$Q_i$`, `$\\langle n \\rangle$` -- and
    matplotlib raises while *drawing* a `$...$` it cannot parse, which takes
    the figure down rather than showing a bad label.  A string it refuses is
    escaped so the dollars come out as dollars, and the operator sees what is
    wrong instead of an empty window.
    """
    if '$' not in text:
        return text
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib import mathtext
        mathtext.MathTextParser('agg').parse(text, 72, FontProperties())
    except Exception:  # noqa: BLE001 -- any parse failure means "not maths"
        return text.replace('$', r'\$')
    return text


class PlotLabels:
    """What the figure says: title, axis labels, legend.

    Everything here is empty by default, and empty means "whatever the data
    says" -- the dataset's title, the column labels, a legend only where one
    is needed.  Typing something replaces that one piece and leaves the rest
    automatic, so a figure keeps following the data until it is told not to.

    Legend entries are keyed by the label matplotlib would have used, not by
    the column name: a complex trace drawn as Re and Im is two entries, and
    both have to be nameable.
    """

    def __init__(self) -> None:
        #: figure title; empty takes the one from the dataset
        self.title: str = ''
        #: whether to show a title at all
        self.showTitle: bool = True
        #: x and y axis labels; empty takes them from the columns
        self.xLabel: str = ''
        self.yLabel: str = ''
        #: where the legend goes; see :data:`LEGEND_LOCATIONS`
        self.legendLocation: str = LEGEND_AUTO
        #: automatic legend entry -> what to show instead
        self.legendNames: Dict[str, str] = {}

    def nameFor(self, label: str) -> str:
        """What to write in the legend for a trace matplotlib would call this."""
        return renderableText(self.legendNames.get(label) or label)

    def legendKeywords(self) -> Optional[Dict[str, Any]]:
        """``Axes.legend`` keywords, or ``None`` for no legend at all."""
        if self.legendLocation == LEGEND_NONE:
            return None
        if self.legendLocation == LEGEND_OUTSIDE:
            # Beside the axes: with six curves on one plot there is often no
            # corner left that hides nothing.
            return dict(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0, fontsize='small')
        loc = self.legendLocation or 'upper right'
        return dict(loc=loc, fontsize='small')


class PlotStyle:
    """How the traces are drawn: point size, line widths, fit color.

    The defaults come from the matplotlib settings in ``plottr/config`` so that
    the config file stays the one place to change the overall look; the toolbar
    moves them per plot from there.
    """

    def __init__(self) -> None:
        import matplotlib as mpl

        #: marker size of the data points (0 draws no markers)
        self.markerSize: float = float(mpl.rcParams.get('lines.markersize', 3))
        #: width of the line through the data points (0 draws no line)
        self.lineWidth: float = float(mpl.rcParams.get('lines.linewidth', 1))
        #: width of a fit curve.  Thicker than the data by default: it is the
        #: line the eye is meant to follow.
        self.fitLineWidth: float = self.lineWidth * 1.5
        #: color of fit curves; empty means "same as the data it fits"
        self.fitColor: str = FIT_COLOR_AUTO

    def dataOptions(self) -> Dict[str, Any]:
        """matplotlib keyword arguments for a measured trace."""
        options: Dict[str, Any] = {
            'markersize': self.markerSize,
            'linewidth': self.lineWidth,
        }
        if self.markerSize <= 0:
            options['marker'] = ''
        return options

    def fitOptions(self) -> Dict[str, Any]:
        """matplotlib keyword arguments for a fit curve.

        A fit is a model, not a measurement: it gets no markers, so that the
        points on the plot are the ones that were actually measured.
        """
        options: Dict[str, Any] = {
            'marker': '',
            'linewidth': self.fitLineWidth,
        }
        if self.fitColor:
            options['color'] = self.fitColor
        return options

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
            style = dict(self.style.dataOptions())
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
        im = colorplot2d(axes[0], x, y, z, plotType=self.plotType)
        if im is None:
            return None
        cb = self.fig.colorbar(im, ax=axes[0], shrink=0.75, pad=0.02)
        lbl = plotItem.labels[-1] if isinstance(plotItem.labels, list) and len(plotItem.labels) > 0 else ''
        cb.set_label(lbl)
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

        form = QtWidgets.QFormLayout(self)
        form.addRow('Point size', self.pointSize)
        form.addRow('Line width', self.lineWidth)
        form.addRow('Fit width', self.fitLineWidth)
        form.addRow('Fit color', self.fitColor)

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

        # Linear or logarithmic axes.  Qi against the photon number is read on
        # a log x axis; so is anything spanning decades.
        self.xScaleBox = QtWidgets.QComboBox()
        self.yScaleBox = QtWidgets.QComboBox()
        for box in (self.xScaleBox, self.yScaleBox):
            for name, value in AXIS_SCALES:
                box.addItem(name, value)
            box.currentIndexChanged.connect(self._emitAxisScales)
        self.xScaleBox.setToolTip(
            'Scale of the x axis.  A log scale is ignored where the data has '
            'no positive values.')
        self.yScaleBox.setToolTip('Scale of the y axis.')

        scaleForm = QtWidgets.QWidget()
        scaleLayout = QtWidgets.QFormLayout(scaleForm)
        scaleLayout.addRow('x axis', self.xScaleBox)
        scaleLayout.addRow('y axis', self.yScaleBox)
        scaleMenu = QtWidgets.QMenu(parent=self)
        scaleAction = QtWidgets.QWidgetAction(scaleMenu)
        scaleAction.setDefaultWidget(scaleForm)
        scaleMenu.addAction(scaleAction)
        self.scaleButton = QtWidgets.QToolButton()
        self.scaleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.scaleButton.setText('Scale')
        self.scaleButton.setToolTip('Linear or logarithmic axes.')
        self.scaleButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.scaleButton.setMenu(scaleMenu)
        self.addWidget(self.scaleButton)
        self._scaleMenu = scaleMenu

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

    @Slot()
    def _emitAxisScales(self) -> None:
        self.axisScaleSelected.emit(str(self.xScaleBox.currentData()),
                                    str(self.yScaleBox.currentData()))

    def setAxisScales(self, xScale: str, yScale: str) -> None:
        """Show these scales without emitting anything."""
        for box, value in ((self.xScaleBox, xScale), (self.yScaleBox, yScale)):
            index = box.findData(value)
            if index >= 0:
                box.blockSignals(True)
                box.setCurrentIndex(index)
                box.blockSignals(False)

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
