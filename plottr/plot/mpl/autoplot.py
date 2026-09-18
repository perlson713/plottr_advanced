"""``plottr.plot.mpl.autoplot`` -- This module contains the tools for automatic plotting with matplotlib.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, Any, Type, cast
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


def _hasPositiveValues(values: Any) -> bool:
    """Whether a log axis can show this data at all."""
    array = np.asanyarray(values)
    if not np.issubdtype(array.dtype, np.number):
        return False
    with np.errstate(invalid='ignore'):
        return bool(np.any(np.isfinite(array) & (array > 0)))


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

        if isinstance(axes, list) and len(labels) == 2 and len(set(labels[1])) > 1:
            axes[0].legend(loc='upper right', fontsize='small')

        if isinstance(axes, list) and len(axes) > 1:
            if len(labels) > 2 and len(set(labels[2])) == 1:
                axes[1].set_ylabel(labels[2][0])

        if isinstance(axes, list):
            for ax in axes:
                self.applyAxisScales(subPlotId, ax)
        return None

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

        # other options
        self.addSeparator()

        self.plotReal = self.addAction('Real')
        self.plotReal.setCheckable(True)
        self.plotReal.triggered.connect(
            lambda: self.selectComplexType(ComplexRepresentation.real))

        self.plotReIm = self.addAction('Re/Im')
        self.plotReIm.setCheckable(True)
        self.plotReIm.triggered.connect(
            lambda: self.selectComplexType(ComplexRepresentation.realAndImag))

        self.plotReImSep = self.addAction('Split Re/Im')
        self.plotReImSep.setCheckable(True)
        self.plotReImSep.triggered.connect(
            lambda: self.selectComplexType(ComplexRepresentation.realAndImagSeparate))

        self.plotMag = self.addAction('Mag')
        self.plotMag.setCheckable(True)
        self.plotMag.setToolTip('Magnitude only, in one panel.')
        self.plotMag.triggered.connect(
            lambda: self.selectComplexType(ComplexRepresentation.mag))

        self.plotPhase = self.addAction('Phase')
        self.plotPhase.setCheckable(True)
        self.plotPhase.setToolTip('Phase only, in one panel.')
        self.plotPhase.triggered.connect(
            lambda: self.selectComplexType(ComplexRepresentation.phase))

        self.plotMagPhase = self.addAction('Mag/Phase')
        self.plotMagPhase.setCheckable(True)
        self.plotMagPhase.triggered.connect(
            lambda: self.selectComplexType(ComplexRepresentation.magAndPhase))

        self.plotComplexPlane = self.addAction('Complex plane')
        self.plotComplexPlane.setCheckable(True)
        self.plotComplexPlane.triggered.connect(
            lambda: self.selectComplexType(ComplexRepresentation.complexPlane))

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
        self.updatePlot()
