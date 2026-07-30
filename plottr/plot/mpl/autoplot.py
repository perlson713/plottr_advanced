"""``plottr.plot.mpl.autoplot`` -- This module contains the tools for automatic plotting with matplotlib.
"""

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, Any, Type, cast
from types import TracebackType

import numpy as np
from matplotlib import rc_context, colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable

from plottr import QtWidgets, QtCore, QtGui, Signal, Slot
from plottr.data.datadict import DataDictBase
from plottr.icons import (get_singleTracePlotIcon, get_multiTracePlotIcon, get_imagePlotIcon,
                          get_colormeshPlotIcon, get_scatterPlot2dIcon)
from plottr.gui.tools import dpiScalingFactor
from .plotting import (PlotType, colorplot2d, square_axes, AxesOptions,
                       apply_axes_options_to_figure, AXIS_SCALES,
                       TICK_DIRECTIONS, LabelOptions, apply_label_options,
                       apply_title, LEGEND_LOCATIONS, TraceOptions, TraceStyle,
                       apply_trace_options, LINE_STYLES, MARKERS,
                       ColorOptions, apply_color_options, ExportSpec,
                       save_figure, FIGURE_WIDTH_PRESETS, EXPORT_FORMATS,
                       figure_lines)
from .widgets import MPLPlotWidget
from ..base import AutoFigureMaker as BaseFM, PlotDataType, \
    PlotItem, ComplexRepresentation, determinePlotDataType, PlotWidgetContainer, \
    errorBarData, plottableDependents, ERROR_BAR_KEY, ERROR_BAR_X_KEY

logger = logging.getLogger(__name__)


#: accent / neutral colors shared by the autoplot GUI styling.
_ACCENT = '#2563eb'
_ACCENT_DARK = '#1d4ed8'
_ACCENT_TINT = '#e8edfb'
_ACCENT_TINT_BORDER = '#cdd9f7'

#: stylesheet for the plot-options toolbar (segmented-control look).
TOOLBAR_STYLESHEET = f"""
QToolBar#autoPlotToolBar {{
    background: #f5f6f8;
    border: none;
    border-bottom: 1px solid #e2e5ea;
    padding: 4px 6px;
    spacing: 3px;
}}
QToolBar#autoPlotToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 10px;
    color: #3a3f4b;
    font-weight: 500;
}}
QToolBar#autoPlotToolBar QToolButton:hover {{
    background: {_ACCENT_TINT};
    border: 1px solid {_ACCENT_TINT_BORDER};
}}
QToolBar#autoPlotToolBar QToolButton:checked {{
    background: {_ACCENT};
    border: 1px solid {_ACCENT};
    color: #ffffff;
}}
QToolBar#autoPlotToolBar QToolButton:checked:hover {{
    background: {_ACCENT_DARK};
}}
QToolBar#autoPlotToolBar QToolButton:disabled {{
    color: #b8bcc6;
}}
QToolBar#autoPlotToolBar::separator {{
    background: #e2e5ea;
    width: 1px;
    margin: 5px 6px;
}}
QLabel#tbSection {{
    color: #8b909c;
    font-weight: 700;
    padding: 0px 5px 0px 3px;
}}
QComboBox#complexCombo {{
    background: #ffffff;
    border: 1px solid #d4d8e0;
    border-radius: 6px;
    padding: 3px 8px;
    min-width: 96px;
    color: #3a3f4b;
}}
QComboBox#complexCombo:hover {{
    border: 1px solid {_ACCENT_TINT_BORDER};
}}
QComboBox#complexCombo:focus {{
    border: 1px solid {_ACCENT};
}}
QComboBox#complexCombo::drop-down {{
    border: none;
    width: 18px;
}}
"""

#: stylesheet for the axes-options dialog.
DIALOG_STYLESHEET = f"""
QDialog#axesOptionsDialog {{
    background: #ffffff;
}}
QDialog#axesOptionsDialog QGroupBox {{
    font-weight: 600;
    color: #3a3f4b;
    border: 1px solid #e2e5ea;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 10px 8px 10px;
}}
QDialog#axesOptionsDialog QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0px 5px;
    color: {_ACCENT};
}}
QDialog#axesOptionsDialog QLineEdit,
QDialog#axesOptionsDialog QComboBox {{
    border: 1px solid #d4d8e0;
    border-radius: 6px;
    padding: 3px 6px;
    background: #ffffff;
    selection-background-color: {_ACCENT};
}}
QDialog#axesOptionsDialog QLineEdit:focus,
QDialog#axesOptionsDialog QComboBox:focus {{
    border: 1px solid {_ACCENT};
}}
QDialog#axesOptionsDialog QPushButton {{
    background: {_ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 5px 18px;
    font-weight: 600;
}}
QDialog#axesOptionsDialog QPushButton:hover {{
    background: {_ACCENT_DARK};
}}
QDialog#axesOptionsDialog QPushButton:pressed {{
    background: #1e40af;
}}
QDialog#axesOptionsDialog QCheckBox {{
    spacing: 6px;
}}
"""


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

        # complex-plane plots (Real vs Imag) default to a square plot range
        # with equal aspect ratio, so that circles stay circular.
        if self.complexRepresentation is ComplexRepresentation.complexPlane \
                and isinstance(axes, list):
            for ax in axes:
                square_axes(ax)
        return None

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
        # both keys must be popped: anything left over is splatted into
        # Axes.plot, which rejects unknown kwargs.
        yerr = plotOptions.pop(ERROR_BAR_KEY, None)
        xerr = plotOptions.pop(ERROR_BAR_X_KEY, None)
        line = axes[0].plot(x, y, label=lbl, **plotOptions)
        if yerr is not None or xerr is not None:
            axes[0].errorbar(x, y, yerr=yerr, xerr=xerr, fmt='none',
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

    #: signal emitted when the user requests the axes-options dialog
    axesOptionsRequested = Signal()

    #: signal emitted when the phase unit (degrees vs radians) has changed
    phaseDegreesSelected = Signal(bool)

    #: signal emitted when phase unwrapping has been toggled
    phaseUnwrapSelected = Signal(bool)

    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        """Constructor for :class:`AutoPlotToolBar`"""

        super().__init__(name, parent=parent)

        self.setObjectName('autoPlotToolBar')
        self.setStyleSheet(TOOLBAR_STYLESHEET)

        self._addSectionLabel('Plot')

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
        self._addSectionLabel('Complex')

        #: complex representation is chosen from a compact dropdown; the enum
        #: member is stored as the item's user data.
        self.complexCombo = QtWidgets.QComboBox()
        self.complexCombo.setObjectName('complexCombo')
        self.complexCombo.setToolTip('How to represent complex-valued data')
        self.complexCombo.activated.connect(self._complexComboActivated)
        self.addWidget(self.complexCombo)

        #: phase unit; only meaningful for representations that show a phase.
        self.phaseCombo = QtWidgets.QComboBox()
        # reuse the complexCombo styling rule
        self.phaseCombo.setObjectName('complexCombo')
        self.phaseCombo.setToolTip('Unit used for the phase panel')
        self.phaseCombo.addItem('rad', False)
        self.phaseCombo.addItem('deg', True)
        self.phaseCombo.activated.connect(self._phaseComboActivated)
        self.addWidget(self.phaseCombo)

        self.unwrapPhase = self.addAction('Unwrap')
        self.unwrapPhase.setCheckable(True)
        self.unwrapPhase.setChecked(False)
        self.unwrapPhase.setToolTip('Remove 2-pi jumps from the phase')
        self.unwrapPhase.triggered.connect(
            lambda: self.phaseUnwrapSelected.emit(self.unwrapPhase.isChecked()))

        self.addSeparator()
        self._addSectionLabel('Display')

        self.showErrorBars = self.addAction('Error bars')
        self.showErrorBars.setCheckable(True)
        self.showErrorBars.setChecked(True)
        self.showErrorBars.triggered.connect(
            lambda: self.errorBarsSelected.emit(self.showErrorBars.isChecked()))

        self.addSeparator()

        self.figureOptions = self.addAction('Figure options…')
        self.figureOptions.setToolTip(
            'Axes, labels, traces, colors and publication export '
            '(kept across re-draws).')
        self.figureOptions.triggered.connect(
            lambda: self.axesOptionsRequested.emit())

        self.plotTypeActions = OrderedDict({
            PlotType.multitraces: self.plotasMultiTraces,
            PlotType.singletraces: self.plotasSingleTraces,
            PlotType.image: self.plotasImage,
            PlotType.colormesh: self.plotasMesh,
            PlotType.scatter2d: self.plotasScatter2d,
        })

        self._currentPlotType = PlotType.empty
        self._currentlyAllowedPlotTypes: Tuple[PlotType, ...] = ()

        self._currentComplex = ComplexRepresentation.realAndImag
        self._currentlyAllowedComplexTypes: Tuple[ComplexRepresentation, ...] = ()
        self._updatePhaseControlsEnabled()

    #: short, compact labels for the complex-representation dropdown.
    _complexLabels = OrderedDict([
        (ComplexRepresentation.real, 'Real'),
        (ComplexRepresentation.realAndImag, 'Re / Im'),
        (ComplexRepresentation.realAndImagSeparate, 'Re / Im (split)'),
        (ComplexRepresentation.magAndPhase, 'Mag / Phase'),
        (ComplexRepresentation.log_MagAndPhase, 'dB / Phase'),
        (ComplexRepresentation.complexPlane, 'Complex plane'),
    ])

    def _complexLabel(self, comp: ComplexRepresentation) -> str:
        return self._complexLabels.get(comp, str(comp.value))

    #: representations that produce a phase panel (phase options apply to them)
    _phaseRepresentations = (
        ComplexRepresentation.magAndPhase,
        ComplexRepresentation.log_MagAndPhase,
    )

    @Slot(int)
    def _complexComboActivated(self, index: int) -> None:
        comp = self.complexCombo.itemData(index)
        if comp is not None:
            self.selectComplexType(comp)

    @Slot(int)
    def _phaseComboActivated(self, index: int) -> None:
        degrees = self.phaseCombo.itemData(index)
        if degrees is not None:
            self.phaseDegreesSelected.emit(bool(degrees))

    def _updatePhaseControlsEnabled(self) -> None:
        """Grey out the phase controls unless a phase is actually shown."""
        showsPhase = self._currentComplex in self._phaseRepresentations
        self.phaseCombo.setEnabled(showsPhase)
        self.unwrapPhase.setEnabled(showsPhase)

    def _addSectionLabel(self, text: str) -> None:
        """Add a small, muted section header to the toolbar."""
        label = QtWidgets.QLabel(text.upper())
        label.setObjectName('tbSection')
        self.addWidget(label)

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

    def _setComboToCurrent(self) -> None:
        """Sync the dropdown to :attr:`_currentComplex` without emitting."""
        index = self.complexCombo.findData(self._currentComplex)
        if index >= 0:
            self.complexCombo.blockSignals(True)
            self.complexCombo.setCurrentIndex(index)
            self.complexCombo.blockSignals(False)

    def selectComplexType(self, comp: ComplexRepresentation) -> None:
        """makes sure that the selected `comp` is shown in the dropdown.

        This method can be used to catch a trigger from the UI or to set the
        selection programmatically. If the active complex representation
        changed, we emit :attr:`complexRepresentationSelected`.
        """
        changed = comp is not self._currentComplex
        self._currentComplex = comp
        self._setComboToCurrent()
        self._updatePhaseControlsEnabled()
        if changed:
            self.complexRepresentationSelected.emit(self._currentComplex)

    def setAllowedComplexTypes(self, *complexOptions: ComplexRepresentation) -> None:
        """Populate the dropdown with the allowed complex representations.

        If the current selection is no longer allowed, select the first
        allowed one instead (and emit the change).
        """

        if complexOptions == self._currentlyAllowedComplexTypes:
            return

        self.complexCombo.blockSignals(True)
        self.complexCombo.clear()
        for comp in complexOptions:
            self.complexCombo.addItem(self._complexLabel(comp), comp)
        self.complexCombo.blockSignals(False)

        if self._currentComplex not in complexOptions:
            self._currentComplex = complexOptions[0] if complexOptions \
                else ComplexRepresentation.realAndImag
            self._setComboToCurrent()
            self._updatePhaseControlsEnabled()
            self.complexRepresentationSelected.emit(self._currentComplex)
        else:
            self._setComboToCurrent()
            self._updatePhaseControlsEnabled()

        self._currentlyAllowedComplexTypes = complexOptions


class FigureOptionsDialog(QtWidgets.QDialog):
    """A modeless dialog to configure a figure for publication.

    Grouped into tabs: axes/ticks, labels & legend, per-trace styling, the
    2D color scale, and export settings. Every control feeds one of the
    option dataclasses in :mod:`.plotting`, which the plot widget re-applies
    after each redraw, so nothing set here is lost when the data updates.
    """

    #: emitted whenever the user changes any option. The plot widget then
    #: reads the current values back through the ``*Options`` accessors.
    optionsChanged = Signal()

    #: emitted with a file path when the user asks to export the figure.
    exportRequested = Signal(str)

    _gridChoices = OrderedDict([
        ('default', None),
        ('on', True),
        ('off', False),
    ])
    _boolChoices = OrderedDict([
        ('default', None),
        ('on', True),
        ('off', False),
    ])

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)
        self.setObjectName('axesOptionsDialog')
        self.setStyleSheet(DIALOG_STYLESHEET)
        self.setWindowTitle('Figure options')
        self._updating = False
        self._traceStyles: Dict[int, TraceStyle] = {}
        self._currentTrace = 0

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._buildAxesTab(), 'Axes')
        tabs.addTab(self._buildLabelsTab(), 'Labels')
        tabs.addTab(self._buildTracesTab(), 'Traces')
        tabs.addTab(self._buildColorTab(), 'Color')
        tabs.addTab(self._buildExportTab(), 'Export')

        self.resetButton = QtWidgets.QPushButton('Reset')
        self.resetButton.clicked.connect(self.resetOptions)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.resetButton)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(tabs)
        layout.addLayout(buttons)

        self._connectSignals()

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _numberEdit(placeholder: str = 'auto') -> QtWidgets.QLineEdit:
        w = QtWidgets.QLineEdit()
        w.setPlaceholderText(placeholder)
        w.setValidator(QtGui.QDoubleValidator(w))
        return w

    def _buildAxesTab(self) -> QtWidgets.QWidget:
        self.xscale = QtWidgets.QComboBox()
        self.xscale.addItems(['default'] + list(AXIS_SCALES))
        self.yscale = QtWidgets.QComboBox()
        self.yscale.addItems(['default'] + list(AXIS_SCALES))
        self.xmin = self._numberEdit()
        self.xmax = self._numberEdit()
        self.ymin = self._numberEdit()
        self.ymax = self._numberEdit()
        self.grid = QtWidgets.QComboBox()
        self.grid.addItems(list(self._gridChoices.keys()))
        self.equalAspect = QtWidgets.QCheckBox('equal (square) aspect ratio')

        self.minorTicks = QtWidgets.QComboBox()
        self.minorTicks.addItems(list(self._boolChoices.keys()))
        self.tickDirection = QtWidgets.QComboBox()
        self.tickDirection.addItems(['default'] + list(TICK_DIRECTIONS))
        self.ticksAllSides = QtWidgets.QComboBox()
        self.ticksAllSides.addItems(list(self._boolChoices.keys()))
        self.noOffsetText = QtWidgets.QCheckBox('no offset / scientific text')

        xForm = QtWidgets.QFormLayout()
        xForm.addRow('scale', self.xscale)
        xForm.addRow('min', self.xmin)
        xForm.addRow('max', self.xmax)
        xBox = QtWidgets.QGroupBox('X axis')
        xBox.setLayout(xForm)

        yForm = QtWidgets.QFormLayout()
        yForm.addRow('scale', self.yscale)
        yForm.addRow('min', self.ymin)
        yForm.addRow('max', self.ymax)
        yBox = QtWidgets.QGroupBox('Y axis')
        yBox.setLayout(yForm)

        tForm = QtWidgets.QFormLayout()
        tForm.addRow('grid', self.grid)
        tForm.addRow('aspect', self.equalAspect)
        tForm.addRow('minor ticks', self.minorTicks)
        tForm.addRow('direction', self.tickDirection)
        tForm.addRow('all sides', self.ticksAllSides)
        tForm.addRow('numbers', self.noOffsetText)
        tBox = QtWidgets.QGroupBox('Ticks and display')
        tBox.setLayout(tForm)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(xBox)
        lay.addWidget(yBox)
        lay.addWidget(tBox)
        lay.addStretch(1)
        return w

    def _buildLabelsTab(self) -> QtWidgets.QWidget:
        self.showTitle = QtWidgets.QCheckBox('show title')
        self.title = QtWidgets.QLineEdit()
        self.title.setPlaceholderText('(file name)')
        self.xlabel = QtWidgets.QLineEdit()
        self.ylabel = QtWidgets.QLineEdit()
        self.colorbarLabel = QtWidgets.QLineEdit()
        for w_ in (self.xlabel, self.ylabel, self.colorbarLabel):
            w_.setPlaceholderText('(from data)')

        self.showLegend = QtWidgets.QComboBox()
        self.showLegend.addItems(list(self._boolChoices.keys()))
        self.legendLocation = QtWidgets.QComboBox()
        self.legendLocation.addItems(['default'] + list(LEGEND_LOCATIONS))
        self.legendFrame = QtWidgets.QComboBox()
        self.legendFrame.addItems(list(self._boolChoices.keys()))
        self.legendColumns = QtWidgets.QSpinBox()
        self.legendColumns.setRange(0, 8)
        self.legendColumns.setSpecialValueText('default')

        tForm = QtWidgets.QFormLayout()
        tForm.addRow('', self.showTitle)
        tForm.addRow('title', self.title)
        tForm.addRow('x label', self.xlabel)
        tForm.addRow('y label', self.ylabel)
        tForm.addRow('colorbar', self.colorbarLabel)
        tBox = QtWidgets.QGroupBox('Text')
        tBox.setLayout(tForm)

        lForm = QtWidgets.QFormLayout()
        lForm.addRow('show', self.showLegend)
        lForm.addRow('location', self.legendLocation)
        lForm.addRow('frame', self.legendFrame)
        lForm.addRow('columns', self.legendColumns)
        lBox = QtWidgets.QGroupBox('Legend')
        lBox.setLayout(lForm)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(tBox)
        lay.addWidget(lBox)
        lay.addStretch(1)
        return w

    def _buildTracesTab(self) -> QtWidgets.QWidget:
        self.traceSelector = QtWidgets.QComboBox()
        self.traceColor = QtWidgets.QLineEdit()
        self.traceColor.setPlaceholderText('(default) e.g. C0 or #1f77b4')
        self.traceLinestyle = QtWidgets.QComboBox()
        self.traceLinestyle.addItems(['default'] + list(LINE_STYLES))
        self.traceLinewidth = self._numberEdit('default')
        self.traceMarker = QtWidgets.QComboBox()
        self.traceMarker.addItems(['default'] + list(MARKERS))
        self.traceMarkersize = self._numberEdit('default')
        self.traceAlpha = self._numberEdit('default')

        self.markerLimit = QtWidgets.QSpinBox()
        self.markerLimit.setRange(0, 1000000)
        self.markerLimit.setSpecialValueText('off')
        self.markerLimit.setValue(TraceOptions().markerLimit or 0)
        self.markerLimit.setToolTip(
            'Traces with more points than this are drawn without markers, '
            'so dense sweeps stay legible in print.')

        sForm = QtWidgets.QFormLayout()
        sForm.addRow('trace', self.traceSelector)
        sForm.addRow('color', self.traceColor)
        sForm.addRow('line style', self.traceLinestyle)
        sForm.addRow('line width', self.traceLinewidth)
        sForm.addRow('marker', self.traceMarker)
        sForm.addRow('marker size', self.traceMarkersize)
        sForm.addRow('alpha', self.traceAlpha)
        sBox = QtWidgets.QGroupBox('Per-trace style')
        sBox.setLayout(sForm)

        gForm = QtWidgets.QFormLayout()
        gForm.addRow('drop markers above', self.markerLimit)
        gBox = QtWidgets.QGroupBox('All traces')
        gBox.setLayout(gForm)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(sBox)
        lay.addWidget(gBox)
        lay.addStretch(1)
        return w

    def _buildColorTab(self) -> QtWidgets.QWidget:
        self.colormap = QtWidgets.QComboBox()
        self.colormap.setEditable(True)
        self.colormap.addItems(['default'] + sorted(colormaps()))
        self.vmin = self._numberEdit()
        self.vmax = self._numberEdit()
        self.symmetric = QtWidgets.QCheckBox('symmetric around center')
        self.symmetricCenter = self._numberEdit('0')

        form = QtWidgets.QFormLayout()
        form.addRow('colormap', self.colormap)
        form.addRow('min', self.vmin)
        form.addRow('max', self.vmax)
        form.addRow('', self.symmetric)
        form.addRow('center', self.symmetricCenter)
        box = QtWidgets.QGroupBox('Color scale (2D plots)')
        box.setLayout(form)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(box)
        lay.addStretch(1)
        return w

    def _buildExportTab(self) -> QtWidgets.QWidget:
        self.sizePreset = QtWidgets.QComboBox()
        self.sizePreset.addItems(list(FIGURE_WIDTH_PRESETS.keys()))
        self.exportWidth = self._numberEdit('auto')
        self.exportHeight = self._numberEdit('auto')
        self.exportDpi = QtWidgets.QSpinBox()
        self.exportDpi.setRange(50, 2400)
        self.exportDpi.setValue(300)
        self.exportFormat = QtWidgets.QComboBox()
        self.exportFormat.addItems(list(EXPORT_FORMATS))
        self.exportTransparent = QtWidgets.QCheckBox('transparent background')
        self.exportFontSize = self._numberEdit('8')
        self.exportFontFamily = QtWidgets.QLineEdit()
        self.exportFontFamily.setPlaceholderText('(current)')

        self.exportButton = QtWidgets.QPushButton('Export figure…')
        self.exportButton.clicked.connect(self._chooseExportFile)

        form = QtWidgets.QFormLayout()
        form.addRow('preset', self.sizePreset)
        form.addRow('width (in)', self.exportWidth)
        form.addRow('height (in)', self.exportHeight)
        form.addRow('dpi', self.exportDpi)
        form.addRow('format', self.exportFormat)
        form.addRow('', self.exportTransparent)
        form.addRow('font size (pt)', self.exportFontSize)
        form.addRow('font family', self.exportFontFamily)
        box = QtWidgets.QGroupBox('Output')
        box.setLayout(form)

        note = QtWidgets.QLabel(
            'The figure is rendered at exactly this size, independently of '
            'the window, with fonts embedded as TrueType.')
        note.setWordWrap(True)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(box)
        lay.addWidget(note)
        lay.addWidget(self.exportButton)
        lay.addStretch(1)
        return w

    def _connectSignals(self) -> None:
        combos = [self.xscale, self.yscale, self.grid, self.minorTicks,
                  self.tickDirection, self.ticksAllSides, self.showLegend,
                  self.legendLocation, self.legendFrame, self.colormap]
        for c in combos:
            c.currentIndexChanged.connect(self._emit)
        checks = [self.equalAspect, self.noOffsetText, self.showTitle,
                  self.symmetric]
        for ch in checks:
            ch.toggled.connect(self._emit)
        edits = [self.xmin, self.xmax, self.ymin, self.ymax, self.title,
                 self.xlabel, self.ylabel, self.colorbarLabel, self.vmin,
                 self.vmax, self.symmetricCenter]
        for e in edits:
            e.editingFinished.connect(self._emit)
        self.legendColumns.valueChanged.connect(self._emit)
        self.markerLimit.valueChanged.connect(self._emit)

        # trace styling
        self.traceSelector.currentIndexChanged.connect(self._traceSelected)
        for wdg in (self.traceLinestyle, self.traceMarker):
            wdg.currentIndexChanged.connect(self._traceStyleEdited)
        for wdg2 in (self.traceColor, self.traceLinewidth,
                     self.traceMarkersize, self.traceAlpha):
            wdg2.editingFinished.connect(self._traceStyleEdited)

        self.sizePreset.currentIndexChanged.connect(self._presetSelected)

    # -- value conversion ----------------------------------------------------
    @staticmethod
    def _floatOrNone(text: str) -> Optional[float]:
        text = text.strip()
        if text == '':
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _textOrNone(text: str) -> Optional[str]:
        return text if text.strip() != '' else None

    @staticmethod
    def _choiceOrNone(text: str) -> Optional[str]:
        return None if text == 'default' else text

    def _boolChoice(self, combo: QtWidgets.QComboBox) -> Optional[bool]:
        return self._boolChoices[combo.currentText()]

    # -- accessors -----------------------------------------------------------
    def axesOptions(self) -> AxesOptions:
        return AxesOptions(
            xscale=self._choiceOrNone(self.xscale.currentText()),
            yscale=self._choiceOrNone(self.yscale.currentText()),
            xmin=self._floatOrNone(self.xmin.text()),
            xmax=self._floatOrNone(self.xmax.text()),
            ymin=self._floatOrNone(self.ymin.text()),
            ymax=self._floatOrNone(self.ymax.text()),
            grid=self._gridChoices[self.grid.currentText()],
            equalAspect=self.equalAspect.isChecked(),
            minorTicks=self._boolChoice(self.minorTicks),
            tickDirection=self._choiceOrNone(self.tickDirection.currentText()),
            ticksAllSides=self._boolChoice(self.ticksAllSides),
            noOffsetText=self.noOffsetText.isChecked() or None,
        )

    def labelOptions(self) -> LabelOptions:
        columns = self.legendColumns.value()
        return LabelOptions(
            title=self._textOrNone(self.title.text()),
            showTitle=self.showTitle.isChecked(),
            xlabel=self._textOrNone(self.xlabel.text()),
            ylabel=self._textOrNone(self.ylabel.text()),
            colorbarLabel=self._textOrNone(self.colorbarLabel.text()),
            showLegend=self._boolChoice(self.showLegend),
            legendLocation=self._choiceOrNone(
                self.legendLocation.currentText()),
            legendFrame=self._boolChoice(self.legendFrame),
            legendColumns=columns if columns > 0 else None,
        )

    def traceOptions(self) -> TraceOptions:
        limit = self.markerLimit.value()
        return TraceOptions(styles=dict(self._traceStyles),
                            markerLimit=limit if limit > 0 else None)

    def colorOptions(self) -> ColorOptions:
        return ColorOptions(
            colormap=self._choiceOrNone(self.colormap.currentText()),
            vmin=self._floatOrNone(self.vmin.text()),
            vmax=self._floatOrNone(self.vmax.text()),
            symmetric=self.symmetric.isChecked(),
            symmetricCenter=self._floatOrNone(
                self.symmetricCenter.text()) or 0.0,
        )

    def exportSpec(self) -> ExportSpec:
        return ExportSpec(
            width=self._floatOrNone(self.exportWidth.text()),
            height=self._floatOrNone(self.exportHeight.text()),
            dpi=self.exportDpi.value(),
            format=self.exportFormat.currentText(),
            transparent=self.exportTransparent.isChecked(),
            fontSize=self._floatOrNone(self.exportFontSize.text()) or 8.0,
            fontFamily=self._textOrNone(self.exportFontFamily.text()),
        )

    # -- trace handling ------------------------------------------------------
    def setTraceNames(self, names: List[str]) -> None:
        """Populate the per-trace selector with the current traces."""
        current = self.traceSelector.currentIndex()
        self.traceSelector.blockSignals(True)
        self.traceSelector.clear()
        for i, n in enumerate(names):
            self.traceSelector.addItem(n or f'trace {i}', i)
        if 0 <= current < len(names):
            self.traceSelector.setCurrentIndex(current)
        self.traceSelector.blockSignals(False)
        self._currentTrace = max(0, self.traceSelector.currentIndex())
        self._showTraceStyle()

    def _showTraceStyle(self) -> None:
        style = self._traceStyles.get(self._currentTrace, TraceStyle())
        self._updating = True
        self.traceColor.setText(style.color or '')
        self.traceLinestyle.setCurrentText(
            'default' if style.linestyle is None else style.linestyle)
        self.traceLinewidth.setText(
            '' if style.linewidth is None else repr(style.linewidth))
        self.traceMarker.setCurrentText(
            'default' if style.marker is None else style.marker)
        self.traceMarkersize.setText(
            '' if style.markersize is None else repr(style.markersize))
        self.traceAlpha.setText(
            '' if style.alpha is None else repr(style.alpha))
        self._updating = False

    @Slot(int)
    def _traceSelected(self, index: int) -> None:
        self._currentTrace = max(0, index)
        self._showTraceStyle()

    @Slot()
    def _traceStyleEdited(self) -> None:
        if self._updating:
            return
        style = TraceStyle(
            color=self._textOrNone(self.traceColor.text()),
            linestyle=self._choiceOrNone(self.traceLinestyle.currentText()),
            linewidth=self._floatOrNone(self.traceLinewidth.text()),
            marker=self._choiceOrNone(self.traceMarker.currentText()),
            markersize=self._floatOrNone(self.traceMarkersize.text()),
            alpha=self._floatOrNone(self.traceAlpha.text()),
        )
        self._traceStyles[self._currentTrace] = style
        self._emit()

    # -- export --------------------------------------------------------------
    @Slot(int)
    def _presetSelected(self, index: int) -> None:
        width = FIGURE_WIDTH_PRESETS.get(self.sizePreset.currentText())
        self._updating = True
        self.exportWidth.setText('' if width is None else f'{width:.4g}')
        self._updating = False
        self._emit()

    @Slot()
    def _chooseExportFile(self) -> None:
        fmt = self.exportFormat.currentText()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export figure', f'figure.{fmt}',
            f'{fmt.upper()} (*.{fmt});;All files (*)')
        if path:
            self.exportRequested.emit(path)

    # -- misc ----------------------------------------------------------------
    @Slot()
    def resetOptions(self) -> None:
        """Reset every field to its neutral (do-nothing) value."""
        self._updating = True
        for combo in (self.xscale, self.yscale, self.grid, self.minorTicks,
                      self.tickDirection, self.ticksAllSides, self.showLegend,
                      self.legendLocation, self.legendFrame, self.colormap,
                      self.traceLinestyle, self.traceMarker):
            combo.setCurrentText('default')
        for edit in (self.xmin, self.xmax, self.ymin, self.ymax, self.title,
                     self.xlabel, self.ylabel, self.colorbarLabel, self.vmin,
                     self.vmax, self.symmetricCenter, self.traceColor,
                     self.traceLinewidth, self.traceMarkersize,
                     self.traceAlpha):
            edit.setText('')
        for check in (self.equalAspect, self.noOffsetText, self.showTitle,
                      self.symmetric):
            check.setChecked(False)
        self.legendColumns.setValue(0)
        self.markerLimit.setValue(200)
        self._traceStyles.clear()
        self._updating = False
        self._emit()

    @Slot()
    def _emit(self) -> None:
        if not self._updating:
            self.optionsChanged.emit()


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

        # phase display options (only relevant for mag/phase representations)
        self.phaseDegrees = False
        self.phaseUnwrap = False

        # User-adjustable figure options. All of these persist across the
        # automatic re-drawing that happens on every data/option change.
        self.axesOptions = AxesOptions()
        self.labelOptions = LabelOptions()
        self.traceOptions = TraceOptions()
        self.colorOptions = ColorOptions()
        self.exportSpec = ExportSpec()
        self._optionsDialog: Optional[FigureOptionsDialog] = None

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
        self.plotOptionsToolBar.axesOptionsRequested.connect(
            self._showAxesOptionsDialog
        )
        self.plotOptionsToolBar.phaseDegreesSelected.connect(
            self._phaseDegreesFromToolBar
        )
        self.plotOptionsToolBar.phaseUnwrapSelected.connect(
            self._phaseUnwrapFromToolBar
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
                    ComplexRepresentation.magAndPhase,
                    ComplexRepresentation.log_MagAndPhase,
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

    @Slot(bool)
    def _phaseDegreesFromToolBar(self, degrees: bool) -> None:
        if degrees is not self.phaseDegrees:
            self.phaseDegrees = degrees
            self._plotData()

    @Slot(bool)
    def _phaseUnwrapFromToolBar(self, unwrap: bool) -> None:
        if unwrap is not self.phaseUnwrap:
            self.phaseUnwrap = unwrap
            self._plotData()

    @Slot()
    def _showAxesOptionsDialog(self) -> None:
        """Open (or raise) the modeless figure-options dialog."""
        if self._optionsDialog is None:
            self._optionsDialog = FigureOptionsDialog(self)
            self._optionsDialog.optionsChanged.connect(self._optionsChanged)
            self._optionsDialog.exportRequested.connect(self._exportRequested)
        self._optionsDialog.setTraceNames(self.traceNames())
        self._optionsDialog.show()
        self._optionsDialog.raise_()
        self._optionsDialog.activateWindow()

    def traceNames(self) -> List[str]:
        """Labels of the traces currently drawn, for the per-trace UI."""
        return [str(line.get_label()) for line in figure_lines(self.plot.fig)]

    @Slot()
    def _optionsChanged(self) -> None:
        """Pull every option group back from the dialog and re-draw."""
        dialog = self._optionsDialog
        if dialog is None:
            return
        self.axesOptions = dialog.axesOptions()
        self.labelOptions = dialog.labelOptions()
        self.traceOptions = dialog.traceOptions()
        self.colorOptions = dialog.colorOptions()
        self.exportSpec = dialog.exportSpec()
        self._plotData()
        dialog.setTraceNames(self.traceNames())

    @Slot(str)
    def _exportRequested(self, filepath: str) -> None:
        if self._optionsDialog is not None:
            self.exportSpec = self._optionsDialog.exportSpec()
        try:
            self.exportFigure(filepath, self.exportSpec)
        except Exception as e:
            logger.error(f"Could not export figure to {filepath}: {e}")
            QtWidgets.QMessageBox.warning(
                self, 'Export failed', f'Could not write {filepath}:\n{e}')

    def _canPlot(self) -> bool:
        """Whether there is currently something plottable."""
        if self.plotDataType is PlotDataType.unknown:
            logger.debug("No plottable data.")
            return False
        if self.plotType is PlotType.empty:
            logger.debug("No plot routine determined.")
            return False
        return self.data is not None

    def _buildFigure(self, fig: Figure) -> None:
        """Draw the current data into `fig` and apply all user options.

        Factored out of :meth:`_plotData` so that the very same drawing code
        can render either the on-screen canvas or a separate, exactly-sized
        figure for export.

        :param fig: the figure to draw into. It is cleared first.
        """
        assert self.data is not None

        kw: Dict[str, Any] = {}
        with FigureMaker(fig) as fm:
            fm.plotType = self.plotType
            if not self.dataIsComplex():
                fm.complexRepresentation = ComplexRepresentation.real
            else:
                fm.complexRepresentation = self.complexRepresentation
            fm.phaseDegrees = self.phaseDegrees
            fm.phaseUnwrap = self.phaseUnwrap

            indeps = self.data.axes()
            for dn in plottableDependents(self.data):
                dvals = self.data.data_vals(dn)
                yerr = errorBarData(self.data, dn) if self.showErrorBars else None
                if yerr is not None:
                    kw[ERROR_BAR_KEY] = yerr
                else:
                    kw.pop(ERROR_BAR_KEY, None)
                fm.addData(
                    *[np.asanyarray(self.data.data_vals(n)) for n in indeps] + [dvals],
                    labels=[str(self.data.label(n)) for n in indeps] + [str(self.data.label(dn))],
                    plotDataType=self.plotDataType,
                    **kw)

        # re-apply user options so they survive automatic re-drawing.
        apply_axes_options_to_figure(fig, self.axesOptions)
        apply_trace_options(fig, self.traceOptions)
        apply_color_options(fig, self.colorOptions)
        apply_label_options(fig, self.labelOptions)
        apply_title(fig, self.labelOptions, default=self.defaultTitle())

    def defaultTitle(self) -> str:
        """Title to use when the user has not set one explicitly."""
        if self.data is None or not self.data.has_meta('title'):
            return ''
        # the loader puts the full file path in here, which is not something
        # anyone wants printed on a figure.
        return os.path.basename(str(self.data.meta_val('title')))

    def _plotData(self) -> None:
        """Plot the data using previously determined data and plot types."""
        if not self._canPlot():
            return
        assert self.data is not None

        self._buildFigure(self.plot.fig)
        self.setMeta(self.data)
        self.updatePlot()

    def exportFigure(self, filepath: str,
                     spec: Optional[ExportSpec] = None) -> bool:
        """Render the current plot into a file at an exact physical size.

        The figure is drawn into a *fresh* figure of the size given by `spec`,
        under export-specific rcParams (TrueType font embedding, a fixed font
        size). The on-screen canvas is untouched, so the exported result does
        not depend on the window size or the monitor's DPI.

        :param filepath: where to write the file.
        :param spec: export settings; defaults to :attr:`exportSpec`.
        :return: whether anything was written.
        """
        if not self._canPlot():
            return False

        if spec is None:
            spec = self.exportSpec

        with rc_context(spec.rcParams()):
            fig = Figure(figsize=spec.size(), dpi=spec.dpi,
                         constrained_layout=True)
            # a canvas is required for savefig to pick the right backend
            FigureCanvasAgg(fig)
            self._buildFigure(fig)
            save_figure(fig, filepath, spec)
        return True
