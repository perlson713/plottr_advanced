"""
``plottr.plot.mpl`` -- matplotlib plotting system for plottr.
Contains the following main objects:

Base UI elements
----------------

* :class:`.widgets.MPLPlot` (``matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg``) --
  Figure and Canvas widget.
  The most elementary Qt widget that contains the the matplotlib figure instance.

* :class:`.widgets.MPLPlotWidget` (:class:`plottr.plot.base.PlotWidget`) --
  A widget that contains the figure/canvas

General plotting functionality
------------------------------

* :class:`.plotting.PlotType` --
  Enum for currently implemented plot types in automatic plotting.

Automatic plotting
------------------

* :class:`.autoplot.AutoPlot` (:class:`.mpl.widgets.PlotWidget`) --
  PlotWidget that allows user selection of plot types and plots using
  :class:`.autoplot.FigureMaker`.

* :class:`.autoplot.FigureMaker` (:class:`.base.AutoFigureMaker`) --
  Matplotlib implementation of the figure manager.

Utilities
---------
* :func:`.widgets.figureDialog` --
  make a dialog window containing a plot widget.

Configuration
-------------
This module looks for a file `plottr_default.mplstyle` in the plottr config
directories and applies it to matplotlib plots using `pyplot.style.use`.

"""
import logging
import os

from matplotlib import rcParams, colormaps, pyplot as plt

from plottr import configFiles
from .autoplot import AutoPlot, FigureMaker
from .widgets import MPLPlot, MPLPlotWidget


logger = logging.getLogger(__name__)

#: name of the style file looked for in the plottr config directories.
DEFAULT_STYLE_FILE = 'plottr_default.mplstyle'

#: style shipped with plottr that produces publication-ready figures.
PUBLICATION_STYLE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'styles', 'publication.mplstyle')


def applyDefaultStyle() -> None:
    """Apply the user's default matplotlib style, if they have one.

    Looks for :data:`DEFAULT_STYLE_FILE` in the plottr config directories and
    applies the files found, in order of increasing priority. Missing or
    broken style files are logged and skipped rather than raised, so a bad
    style can never stop plottr from starting.
    """
    for filepath in configFiles(DEFAULT_STYLE_FILE):
        try:
            plt.style.use(filepath)
            logger.debug(f"Applied matplotlib style from {filepath}.")
        except Exception as e:
            logger.warning(f"Could not apply style file {filepath}: {e}")


def applyPublicationStyle() -> None:
    """Apply the publication style shipped with plottr.

    Sets serif fonts, inward ticks and TrueType font embedding -- the settings
    most journals expect. Intended to be applied on top of the defaults.
    """
    try:
        plt.style.use(PUBLICATION_STYLE_FILE)
    except Exception as e:
        logger.warning(f"Could not apply the publication style: {e}")


applyDefaultStyle()

