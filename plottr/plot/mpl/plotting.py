"""
``plottr.plot.mpl.plotting`` -- Plotting tools (mostly used in Autoplot)
"""

from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
from matplotlib import colors, rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.cm import ScalarMappable

from plottr.utils import num
from plottr.utils.num import centers2edges_2d, interp_meshgrid_2d

__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'


@unique
class PlotType(Enum):
    """Plot types currently supported in Autoplot."""

    #: no plot defined
    empty = auto()

    #: a single 1D line/scatter plot per panel
    singletraces = auto()

    #: multiple 1D lines/scatter plots per panel
    multitraces = auto()

    #: image plot of 2D data
    image = auto()

    #: colormesh plot of 2D data
    colormesh = auto()

    #: 2D scatter plot
    scatter2d = auto()


class SymmetricNorm(colors.Normalize):
    """Color norm that's symmetric and linear around a center value."""

    def __init__(self, vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 vcenter: float = 0,
                 clip: bool = False):
        super().__init__(vmin, vmax, clip)
        self.vcenter = vcenter

    # this only overrides one of the overloaded signatures and therefor fails
    # however on 3.8 this does not fail.
    def __call__(self, value: float, clip: Optional[bool] = None) -> float:  # type: ignore[override,unused-ignore]
        vlim = max(abs(self.vmin - self.vcenter), abs(self.vmax - self.vcenter))
        self.vmax: float = vlim + self.vcenter
        self.vmin: float = -vlim + self.vcenter
        return super().__call__(value, clip)


# 2D plots
def colorplot2d(ax: Axes,
                x: Union[np.ndarray, np.ma.MaskedArray],
                y: Union[np.ndarray, np.ma.MaskedArray],
                z: Union[np.ndarray, np.ma.MaskedArray],
                plotType: PlotType = PlotType.image,
                axLabels: Tuple[Optional[str], Optional[str], Optional[str]] = ('', '', ''),
                **kw: Any) -> Optional[ScalarMappable]:
    """make a 2d colorplot. what plot is made, depends on `plotType`.
    Any of the 2d plot types in :class:`PlotType` works.

    :param ax: matplotlib subPlots to plot in
    :param x: x coordinates (meshgrid)
    :param y: y coordinates (meshgrid)
    :param z: z data
    :param plotType: the plot type
    :param axLabels: labels for the x, y subPlots, and the colorbar.

    all keywords are passed to the actual plotting functions, depending on the ``plotType``:

    - :attr:`PlotType.image` --
        :func:`plotImage`
    - :attr:`PlotType.colormesh` --
        :func:`ppcolormesh_from_meshgrid`
    - :attr:`PlotType.scatter2d` --
        matplotlib's `scatter`
    """
    cmap = kw.pop('cmap', rcParams['image.cmap'])

    # first we need to check if our grid can be plotted nicely.
    if plotType in [PlotType.image, PlotType.colormesh]:
        x = x.astype(float)
        y = y.astype(float)
        z = z.astype(float)

        # first check if we need to fill some masked values in
        if isinstance(x, np.ma.MaskedArray) and np.ma.is_masked(x):
            x = x.filled(np.nan)
        if isinstance(y, np.ma.MaskedArray) and np.ma.is_masked(y):
            y = y.filled(np.nan)
        if isinstance(z, np.ma.MaskedArray) and np.ma.is_masked(z):
            z = z.filled(np.nan)

        # next: try some surgery, if possible
        if np.all(num.is_invalid(x)) or np.all(num.is_invalid(y)):
            return None
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            x, y = interp_meshgrid_2d(x, y)
        if np.any(num.is_invalid(x)) or np.any(num.is_invalid(y)):
            x, y, z = num.crop2d(x, y, z)

        # next, check if the resulting grids are even still plottable
        for g in x, y, z:
            if g.size == 0:
                return None
            elif len(g.shape) < 2:
                return None

            # special case: if we have a single line, a pcolor-type plot won't work.
            elif min(g.shape) < 2:
                plotType = PlotType.scatter2d
    im: Optional[ScalarMappable]
    if plotType is PlotType.image:
        im = plotImage(ax, x, y, z, cmap=cmap, **kw)
    elif plotType is PlotType.colormesh:
        im = ppcolormesh_from_meshgrid(ax, x, y, z, cmap=cmap, **kw)
    elif plotType is PlotType.scatter2d:
        im = ax.scatter(x.ravel(), y.ravel(), c=z.ravel(), cmap=cmap, **kw)
    else:
        im = None

    if im is None:
        return None

    if axLabels[0]:
        ax.set_xlabel(axLabels[0])
    if axLabels[1]:
        ax.set_ylabel(axLabels[1])
    return im


def ppcolormesh_from_meshgrid(ax: Axes, x: np.ndarray, y: np.ndarray,
                              z: np.ndarray, **kw: Any) -> Optional[ScalarMappable]:
    r"""Plot a pcolormesh with some reasonable defaults.
    Input are the corresponding arrays from a 2D ``MeshgridDataDict``.

    Will attempt to fix missing points in the coordinates.

    :param ax: subPlots to plot the colormesh into.
    :param x: x component of the meshgrid coordinates
    :param y: y component of the meshgrid coordinates
    :param z: data values
    :returns: the image returned by `pcolormesh`.

    Keywords are passed on to `pcolormesh`.
    """
    # the meshgrid we have describes coordinates, but for plotting
    # with pcolormesh we need vertices.
    try:
        x = centers2edges_2d(x)
        y = centers2edges_2d(y)
    except:
        return None

    im = ax.pcolormesh(x, y, z, **kw)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return im


def plotImage(ax: Axes, x: np.ndarray, y: np.ndarray,
              z: np.ndarray, **kw: Any) -> AxesImage:
    """Plot 2d meshgrid data as image.

    :param ax: matplotlib subPlots to plot the image in.
    :param x: x coordinates (as meshgrid)
    :param y: y coordinates
    :param z: z values
    :returns: the image object returned by `imshow`

    All keywords are passed to `imshow`.
    """
    ax.grid(False)
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()

    extentx = [x0, x1]
    if x0 > x1:
        extentx = extentx[::-1]
    if x0 == x1:
        extentx = [x0, x0 + 1]
    extenty = [y0, y1]
    if y0 > y1:
        extenty = extenty[::-1]
    if y0 == y1:
        extenty = [y0, y0 + 1]
    extent = cast(Tuple[float,float,float,float], tuple(extentx + extenty))

    if x.shape[0] > 1:
        # in image mode we have to be a little careful:
        # if the x/y subPlots are specified with decreasing values we need to
        # flip the image. otherwise we'll end up with an axis that has the
        # opposite ordering from the data.
        z = z if x[0, 0] < x[1, 0] else z[::-1, :]

    if y.shape[1] > 1:
        z = z if y[0, 0] < y[0, 1] else z[:, ::-1]

    im = ax.imshow(z.T, aspect='auto', origin='lower',
                   extent=extent, **kw)
    return im


# Axis/display helpers -------------------------------------------------------

#: matplotlib axis scales we expose in the GUI.
AXIS_SCALES: Tuple[str, ...] = ('linear', 'log', 'symlog')


def square_axes(ax: Axes) -> None:
    """Give an axes a square plot range with equal aspect ratio.

    The current x and y ranges are replaced by a single, common range
    (the larger of the two spans, centered on the data), and the aspect
    ratio is set to ``'equal'``. This makes, e.g., a circle in the complex
    plane appear as a circle rather than an ellipse.

    :param ax: the axes to modify.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    half = 0.5 * max(abs(x1 - x0), abs(y1 - y0))
    if half == 0 or not np.isfinite(half):
        half = 1.0
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    ax.set_aspect('equal', adjustable='box')


def data_axes(fig: Figure) -> List[Axes]:
    """Return the "data" axes of a figure, i.e. everything except colorbars.

    :param fig: the figure to inspect.
    :return: list of axes that hold plotted data.
    """
    return [ax for ax in fig.axes if ax.get_label() != '<colorbar>']


@dataclass
class AxesOptions:
    """User-adjustable matplotlib axes properties.

    Any attribute left at its "unset" value (``None`` for most, ``False`` for
    :attr:`equalAspect`) leaves the corresponding matplotlib default/automatic
    behaviour untouched. Instances are meant to be re-applied to freshly drawn
    axes so that user choices survive plottr's automatic re-drawing.
    """

    #: x-axis scale, one of :data:`AXIS_SCALES`; ``None`` leaves it as-is.
    xscale: Optional[str] = None
    #: y-axis scale, one of :data:`AXIS_SCALES`; ``None`` leaves it as-is.
    yscale: Optional[str] = None
    #: lower x limit; ``None`` means auto-scale.
    xmin: Optional[float] = None
    #: upper x limit; ``None`` means auto-scale.
    xmax: Optional[float] = None
    #: lower y limit; ``None`` means auto-scale.
    ymin: Optional[float] = None
    #: upper y limit; ``None`` means auto-scale.
    ymax: Optional[float] = None
    #: whether to draw the grid; ``None`` leaves the rcParams default.
    grid: Optional[bool] = None
    #: whether to force an equal (square) aspect ratio.
    equalAspect: bool = False

    def isDefault(self) -> bool:
        """Whether all options are at their neutral (do-nothing) value."""
        return (
            self.xscale is None and self.yscale is None
            and self.xmin is None and self.xmax is None
            and self.ymin is None and self.ymax is None
            and self.grid is None and not self.equalAspect
        )


def apply_axes_options(ax: Axes, options: AxesOptions) -> None:
    """Apply :class:`AxesOptions` to a single matplotlib axes.

    Only options that differ from their neutral value have an effect, so this
    can be called on any axes without clobbering settings the user did not
    explicitly request.

    :param ax: the axes to modify.
    :param options: the options to apply.
    """
    if options.xscale is not None:
        ax.set_xscale(options.xscale)
    if options.yscale is not None:
        ax.set_yscale(options.yscale)

    if options.grid is not None:
        ax.grid(options.grid)

    if options.equalAspect:
        ax.set_aspect('equal', adjustable='box')

    # limits are applied last so that they win over any autoscaling triggered
    # by the settings above. A value of None keeps the current (auto) limit.
    if options.xmin is not None or options.xmax is not None:
        ax.set_xlim(left=options.xmin, right=options.xmax)
    if options.ymin is not None or options.ymax is not None:
        ax.set_ylim(bottom=options.ymin, top=options.ymax)


def apply_axes_options_to_figure(fig: Figure, options: AxesOptions) -> None:
    """Apply :class:`AxesOptions` to all data axes of a figure.

    Colorbar axes are left untouched.

    :param fig: the figure whose data axes should be modified.
    :param options: the options to apply.
    """
    if options.isDefault():
        return
    for ax in data_axes(fig):
        apply_axes_options(ax, options)
