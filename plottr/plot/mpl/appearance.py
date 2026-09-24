"""``plottr.plot.mpl.appearance`` -- what a figure looks like, as plain objects.

Everything the toolbar can change about a figure lives here: point sizes and
colors, the title and the legend, the axis limits, the colormap of a 2-D plot,
and the settings a figure is exported with.  The widgets that edit these are in
``plottr.plot.mpl.autoplot``; keeping the values apart from the widgets is what
makes them saveable -- :func:`appearanceToDict` writes the lot to a file so the
next dataset can be drawn the same way, rather than retyped.

Two rules run through all of it:

* **Empty means automatic.**  A blank field, an empty string, a ``None`` limit:
  all mean "whatever the data says".  A figure keeps following the data until
  it is told not to, one field at a time.
* **Nothing here knows about Qt or about the data.**  These are values and the
  matplotlib keywords they turn into, which is why they can be tested without a
  window and written to a file without a converter.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    'FIT_COLOR_AUTO', 'LEGEND_AUTO', 'LEGEND_NONE', 'LEGEND_OUTSIDE',
    'LEGEND_LOCATIONS', 'LINE_STYLES', 'MARKERS', 'COLORMAPS',
    'FigureExport', 'PlotAxes', 'PlotColors', 'PlotLabels', 'PlotStyle',
    'TraceStyle', 'appearanceToDict', 'applyAppearance', 'parseNumber',
    'renderableText',
]


#: ``fitColor`` value meaning "whatever color the data got".
FIT_COLOR_AUTO = ''

#: ``legendLocation`` values that are not matplotlib ``loc`` strings.
LEGEND_AUTO = ''            #: a legend only where one is needed
LEGEND_NONE = 'none'        #: never
LEGEND_OUTSIDE = 'outside'  #: beside the axes, where it hides no data

#: Where the legend can go.  Everything after the first three is matplotlib's
#: own ``loc``, spelled the way matplotlib spells it.
LEGEND_LOCATIONS: Tuple[Tuple[str, str], ...] = (
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

#: Point shapes offered per trace.  ``''`` keeps whatever the overall style
#: says; ``'none'`` is our spelling of "no marker at all" (matplotlib wants an
#: empty string there, which would be indistinguishable from "automatic").
MARKERS: Tuple[Tuple[str, str], ...] = (
    ('Automatic', ''),
    ('Circle', 'o'),
    ('Square', 's'),
    ('Triangle', '^'),
    ('Diamond', 'D'),
    ('Star', '*'),
    ('Plus', '+'),
    ('Cross', 'x'),
    ('Point', '.'),
    ('None', 'none'),
)

#: Line styles offered per trace, same convention as :data:`MARKERS`.
LINE_STYLES: Tuple[Tuple[str, str], ...] = (
    ('Automatic', ''),
    ('Solid', '-'),
    ('Dashed', '--'),
    ('Dash-dot', '-.'),
    ('Dotted', ':'),
    ('None', 'none'),
)

#: Colormaps offered for 2-D plots.  The perceptually uniform ones first,
#: because a rainbow map invents structure that is not in the data; the
#: diverging ones are for quantities with a meaningful zero.
COLORMAPS: Tuple[Tuple[str, str], ...] = (
    ('Default', ''),
    ('Viridis', 'viridis'),
    ('Plasma', 'plasma'),
    ('Inferno', 'inferno'),
    ('Magma', 'magma'),
    ('Cividis', 'cividis'),
    ('Greys', 'Greys'),
    ('Blues', 'Blues'),
    ('Coolwarm (diverging)', 'coolwarm'),
    ('Red-blue (diverging)', 'RdBu_r'),
    ('Seismic (diverging)', 'seismic'),
)


def renderableText(text: str) -> str:
    """``text``, with maths matplotlib cannot parse turned into plain text.

    Labels are worth typing maths into -- `$Q_i$`, `$\\langle n \\rangle$` --
    and matplotlib raises while *drawing* a `$...$` it cannot parse, which
    takes the figure down rather than showing a bad label.  A string it refuses
    is escaped so the dollars come out as dollars, and the operator sees what
    is wrong instead of an empty window.
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


def parseNumber(text: str) -> Optional[float]:
    """A number out of what was typed, or ``None`` for an empty or bad field.

    ``None`` is the automatic limit, so a half-typed number reads as
    "automatic" rather than as an error box: the operator is in the middle of
    typing, and the figure should not change under them.
    """
    text = (text or '').strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _formatNumber(value: Optional[float]) -> str:
    """The text for a limit, short enough to read back in a small field."""
    if value is None:
        return ''
    return repr(float(value)) if abs(value) < 1e-4 else f'{value:g}'


class TraceStyle:
    """Color, point shape and line style of one trace.

    Each is empty by default, meaning "what the rest of the style says", which
    for the color is matplotlib's own cycle.  A comparison of six resonators is
    exactly where this matters: the cycle decides which one is orange, and the
    order the datasets were added decides the cycle.
    """

    def __init__(self, color: str = '', marker: str = '',
                 lineStyle: str = '') -> None:
        self.color = color
        self.marker = marker
        self.lineStyle = lineStyle

    def isEmpty(self) -> bool:
        return not (self.color or self.marker or self.lineStyle)

    def options(self) -> Dict[str, Any]:
        """matplotlib keyword arguments for this trace."""
        options: Dict[str, Any] = {}
        if self.color:
            options['color'] = self.color
        if self.marker:
            options['marker'] = '' if self.marker == 'none' else self.marker
        if self.lineStyle:
            options['linestyle'] = 'none' if self.lineStyle == 'none' \
                else self.lineStyle
        return options

    def toDict(self) -> Dict[str, str]:
        return {k: v for k, v in (('color', self.color),
                                  ('marker', self.marker),
                                  ('lineStyle', self.lineStyle)) if v}

    @classmethod
    def fromDict(cls, values: Dict[str, Any]) -> 'TraceStyle':
        return cls(color=str(values.get('color', '')),
                   marker=str(values.get('marker', '')),
                   lineStyle=str(values.get('lineStyle', '')))


class PlotStyle:
    """How the traces are drawn: point size, line widths, fit color.

    The defaults come from the matplotlib settings in ``plottr/config`` so that
    the config file stays the one place to change the overall look; the toolbar
    moves them per plot from there.  :attr:`traces` overrides them for one
    trace at a time, keyed by the name matplotlib puts in the legend.
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
        #: per-trace overrides, keyed by the legend label
        self.traces: Dict[str, TraceStyle] = {}

    def traceStyle(self, label: str) -> TraceStyle:
        """The overrides for this trace, making an empty one if there is none."""
        return self.traces.setdefault(label, TraceStyle())

    def dataOptions(self, label: str = '') -> Dict[str, Any]:
        """matplotlib keyword arguments for a measured trace."""
        options: Dict[str, Any] = {
            'markersize': self.markerSize,
            'linewidth': self.lineWidth,
        }
        if self.markerSize <= 0:
            options['marker'] = ''
        if label in self.traces:
            options.update(self.traces[label].options())
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

    def toDict(self) -> Dict[str, Any]:
        return dict(
            markerSize=self.markerSize,
            lineWidth=self.lineWidth,
            fitLineWidth=self.fitLineWidth,
            fitColor=self.fitColor,
            traces={name: trace.toDict()
                    for name, trace in self.traces.items()
                    if not trace.isEmpty()},
        )

    def update(self, values: Dict[str, Any]) -> None:
        for name in ('markerSize', 'lineWidth', 'fitLineWidth'):
            if name in values:
                setattr(self, name, float(values[name]))
        if 'fitColor' in values:
            self.fitColor = str(values['fitColor'])
        if 'traces' in values:
            self.traces = {str(name): TraceStyle.fromDict(trace)
                           for name, trace in dict(values['traces']).items()}


class PlotLabels:
    """What the figure says: title, axis labels, legend, colorbar.

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
        #: label beside the colorbar of a 2-D plot; empty takes the column's
        self.colorbarLabel: str = ''
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

    def toDict(self) -> Dict[str, Any]:
        return dict(title=self.title, showTitle=self.showTitle,
                    xLabel=self.xLabel, yLabel=self.yLabel,
                    colorbarLabel=self.colorbarLabel,
                    legendLocation=self.legendLocation,
                    legendNames=dict(self.legendNames))

    def update(self, values: Dict[str, Any]) -> None:
        for name in ('title', 'xLabel', 'yLabel', 'colorbarLabel',
                     'legendLocation'):
            if name in values:
                setattr(self, name, str(values[name]))
        if 'showTitle' in values:
            self.showTitle = bool(values['showTitle'])
        if 'legendNames' in values:
            self.legendNames = {str(k): str(v)
                                for k, v in dict(values['legendNames']).items()}


class PlotAxes:
    """Where the axes start and stop.

    ``None`` is the automatic limit.  Panning and zooming with the matplotlib
    toolbar is fine for looking, but a figure that has to be made again -- the
    same range for six resonators, or the same one next month -- needs numbers
    that can be typed and read back.

    Only the first panel is covered.  A split Re/Im plot has two, and the
    second one's range follows its own data; asking for a limit "on the y axis"
    of a figure that has two different y axes is a question without an answer.
    """

    def __init__(self) -> None:
        self.xMin: Optional[float] = None
        self.xMax: Optional[float] = None
        self.yMin: Optional[float] = None
        self.yMax: Optional[float] = None

    def isEmpty(self) -> bool:
        return all(v is None for v in
                   (self.xMin, self.xMax, self.yMin, self.yMax))

    def texts(self) -> Dict[str, str]:
        """The four limits as they go into the form."""
        return {'xMin': _formatNumber(self.xMin),
                'xMax': _formatNumber(self.xMax),
                'yMin': _formatNumber(self.yMin),
                'yMax': _formatNumber(self.yMax)}

    def setFromAxes(self, ax: Any) -> None:
        """Take the limits the plot happens to have now.

        What the zoom button is for: get the view right by eye, then keep it as
        numbers.
        """
        self.xMin, self.xMax = (float(v) for v in ax.get_xlim())
        self.yMin, self.yMax = (float(v) for v in ax.get_ylim())

    def applyTo(self, ax: Any) -> None:
        """Set the limits that were asked for, leaving the rest automatic."""
        if self.xMin is not None or self.xMax is not None:
            ax.set_xlim(left=self.xMin, right=self.xMax)
        if self.yMin is not None or self.yMax is not None:
            ax.set_ylim(bottom=self.yMin, top=self.yMax)

    def toDict(self) -> Dict[str, Any]:
        return dict(xMin=self.xMin, xMax=self.xMax,
                    yMin=self.yMin, yMax=self.yMax)

    def update(self, values: Dict[str, Any]) -> None:
        for name in ('xMin', 'xMax', 'yMin', 'yMax'):
            if name in values:
                value = values[name]
                setattr(self, name, None if value is None else float(value))


class PlotColors:
    """The color scale of a 2-D plot.

    A colormap is not decoration: a rainbow map invents edges that are not in
    the data, and a color range chosen by the outlier of a sweep flattens
    everything else.  Both are fixed from here.
    """

    def __init__(self) -> None:
        #: name of the colormap; empty keeps the one from ``plottr/config``
        self.colormap: str = ''
        #: ends of the color scale; ``None`` takes them from the data
        self.vMin: Optional[float] = None
        self.vMax: Optional[float] = None

    def keywords(self) -> Dict[str, Any]:
        """Keywords for :func:`.colorplot2d`."""
        options: Dict[str, Any] = {}
        if self.colormap:
            options['cmap'] = self.colormap
        if self.vMin is not None:
            options['vmin'] = self.vMin
        if self.vMax is not None:
            options['vmax'] = self.vMax
        return options

    def texts(self) -> Dict[str, str]:
        return {'vMin': _formatNumber(self.vMin),
                'vMax': _formatNumber(self.vMax)}

    def toDict(self) -> Dict[str, Any]:
        return dict(colormap=self.colormap, vMin=self.vMin, vMax=self.vMax)

    def update(self, values: Dict[str, Any]) -> None:
        if 'colormap' in values:
            self.colormap = str(values['colormap'])
        for name in ('vMin', 'vMax'):
            if name in values:
                value = values[name]
                setattr(self, name, None if value is None else float(value))


#: File formats a figure can be written in.  Vector first: a figure for a paper
#: should not be a photograph of a figure.
EXPORT_FORMATS: Tuple[Tuple[str, str], ...] = (
    ('PDF (vector)', 'pdf'),
    ('SVG (vector)', 'svg'),
    ('PNG', 'png'),
    ('EPS (vector)', 'eps'),
)


class FigureExport:
    """Size, resolution and format a figure is written out with.

    The save button of the matplotlib toolbar writes the figure at the size the
    window happens to have, so the same plot saved from a maximised window and
    from a small one come out with text of quite different relative size.  A
    figure for a paper has a width -- one column, two columns -- and the text
    has to be readable at it.
    """

    #: Widths that come up: one column and two columns of a two-column paper,
    #: and a slide.
    PRESETS: Tuple[Tuple[str, float, float], ...] = (
        ('One column (86 x 60 mm)', 86.0, 60.0),
        ('Two columns (180 x 110 mm)', 180.0, 110.0),
        ('Slide (254 x 143 mm)', 254.0, 143.0),
    )

    def __init__(self) -> None:
        #: figure size in millimetres
        self.width: float = 120.0
        self.height: float = 85.0
        #: resolution, for the raster formats
        self.dpi: int = 300
        #: leave the background out, for a slide that is not white
        self.transparent: bool = False
        #: ``pdf``, ``svg``, ``png`` or ``eps``
        self.format: str = 'pdf'
        #: crop to what is actually drawn
        self.tight: bool = True

    def inches(self) -> Tuple[float, float]:
        return self.width / 25.4, self.height / 25.4

    def saveKeywords(self) -> Dict[str, Any]:
        options: Dict[str, Any] = dict(dpi=self.dpi,
                                       transparent=self.transparent,
                                       format=self.format)
        if self.tight:
            options['bbox_inches'] = 'tight'
        if not self.transparent:
            options['facecolor'] = 'white'
        return options

    def toDict(self) -> Dict[str, Any]:
        return dict(width=self.width, height=self.height, dpi=self.dpi,
                    transparent=self.transparent, format=self.format,
                    tight=self.tight)

    def update(self, values: Dict[str, Any]) -> None:
        for name in ('width', 'height'):
            if name in values:
                setattr(self, name, float(values[name]))
        if 'dpi' in values:
            self.dpi = int(values['dpi'])
        for name in ('transparent', 'tight'):
            if name in values:
                setattr(self, name, bool(values[name]))
        if 'format' in values:
            self.format = str(values['format'])


#: Bumped when the meaning of a saved field changes, so that a file written by
#: a later version is refused rather than half understood.
APPEARANCE_VERSION = 1


def appearanceToDict(style: PlotStyle, labels: PlotLabels, axes: PlotAxes,
                     colors: PlotColors, export: FigureExport,
                     xScale: str = 'linear', yScale: str = 'linear') \
        -> Dict[str, Any]:
    """Everything the toolbar can change, as something writable to a file.

    This is what makes the rest of the panel worth using: a figure that took
    ten fields to get right can be given to the next dataset instead of being
    typed again.
    """
    return {
        'version': APPEARANCE_VERSION,
        'style': style.toDict(),
        'labels': labels.toDict(),
        'axes': axes.toDict(),
        'colors': colors.toDict(),
        'export': export.toDict(),
        'scales': {'x': xScale, 'y': yScale},
    }


def applyAppearance(values: Dict[str, Any], style: PlotStyle,
                    labels: PlotLabels, axes: PlotAxes, colors: PlotColors,
                    export: FigureExport) -> Tuple[str, str]:
    """Put a saved appearance back onto these objects.

    Missing sections are left alone rather than reset, so a file saved by an
    older version still applies as far as it goes.

    :return: the axis scales the file asks for.
    :raises ValueError: if the file was written by a later version.
    """
    version = int(values.get('version', APPEARANCE_VERSION))
    if version > APPEARANCE_VERSION:
        raise ValueError(
            f'this file was written by a newer plottr (format {version}, '
            f'this one understands {APPEARANCE_VERSION})')

    for name, target in (('style', style), ('labels', labels),
                         ('axes', axes), ('colors', colors),
                         ('export', export)):
        section = values.get(name)
        if isinstance(section, dict):
            target.update(section)

    scales = values.get('scales') or {}
    return str(scales.get('x', 'linear')), str(scales.get('y', 'linear'))
