"""The look of the application: palette, spacing, and widget styling.

Qt's default look is whatever the platform hands it, which on Windows is a grey
that fights with the white of a matplotlib figure and puts controls at
different sizes in different dialogs.  The point of this module is one light
theme, applied once at startup, that the plots sit inside of without clashing.

It is deliberately a *light* theme: a figure is white, and a white figure in a
dark window is a lamp in a dark room.  Saved figures also go into papers and
talks, where the surroundings are light.

The text follows the window: a window dragged out to a second screen, or a
plot opened at half the screen, should read at the same apparent size.  Qt has
no such thing built in, so :func:`applyTheme` watches every window and sets the
font from its size.  (The text *inside* the figure is matplotlib's, and follows
the canvas on its own -- see ``plot/mpl/widgets.py``.)

Nothing here changes what is plotted -- only the widgets around it.  The
matplotlib settings live in ``plottr/config``.
"""

from typing import Any, Optional

from plottr import QtCore

from plottr import QtGui, QtWidgets

__all__ = ['ACCENT', 'applyTheme', 'fontSizeForWindow', 'palette',
           'styleSheet']

#: The one saturated color, used for selection and for the focused control.
#: Taken from the first color of the plot cycle so that the window and the
#: figures inside it belong together.
ACCENT = '#1f77b4'

#: Background of the window, and of the panels sitting on it.
WINDOW = '#f4f5f7'
PANEL = '#ffffff'
BORDER = '#d6d9de'
TEXT = '#1c1e21'
MUTED = '#6b7280'

#: Window size the base font size is meant for, and the base size itself.
#: Sizes scale from here, within bounds that keep a very small window readable
#: and a very large one from turning into a poster.
REFERENCE_WINDOW = (1280, 800)
BASE_FONT_PX = 10

#: Text grows more slowly than the window: at this exponent, doubling the
#: window makes the text about 60% larger, which keeps a large window from
#: turning into a poster while still being visibly bigger.
FONT_SCALE_EXPONENT = 0.7

#: Bounds on that scale, so a tiny window stays readable and a wall display
#: does not fill with menu text.
FONT_SCALE_RANGE = (0.8, 1.7)


def fontSizeForWindow(width: int, height: int, scaling: float = 1.0) -> int:
    """Font size in pixels for a window of this size.

    Both dimensions matter and the smaller ratio wins: a window made wide but
    left short has no more room for text than a short one.
    """
    if width <= 0 or height <= 0:
        return int(round(BASE_FONT_PX * scaling))
    ratio = min(width / REFERENCE_WINDOW[0], height / REFERENCE_WINDOW[1])
    scale = ratio ** FONT_SCALE_EXPONENT
    low, high = FONT_SCALE_RANGE
    scale = max(low, min(high, scale))
    return max(8, int(round(BASE_FONT_PX * scaling * scale)))


#: Marks the block this module appends to a window's own stylesheet, so that
#: it can be replaced without disturbing what the window set itself.
FONT_BLOCK = '/* plottr: font follows the window size */'


class _WindowFontScaler(QtCore.QObject):
    """Keeps every window's font in step with its size.

    An application-wide event filter rather than something each window has to
    remember to install: windows are made in several places (the plot apps, the
    file browser, the dialogs), and one that forgot would be the odd one out.

    The size goes into the window's *stylesheet* rather than through
    ``setFont``.  With an application stylesheet in play, Qt resolves each
    styled widget's font from the stylesheet, and a font set on the window no
    longer reaches its children (checked: the child of a 17px window reported a
    font of its own).  A rule on the window does reach them, because a widget's
    own stylesheet wins over the application's.
    """

    def __init__(self, scaling: float = 1.0) -> None:
        super().__init__()
        self.scaling = scaling

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if event.type() == QtCore.QEvent.Resize and isinstance(
                obj, QtWidgets.QWidget) and obj.isWindow():
            self.applyTo(obj)
        return False

    def applyTo(self, window: Any) -> None:
        size = fontSizeForWindow(window.width(), window.height(), self.scaling)
        sheet = window.styleSheet() or ''
        own = sheet.split(FONT_BLOCK)[0]
        block = f'{FONT_BLOCK}\nQWidget {{ font-size: {size}px; }}\n'
        if sheet == own + block:
            return  # already at this size; re-applying re-polishes every widget
        window.setStyleSheet(own + block)


def palette() -> QtGui.QPalette:
    """The application palette: light, low contrast except where it matters."""
    p = QtGui.QPalette()
    color = QtGui.QColor
    p.setColor(QtGui.QPalette.Window, color(WINDOW))
    p.setColor(QtGui.QPalette.WindowText, color(TEXT))
    p.setColor(QtGui.QPalette.Base, color(PANEL))
    p.setColor(QtGui.QPalette.AlternateBase, color(WINDOW))
    p.setColor(QtGui.QPalette.Text, color(TEXT))
    p.setColor(QtGui.QPalette.Button, color(PANEL))
    p.setColor(QtGui.QPalette.ButtonText, color(TEXT))
    p.setColor(QtGui.QPalette.Highlight, color(ACCENT))
    p.setColor(QtGui.QPalette.HighlightedText, color('#ffffff'))
    p.setColor(QtGui.QPalette.ToolTipBase, color(PANEL))
    p.setColor(QtGui.QPalette.ToolTipText, color(TEXT))
    p.setColor(QtGui.QPalette.PlaceholderText, color(MUTED))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, color(MUTED))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, color(MUTED))
    return p


def styleSheet(scaling: float = 1.0) -> str:
    """The application stylesheet.

    :param scaling: display scaling, so that the sizes here follow the ones
        the rest of the application computes from the screen's DPI.
    """
    radius = 4
    base = fontSizeForWindow(*REFERENCE_WINDOW, scaling)
    return f"""
    QWidget {{
        color: {TEXT};
        font-size: {base}px;
    }}
    QMainWindow, QDialog {{
        background: {WINDOW};
    }}

    /* Panels: white cards on the window's grey, with room to breathe. */
    QDockWidget {{
        titlebar-close-icon: none;
    }}
    QDockWidget::title {{
        background: {WINDOW};
        color: {MUTED};
        padding: 4px 8px;
        border-bottom: 1px solid {BORDER};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    QGroupBox {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: {radius}px;
        margin-top: 10px;
        padding: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {MUTED};
    }}

    /* Toolbars: flat, with the checked state actually visible. */
    QToolBar {{
        background: {PANEL};
        border: none;
        border-bottom: 1px solid {BORDER};
        spacing: 2px;
        padding: 2px 4px;
    }}
    QToolBar QToolButton {{
        border: 1px solid transparent;
        border-radius: {radius}px;
        padding: 3px 6px;
    }}
    QToolBar QToolButton:hover {{
        background: {WINDOW};
        border-color: {BORDER};
    }}
    QToolBar QToolButton:checked {{
        background: #e4eef8;
        border-color: {ACCENT};
        color: {ACCENT};
    }}
    QToolBar QToolButton:disabled {{
        color: {MUTED};
    }}
    QToolBar::separator {{
        background: {BORDER};
        width: 1px;
        margin: 4px 6px;
    }}

    /* Buttons and entry fields: same height, same corner, visible focus. */
    QPushButton {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: {radius}px;
        padding: 4px 12px;
    }}
    QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
    QPushButton:pressed {{ background: #e4eef8; }}
    QPushButton:disabled {{ color: {MUTED}; border-color: {BORDER}; }}

    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: {radius}px;
        padding: 3px 6px;
        selection-background-color: {ACCENT};
        selection-color: #ffffff;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QPlainTextEdit:focus, QTextEdit:focus {{
        border-color: {ACCENT};
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}

    QCheckBox, QRadioButton {{ spacing: 6px; }}

    /* Lists and tables: quiet grid, clear selection. */
    QListView, QTreeView, QTableView {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: {radius}px;
        alternate-background-color: {WINDOW};
        selection-background-color: {ACCENT};
        selection-color: #ffffff;
        outline: none;
    }}
    QHeaderView::section {{
        background: {WINDOW};
        color: {MUTED};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: 4px 6px;
    }}

    QTabBar::tab {{
        background: transparent;
        color: {MUTED};
        padding: 5px 12px;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:selected {{
        color: {ACCENT};
        border-bottom-color: {ACCENT};
    }}

    QScrollBar:vertical, QScrollBar:horizontal {{
        background: transparent;
        border: none;
        width: 10px;
        height: 10px;
        margin: 0;
    }}
    QScrollBar::handle {{
        background: #c7cbd1;
        border-radius: 5px;
        min-height: 24px;
        min-width: 24px;
    }}
    QScrollBar::handle:hover {{ background: {MUTED}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QMenu {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: {radius}px;
        padding: 4px;
    }}
    QMenu::item {{ padding: 4px 18px 4px 12px; border-radius: 3px; }}
    QMenu::item:selected {{ background: #e4eef8; color: {ACCENT}; }}

    QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
    QToolTip {{
        background: {PANEL};
        color: {TEXT};
        border: 1px solid {BORDER};
        padding: 3px 6px;
    }}
    """


def applyTheme(app: Optional[Any] = None, scaling: float = 1.0) -> None:
    """Apply the theme to an application, and keep the font following windows.

    Safe to call more than once, and safe to call without an application (it
    then does nothing) so that importing a widget never depends on it.
    """
    if app is None:
        app = QtWidgets.QApplication.instance()
    if app is None:
        return
    try:
        app.setStyle('Fusion')  # the same starting point on every platform
    except Exception:  # noqa: BLE001 -- a missing style is not worth failing on
        pass
    app.setPalette(palette())
    app.setStyleSheet(styleSheet(scaling))

    # Held on the application: an event filter that is garbage collected stops
    # filtering, silently.
    scaler = getattr(app, '_plottrFontScaler', None)
    if scaler is None:
        scaler = _WindowFontScaler(scaling)
        app.installEventFilter(scaler)
        app._plottrFontScaler = scaler
    scaler.scaling = scaling
