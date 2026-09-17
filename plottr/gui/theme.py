"""The look of the application: palette, spacing, and widget styling.

Qt's default look is whatever the platform hands it, which on Windows is a grey
that fights with the white of a matplotlib figure and puts controls at
different sizes in different dialogs.  The point of this module is one light
theme, applied once at startup, that the plots sit inside of without clashing.

It is deliberately a *light* theme: a figure is white, and a white figure in a
dark window is a lamp in a dark room.  Saved figures also go into papers and
talks, where the surroundings are light.

Nothing here changes what is plotted -- only the widgets around it.  The
matplotlib settings live in ``plottr/config``.
"""

from typing import Any, Optional

from plottr import QtGui, QtWidgets

__all__ = ['ACCENT', 'applyTheme', 'palette', 'styleSheet']

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
    font = max(10, int(round(10 * scaling)))
    small = max(9, int(round(9 * scaling)))
    radius = 4
    return f"""
    QWidget {{
        color: {TEXT};
        font-size: {font}px;
    }}
    QMainWindow, QDialog {{
        background: {WINDOW};
    }}

    /* Panels: white cards on the window's grey, with room to breathe. */
    QDockWidget {{
        titlebar-close-icon: none;
        font-size: {small}px;
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
    """Apply the theme to an application.

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
