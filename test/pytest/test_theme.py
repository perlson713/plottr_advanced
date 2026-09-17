"""Tests for the application theme.

Little to assert about how something looks; what matters is that applying the
theme never fails and never depends on an application being there, because it
is called from every entry point and from tests that have no window.
"""

from plottr import QtWidgets
from plottr.gui.theme import ACCENT, applyTheme, palette, styleSheet


def test_stylesheet_is_valid_qss_ish():
    sheet = styleSheet()
    assert sheet.count('{') == sheet.count('}')
    assert ACCENT in sheet


def test_scaling_changes_the_font_size():
    assert styleSheet(1.0) != styleSheet(2.0)
    assert 'font-size: 20px' in styleSheet(2.0)


def test_apply_sets_palette_and_stylesheet(qtbot):
    app = QtWidgets.QApplication.instance()
    assert app is not None
    applyTheme(app)
    assert app.styleSheet() != ''
    assert app.palette().color(QtWidgets.QApplication.palette().Window).isValid()

    # 2 回呼んでも壊れない（入口ごとに呼ばれる）
    applyTheme(app)


def test_apply_without_an_application_does_nothing(monkeypatch):
    monkeypatch.setattr(QtWidgets.QApplication, 'instance', staticmethod(lambda: None))
    applyTheme()   # 例外を出さないこと


def test_palette_is_light():
    window = palette().color(QtWidgets.QApplication.palette().Window)
    # 明るいテーマ: 図（白）を暗い窓に置かない
    assert window.lightness() > 200


# ---------------------------------------------------------------------------
# ウィンドウの大きさに合わせたフォント。
# ---------------------------------------------------------------------------


def test_font_grows_with_the_window_but_slower_than_it():
    from plottr.gui.theme import REFERENCE_WINDOW, fontSizeForWindow

    base = fontSizeForWindow(*REFERENCE_WINDOW)
    small = fontSizeForWindow(800, 500)
    large = fontSizeForWindow(2560, 1600)

    assert small < base < large
    # 窓が 2 倍でも文字は 2 倍にしない（見出しだらけの画面にしない）
    assert large < 2 * base


def test_a_wide_but_short_window_does_not_get_big_text():
    """短辺で決める。横に広いだけの窓に文字を置く余地は増えない。"""
    from plottr.gui.theme import fontSizeForWindow

    assert fontSizeForWindow(3000, 500) == fontSizeForWindow(1000, 500)


def test_font_size_is_bounded():
    from plottr.gui.theme import FONT_SCALE_RANGE, fontSizeForWindow

    tiny = fontSizeForWindow(200, 150)
    huge = fontSizeForWindow(8000, 5000)
    assert tiny >= 8
    assert huge <= fontSizeForWindow(1280, 800) * FONT_SCALE_RANGE[1] + 1
    # 0 やマイナスでも落ちない
    assert fontSizeForWindow(0, 0) > 0


def test_the_window_font_follows_a_resize(qtbot):
    from plottr import QtGui

    app = QtWidgets.QApplication.instance()
    applyTheme(app)

    window = QtWidgets.QMainWindow()
    label = QtWidgets.QLabel('x')
    window.setCentralWidget(label)
    qtbot.addWidget(window)
    window.resize(820, 560)
    window.show()
    qtbot.waitExposed(window)
    smallText = QtGui.QFontMetrics(label.font()).height()

    window.resize(2200, 1400)
    app.processEvents()
    largeText = QtGui.QFontMetrics(label.font()).height()

    # 子ウィジェットまで届いていること（届かない実装を 1 度踏んでいる）
    assert largeText > smallText
