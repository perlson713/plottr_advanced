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
