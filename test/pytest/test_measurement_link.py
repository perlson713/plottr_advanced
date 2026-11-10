"""How plottr finds the measurement repository's ``dataset_refit``.

These run without qcodes_measurement: they build the folder layouts in a
temporary directory.  The real link is exercised by test_resonator_node.py
and test_dependent_axis_node.py (skipped when the repository is not found).
"""
import types

from plottr.node import resonator
from plottr.node.resonator import MEASUREMENT_INTERFACE, _interfaceNote, _locate


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('')


def test_repository_root_with_support_package(tmp_path):
    # 今の qcodes_measurement: support/dataset_refit.py（パッケージとして import）
    _touch(tmp_path / 'support' / '__init__.py')
    _touch(tmp_path / 'support' / 'dataset_refit.py')
    assert _locate(tmp_path) == (tmp_path, 'support.dataset_refit')


def test_support_folder_itself(tmp_path):
    # RESONATOR_FIT_PATH に support フォルダを指定した場合も、親をパスに入れる
    # （dataset_refit は `from support import resonator_fit` するため）。
    _touch(tmp_path / 'support' / '__init__.py')
    _touch(tmp_path / 'support' / 'dataset_refit.py')
    assert _locate(tmp_path / 'support') == (tmp_path, 'support.dataset_refit')


def test_old_flat_layout(tmp_path):
    # support パッケージに分ける前（2026-10-29 より前）の構成。
    _touch(tmp_path / 'dataset_refit.py')
    assert _locate(tmp_path) == (tmp_path, 'dataset_refit')


def test_unrelated_folder(tmp_path):
    assert _locate(tmp_path) is None
    assert _locate(tmp_path / 'missing') is None


def test_interface_mismatch_names_the_side_that_is_behind():
    current = types.SimpleNamespace(PLOTTR_INTERFACE=MEASUREMENT_INTERFACE)
    old = types.SimpleNamespace()  # 定数が無い版は 1
    new = types.SimpleNamespace(PLOTTR_INTERFACE=MEASUREMENT_INTERFACE + 1)
    assert _interfaceNote(current) == ''
    assert 'update qcodes_measurement' in _interfaceNote(old)
    assert 'update plottr_advanced' in _interfaceNote(new)


def test_not_found_message_points_at_the_fixes(tmp_path, monkeypatch):
    monkeypatch.setattr(resonator, '_MODULE', None)
    monkeypatch.setenv('RESONATOR_FIT_PATH', str(tmp_path))
    message = resonator._whyNotImported(
        ModuleNotFoundError("No module named 'support'", name='support'),
        [tmp_path], [], str(tmp_path))
    assert 'support/dataset_refit.py' in message
    assert 'install_plottr_link.py' in message
