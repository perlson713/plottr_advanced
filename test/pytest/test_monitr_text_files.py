"""Tests for reading the text files that sit next to a dataset.

A dataset folder collects text from several writers.  ``DDH5Writer.backup_file``
copies the measurement scripts in byte for byte, and those are UTF-8 with
non-ASCII comments; text this viewer wrote before it named an encoding is in
the machine's locale encoding (cp932 on a Japanese Windows).

Reading with the locale encoding raised UnicodeDecodeError on the first
non-ASCII character of a UTF-8 file, and because that happens while the
right-hand file panel is being populated, one backed-up script took the whole
panel down::

    File "monitr.py", line 2449, in __init__
      self.file_text = file.read()
    UnicodeDecodeError: 'cp932' codec can't decode byte 0x8f in position 1527
"""

import locale

import pytest

from plottr.apps.monitr import read_text_file

JAPANESE = '# 共振器のフィット\nQl は円フィットから求める\n'


def test_reads_utf8(tmp_path):
    """The case that crashed: a script backed up into the dataset folder."""
    path = tmp_path / 'resonator_powersweep_repeat.py'
    path.write_bytes(JAPANESE.encode('utf-8'))
    assert read_text_file(path) == JAPANESE


def test_reads_the_locale_encoding_too(tmp_path):
    """Comments this viewer wrote earlier are in the locale encoding."""
    path = tmp_path / 'comment.md'
    path.write_bytes(JAPANESE.encode('cp932'))

    # 読み手のロケールが cp932 のときだけ、その解釈に落ちれば正しく読める。
    if locale.getpreferredencoding(False).lower() not in ('cp932', 'shift_jis'):
        pytest.skip('locale is not cp932, so cp932 bytes are not expected to decode')
    assert read_text_file(path) == JAPANESE


def test_never_raises_on_undecodable_bytes(tmp_path):
    """Whatever the bytes are, the file panel must still come up."""
    path = tmp_path / 'broken.md'
    path.write_bytes(b'ok \x8f\xff\xfe then')
    text = read_text_file(path)
    assert text.startswith('ok ') and text.endswith(' then')


def test_ascii_is_unchanged(tmp_path):
    path = tmp_path / 'plain.md'
    path.write_text('nothing special here\n')
    assert read_text_file(path) == 'nothing special here\n'


def test_empty_file(tmp_path):
    path = tmp_path / 'empty.md'
    path.write_bytes(b'')
    assert read_text_file(path) == ''
