# plottr: modular data plotting and processing

[![PyPi version](https://badge.fury.io/py/plottr.svg)](https://badge.fury.io/py/plottr)
[![PyPI python versions](https://img.shields.io/pypi/pyversions/plottr.svg)](https://pypi.python.org/pypi/plottr/)
[![Docs](https://img.shields.io/badge/read%20-thedocs-ff66b4.svg)](https://plottr.readthedocs.io/en/latest/)
[![Build on GitHub actions](https://github.com/toolsforexperiments/plottr/workflows/Python%20application/badge.svg?branch=master)](https://github.com/toolsforexperiments/plottr/actions)

A framework for inspecting data, based on flowcharts from *pyqtgraph*.
*plottr*'s main aim is to allow the user to define custom data processing flows and plotting.
A particular use case is data filtering and plotting.

## Documentation: 
https://plottr.readthedocs.io (work in progress...)

# plottr_advanced（このフォーク）について

公開版の [plottr](https://github.com/toolsforexperiments/plottr) をもとに、共振器測定
（[qcodes_measurement](https://github.com/perlson713/qcodes_measurement)）で使いやすいよう拡張したものです。
主な追加: 複素平面表示とエラーバー（`<dep>_err` の自動認識）、`Resonator fit` / `Photon axis` /
`Compare datasets` ノード、`Scale units`、図の `Style` / `Labels` / `Axes` / `Colors` / `Export...` /
`Appearance`、monitr のテキスト読み込み（UTF-8）。

## インストール（qcodes_measurement と並べる）

qcodes_measurement と**同じフォルダに並べてクローン**し、編集可能インストールにしてください。

```powershell
git clone https://github.com/perlson713/qcodes_measurement.git
git clone https://github.com/perlson713/plottr_advanced.git
cd qcodes_measurement
python -m pip install -r requirements.txt
python -m pip install -e ..\plottr_advanced     # requirements.txt が入れたコピーを置き換える
python support\check_plottr_link.py             # 結び付きと版の一致を確認
```

`pip install git+https://...plottr_advanced.git`（qcodes_measurement の `requirements.txt` も
これ）は、**その時点の `main` を site-packages にコピーするだけ**です。後でこのリポジトリを更新しても、
そのコピーは古いままです。どれが動いているかは
`python -c "import plottr; print(plottr.__file__)"` で確認できます。

## qcodes_measurement との結び付き

`Resonator fit` と `Photon axis` は、フィットと光子数の計算を qcodes_measurement の
`support/dataset_refit.py` に任せます（物理をこちらに書き写すと、検証済みの実装と黙って食い違うため）。
探す順序は `plottr/node/resonator.py` の `resonator_fit_module()` にあります:

1. 環境変数 `RESONATOR_FIT_PATH`（qcodes_measurement のフォルダ。`support` を指しても可）
2. このリポジトリの隣の `../qcodes_measurement`（上のインストールならこれで見つかる）
3. `sys.path` 上の `support.dataset_refit`（qcodes_measurement の `support/install_plottr_link.py`
   が置く `.pth`、または `PYTHONPATH`）。`support/` に分ける前の旧構成（直下の `dataset_refit.py`）も読めます

見つかればノードの `Status` は `using ...\support\dataset_refit.py`、見つからなければ探したフォルダと直し方を出します。

### 版の照合

接点の API は `plottr.node.resonator.MEASUREMENT_INTERFACE`（こちら）と
`dataset_refit.PLOTTR_INTERFACE`（qcodes_measurement）で番号を付けてあり、いまは **2** です。
食い違うと `Status` の末尾に `update qcodes_measurement` / `update plottr_advanced` と、
**どちらが古いか**が出ます。**API を変えたら両方の番号を上げてください。**

| 版 | 内容 |
|---|---|
| 1 | `fittable_dependents`・`default_dependent`・`fit_datadict`・`recompute_photons` |
| 2 | `to_record_layout` — フィット結果を 1 レコード＝1 掃引の形に戻す |

### 2026-11-10 の整合修正

qcodes_measurement と別々に改変した結果、次の食い違いが出ていたので直しました
（詳細は qcodes_measurement の README の「plottr_advancedとの整合」）。

- **`Resonator fit` / `Photon axis` が測定リポジトリを見つけられなかった。** 2026-10-29 に向こうで
  `dataset_refit.py` が `support/` パッケージへ移ったのに、こちらは `../qcodes_measurement/dataset_refit.py`
  を探していた。上の順序で `support.dataset_refit` を探すようにした。
- **ノードを通した後、`Qi` 対パワーが表示できなかった。** フィットはデータを展開（1 点＝1 行）して返すので、
  パワーごとの量が周波数の点数ぶん複製され、Grid ノードが `power is expected to be 0th axis but has no
  variation along that axis` で止まっていた。ノードの出力を `to_record_layout` で元の形に戻す。
  当て直しのファイル（`refit_dataset.py`・`refit_raw_data.ipynb`）も同じ理由で表示できなかったので、
  向こうも保存前に同じ処理をするようにした。
- **テストが食い違いを見逃していた。** `test/pytest/test_resonator_node.py` と
  `test_dependent_axis_node.py` は測定リポジトリが見つからないと**スキップ**する。探索が外れていたため
  全部スキップされていた。探し方は `test/pytest/test_measurement_link.py`（測定リポジトリ不要）で押さえ、
  Grid が Q 対パワーを格子にできることは `test_per_power_columns_can_be_gridded` で押さえる。

```bash
# 両リポジトリを並べた状態で（スキップされずに走ることを確認する）
pytest -rs test/pytest/test_measurement_link.py test/pytest/test_resonator_node.py test/pytest/test_dependent_axis_node.py
```

# Quickstart

## Installation

Plottr is installable from pypi with `pip install plottr`

Plottr requires either the PyQt5 or Pyside2 gui framework.
To install with PyQt5 or Pyside2 backend you can do
``pip install plottr[PyQt5]`` or ``pip install plottr[Pyside2]`` 

Note that if  you have installed ``pyqt`` from ``(Ana)Conda`` you should not use any of these
targets but do ``pip install plottr`` or install Plottr from conda forge:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install plottr
```

To install from source: clone the repo, and install using `pip install -e .`

## inspectr: QCoDeS dataset inspection and (live) plotting

You can use the `inspectr` tool to get a simple overview over QCoDeS database
files, and plot datasets contained in the database.

To use: run `plottr-inspectr [--dbpath <path to your .db file>]` 
 
As an alternative from the root of the cloned plottr repository, run `python apps/inspectr.py [--dbpath <path to your .db file>]`

For basic instructions, check out the Notebook *Live plotting qcodes data* under `/doc/examples`.

## Some notes on installing

Note: this package is not compatible with the original `plottr` tool.
You might want to install freshly if you still use the old version.

## Requirements:
* python >= 3.8
* the usual: numpy, mpl, ...
* pandas >= 0.22
* xarray
* pyqtgraph >= 0.12.1

# Recent changes:

See the [plottr GitHub Releases page](https://github.com/toolsforexperiments/plottr/releases) for versions and their changelogs

## v0.10.0 2022-07-14

### Added 

- added a generalized way for fitting data (or similar procedures that generate analysis parameters from datasets) (#293)
- ... and more, see here https://github.com/toolsforexperiments/plottr/releases/tag/v0.10.0

## v0.9.1 2022-01-12

### Added

- Fixed a crash bug with python 3.10 (#257)

## v0.9.0 2021-12-20

### Added

* Add new analysis node for creating histogram of data by @wpfff in https://github.com/toolsforexperiments/plottr/pull/241

### Fixed

* bugfix: prevent flushing when we are updating data. by @wpfff in https://github.com/toolsforexperiments/plottr/pull/243
* fix type checking with mypy 920 by @jenshnielsen in https://github.com/toolsforexperiments/plottr/pull/248
* Bump minimum version of pyqtgraph by @jenshnielsen in https://github.com/toolsforexperiments/plottr/pull/247

## v0.8.1 2021-11-30

### Added

- Test with python 3.10 and mark as supported (#238)

### Fixed

- add missing init file to config dir (#236)
- Use a regular install in tests (#237)

## v0.8.0 2021-11-11

### Added

- Inspectr: tag a run with a star (⭐) or cross (❌) icon, filter by those,
  also show dataset metadata next to parameters and snapshot (#229)
- Improvements to monitr: more stability in adding data to ddh5, better
  performance by making data loading multithreaded and running reach plot
  window in a separate process (#219)
- Added pyqtgraph backend for plotting that can be used instead of matplotlib
  (Example for how to select can be found in test/apps/autoplot_app.py) (#215, #218)

### Fixed

- Fix/invaliddata: small fixes when data contains a lot of invalid entries (#226)
- Fix in shape recognition when adding data (#220)

### Behind the scenes

- Add minimal versions to dependencies (#201)
- Make the .gitignore proper (#73)
- add dependabot (#208)
- Fix typechecking with mypy 0.9xx (#207)
- clarify install instructions wrt qt and mention conda forge (#202)

## 2021-06-08

### Added

- refactoring the plotting system (#166)
- Add version log message to main ``__init__`` (#175)

### Fixed

- Fix crop if less than one row is not nan (#198)
- Fix rgba error (#199)
- Allow empty dataset if datadict is none (#195)

### Behind the scenes

- Modernize setup files (#194)
- packaging cleanups (#177)
- upgrade versioneer to 0.19 (#176)

## 2021-02-16

### Added

- Add copy content features to inspectr and autoplot windows, specifically
  - a new Copy pop up menu for copying content of cells in inspectr
  - a new Copy metadata button in plot window for copying info about the dataset to clipboard

### Fixed

- remove redundant information between the optional "info" box on the plot and the plot title

## 2021-02-08

- Drop support for Python 3.6 and support type-checking with qcodes 0.21.0
- Fix type-checking with numpy 1.20

### Fixed
- Fixed y-axis to not show axis-label if more than one plot is selected in 1D single-plot show.

## 2020-08-21

- Workaround for bug with pyqt installed via conda that would result in blank icons.
- Add experimental support for using Pyside2 as an alternative to PyQt5

## 2020-08-06

### Added

- Entry points for inspectr (plottr-inspectr) and autoplot (plottr-autoplot-ddh5) 
- LICENSE file has been added (no change to license of the code)
- setup.py has been tweeked to ensure that sdist and bdist_wheel packages are generated correctly

## 2020-06-05

### Added

- Usable version of hdf5 file support. can use the apps/monitr.py script to launch a tool that allows easy (live) plotting.

## 2020-04-24

### Fixed

- there were several issues/annoyances that caused (minor) issues with grid handling and plotting in rare-ish circumstances. Most of them should be fixed now.

### Added

- A few docs and examples, mostly about grid usage.

## 2020-04-17

### Changed

- the main repo has now moved to to a different organization: https://github.com/toolsforexperiments/plottr 
  the old repo (https://github.com/data-plottr/plottr) will be kept in sync for a while, but deleted eventually.
