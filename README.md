# Petrel Detection — Grey-faced Petrel Call Detection System

Automated detection and annotation of Grey-faced Petrel (*Pterodroma gouldi*) calls using the [vak](https://github.com/vocalpy/vak) framework and TweetyNet neural network, with a custom loss function:

```
Total Loss = α · CrossEntropy + β · BoundaryLoss + γ · FalsePositiveLoss
```

By tuning β and γ, the model can be optimised to balance between Miss Rate and False Discovery Rate (FDR) depending on your research needs.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start](#3-quick-start)
4. [Configuration Reference](#4-configuration-reference)
5. [Script Reference](#5-script-reference)
6. [Results Analysis](#6-results-analysis)

---

## 1. Environment Setup

### 1.1 Create the Conda Environment

Follow the official vak installation guide: https://vak.readthedocs.io/en/latest/get_started/installation.html

```bash
# Option A — reproduce the exact environment used in this project
conda env create -f environment.yml

# Option B — create a fresh environment and install vak manually
conda create -n vak-env python=3.11
conda activate vak-env
pip install vak pydub tomlkit pandas openpyxl
```

### 1.2 Install ffmpeg

`pydub` requires `ffmpeg` for audio segmentation:

```bash
conda activate vak-env
conda install -c conda-forge ffmpeg
```

### 1.3 Replace with the Custom vak Library

After installation, **replace the default vak package** with the modified version from this repository to enable the custom loss function.

**Windows:**
```
Anaconda\envs\vak-env\Lib\site-packages\vak\
```

**Linux / macOS:**
```
~/miniconda3/envs/vak-env/lib/python3.11/site-packages/vak/
```

Copy the `vak/` folder from this repository and overwrite the directory above.

### 1.4 Platform Notes

**Linux or macOS / no GPU:** In `config.py`, set:

```python
ACCELERATOR = "cpu"
DEVICES     = 1
```

---

## 2. Repository Structure

```
petrel_project/
│
├── config.py                          ← ★ Master config — edit only this file
├── config_train.toml                  ← vak training configuration
├── config_predict.toml                ← vak prediction configuration
├── environment.yml                    ← Conda environment specification
│
├── petrel_detection/                  ← All scripts
│   ├── prepare_data.py                ← Step 1: Segment audio and run vak prep
│   ├── run_experiment.py              ← Step 2: Train all hyperparameter combinations
│   ├── batch_predict.py               ← Step 3: Select model and run batch prediction
│   ├── generate_tables.py             ← Convert predictions to Raven annotation tables
│   ├── post_hoc.py                    ← Compute FDR, Miss Rate, Precision, Recall
│   ├── downstream.py                  ← Convert timestamps for downstream analysis
│   ├── w_tune.py                      ← Rank experiments by weighted score
│   ├── test_postprocess.py            ← Test min_segment_dur thresholds
│   ├── extract_fp_tp.py               ← Export TP/FP detections for manual review
│   └── analyze_audio_gaps.py          ← Check for gaps between recorder files
│
├── vak/                               ← Modified vak source code
│   └── ...                            (copy to site-packages/vak after installation)
│
│   ── Provide these directories with your own data ──
│
├── raw_audio/
│   ├── train/                         ← Original long .wav files for training
│   ├── test/                          ← Original long .wav file(s) for evaluation
│   └── predict/                       ← Original long .wav files for prediction
│
├── raw_annot/
│   ├── train/                         ← Raven Pro .txt annotation files for training
│   └── test/                          ← Raven Pro .txt annotation file(s) for evaluation
│
│   recording_start_times.xlsx         ← Recorder start times for downstream analysis
│
│   ── The following directories are created automatically by the scripts ──
│
├── segment_data/
│   ├── train_data/                    ← 30-second segments + annotation CSV files
│   ├── test_data/<recorder>/          ← Segments for the test recording
│   └── predict_data/<recorder>/       ← Segments per recorder (predict)
│
├── prep/                              ← vak-prepared datasets
│   ├── train/
│   ├── test/
│   └── predict/
│
├── train/
│   └── results/
│       └── results_XXXXXX/            ← One folder per training run
│           ├── labelmap.json
│           ├── FramesStandardizer/
│           └── TweetyNet/checkpoints/
│               └── max-val-acc-checkpoint.pt
│
└── results/
    ├── test/                           ← Per-experiment .annot.csv outputs (run_experiment.py)
    ├── predict/                        ← Full-night .annot.csv outputs (batch_predict.py)
    ├── raven_tables/                   ← Converted Raven annotation tables
    ├── metric_tables/                  ← GT vs Pred comparison tables
    └── stats_csv/                      ← Time-of-arrival statistics (downstream.py)
```

---

## 3. Quick Start

### Step 0 — Configure

Open `config.py` and update the settings to match your system. **This is the only file you need to edit.**

The minimum required changes are:

```python
# Interpreter path
VAK_PYTHON = r"D:\Anaconda\envs\vak-env\python.exe"

# Project root
PROJECT_ROOT = r"E:\petrel_project"

# Raw input data directories
RAW_AUDIO_TRAIN   = r"E:\petrel_project\raw_audio\train"
RAW_AUDIO_TEST    = r"E:\petrel_project\raw_audio\test"
RAW_AUDIO_PREDICT = r"E:\petrel_project\raw_audio\predict"
RAW_ANNOT_TRAIN   = r"E:\petrel_project\raw_annot\train"
RAW_ANNOT_TEST    = r"E:\petrel_project\raw_annot\test"

# Name of the test recording (filename stem, no extension)
# Used to locate the prep/test/ dataset and name output files
TEST_RECORDING_NAME = "r02_250421081501p2"

# Ground truth annotation and duration for hyperparameter evaluation
GT_PATH            = r"E:\petrel_project\raw_annot\test\r02_250421081501p2.txt"
AUDIO_DURATION_SEC = 8040.0

# Path to tweetynet.py in the vak environment
TWEETYNET_PY = r"D:\Anaconda\envs\vak-env\Lib\site-packages\vak\models\tweetynet.py"
```

All output directories are created automatically. The `EXPERIMENTS` list in `config.py` controls which `(β, γ)` combinations are trained.

---

### Step 1 — Prepare Data

Segment audio files and run `vak prep` to build the datasets.

```bash
conda activate vak-env
cd E:\petrel_project

# Training data — annotation files required
python petrel_detection/prepare_data.py --mode train

# Test data — one recording used for evaluating hyperparameter combinations
python petrel_detection/prepare_data.py --mode test

# Prediction data — no annotation files required
python petrel_detection/prepare_data.py --mode predict
```

Input paths are read from `config.py`. Each mode can be overridden on the command line without editing `config.py`:

```bash
python petrel_detection/prepare_data.py --mode train --audio_dir E:/other/audio --annot_dir E:/other/annot
```

**What this does for each mode:**

| Mode | Segments audio | Generates annotation CSVs | Runs `vak prep` using |
|------|:-:|:-:|---|
| `train` | ✓ | ✓ (from Raven `.txt`) | `config_train.toml` |
| `test` | ✓ | — | `config_predict.toml` → `prep/test/` |
| `predict` | ✓ | — | `config_predict.toml` → `prep/predict/` |

---

### Step 2 — Train and Evaluate Hyperparameter Combinations

```bash
python petrel_detection/run_experiment.py
```

This script sweeps all `(β, γ)` combinations defined in `config.py`. For each combination it:

1. Updates the loss function weights in `tweetynet.py`
2. Trains the model (`vak train`)
3. Runs prediction on the test recording (`vak predict`) using the prep dataset in `prep/test/`
4. Generates Raven annotation tables
5. Evaluates against the ground truth (`GT_PATH`) and computes FDR, Miss Rate, and Recall
6. Appends results to `experiment_summary.csv`

Already-completed combinations are skipped automatically on re-run. A ranked summary is printed at the end:

```
Ranking (score = FDR + 0.2 × Miss_rate, lower = better):

tag        beta  gamma    FDR  Miss_rate  Recall  score
b02_g05    0.2   0.5    0.123      0.210   0.790  0.165
b02_g03    0.2   0.3    0.145      0.198   0.802  0.185
...

✅ Best combination: b02_g05
```

---

### Step 3 — Batch Prediction

Once the best model has been identified, run batch prediction on all overnight recordings:

```bash
python petrel_detection/batch_predict.py
```

An interactive menu lists all available trained models, sorted newest first:

```
Available trained models (newest first):

  [0] results_260417_204919  (2026-04-17 20:49)  [latest]
  [1] results_260415_162603  (2026-04-15 16:26)
  [2] results_260412_091234  (2026-04-12 09:12)

Enter model number (or press Enter for latest):
```

After selection, the script iterates over every recorder in `segment_data/predict_data/`, updates `config_predict.toml` with the chosen model and the corresponding prep dataset, and runs `vak predict` for each. No re-preparation of data is required. Raven annotation tables are generated automatically.

To specify a model directly without the menu:

```bash
python petrel_detection/batch_predict.py --model results_260417_204919
```

To also generate downstream time-of-arrival statistics after prediction:

```bash
python petrel_detection/batch_predict.py --downstream
```

---

## 4. Configuration Reference

All pipeline settings are controlled by `config.py`.

| Parameter | Description |
|-----------|-------------|
| `VAK_PYTHON` | Path to the Python interpreter in the `vak-env` conda environment |
| `PROJECT_ROOT` | Root directory of the project |
| `RAW_AUDIO_TRAIN/TEST/PREDICT` | Directories containing original (unsegmented) `.wav` files |
| `RAW_ANNOT_TRAIN/TEST` | Directories containing Raven Pro `.txt` annotation files |
| `SEGMENT_DURATION` | Segment length in seconds (default: 30) |
| `SEGMENT_ROOT` | Root directory where segmented audio is written |
| `TRAIN_TOML` / `PREDICT_TOML` | Paths to the vak configuration files |
| `PREP_TRAIN/TEST/PREDICT_DIR` | Output directories for `vak prep` datasets |
| `TRAIN_DUR` / `VAL_DUR` / `TEST_DUR` | Duration splits for the training dataset (seconds) |
| `LABELSET` | Annotation label for the target vocalisation (default: `"p"`) |
| `TRAIN_RESULTS_ROOT` | Directory where training results (`results_XXXXXX/`) are saved |
| `TEST_RECORDING_NAME` | Filename stem of the test recording (e.g. `"r02_250421081501p2"`). Used by `run_experiment.py` to locate the prep dataset and name output files. |
| `EXPERIMENTS` | List of `{"beta": ..., "gamma": ...}` dicts defining the hyperparameter sweep |
| `TWEETYNET_PY` | Path to `tweetynet.py` in the vak environment (updated before each training run) |
| `GT_PATH` | Raven `.txt` ground truth file used for evaluation in `run_experiment.py` |
| `AUDIO_DURATION_SEC` | Duration of the test recording in seconds |
| `RANKING_W` | Weight `w` in `score = FDR + w × Miss_rate` used for experiment ranking |
| `IOU_THRESHOLD` | Minimum IoU for a detection to count as a true positive (default: 0.3) |
| `EXPERIMENT_CSV_DIR` | Output directory for per-experiment `.annot.csv` files (`run_experiment.py`) |
| `PREDICT_CSV_DIR` | Output directory for full-night `.annot.csv` files (`batch_predict.py`) |
| `RAVEN_TABLE_DIR` | Output directory for converted Raven annotation tables |
| `METRIC_TABLE_DIR` | Output directory for GT vs Pred comparison tables |
| `SUMMARY_CSV` | Path to the experiment summary CSV |
| `RECORDING_START_XLSX` | Excel file mapping recorder filenames to recording start times (downstream analysis) |
| `DOWNSTREAM_SUFFIX` | Suffix of Raven table files to process in `downstream.py` |
| `DOWNSTREAM_INPUT_DIR` / `DOWNSTREAM_OUTPUT_DIR` | Input/output directories for `downstream.py` |
| `ACCELERATOR` | Hardware accelerator: `"gpu"` or `"cpu"` |
| `DEVICES` | Device index: `[0]` for first GPU, `1` (integer) for CPU |
| `PETREL_SCRIPTS_DIR` | Directory containing the `petrel_detection/` scripts |

---

## 5. Script Reference

### Main Pipeline Scripts

| Script | Description |
|--------|-------------|
| `prepare_data.py` | Segments audio into 30-second clips, generates annotation CSVs for training data, updates the relevant toml config, and runs `vak prep`. Supports `--mode train`, `test`, or `predict`. |
| `run_experiment.py` | Sweeps all `(β, γ)` combinations: trains the model, predicts on the test recording, evaluates against ground truth, and writes `experiment_summary.csv`. Skips already-completed runs on re-run. |
| `batch_predict.py` | Interactive model selection followed by batch prediction across all recorders in `segment_data/predict_data/`. Generates Raven annotation tables on completion. Optionally runs downstream statistics with `--downstream`. |

### Utility Scripts

| Script | Description |
|--------|-------------|
| `generate_tables.py` | Converts `.annot.csv` prediction outputs to Raven Pro-compatible annotation tables. Batch supported. |
| `post_hoc.py` | Compares manual annotations with model predictions and computes FDR, Miss Rate, Precision, Recall, and TP/FP counts. |
| `downstream.py` | Converts Raven annotation timestamps to seconds elapsed since 06:00 for downstream temporal analysis. Can be run standalone or called from `batch_predict.py`. |
| `w_tune.py` | Reads `experiment_summary.csv` and re-ranks experiments using a configurable weighted score. |
| `test_postprocess.py` | Tests `min_segment_dur` thresholds (0.05 / 0.08 / 0.10 / 0.15 s) to find the best post-processing setting. |
| `extract_fp_tp.py` | Separates TP and FP detections from evaluation results and exports Raven-format tables for manual inspection in Raven Pro. |
| `analyze_audio_gaps.py` | Detects time gaps between recorder audio files and exports a report to `audio_gap_report.csv`. |

---

## 6. Results Analysis

### Reviewing Experiment Results

After `run_experiment.py` completes, `experiment_summary.csv` contains metrics for all hyperparameter combinations. To re-rank with a different weighting:

```bash
python petrel_detection/w_tune.py
```

To find the optimal post-processing threshold:

```bash
python petrel_detection/test_postprocess.py
```

### Manual Inspection of Detections

```bash
python petrel_detection/extract_fp_tp.py
```

Produces `fp_<tag>.txt` and `tp_<tag>.txt` files importable into Raven Pro for visual inspection of false positive and true positive detections.

### Recording Continuity Check

```bash
python petrel_detection/analyze_audio_gaps.py
```

Outputs `audio_gap_report.csv` identifying any missing intervals between consecutive recorder files.
