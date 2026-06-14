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
pip install vak
pip install pydub tomlkit pandas
```

### 1.2 Install Additional Dependencies

`pydub` requires `ffmpeg` for audio processing. Install it via conda:

```bash
conda activate vak-env
conda install -c conda-forge ffmpeg
```

### 1.3 Replace with the Custom vak Library

After installation, **replace the default vak package** with the modified version from this repository to enable the custom loss function:

```
# Copy the vak/ folder from this repository and overwrite:
Anaconda\envs\vak-env\Lib\site-packages\vak\
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
│   ├── test/                          ← Original long .wav files for testing
│   └── predict/                       ← Original long .wav files for prediction
│
├── raw_annot/
│   ├── train/                         ← Raven Pro .txt annotation files for training
│   └── test/                          ← Raven Pro .txt annotation files for evaluation
│
│   ── The following directories are created automatically by the scripts ──
│
├── segment_data/
│   ├── train_data/                    ← 30-second segments + annotation CSV files
│   ├── test_data/<recorder>/          ← Segments per recorder (test)
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
    ├── predict/                        ← Raw .annot.csv prediction outputs
    ├── test/                           ← Test-set prediction outputs
    ├── raven_tables/                   ← Converted Raven annotation tables
    ├── raven_tables_test/
    ├── metric_tables/                  ← GT vs Pred comparison tables
    └── stats_csv/                      ← Time-of-arrival statistics
```

---

## 3. Quick Start

### Step 0 — Configure

Open `config.py` and update the paths to match your system. **This is the only file you need to edit.**

```python
# config.py — minimum required changes

VAK_PYTHON    = r"D:\Anaconda\envs\vak-env\python.exe"  # your Python path
PROJECT_ROOT  = r"E:\petrel_project"                     # your project root

# Directories containing your original audio and annotation files
RAW_AUDIO_TRAIN   = r"E:\petrel_project\raw_audio\train"
RAW_AUDIO_TEST    = r"E:\petrel_project\raw_audio\test"
RAW_AUDIO_PREDICT = r"E:\petrel_project\raw_audio\predict"
RAW_ANNOT_TRAIN   = r"E:\petrel_project\raw_annot\train"
RAW_ANNOT_TEST    = r"E:\petrel_project\raw_annot\test"

# Ground truth file and duration used during hyperparameter evaluation
GT_PATH            = r"E:\petrel_project\raw_annot\test\your_gt_file.txt"
AUDIO_DURATION_SEC = 8040.0
```

All output directories are created automatically. Experiment hyperparameter combinations can be adjusted in the `EXPERIMENTS` list in `config.py`.

---

### Step 1 — Prepare Data

Segment all audio files and run `vak prep` to build the datasets.

```bash
conda activate vak-env
cd E:\petrel_project

# Training data (annotation files required)
python petrel_detection/prepare_data.py --mode train

# Test data (annotation files used for evaluation)
python petrel_detection/prepare_data.py --mode test

# Prediction data (no annotation files required)
python petrel_detection/prepare_data.py --mode predict
```

Each command reads its input paths from `config.py`. To override a path for a single run without editing `config.py`:

```bash
python petrel_detection/prepare_data.py --mode train --audio_dir E:/other/audio --annot_dir E:/other/annot
```

**What this does:**
- Splits each `.wav` file into 30-second segments
- For training data, generates a `.csv` annotation file per segment from the corresponding Raven `.txt` file
- Updates `config_train.toml` or `config_predict.toml` with the correct paths
- Runs `vak prep` to build the dataset

---

### Step 2 — Train and Evaluate Hyperparameter Combinations

```bash
python petrel_detection/run_experiment.py
```

This script automatically sweeps all `(β, γ)` combinations defined in `config.py`, running the full pipeline for each:

1. Updates the loss function weights in `tweetynet.py`
2. Trains the model (`vak train`)
3. Runs prediction on the evaluation recording (`vak predict`)
4. Generates Raven annotation tables
5. Computes FDR, Miss Rate, and Recall against the ground truth
6. Appends results to `experiment_summary.csv`

Already-completed experiments are skipped automatically on re-run. A ranked summary is printed at the end:

```
Ranking (score = FDR + 0.2 × Miss_rate, lower = better):

tag        beta  gamma    FDR  Miss_rate  Recall  score
b02_g05    0.2   0.5    0.123      0.210   0.790  0.165
b02_g03    0.2   0.3    0.145      0.198   0.802  0.185
...
```

---

### Step 3 — Batch Prediction

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

After selection, the script iterates over every recorder in `segment_data/predict_data/`, updates `config_predict.toml`, and runs `vak predict`. No re-preparation of data is required. Raven annotation tables are generated automatically on completion.

To specify a model directly without the menu:

```bash
python petrel_detection/batch_predict.py --model results_260417_204919
```

To also generate downstream time-of-arrival statistics:

```bash
python petrel_detection/batch_predict.py --downstream
```

---

## 4. Configuration Reference

All pipeline settings are controlled by `config.py`. The table below lists every parameter.

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
| `EXPERIMENTS` | List of `{"beta": ..., "gamma": ...}` dicts to sweep in `run_experiment.py` |
| `TWEETYNET_PY` | Path to `tweetynet.py` in the vak environment (updated before each training run) |
| `GT_PATH` | Ground truth Raven `.txt` file used for evaluation in `run_experiment.py` |
| `AUDIO_DURATION_SEC` | Duration of the evaluation recording in seconds |
| `RANKING_W` | Weight `w` in `score = FDR + w × Miss_rate` used for experiment ranking |
| `IOU_THRESHOLD` | Minimum IoU for a detection to count as a true positive (default: 0.3) |
| `PREDICT_CSV_DIR` / `RAVEN_TABLE_DIR` / etc. | Output directories for prediction results and derived files |
| `PETREL_SCRIPTS_DIR` | Directory containing the `petrel_detection/` scripts |

---

## 5. Script Reference

### Main Pipeline Scripts

| Script | Description |
|--------|-------------|
| `prepare_data.py` | Segments audio, generates annotation CSVs, updates toml config, and runs `vak prep`. Supports `--mode train`, `test`, or `predict`. |
| `run_experiment.py` | Sweeps all `(β, γ)` combinations in `config.py`: trains, predicts, evaluates, and writes `experiment_summary.csv`. Skips already-completed runs. |
| `batch_predict.py` | Interactive model selection followed by batch prediction across all prepared predict recordings. Generates Raven tables on completion. |


### Utility Scripts

| Script | Description |
|--------|-------------|
| `generate_tables.py` | Converts `.annot.csv` prediction outputs to Raven Pro-compatible annotation tables. Batch supported. |
| `post_hoc.py` | Compares manual annotations with model predictions and computes FDR, Miss Rate, Precision, Recall, and TP/FP counts. |
| `downstream.py` | Converts Raven annotation timestamps to seconds elapsed since 18:00 for downstream temporal analysis. Batch supported. |
| `w_tune.py` | Reads `experiment_summary.csv` and ranks experiments using a configurable weighted score. |
| `test_postprocess.py` | Tests `min_segment_dur` thresholds (0.05 / 0.08 / 0.10 / 0.15 s) to find the best post-processing setting. |
| `extract_fp_tp.py` | Separates TP and FP detections from evaluation results and exports Raven-format tables for manual inspection in Raven Pro. |
| `analyze_audio_gaps.py` | Detects time gaps between recorder audio files and exports a report to `audio_gap_report.csv`. |

---

## 6. Results Analysis

### Reviewing Experiment Results

After `run_experiment.py` completes, open `experiment_summary.csv` for a full comparison across all hyperparameter combinations, or run:

```bash
python petrel_detection/w_tune.py
```

to re-rank with a different weighting between FDR and Miss Rate.

To find the optimal post-processing threshold:

```bash
python petrel_detection/test_postprocess.py
```

### Manual Inspection of Detections

```bash
python petrel_detection/extract_fp_tp.py
```

Produces `fp_<tag>.txt` and `tp_<tag>.txt` files importable into Raven Pro for visual inspection of spectrogram regions corresponding to false positives and true positives.

### Recording Continuity Check

```bash
python petrel_detection/analyze_audio_gaps.py
```

Outputs `audio_gap_report.csv` identifying any missing intervals between consecutive recorder files.
