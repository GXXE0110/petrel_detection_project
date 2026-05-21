# Petrel Detection — Grey‑faced Petrel Call Detection System

Automated detection and annotation of Grey‑faced Petrel (*Pterodroma gouldi*) calls using the [vak](https://github.com/vocalpy/vak) framework and TweetyNet neural network, with a custom loss function:

```
Total Loss = α · CrossEntropy + β · BoundaryLoss + γ · FalsePositiveLoss
```

By tuning β and γ, the model can be optimised to balance between Miss Rate and False Discovery Rate (FDR) depending on your research needs.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Repository Structure](#2-repository-structure)
3. [Script Reference](#3-script-reference)
4. [Training Workflow](#4-training-workflow)
5. [Prediction Workflow](#5-prediction-workflow)
6. [Batch Prediction for Multiple Recorders](#6-batch-prediction-for-multiple-recorders)
7. [Results Analysis](#7-results-analysis)

---

## 1. Environment Setup

### 1.1 Install vak

Follow the official installation guide:

> https://vak.readthedocs.io/en/latest/get_started/installation.html

We recommend creating a dedicated Anaconda environment:

```bash
conda create -n vak-env python=3.11
conda activate vak-env
pip install vak
```

### 1.2 Replace with the Custom vak Library

After installation, **replace the default vak package** in your Anaconda environment with the modified version from this repository to enable the custom loss function:

```
Anaconda\envs\vak-env\Lib\site-packages\vak
```

Simply copy the `vak/` folder from this repository and overwrite the directory above.

---

## 2. Repository Structure

```
├── vak/                        # Modified vak library with custom loss function
├── petrel_detection/           # Experiment and utility scripts
│   ├── split_data.py
│   ├── generate_tables.py
│   ├── post_hoc.py
│   ├── run_experiment.py
│   ├── w_tune.py
│   ├── test_postprocess.py
│   ├── extract_fp_tp.py
│   ├── edit_configure.py
│   ├── downstream.py
│   └── analyze_audio_gaps.py
├── config_train.toml           # Training configuration file
└── config_predict.toml         # Prediction configuration file
```

---

## 3. Script Reference

| Script | Description |
|--------|-------------|
| `split_data.py` | Splits long audio files into 30-second segments and generates corresponding annotation CSVs |
| `generate_tables.py` | Converts model output `.annot.csv` files back into Raven Pro annotation tables (batch supported) |
| `post_hoc.py` | Compares manual annotations against model predictions; outputs FDR, Miss Rate, Precision, and Recall |
| `run_experiment.py` | Runs the full experiment pipeline: train → predict → evaluate → summarise results to `experiment_summary.csv` |
| `w_tune.py` | Reads `experiment_summary.csv` and ranks experiments by a weighted score to identify the optimal hyperparameter combination |
| `test_postprocess.py` | Tests different `min_segment_dur` thresholds (0.05 / 0.08 / 0.10 / 0.15) to find the best post-processing filter |
| `extract_fp_tp.py` | Separates TP and FP detections from evaluation results and exports them as Raven-format files for manual review |
| `edit_configure.py` | Automatically updates data paths in `config_predict.toml` to reduce manual errors during batch prediction |
| `downstream.py` | Converts Raven annotation timestamps into seconds elapsed since 06:00, for downstream temporal analysis (batch supported) |
| `analyze_audio_gaps.py` | Checks for gaps or missing recordings between recorder files; outputs `audio_gap_report.csv` |

---

## 4. Training Workflow

Open Anaconda Prompt, activate the environment, and navigate to your project directory:

```bash
conda activate vak-env
cd E:/petrel_project
```

### Step 1: Prepare the Training Dataset

Run `split_data.py` to segment long audio files into 30-second clips. Raven annotation `.txt` files must be provided alongside the audio.

```bash
python petrel_detection/split_data.py
```

**Input:** Long `.wav` audio + Raven annotation `.txt`  
**Output:** Segment files such as `seg001.wav` and `seg001.wav.csv`

### Step 2: Edit the Training Configuration File

Open `config_train.toml` and update the paths and parameters:

```toml
data_dir = "E:/petrel_project/train_data"    # Directory containing segmented audio and annotations
output_dir = "E:/petrel_project/prep/train"  # Output directory for prepared training data
labelset = "p"                               # Label of the target vocalisation
train_dur = 1000                             # Training set duration (seconds)
val_dur = 300                                # Validation set duration (seconds)
test_dur = 500                               # Test set duration (seconds)
```

> **Note:** If a `path = "..."` field already exists under `[vak.train.dataset]`, delete it before proceeding.

### Step 3: Prepare the Training Dataset

```bash
vak prep config_train.toml
```

### Step 4: Train the Model

Run the experiment script to automatically sweep β / γ hyperparameter combinations and collect evaluation results:

```bash
python petrel_detection/run_experiment.py
```

This script handles the full pipeline automatically: updates loss function hyperparameters → trains model → runs prediction on the test set → calls `post_hoc.py` to compute metrics → writes all results to `experiment_summary.csv`.

### Step 5 (Optional): Select the Best Model

```bash
# Rank experiments by weighted score
python petrel_detection/w_tune.py

# Find the optimal post-processing threshold
python petrel_detection/test_postprocess.py
```

---

## 5. Prediction Workflow

### Step 1: Prepare the Prediction Dataset

Run `split_data.py` on the audio files you want to predict (no annotation files required):

```bash
python petrel_detection/split_data.py
```

Each audio file will produce its own sub-folder of segments.

### Step 2: Edit the Prediction Configuration File (Data Paths)

Open `config_predict.toml` and set:

```toml
data_dir = "E:/petrel_project/predict_data"    # Directory of segmented prediction audio
output_dir = "E:/petrel_project/prep/predict"  # Output directory for prepared prediction data
```

> **Note:** If a `path = "..."` field already exists under `[vak.predict.dataset]`, delete it before proceeding.

### Step 3: Prepare the Prediction Dataset

```bash
vak prep config_predict.toml
```

### Step 4: Edit the Prediction Configuration File (Model Paths)

Fill in the paths from your chosen training result folder (found under `train/results/results_XXXXXX/`):

```toml
checkpoint_path = "E:/petrel_project/train/results/results_260325_191545/TweetyNet/checkpoints/max-val-acc-checkpoint.pt"
labelmap_path = "E:/petrel_project/train/results/results_260325_191545/labelmap.json"
frames_standardizer_path = "E:/petrel_project/train/results/results_260325_191545/FramesStandardizer"
output_dir = "E:/petrel_project/results/predict"
annot_csv_filename = "results.annot.csv"
```

### Step 5: Run Prediction

```bash
vak predict config_predict.toml
```

### Step 6: Generate Raven Annotation Tables

```bash
python petrel_detection/generate_tables.py
```

Converts `results.annot.csv` into Raven Pro-compatible annotation tables (batch supported).

### Step 7: Generate Downstream Statistics Fields

```bash
python petrel_detection/downstream.py
```

Converts annotation timestamps to seconds elapsed since 06:00, merges and sorts all outputs across recorders (batch supported).

---

## 6. Batch Prediction for Multiple Recorders

When processing overnight recordings from multiple recorders, Steps 3 (prep) and 5 (predict) must be repeated for each audio file, updating `config_predict.toml` each time.

Use `edit_configure.py` to automate configuration updates and avoid manual errors:

**1.** Open `edit_configure.py` and set the recorder name:

```python
new_name = "r10_250505102732"   # Replace with the current recorder's folder name
```

**2.** Run the script to automatically update the config file:

```bash
python petrel_detection/edit_configure.py
```

**3.** Run prep and predict in Anaconda Prompt:

```bash
vak prep config_predict.toml
vak predict config_predict.toml
```

**4.** Repeat steps 1–3 for each recorder.

**5.** Once all recorders are processed, run:

```bash
python petrel_detection/generate_tables.py
python petrel_detection/downstream.py
```

---

## 7. Results Analysis

### Model Evaluation

```bash
python petrel_detection/post_hoc.py
```

Outputs FDR, Miss Rate, Precision, Recall, and TP / FP classification for each prediction.

### Manual Review of FP / TP Detections

```bash
python petrel_detection/extract_fp_tp.py
```

Produces `fp_b02_g0.txt` and `tp_b02_g0.txt` — importable into Raven Pro for manual inspection of what the model got right and wrong.

### Recording Continuity Check

```bash
python petrel_detection/analyze_audio_gaps.py
```

Outputs `audio_gap_report.csv` to identify any gaps or missing recordings between recorder files.
