# =============================================================================
# config.py — Master Configuration for Petrel Detection Project
# =============================================================================
# Edit ONLY this file before running any scripts.
# All other scripts import their settings from here.
#
# Workflow:
#   Step 1:  python petrel_detection/prepare_data.py   (--mode train/test/predict)
#   Step 2:  python petrel_detection/run_experiment.py
#   Step 3:  python petrel_detection/batch_predict.py
# =============================================================================


# ── 1. Environment ────────────────────────────────────────────────────────────

# Path to the Python interpreter inside your vak conda environment
VAK_PYTHON = r"D:\Anaconda\envs\vak-env\python.exe"


# ── 2. Project Root ───────────────────────────────────────────────────────────

# Root directory of the project (all other paths are relative to this)
PROJECT_ROOT = r"E:\petrel_project"


# ── 3. Raw Input Data ─────────────────────────────────────────────────────────
# Directories containing your original (un-split) audio and annotation files.
# Audio: .wav files; Annotations: Raven Pro .txt files (tab-separated).

RAW_AUDIO_TRAIN   = r"E:\petrel_project\raw_audio\train"
RAW_AUDIO_TEST    = r"E:\petrel_project\raw_audio\test"
RAW_AUDIO_PREDICT = r"E:\petrel_project\raw_audio\predict"

RAW_ANNOT_TRAIN   = r"E:\petrel_project\raw_annot\train"   # Required for training
RAW_ANNOT_TEST    = r"E:\petrel_project\raw_annot\test"    # Required for test evaluation


# ── 4. Segmentation ───────────────────────────────────────────────────────────

# Duration of each audio segment in seconds
SEGMENT_DURATION = 30

# Root directory where split segments will be written
# Subdirectories (train_data/, test_data/, predict_data/) are created automatically
SEGMENT_ROOT = r"E:\petrel_project\segment_data"


# ── 5. vak Configuration Files ────────────────────────────────────────────────

TRAIN_TOML   = r"E:\petrel_project\config_train.toml"
PREDICT_TOML = r"E:\petrel_project\config_predict.toml"


# ── 6. vak Prep Output Directories ───────────────────────────────────────────

PREP_TRAIN_DIR   = r"E:\petrel_project\prep\train"
PREP_TEST_DIR    = r"E:\petrel_project\prep\test"
PREP_PREDICT_DIR = r"E:\petrel_project\prep\predict"


# ── 7. Training Parameters ────────────────────────────────────────────────────

# Duration splits for the training dataset (seconds)
TRAIN_DUR = 800
VAL_DUR   = 300
TEST_DUR  = 500

# Target vocalization label used in annotation CSV files
LABELSET = "p"

# Directory where each training run saves its results (results_XXXXXX subfolders)
TRAIN_RESULTS_ROOT = r"E:\petrel_project\train\results"


# ── 8. Experiment Hyperparameter Combinations (run_experiment.py) ─────────────
# Each dict specifies one (beta, gamma) combination to train and evaluate.
# beta  controls boundary loss weight
# gamma controls false-positive penalty weight
# beta=0, gamma=0 → standard cross-entropy only (baseline)

EXPERIMENTS = [
    {"beta": 0.1, "gamma": 0.2},
    {"beta": 0.1, "gamma": 0.3},
    {"beta": 0.1, "gamma": 0.5},
    {"beta": 0.2, "gamma": 0.2},
    {"beta": 0.2, "gamma": 0.3},
    {"beta": 0.2, "gamma": 0.5},
    {"beta": 0.2, "gamma": 0.0},   # Compare model (boundary loss only)
    {"beta": 0.0, "gamma": 0.5},   # FP penalty only
    {"beta": 0.3, "gamma": 0.5},
    {"beta": 0.2, "gamma": 0.6},
    {"beta": 0.1, "gamma": 0.6},
]

# Path to tweetynet.py in the vak environment (used to update beta/gamma before each run)
TWEETYNET_PY = r"D:\Anaconda\envs\vak-env\Lib\site-packages\vak\models\tweetynet.py"

# Ground truth annotation file used during run_experiment.py evaluation
# (the Raven .txt file for the audio that was used as the test set in training)
GT_PATH = r"E:\petrel_project\raw_annot\test\r02_250421081501p2.txt"

# Total duration of the test audio used in run_experiment.py (seconds)
AUDIO_DURATION_SEC = 8040.0

# Ranking weight: score = FDR + RANKING_W * Miss_rate  (lower = better)
RANKING_W = 0.2

# IoU threshold for a detection to count as a true positive in post-hoc evaluation
IOU_THRESHOLD = 0.3


# ── 9. Result Output Directories ─────────────────────────────────────────────

PREDICT_CSV_DIR    = r"E:\petrel_project\results\predict"
RAVEN_TABLE_DIR    = r"E:\petrel_project\results\raven_tables"
METRIC_TABLE_DIR   = r"E:\petrel_project\results\metric_tables"
STATS_CSV_DIR      = r"E:\petrel_project\results\stats_csv"
SUMMARY_CSV        = r"E:\petrel_project\experiment_summary.csv"

# For test-set evaluation (train_and_evaluate.py)
PREDICT_CSV_DIR_TEST  = r"E:\petrel_project\results\test"
RAVEN_TABLE_DIR_TEST  = r"E:\petrel_project\results\raven_tables_test"
METRIC_TABLE_DIR_TEST = r"E:\petrel_project\results\metric_tables_test"
EVAL_SUMMARY_CSV      = r"E:\petrel_project\evaluation_summary.csv"


# ── 10. Script Locations ──────────────────────────────────────────────────────

# Directory containing petrel_detection scripts (generate_tables, post_hoc, downstream, etc.)
PETREL_SCRIPTS_DIR = r"E:\petrel_project\petrel_detection"

# ── 11. Downstream Statistics ─────────────────────────────────────────────────

# Excel file mapping recorder filenames to recording start times
RECORDING_START_XLSX = r"E:\petrel_project\recording_start_times.xlsx"

# Suffix pattern of Raven table files to process (update to match current prediction tag)
DOWNSTREAM_SUFFIX = "_b02_g05_full_predict.txt"

# Input: Raven tables directory to process
DOWNSTREAM_INPUT_DIR = r"E:\petrel_project\results\raven_tables\night_1"

# Output: where per-recorder CSV and merged master file are saved
DOWNSTREAM_OUTPUT_DIR = r"E:\petrel_project\results\stats_csv\night_1_b02_g05"

# ── 12. Hardware ──────────────────────────────────────────────────────────────
ACCELERATOR = "gpu"   # Change to "cpu" if no GPU available
DEVICES     = [0]     # Change to 1 (integer) if using CPU
