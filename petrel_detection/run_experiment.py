"""
run_experiment.py
Run 11 groups of hyperparameter experiments with one click:
Train → Update predict toml → Predict → Generate annotation tables → Post-hoc evaluation → Summary CSV
"""

import os
import sys
import subprocess
import pandas as pd
import tomlkit
from pathlib import Path
import shutil

# ── Path Config (Only change here) ───────────────────────────────────────
VAK_PYTHON       = r"D:\Anaconda\envs\vak-env\python.exe"
TRAIN_TOML       = r"E:\petrel_project\config_train.toml"
PREDICT_TOML     = r"E:\petrel_project\config_predict.toml"

# Training results root (each training creates results_XXXXXX subdir)
TRAIN_RESULTS_ROOT = r"E:\petrel_project\train\results"

# Fixed ground truth path
GT_PATH            = r"E:\petrel\01p2.txt"
AUDIO_DURATION_SEC = 8040.0

# Prediction outputs
PREDICT_CSV_DIR  = r"E:\petrel_project\results\predict"
RAVEN_TABLE_DIR  = r"E:\petrel_project\results\raven_tables"
METRIC_TABLE_DIR = r"E:\petrel_project\results\metric_tables"
SUMMARY_CSV      = r"E:\petrel_project\experiment_summary.csv"

SEGMENT_DURATION = 30   # Segment duration in seconds

# ── Experiment Combinations ───────────────────────────────────────────────
EXPERIMENTS = [
    {"beta": 0.1, "gamma": 0.2},
    {"beta": 0.1, "gamma": 0.3},
    {"beta": 0.1, "gamma": 0.5},
    {"beta": 0.2, "gamma": 0.2},
    {"beta": 0.2, "gamma": 0.3},
    {"beta": 0.2, "gamma": 0.5},
    {"beta": 0.2, "gamma": 0},
    {"beta": 0, "gamma": 0.5},
    {"beta": 0.3, "gamma": 0.5},
    {"beta": 0.2, "gamma": 0.6},
]

# ── Import Custom Scripts ─────────────────────────────────────────────────
sys.path.insert(0, r"E:\py.ex\return_raven\toraven")
from generate_tables import run_generate_tables
from post_hoc import run_posthoc


# ─────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────

def make_tag(beta: float, gamma: float, delta: float = 0.0) -> str:
    """Generate experiment tag from hyperparameters"""
    b = str(beta).replace('.', '')
    g = str(gamma).replace('.', '')
    if delta == 0.0:
        return f"b{b}_g{g}"
    d = str(delta).replace('.', '')
    return f"b{b}_g{g}_d{d}"


def update_loss_params(beta: float, gamma: float):
    """Update loss parameters in tweetynet.py default_config"""
    path = Path(r"D:\Anaconda\envs\vak-env\Lib\site-packages\vak\models\tweetynet.py")
    content = path.read_text(encoding="utf-8")

    import re
    content = re.sub(
        r'("beta":\s*)[\d.]+',
        rf'\g<1>{beta}',
        content
    )
    content = re.sub(
        r'("gamma":\s*)[\d.]+',
        rf'\g<1>{gamma}',
        content
    )

    path.write_text(content, encoding="utf-8")
    print(f"  ✓ Updated tweetynet.py: beta={beta}, gamma={gamma}")

    content_check = path.read_text(encoding="utf-8")
    if f'"beta": {beta}' in content_check and f'"gamma": {gamma}' in content_check:
        print(f"  ✓ Verification passed")
    else:
        raise RuntimeError("Failed to update tweetynet.py, please check manually")


def find_latest_results_dir() -> Path:
    """Find the latest results_XXXXXX directory"""
    root = Path(TRAIN_RESULTS_ROOT)
    candidates = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("results_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No results_* found in {TRAIN_RESULTS_ROOT}")
    return candidates[-1]


def find_checkpoint(results_dir: Path) -> str:
    """Find checkpoint file in training results"""
    ckpt_dir = results_dir / "TweetyNet" / "checkpoints"
    for name in ["max-val-acc-checkpoint.pt"]:
        ckpt = ckpt_dir / name
        if ckpt.exists():
            return str(ckpt)
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def update_predict_toml(checkpoint_path: str, labelmap_path: str, standardizer_path: str, tag: str):
    """Auto-update paths in predict.toml after training"""
    with open(PREDICT_TOML, "r", encoding="utf-8") as f:
        config = tomlkit.load(f)

    config["vak"]["predict"]["checkpoint_path"]          = checkpoint_path
    config["vak"]["predict"]["labelmap_path"]            = labelmap_path
    config["vak"]["predict"]["frames_standardizer_path"] = standardizer_path
    config["vak"]["predict"]["annot_csv_filename"] = f"01p2_{tag}.annot.csv"

    with open(PREDICT_TOML, "w", encoding="utf-8") as f:
        tomlkit.dump(config, f)

    print(f"  ✓ Updated predict toml")
    print(f"    checkpoint:   {checkpoint_path}")
    print(f"    labelmap:     {labelmap_path}")
    print(f"    standardizer: {standardizer_path}")
    print(f"    annot_csv_filename: 01p2_{tag}.annot.csv")


def find_predict_csv() -> str:
    """Find the latest .annot.csv prediction file"""
    csvs = list(Path(PREDICT_CSV_DIR).glob("*.annot.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .annot.csv found in {PREDICT_CSV_DIR}")
    return str(sorted(csvs, key=lambda p: p.stat().st_mtime)[-1])


# ─────────────────────────────────────────────────────────────────────────
# Single Experiment Pipeline
# ─────────────────────────────────────────────────────────────────────────

def run_one_experiment(beta: float, gamma: float) -> dict:
    tag = make_tag(beta, gamma)

    # Skip if already completed
    if os.path.exists(SUMMARY_CSV):
        existing = pd.read_csv(SUMMARY_CSV)
        if tag in existing["tag"].values:
            print(f"\n⏭ Skipping [{tag}], results already exist")
            return existing[existing["tag"] == tag].iloc[0].to_dict()

    print(f"\n{'=' * 60}")
    print(f"Experiment: beta={beta}, gamma={gamma}  [{tag}]")
    print(f"{'=' * 60}")

    # Step 1: Update loss hyperparameters
    print("\n[1/5] Updating loss hyperparameters...")
    update_loss_params(beta=beta, gamma=gamma)

    # Step 2: Training
    print("\n[2/5] Starting training...")
    subprocess.run(
        [VAK_PYTHON, "-m", "vak", "train", TRAIN_TOML],
        check=True,
    )

    # Step 3: Update predict config
    print("\n[3/5] Updating predict toml paths...")
    results_dir       = find_latest_results_dir()
    checkpoint_path   = find_checkpoint(results_dir)
    labelmap_path     = str(results_dir / "labelmap.json")
    standardizer_path = str(results_dir / "FramesStandardizer")

    update_predict_toml(
        checkpoint_path=checkpoint_path,
        labelmap_path=labelmap_path,
        standardizer_path=standardizer_path,
        tag=tag,
    )

    if os.path.exists(PREDICT_CSV_DIR):
        shutil.rmtree(PREDICT_CSV_DIR)
    os.makedirs(PREDICT_CSV_DIR)
    print("  ✓ Cleared prediction output directory")

    # Step 4: Prediction
    print("\n[4/5] Starting prediction...")
    subprocess.run(
        [VAK_PYTHON, "-m", "vak", "predict", PREDICT_TOML],
        check=True,
    )

    # Step 5: Generate Raven tables
    print("\n[5/5] Generating Raven annotation tables...")
    predict_csv = find_predict_csv()
    print(f"  Using prediction file: {predict_csv}")

    pred_raven_path = run_generate_tables(
        predict_csv_path=predict_csv,
        output_dir=RAVEN_TABLE_DIR,
        segment_duration=SEGMENT_DURATION,
        tag=tag,
    )
    if pred_raven_path is None:
        raise RuntimeError("Failed to generate tables, check prediction output")

    # Step 6: Post-hoc evaluation
    print("\n[6/6] Running post-hoc evaluation...")
    os.makedirs(METRIC_TABLE_DIR, exist_ok=True)

    summary = run_posthoc(
        gt_path=GT_PATH,
        pred_path=pred_raven_path,
        audio_duration_sec=AUDIO_DURATION_SEC,
        save_gt_path=os.path.join(METRIC_TABLE_DIR, f"gt_{tag}.csv"),
        save_pred_path=os.path.join(METRIC_TABLE_DIR, f"pred_{tag}.csv"),
        perfect_iou_threshold=0.3,
    )

    # Step 7: Append to summary
    summary["tag"]   = tag
    summary["beta"]  = beta
    summary["gamma"] = gamma

    row_df = pd.DataFrame([summary])
    if os.path.exists(SUMMARY_CSV):
        old_df = pd.read_csv(SUMMARY_CSV)
        merged = pd.concat([old_df, row_df], ignore_index=True)
        merged.to_csv(SUMMARY_CSV, index=False)
    else:
        row_df.to_csv(SUMMARY_CSV, index=False)

    print(f"\n  ✓ Results appended to summary: {SUMMARY_CSV}")
    return summary


# ─────────────────────────────────────────────────────────────────────────
# Main Function
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("Starting batch experiments...")
    print(f"{len(EXPERIMENTS)} hyperparameter combinations in total\n")

    failed = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\nProgress: {i}/{len(EXPERIMENTS)}")
        try:
            run_one_experiment(**exp)
        except Exception as e:
            tag = make_tag(exp['beta'], exp['gamma'])
            print(f"\n❌ Experiment [{tag}] failed: {e}")
            failed.append(tag)
            continue

    # Print final ranking
    print(f"\n{'=' * 60}")
    print("All experiments completed")
    print(f"{'=' * 60}")

    if failed:
        print(f"⚠ Failed experiments (excluded from ranking): {failed}")

    if not os.path.exists(SUMMARY_CSV):
        print("No successful experiment results.")
        return

    summary_df = pd.read_csv(SUMMARY_CSV)

    # Ranking metric: FDR + 0.2 * Miss_rate
    summary_df["score"] = summary_df["FDR"] + 0.2 * summary_df["Miss_rate"]
    summary_df = summary_df.sort_values("score").reset_index(drop=True)

    print("\nRanking (score = FDR + 0.3 × Miss_rate, lower = better):\n")
    print(
        summary_df[[
            "tag", "beta", "gamma",
            "FDR", "Miss_rate", "Recall", "score"
        ]].to_string(index=False)
    )

    best = summary_df.iloc[0]
    print(f"\n✅ Best combination: {best['tag']}")
    print(f"   beta={best['beta']}, gamma={best['gamma']}")
    print(f"   FDR={best['FDR']:.4f}, Miss_rate={best['Miss_rate']:.4f}, Recall={best['Recall']:.4f}")


if __name__ == "__main__":
    main()