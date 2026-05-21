# test_postprocess.py
# Test the effect of different min_segment_dur values on the same prediction result

import subprocess
import pandas as pd
import tomlkit
from pathlib import Path
from post_hoc import run_posthoc

PREDICT_TOML = r"E:\petrel_project\config_predict.toml"
VAK_PYTHON = r"D:\Anaconda\envs\vak-env\python.exe"
GT_PATH = r"E:\petrel\01p2.txt"

# Use checkpoint from the best model b02_g05
CHECKPOINT = r"E:\petrel_project\train\results\results_260418_161633\TweetyNet\checkpoints\max-val-acc-checkpoint.pt"
LABELMAP = r"E:\petrel_project\train\results\results_260418_161633\labelmap.json"
STANDARDIZER = r"E:\petrel_project\train\results\results_260418_161633\FramesStandardizer"

durations = [0.05, 0.08, 0.10, 0.15]

results = []
for dur in durations:
    # Load and update config
    with open(PREDICT_TOML, "r", encoding="utf-8") as f:
        config = tomlkit.load(f)

    config["vak"]["predict"]["checkpoint_path"] = CHECKPOINT
    config["vak"]["predict"]["labelmap_path"] = LABELMAP
    config["vak"]["predict"]["frames_standardizer_path"] = STANDARDIZER
    config["vak"]["predict"]["min_segment_dur"] = dur
    config["vak"]["predict"]["annot_csv_filename"] = f"test_dur{str(dur).replace('.', '')}.annot.csv"

    with open(PREDICT_TOML, "w", encoding="utf-8") as f:
        tomlkit.dump(config, f)

    # Run prediction
    subprocess.run([VAK_PYTHON, "-m", "vak", "predict", PREDICT_TOML], check=True)

    # Generate Raven table and run post-hoc evaluation
    from generate_tables import run_generate_tables

    pred_csv = f"E:/petrel_project/results/predict/test_dur{str(dur).replace('.', '')}.annot.csv"
    pred_raven = run_generate_tables(pred_csv, r"E:\petrel_project\results\raven_tables_postproc",
                                     tag=f"dur{str(dur).replace('.', '')}")

    summary = run_posthoc(
        gt_path=GT_PATH,
        pred_path=pred_raven,
        audio_duration_sec=8040.0,
        save_gt_path=None,
        save_pred_path=None,
        perfect_iou_threshold=0.3,
    )
    summary["min_segment_dur"] = dur
    results.append(summary)
    print(f"dur={dur}: FDR={summary['FDR']:.4f}, Miss_rate={summary['Miss_rate']:.4f}")

# Compile and print final results
df = pd.DataFrame(results)
df["score"] = df["FDR"] + 0.3 * df["Miss_rate"]
print(df[["min_segment_dur", "FDR", "Miss_rate", "Recall", "score"]].to_string(index=False))