import os
import pandas as pd

# ── Path Configuration ──────────────────────────────────────
PRED_CSV   = r"E:\petrel_project\results\metric_tables\pred_b02_g0.csv"
OUTPUT_DIR = r"E:\petrel\spectrogram_check"
# ────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_FP = os.path.join(OUTPUT_DIR, "fp_b02_g0.txt")
OUT_TP = os.path.join(OUTPUT_DIR, "tp_b02_g0.txt")

df = pd.read_csv(PRED_CSV)

def to_raven(subset):
    out = pd.DataFrame()
    out["Selection"]      = range(1, len(subset) + 1)
    out["View"]           = "Spectrogram 1"
    out["Channel"]        = 1
    out["Begin Time (s)"] = subset["pred_start"].values
    out["End Time (s)"]   = subset["pred_end"].values
    out["Low Freq (Hz)"]  = 2000
    out["High Freq (Hz)"] = 12000
    out["Annotation"]     = ""
    return out

fp_df = df[df["pred_label"] == "False Positive"]
tp_df = df[df["pred_label"].isin(["Perfect", "Offset"])]

to_raven(fp_df).to_csv(OUT_FP, sep="\t", index=False)
to_raven(tp_df).to_csv(OUT_TP, sep="\t", index=False)

print(f"FP: {len(fp_df)} selections → {OUT_FP}")
print(f"TP: {len(tp_df)} selections → {OUT_TP}")