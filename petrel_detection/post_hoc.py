"""
post_hoc.py
Compare predicted annotation tables with manual ground truth tables,
output metrics including FDR, miss_rate, precision, recall, etc.
"""

import numpy as np
import pandas as pd


def load_raven_table(
    path,
    start_col="Begin Time (s)",
    end_col="End Time (s)"
):
    """Load Raven format annotation table and extract start/end times"""
    df = pd.read_csv(path, sep="\t")

    if start_col not in df.columns or end_col not in df.columns:
        raise ValueError(
            f"{path} missing columns: '{start_col}' or '{end_col}'. "
            f"Actual columns: {list(df.columns)}"
        )

    df = df[[start_col, end_col]].copy()
    df = df.rename(columns={start_col: "start", end_col: "end"})
    df = df.dropna(subset=["start", "end"]).reset_index(drop=True)
    df = df[df["end"] > df["start"]].reset_index(drop=True)

    return df


def apply_buffer(df, buffer_sec=0.02):
    """Apply time buffer to ground truth intervals for flexible matching"""
    out = df.copy()
    out["start_buf"] = out["start"] - buffer_sec
    out["end_buf"]   = out["end"]   + buffer_sec
    return out


def compute_overlap(a_start, a_end, b_start, b_end):
    """Compute overlapping duration between two intervals"""
    return min(a_end, b_end) - max(a_start, b_start)


def compute_iou(a_start, a_end, b_start, b_end):
    """Compute Intersection over Union (IoU) between two intervals"""
    overlap = compute_overlap(a_start, a_end, b_start, b_end)
    if overlap <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return overlap / union


def build_pair_table(gt_df, pred_df, overlap_threshold=0.1):
    """Build all overlapping pairs between ground truth and predictions"""
    pairs = []
    for gi, g in gt_df.iterrows():
        for pi, p in pred_df.iterrows():
            overlap_buf = compute_overlap(
                g["start_buf"], g["end_buf"],
                p["start"],     p["end"]
            )
            if overlap_buf <= 0:
                continue
            iou = compute_iou(
                g["start"], g["end"],
                p["start"], p["end"]
            )
            if iou >= overlap_threshold:
                pairs.append({"gt_idx": gi, "pred_idx": pi, "iou": iou})

    pair_df = pd.DataFrame(pairs)
    if len(pair_df) == 0:
        pair_df = pd.DataFrame(columns=["gt_idx", "pred_idx", "iou"])
    return pair_df


def greedy_one_to_one_matching(pair_df):
    """Greedy one-to-one matching based on highest IoU (no duplicate matches)"""
    if pair_df.empty:
        return pd.DataFrame(columns=["gt_idx", "pred_idx", "iou"])

    pair_df = pair_df.sort_values(
        by=["iou", "gt_idx", "pred_idx"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    used_gt   = set()
    used_pred = set()
    matches   = []

    for _, row in pair_df.iterrows():
        gi  = int(row["gt_idx"])
        pi  = int(row["pred_idx"])
        iou = float(row["iou"])
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append({"gt_idx": gi, "pred_idx": pi, "iou": iou})

    match_df = pd.DataFrame(matches)
    if len(match_df) == 0:
        match_df = pd.DataFrame(columns=["gt_idx", "pred_idx", "iou"])
    return match_df


def classify_gt(gt_df, pair_df, match_df, perfect_iou=0.5):
    """Classify each ground truth annotation: Perfect, Offset, Missed, Structural"""
    gt_results = []
    gt_overlap_counts = pair_df.groupby("gt_idx").size().to_dict()
    gt_match_map = {
        int(r["gt_idx"]): {"pred_idx": int(r["pred_idx"]), "iou": float(r["iou"])}
        for _, r in match_df.iterrows()
    }

    for gi, g in gt_df.iterrows():
        overlap_count = int(gt_overlap_counts.get(gi, 0))
        matched_info  = gt_match_map.get(gi, None)

        if overlap_count == 0:
            code, label = 0, "Missed"
            matched_pred, matched_iou = None, None

        elif overlap_count >= 2:
            code, label = 3, "Structural"
            matched_pred = matched_info["pred_idx"] if matched_info else None
            matched_iou  = matched_info["iou"]      if matched_info else None

        else:
            if matched_info is None:
                code, label = 2, "Offset"
                matched_pred, matched_iou = None, None
            else:
                matched_pred = matched_info["pred_idx"]
                matched_iou  = matched_info["iou"]
                if matched_iou >= perfect_iou:
                    code, label = 1, "Perfect"
                else:
                    code, label = 2, "Offset"

        gt_results.append({
            "gt_idx":           gi,
            "gt_start":         g["start"],
            "gt_end":           g["end"],
            "gt_duration":      g["end"] - g["start"],
            "gt_overlap_count": overlap_count,
            "matched_pred_idx": matched_pred,
            "matched_iou":      matched_iou,
            "gt_code":          code,
            "gt_label":         label,
        })

    return pd.DataFrame(gt_results)


def classify_pred(pred_df, pair_df, match_df, perfect_iou=0.5):
    """Classify each prediction: Perfect, Offset, False Positive, Structural"""
    pred_results = []
    pred_overlap_counts = pair_df.groupby("pred_idx").size().to_dict()
    pred_match_map = {
        int(r["pred_idx"]): {"gt_idx": int(r["gt_idx"]), "iou": float(r["iou"])}
        for _, r in match_df.iterrows()
    }

    for pi, p in pred_df.iterrows():
        overlap_count = int(pred_overlap_counts.get(pi, 0))
        matched_info  = pred_match_map.get(pi, None)

        if overlap_count == 0:
            code, label = 0, "False Positive"
            matched_gt, matched_iou = None, None

        elif overlap_count >= 2:
            code, label = 3, "Structural"
            matched_gt  = matched_info["gt_idx"] if matched_info else None
            matched_iou = matched_info["iou"]    if matched_info else None

        else:
            if matched_info is None:
                code, label = 2, "Offset"
                matched_gt, matched_iou = None, None
            else:
                matched_gt  = matched_info["gt_idx"]
                matched_iou = matched_info["iou"]
                if matched_iou >= perfect_iou:
                    code, label = 1, "Perfect"
                else:
                    code, label = 2, "Offset"

        pred_results.append({
            "pred_idx":           pi,
            "pred_start":         p["start"],
            "pred_end":           p["end"],
            "pred_duration":      p["end"] - p["start"],
            "pred_overlap_count": overlap_count,
            "matched_gt_idx":     matched_gt,
            "matched_iou":        matched_iou,
            "pred_code":          code,
            "pred_label":         label,
        })

    return pd.DataFrame(pred_results)


def compute_summary(gt_cls_df, pred_cls_df, audio_duration_sec=None):
    """Compute final evaluation metrics: Precision, Recall, FDR, Miss rate, etc."""
    gt_total   = len(gt_cls_df)
    pred_total = len(pred_cls_df)

    gt_counts   = gt_cls_df["gt_label"].value_counts().to_dict()
    pred_counts = pred_cls_df["pred_label"].value_counts().to_dict()

    gt_missed    = gt_counts.get("Missed", 0)
    gt_perfect   = gt_counts.get("Perfect", 0)
    gt_offset    = gt_counts.get("Offset", 0)
    gt_structural = gt_counts.get("Structural", 0)

    pred_fp         = pred_counts.get("False Positive", 0)
    pred_perfect    = pred_counts.get("Perfect", 0)
    pred_offset     = pred_counts.get("Offset", 0)
    pred_structural = pred_counts.get("Structural", 0)

    TP = pred_perfect + pred_offset
    FP = pred_fp
    FN = gt_missed

    precision  = TP / pred_total if pred_total > 0 else np.nan
    recall     = TP / gt_total   if gt_total   > 0 else np.nan
    fdr        = FP / pred_total if pred_total > 0 else np.nan
    miss_rate  = FN / gt_total   if gt_total   > 0 else np.nan
    fp_per_min = (FP / (audio_duration_sec / 60.0)
                  if audio_duration_sec and audio_duration_sec > 0
                  else np.nan)

    return {
        "GT_total":          gt_total,
        "Pred_total":        pred_total,
        "TP":                TP,
        "FP":                FP,
        "FN":                FN,
        "Precision":         precision,
        "Recall":            recall,
        "FDR":               fdr,
        "Miss_rate":         miss_rate,
        "FP_per_minute":     fp_per_min,
        "GT_Missed":         gt_missed,
        "GT_Perfect":        gt_perfect,
        "GT_Offset":         gt_offset,
        "GT_Structural":     gt_structural,
        "Pred_FalsePositive": pred_fp,
        "Pred_Perfect":      pred_perfect,
        "Pred_Offset":       pred_offset,
        "Pred_Structural":   pred_structural,
    }


def evaluate_selection_tables(
    gt_path,
    pred_path,
    audio_duration_sec=None,
    buffer_sec=0.02,
    overlap_iou_threshold=0.1,
    perfect_iou_threshold=0.5,
    gt_start_col="Begin Time (s)",
    gt_end_col="End Time (s)",
    pred_start_col="Begin Time (s)",
    pred_end_col="End Time (s)",
    save_gt_result_path=None,
    save_pred_result_path=None,
):
    """Main pipeline to evaluate prediction against ground truth"""
    gt_df      = load_raven_table(gt_path,   gt_start_col,   gt_end_col)
    pred_df    = load_raven_table(pred_path, pred_start_col, pred_end_col)
    gt_buf_df  = apply_buffer(gt_df, buffer_sec=buffer_sec)
    pair_df    = build_pair_table(gt_buf_df, pred_df, overlap_threshold=overlap_iou_threshold)
    match_df   = greedy_one_to_one_matching(pair_df)
    gt_cls_df  = classify_gt(gt_df,   pair_df, match_df, perfect_iou=perfect_iou_threshold)
    pred_cls_df = classify_pred(pred_df, pair_df, match_df, perfect_iou=perfect_iou_threshold)
    summary    = compute_summary(gt_cls_df, pred_cls_df, audio_duration_sec)

    if save_gt_result_path is not None:
        gt_cls_df.to_csv(save_gt_result_path, index=False)
    if save_pred_result_path is not None:
        pred_cls_df.to_csv(save_pred_result_path, index=False)

    return {
        "gt_results":   gt_cls_df,
        "pred_results": pred_cls_df,
        "pairs":        pair_df,
        "matches":      match_df,
        "summary":      summary,
    }


def run_posthoc(
    gt_path,
    pred_path,
    audio_duration_sec,
    save_gt_path,
    save_pred_path,
    perfect_iou_threshold=0.3,
):
    """
    Entry function called by run_experiment.py.

    Returns
    -------
    dict
        Summary dictionary containing all evaluation metrics
    """
    result = evaluate_selection_tables(
        gt_path=gt_path,
        pred_path=pred_path,
        audio_duration_sec=audio_duration_sec,
        buffer_sec=0.02,
        overlap_iou_threshold=0.1,
        perfect_iou_threshold=perfect_iou_threshold,
        save_gt_result_path=save_gt_path,
        save_pred_result_path=save_pred_path,
    )
    summary = result["summary"]

    print(f"  FDR:       {summary['FDR']:.4f}")
    print(f"  Miss_rate: {summary['Miss_rate']:.4f}")
    print(f"  Precision: {summary['Precision']:.4f}")
    print(f"  Recall:    {summary['Recall']:.4f}")

    return summary


if __name__ == "__main__":
    summary = run_posthoc(
        gt_path=r"E:\petrel\01p2.txt",
        pred_path=r"E:\petrel_project\results\raven_tables\r02_250421081501p2_b03_g05_full_predict.txt",
        audio_duration_sec=8040.0,
        save_gt_path=r"E:\petrel_project\results\metric_tables\gt_b03_g05.csv",
        save_pred_path=r"E:\petrel_project\results\metric_tables\pred_b03_g05.csv",
        perfect_iou_threshold=0.3,
    )
    print("\n===== Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")