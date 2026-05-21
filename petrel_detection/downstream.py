"""
batch_raven_to_csv.py
Convert annotation times from _full_predict.txt to seconds relative to 06:00:00 of the same day.
"""

import os
import re
import pandas as pd
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────
RECORDING_START_XLSX = r"E:\petrel_project\recording_start_times.xlsx"
SUFFIX               = "_b02_g0_full_predict.txt"   # ← Only change this each time
# ────────────────────────────────────────────────────────────

# Baseline time: 06:00:00 of the current day
BASELINE_HOUR = 6


def load_start_times(xlsx_path):
    """Read Excel file and return {file_name: start_time string} dictionary"""
    df = pd.read_excel(xlsx_path, dtype=str)
    return dict(zip(df['file_name'].str.strip(), df['start_time'].str.strip()))


def parse_start_time_to_seconds(start_time_str):
    """
    Convert HH:MM:SS.s to seconds relative to 06:00:00 of the same day.
    Example: '08:14:35.852' → 8075.852
    """
    t = datetime.strptime(start_time_str, "%H:%M:%S.%f")
    baseline = t.replace(hour=BASELINE_HOUR, minute=0, second=0, microsecond=0)
    delta = t - baseline
    return delta.total_seconds()


def convert_one_file(txt_path, start_times_dict):
    """
    Process a single _full_predict.txt file, return DataFrame if successful, None if failed.
    """
    basename = os.path.basename(txt_path)

    # Extract recorder ID
    recorder_match = re.match(r'r(\d+)_', basename, re.IGNORECASE)
    if not recorder_match:
        print(f"  ⚠ Skipped (cannot parse recorder ID): {basename}")
        return None
    recorder = int(recorder_match.group(1))

    # Extract file_name (e.g., r01_250505055705)
    file_name_match = re.match(r'(r\d+_\d+)', basename, re.IGNORECASE)
    if not file_name_match:
        print(f"  ⚠ Skipped (cannot parse file_name): {basename}")
        return None
    file_name = file_name_match.group(1)

    # Look up start time from Excel
    if file_name not in start_times_dict:
        print(f"  ⚠ Skipped (not found in Excel: {file_name}): {basename}")
        return None

    start_time_str = start_times_dict[file_name]
    recording_offset_s = parse_start_time_to_seconds(start_time_str)

    # Read annotations, keep only 'p'
    df = pd.read_csv(txt_path, sep='\t')
    df.columns = df.columns.str.strip()

    if 'Annotation' in df.columns:
        df = df[df['Annotation'] == 'p'].copy()

    if df.empty:
        print(f"  ⚠ Skipped (no valid annotations): {basename}")
        return None

    # Convert time: seconds from 6AM + offset within the recording
    out_df = pd.DataFrame({
        'Recorder': recorder,
        'start': (recording_offset_s + df['Begin Time (s)'].astype(float)).round(3).values,
        'end':   (recording_offset_s + df['End Time (s)'].astype(float)).round(3).values,
    })

    return out_df


def batch_convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    start_times_dict = load_start_times(RECORDING_START_XLSX)
    print(f"📋 Loaded {len(start_times_dict)} recording start times\n")

    txt_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(SUFFIX)
    ]

    if not txt_files:
        print(f"❌ No {SUFFIX} files found in: {input_dir}")
        return

    print(f"📂 Found {len(txt_files)} files, starting processing...\n")

    all_dfs = []

    for txt_path in sorted(txt_files):
        print(f"  Processing: {os.path.basename(txt_path)}")
        df = convert_one_file(txt_path, start_times_dict)
        if df is not None:
            base = os.path.basename(txt_path).replace(SUFFIX, '')
            single_path = os.path.join(output_dir, f"{base}_stats.csv")
            df.to_csv(single_path, index=False)
            print(f"    ✅ {len(df)} records → {os.path.basename(single_path)}")
            all_dfs.append(df)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged = merged.sort_values(['Recorder', 'start']).reset_index(drop=True)
        merged_path = os.path.join(output_dir, "all_merged_stats.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\n✅ Merged master file created: {merged_path} (total {len(merged)} records)")
    else:
        print("\n❌ No files processed successfully.")


if __name__ == "__main__":
    batch_convert(
        input_dir  = r"E:\petrel_project\results\raven_tables\night_11",
        output_dir = r"E:\petrel_project\results\stats_csv\b02_g0",
    )