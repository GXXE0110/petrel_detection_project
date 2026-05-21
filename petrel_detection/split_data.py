"""
Prepare training and prediction data.
Split long audio into paired wav + wav.csv files.
"""

import os
import pandas as pd
from pydub import AudioSegment


def split_and_label_data(
        wav_path,
        raven_txt_path=None,
        output_dir=None,
        segment_dur_s=30,
        default_label='p',
        min_segment_ms=1000
):
    """
    Adaptive splitting:
    - If raven_txt_path is provided: generate labeled CSV (for training)
    - If raven_txt_path is not provided: generate empty label CSV (for prediction)
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Load annotations (if available) ---
    raven_df = None
    start_col, end_col, label_col = 'Begin Time (s)', 'End Time (s)', 'Annotation'

    if raven_txt_path and os.path.exists(raven_txt_path):
        print(f"📖 Annotation file detected, loading: {raven_txt_path}")
        raven_df = pd.read_csv(raven_txt_path, sep='\t')
        raven_df[label_col] = raven_df[label_col].fillna(default_label)
        raven_df = raven_df.dropna(subset=[start_col, end_col]).copy()
        raven_df = raven_df[raven_df[end_col] > raven_df[start_col]].copy()
        print(f"✅ Number of valid annotations: {len(raven_df)}")
    else:
        print("🔍 No annotation file provided or path invalid. Generating empty CSV in PREDICTION mode.")

    # --- 2. Load audio ---
    print(f"🎵 Loading audio: {wav_path}")
    audio = AudioSegment.from_wav(wav_path)
    total_len_ms = len(audio)
    file_stem = os.path.splitext(os.path.basename(wav_path))[0]
    segment_len_ms = int(segment_dur_s * 1000)

    # --- 3. Loop and split ---
    for i, start_ms in enumerate(range(0, total_len_ms, segment_len_ms)):
        end_ms = min(start_ms + segment_len_ms, total_len_ms)
        actual_seg_len_ms = end_ms - start_ms

        if actual_seg_len_ms < min_segment_ms:
            continue

        curr_start_s = start_ms / 1000.0
        curr_end_s = end_ms / 1000.0
        actual_seg_len_s = actual_seg_len_ms / 1000.0

        seg_name = f"{file_stem}_seg{i:03d}"
        wav_out = os.path.join(output_dir, f"{seg_name}.wav")
        csv_out = os.path.join(output_dir, f"{seg_name}.wav.csv")

        # A. Save audio segment
        extract = audio[start_ms:end_ms]
        channels = extract.split_to_mono()
        extract_ch1 = channels[0]
        extract_ch1.export(wav_out, format="wav")

        # B. Generate CSV
        final_df = pd.DataFrame(columns=['onset_s', 'offset_s', 'label'])

        if raven_df is not None:
            overlap_mask = (
                    (raven_df[start_col] < curr_end_s) &
                    (raven_df[end_col] > curr_start_s)
            )
            sub_df = raven_df[overlap_mask].copy()

            if not sub_df.empty:
                sub_df['onset_s'] = (sub_df[start_col].clip(lower=curr_start_s) - curr_start_s).clip(lower=0)
                sub_df['offset_s'] = (sub_df[end_col].clip(upper=curr_end_s) - curr_start_s).clip(
                    upper=actual_seg_len_s)
                sub_df['label'] = sub_df[label_col]
                final_df = sub_df[sub_df['offset_s'] > sub_df['onset_s']][['onset_s', 'offset_s', 'label']]

        final_df.to_csv(csv_out, index=False)

    print(f"✨ Processing complete for [{os.path.basename(wav_path)}]! Output directory: {output_dir}")


def batch_split(
        input_dir,
        output_base_dir,
        raven_dir=None,       # Optional: label folder, txt filenames must match wav filenames
        segment_dur_s=30,
        default_label='p',
        min_segment_ms=1000
):
    """
    Batch process all wav files in a folder.
    Results for each wav file are saved into a separate subfolder named after the source file.

    Args:
        input_dir       : Source folder containing wav files
        output_base_dir : Root output directory for all split results
        raven_dir       : (Optional) Folder containing matching .txt annotation files
    """
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print(f"⚠️ No wav files found in {input_dir}.")
        return

    print(f"📂 Found {len(wav_files)} wav files. Starting batch processing...\n")

    for idx, wav_file in enumerate(wav_files, 1):
        file_stem = os.path.splitext(wav_file)[0]
        wav_path = os.path.join(input_dir, wav_file)

        # Each source file → independent subfolder
        output_dir = os.path.join(output_base_dir, file_stem)

        # Auto-match annotation file (if raven_dir is provided)
        raven_txt_path = None
        if raven_dir:
            candidate = os.path.join(raven_dir, f"{file_stem}.txt")
            if os.path.exists(candidate):
                raven_txt_path = candidate

        print(f"[{idx}/{len(wav_files)}] Processing: {wav_file}")
        split_and_label_data(
            wav_path=wav_path,
            raven_txt_path=raven_txt_path,
            output_dir=output_dir,
            segment_dur_s=segment_dur_s,
            default_label=default_label,
            min_segment_ms=min_segment_ms
        )
        print()

    print(f"🎉 All files processed! Results saved in: {output_base_dir}")


# --- Run Settings ---
batch_split(
    input_dir       = r"G:/predict_Ihu",      # Source wav folder
    output_base_dir = r"E:/petrel_project/predict_data",  # Root output directory
    raven_dir       = None,   # Fill when labels are available, e.g. r"E:/petrel/labels"
)