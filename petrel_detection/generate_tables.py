"""
generate_tables.py
Convert vak predict output .annot.csv to Raven-format .txt annotation tables,
and concatenate into a full long audio annotation table with segment time offsets.
"""

import os
import re
import pandas as pd


def reconstruct_raven_final(predict_csv_path, output_dir, segment_duration=30, tag=None):
    """
    Concatenate predicted CSV into a complete Raven annotation table.

    Parameters
    ----------
    predict_csv_path : str
        Path to the .annot.csv file output by vak predict
    output_dir : str
        Output directory
    segment_duration : int or float
        Duration of each audio segment in seconds, default 30
    tag : str, optional
        Experiment tag to distinguish outputs from different runs, e.g. 'b01_g03'

    Returns
    -------
    str or None
        Path to the generated _full_predict.txt file, None if failed
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check whether the file actually exists and whether its size is 0 bytes
    if not os.path.exists(predict_csv_path) or os.path.getsize(predict_csv_path) == 0:
        print(f"⚠️ Warning: File is empty or does not exist (0 bytes). Skipping: {os.path.basename(predict_csv_path)}")
        return None

    # 1. Load prediction data
    try:
        df = pd.read_csv(predict_csv_path)
    except pd.errors.EmptyDataError:
        # Catching exceptions when Pandas reads a completely blank file (or one containing only newline characters)
        print(f"⚠️ Warning: No data to parse (EmptyDataError). Skipping: {os.path.basename(predict_csv_path)}")
        return None

    # If the file has a header row but zero rows, or if the required columns are missing entirely
    if df.empty or 'onset_s' not in df.columns or 'offset_s' not in df.columns:
        print(f"⚠️ Warning: Missing data or required columns. Skipping: {os.path.basename(predict_csv_path)}")
        return None

    # 2. Drop empty rows
    df_clean = df.dropna(subset=['onset_s', 'offset_s']).copy()

    if df_clean.empty:
        print("❌ Warning: No valid syllable boxes found in prediction results!")
        return None

    all_rows = []
    base_file_name = "reconstructed_results"

    # 3. Process by segment
    for audio_path, group in df_clean.groupby('notated_path'):
        file_name = os.path.basename(audio_path)

        # Extract original base filename (remove _segXXX and suffix)
        if base_file_name == "reconstructed_results":
            base_file_name = re.split(r'_seg\d+', file_name)[0]

        match = re.search(r'seg(\d+)', file_name)
        if not match:
            print(f"  ⚠ Skipping file with unparseable seg number: {file_name}")
            continue

        seg_index = int(match.group(1))
        offset_time = seg_index * segment_duration

        # 4. Convert time coordinates
        raven_group = pd.DataFrame()
        raven_group['Begin Time (s)'] = (group['onset_s'] + offset_time).astype(float)
        raven_group['End Time (s)']   = (group['offset_s'] + offset_time).astype(float)
        raven_group['View']           = 'Spectrogram 1'
        raven_group['Channel']        = 1
        raven_group['Low Freq (Hz)']  = 2000.0
        raven_group['High Freq (Hz)'] = 12000.0
        raven_group['Annotation']     = group['label'].values

        all_rows.append(raven_group)

    # 5. Concatenate and export
    if not all_rows:
        print("❌ No valid seg-numbered files matched.")
        return None

    total_df = pd.concat(all_rows).sort_values('Begin Time (s)').reset_index(drop=True)
    total_df.insert(0, 'Selection', range(1, len(total_df) + 1))

    final_columns = [
        'Selection', 'View', 'Channel',
        'Begin Time (s)', 'End Time (s)',
        'Low Freq (Hz)', 'High Freq (Hz)', 'Annotation'
    ]
    total_df = total_df[final_columns]

    # Add tag to filename for experiment distinction
    if tag:
        output_filename = f"{base_file_name}_{tag}_full_predict.txt"
    else:
        output_filename = f"{base_file_name}_full_predict.txt"

    output_path = os.path.join(output_dir, output_filename)
    total_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    print(f"✅ Conversion successful! Full table saved to: {output_path}")

    return output_path


def run_generate_tables(predict_csv_path, output_dir, segment_duration=30, tag=None):
    """
    Entry function called by run_experiment.py.

    Returns
    -------
    str or None
        Path to the generated annotation table
    """
    return reconstruct_raven_final(
        predict_csv_path=predict_csv_path,
        output_dir=output_dir,
        segment_duration=segment_duration,
        tag=tag,
    )


if __name__ == "__main__":
    import glob

    PREDICT_DIR = r"E:\petrel_project\results\predict_b02g0"  # Folder containing annot.csv
    OUTPUT_DIR  = r"E:\petrel_project\results\raven_tables\night_11"
    TAG         = "b02_g0"  # Can be None

    csv_files = glob.glob(os.path.join(PREDICT_DIR, "*.annot.csv"))

    if not csv_files:
        print("⚠️ No .annot.csv files found")
    else:
        print(f"📂 Found {len(csv_files)} files. Starting batch conversion...\n")
        for csv_path in sorted(csv_files):
            print(f"▶ Processing: {os.path.basename(csv_path)}")
            run_generate_tables(
                predict_csv_path=csv_path,
                output_dir=OUTPUT_DIR,
                segment_duration=30,
                tag=TAG,
            )
        print("\n🎉 All tasks completed!")
