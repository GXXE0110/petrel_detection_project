import pandas as pd
import wave
import os
from datetime import datetime, timedelta


def get_wav_duration(file_path):
    """Get duration of a WAV file in seconds"""
    try:
        with wave.open(file_path, 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
        return 0.0
    except Exception as e:
        print(f"Error: Cannot read {file_path} - {e}")
        return 0.0


def analyze_audio_gaps(excel_path, wav_folder, output_dir):
    """Analyze time gaps between consecutive audio recordings per recorder"""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(excel_path)
    df['start_dt'] = pd.to_datetime(df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['start_time'].astype(str))
    df = df.sort_values(by=['rec', 'start_dt'])

    gap_rows = []

    for rec_id, group in df.groupby('rec'):
        group = group.reset_index(drop=True)
        recorder_num = int(re.search(r'\d+', rec_id).group()) if re.search(r'\d+', rec_id) else rec_id
        print(f"\n--- Analyzing recorder: {rec_id} ---")

        for i in range(len(group)):
            row = group.iloc[i]
            wav_path = os.path.join(wav_folder, f"{row['file_name']}.wav")
            duration_sec = get_wav_duration(wav_path)
            end_time = row['start_dt'] + timedelta(seconds=duration_sec)

            print(f"  {row['file_name']} | Duration: {duration_sec:.3f}s | End: {end_time.strftime('%H:%M:%S.%f')[:-3]}")

            if i < len(group) - 1:
                next_start = group.iloc[i + 1]['start_dt']
                gap_sec = (next_start - end_time).total_seconds()

                gap_rows.append({
                    'Recorder': recorder_num,
                    'Prev_End_Time': end_time.strftime('%H:%M:%S.%f')[:-3],
                    'Next_Start_Time': next_start.strftime('%H:%M:%S.%f')[:-3],
                    'Gap_Duration(s)': round(gap_sec, 3),
                })

                print(f"    → Gap: {gap_sec:.3f}s")

    result_df = pd.DataFrame(gap_rows)
    result_df = result_df.sort_values(['Recorder', 'Prev_End_Time']).reset_index(drop=True)

    output_path = os.path.join(output_dir, "audio_gap_report.csv")
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Analysis complete! Results saved to {output_path}")


# --- Configure paths ---
EXCEL_PATH = r"E:\petrel_project\recording_start_times.xlsx"
WAV_FOLDER = r"G:\predict_Ihu"
OUTPUT_DIR = r"E:\petrel_project"

if __name__ == "__main__":
    import re
    analyze_audio_gaps(EXCEL_PATH, WAV_FOLDER, OUTPUT_DIR)