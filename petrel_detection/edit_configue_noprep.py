import os
import glob

# Path to your prediction config file
TOML_PATH = r"E:\petrel_project\config_predict.toml"
# Root directory for prepared prediction data
PREP_ROOT = r"E:\petrel_project\prep\predict"

# Only change this line for each new file
new_name = "r10_250505102732"

# Auto-find the corresponding dataset directory
pattern = os.path.join(PREP_ROOT, f"{new_name}-vak-frame-classification-dataset-generated-*")
candidates = glob.glob(pattern)

if not candidates:
    print(f"❌ Dataset directory for {new_name} not found. Please check if prep is completed.")
    exit()

# Select the most recently generated dataset
dataset_path = sorted(candidates, key=os.path.getmtime, reverse=True)[0]
dataset_path_toml = dataset_path.replace("\\", "\\\\")
print(f"📁 Found dataset path: {dataset_path_toml}")

# Read and modify the TOML config
with open(TOML_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip().startswith("annot_csv_filename"):
        line = f'annot_csv_filename = "{new_name}.annot.csv"\n'
    elif line.strip().startswith("path") and "vak-frame-classification" in line:
        line = f'path = "{dataset_path_toml}"\n'
    new_lines.append(line)

# Write back the updated config
with open(TOML_PATH, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"✅ Config updated successfully, new_name = {new_name}")
