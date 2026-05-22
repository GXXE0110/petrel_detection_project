# Path to your predict configuration file
TOML_PATH = r"E:\petrel_project\config_predict.toml"

# Only change this line for each new dataset
new_name = "r01_250409081452"

# Read the current config file
with open(TOML_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Update the data_dir path
    if line.strip().startswith("data_dir"):
        parts = line.split("/")
        parts[-1] = new_name + '"\n'
        line = "/".join(parts)
    # Update the output annotation CSV filename
    elif line.strip().startswith("annot_csv_filename"):
        line = f'annot_csv_filename = "{new_name}.annot.csv"\n'
    # Remove the old 'path' line that points to the vak dataset
    elif line.strip().startswith("path") and "vak-frame-classification" in line:
        continue
    new_lines.append(line)

# Write the modified config back to file
with open(TOML_PATH, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"✅ Config updated, new_name = {new_name}")