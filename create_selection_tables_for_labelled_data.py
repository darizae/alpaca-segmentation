import json
import pandas as pd
from pathlib import Path
import shutil

# ----------- CONFIGURABLE PARAMETERS -----------
# Path to the JSON file
json_path = Path("/Users/danie/repos/alpaca-segmentation/data/index_hums.json")

# Output directory for selection tables
output_root = Path("/Users/danie/repos/alpaca-segmentation/data/selection_tables")

# Qualities to include
ALLOWED_QUALITIES = [1, 2, 3 , 4]

# Substring to match in the base path (before third .wav)
PATH_FILTER = "387_20201207_cut.wav_60_75.wav"  # e.g., "387_20201207"

# Frequency bounds for Raven Lite
LOW_FREQ = 0
HIGH_FREQ = 4000

# ----------- SCRIPT START -----------
# Recreate the output directory
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True, exist_ok=True)

# Load the JSON data
with open(json_path, 'r') as file:
    hum_objects = json.load(file)

# Organize selections by raw recording name
selection_tables = {}

for obj in hum_objects:
    if obj["quality"] not in ALLOWED_QUALITIES:
        continue

    full_path = obj["path"]

    # Extract only the base part up to the second '.wav'
    parts = full_path.split(".wav")
    if len(parts) < 3:
        continue  # Skip malformed paths
    base_path = ".wav".join(parts[:2]) + ".wav"

    if PATH_FILTER and PATH_FILTER not in base_path:
        continue

    if base_path not in selection_tables:
        selection_tables[base_path] = []

    selection_tables[base_path].append({
        "Selection": len(selection_tables[base_path]) + 1,
        "View": "Spectrogram_1",
        "Channel": 1,
        "Begin time (s)": obj["hum_start_s"],
        "End time (s)": obj["hum_end_s"],
        "Low Freq (Hz)": LOW_FREQ,
        "High Freq (Hz)": HIGH_FREQ,
        "Sound type": "target",
        "Comments": ""
    })

# Write selection tables to output
for raw_name, entries in selection_tables.items():
    df = pd.DataFrame(entries)
    output_file = output_root / f"{raw_name}_selection.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep="\t", index=False)

print("Selection tables created successfully with quality filter:", ALLOWED_QUALITIES)
if PATH_FILTER:
    print("Path filter applied to base path:", PATH_FILTER)
else:
    print("No path filter applied.")
