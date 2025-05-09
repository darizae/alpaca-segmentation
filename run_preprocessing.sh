#!/usr/bin/env bash
set -euo pipefail

# Make sure we’re running from the repo root
cd "$(dirname "$0")"

# 1) build the index
python3 data_preprocessing/build_alpaca_index.py

# 2) generate + cache PNG spectrograms
python3 data_preprocessing/add_png_spectrograms.py

echo "✅ All preprocessing steps complete."
