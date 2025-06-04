#!/usr/bin/env bash
# ------------------------------------------------------------
# upload_to_segmentation_data.sh
#   Upload a local directory to the alpaca-segmentation data folder,
#   keeping the folder name on the remote side
# ------------------------------------------------------------
set -euo pipefail

# ─── user settings ───────────────────────────────────────────
REMOTE_USER="u17184"
REMOTE_HOST="glogin-gpu.hpc.gwdg.de"

# Fixed remote target base directory (no trailing /)
REMOTE_TARGET_BASE="/user.d.arizaecheverri/u17184/.project/dir.project/alpaca-segmentation/data"

# Argument: local directory to upload
LOCAL_DIR="${1:-}"

if [[ -z "$LOCAL_DIR" ]]; then
  echo "✖ Please provide the path to the local folder you want to upload." >&2
  echo "Usage: ./upload_to_segmentation_data.sh <local_folder>" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "✖ Local folder '$LOCAL_DIR' not found." >&2
  exit 1
fi

# Extract just the folder name (no path)
FOLDER_NAME="$(basename "$LOCAL_DIR")"
REMOTE_FULL_PATH="${REMOTE_TARGET_BASE}/${FOLDER_NAME}"

echo "⇪  Uploading '${LOCAL_DIR}'  →  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FULL_PATH}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_FULL_PATH}'"
rsync -av --progress "${LOCAL_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FULL_PATH}/"

echo "✅  Upload complete."
