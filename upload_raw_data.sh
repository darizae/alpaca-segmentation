#!/usr/bin/env bash
# ------------------------------------------------------------
# upload_raw_data.sh
#   Uploads a local file or directory to the alpaca-segmentation data folder.
#   Uses rsync for folders and scp for files.
# ------------------------------------------------------------
set -euo pipefail

# ─── user settings ───────────────────────────────────────────
REMOTE_USER="u17184"
REMOTE_HOST="glogin-gpu.hpc.gwdg.de"
REMOTE_TARGET_SUBPATH=".project/dir.project/alpaca-segmentation/data"

# ─── input validation ────────────────────────────────────────
LOCAL_PATH="${1:-}"

if [[ -z "$LOCAL_PATH" ]]; then
  echo "✖ Please provide the path to the local folder or file you want to upload." >&2
  echo "Usage: ./upload_raw_data.sh <local_path>" >&2
  exit 1
fi

if [[ ! -e "$LOCAL_PATH" ]]; then
  echo "✖ Local path '$LOCAL_PATH' not found." >&2
  exit 1
fi

# ─── destination setup ───────────────────────────────────────
ITEM_NAME="$(basename "$LOCAL_PATH")"
REMOTE_FULL_PATH="${REMOTE_TARGET_SUBPATH}/${ITEM_NAME}"

echo "⇪  Uploading '${LOCAL_PATH}'  →  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FULL_PATH}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ~/${REMOTE_TARGET_SUBPATH}"

# ─── file or directory handling ──────────────────────────────
if [[ -d "$LOCAL_PATH" ]]; then
  echo "📁 Detected directory → using rsync"
  rsync -av --progress -e ssh "${LOCAL_PATH}/" "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_FULL_PATH}/"
elif [[ -f "$LOCAL_PATH" ]]; then
  echo "📦 Detected file → using rsync instead of scp"
  rsync -av --progress -e ssh "${LOCAL_PATH}" "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_FULL_PATH}"
else
  echo "✖ '$LOCAL_PATH' is neither a file nor a directory." >&2
  exit 1
fi

echo "✅ Upload complete."
