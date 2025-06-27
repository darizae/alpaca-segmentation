#!/usr/bin/env bash
# ------------------------------------------------------------
# upload_all_datasets.sh
# Uploads all dataset_* folders from a given root directory
# into a named wrapper folder on the remote server.
# ------------------------------------------------------------
set -euo pipefail

# Catch interruptions (Ctrl+C) cleanly
trap 'echo "✖ Upload interrupted."; exit 1' INT

# ─── user settings ───────────────────────────────────────────
REMOTE_USER="u17184"
REMOTE_HOST="glogin-gpu.hpc.gwdg.de"
REMOTE_BASE_PATH=".project/dir.project/alpaca-segmentation/data"

# ─── input validation ────────────────────────────────────────
LOCAL_ROOT="${1:-}"

if [[ -z "$LOCAL_ROOT" ]]; then
  echo "✖ Please provide the root directory containing dataset_* folders." >&2
  echo "Usage: ./upload_all_datasets.sh <local_dataset_root>" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_ROOT" ]]; then
  echo "✖ '$LOCAL_ROOT' is not a directory or does not exist." >&2
  exit 1
fi

LOCAL_ROOT_ABS="$(cd "$LOCAL_ROOT" && pwd)"
WRAPPER_FOLDER_NAME="$(basename "$LOCAL_ROOT_ABS")"
REMOTE_TARGET_PATH="${REMOTE_BASE_PATH}/${WRAPPER_FOLDER_NAME}"

echo "📦 Will upload datasets from:"
echo "   ${LOCAL_ROOT_ABS}"
echo "🛰️  To remote folder:"
echo "   ${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_TARGET_PATH}"
echo

# ─── remote wrapper folder setup ─────────────────────────────
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ~/${REMOTE_TARGET_PATH}"

# ─── upload all dataset_* folders ────────────────────────────
for dataset_dir in "${LOCAL_ROOT_ABS}"/dataset_*; do
  if [[ ! -d "$dataset_dir" ]]; then
    echo "⚠️  Skipping '$dataset_dir' (not a directory)"
    continue
  fi

  DATASET_NAME="$(basename "$dataset_dir")"
  REMOTE_FULL_PATH="${REMOTE_TARGET_PATH}/${DATASET_NAME}"

  echo "⇪ Uploading '${DATASET_NAME}' → ${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_FULL_PATH}"

  # Create remote dataset folder
  ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ~/${REMOTE_FULL_PATH}"

  # Upload with resume-friendly rsync
  rsync -av --partial --progress -e ssh "${dataset_dir}/" "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_FULL_PATH}/"

  echo "✅ Finished '${DATASET_NAME}'"
  echo
done

echo "🎉 All dataset_* folders uploaded successfully into '${WRAPPER_FOLDER_NAME}' on remote."
