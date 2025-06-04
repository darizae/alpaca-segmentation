#!/usr/bin/env bash
# ------------------------------------------------------------
# sync_prepared_to_server.sh
#   Push a prepared/ clip set to the GPU server
# ------------------------------------------------------------
set -euo pipefail

# ─── user settings ───────────────────────────────────────────
REMOTE_USER="u17184"
REMOTE_HOST="glogin-gpu.hpc.gwdg.de"

# Remote repo root (no trailing /)
REMOTE_REPO="/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca"
LOCAL_PREP_DIR="${1:-data/prepared}"           # arg1 = local folder to sync
DATASET_TAG="${2:-alpaca_Q1Q4_v1}"        # arg2 = remote sub-folder name
# e.g. run:  ./sync_prepared_to_server.sh prepared alpaca_Q1Q4_v2
# ─────────────────────────────────────────────────────────────

if [[ ! -d "$LOCAL_PREP_DIR" ]]; then
  echo "✖ Local folder '$LOCAL_PREP_DIR' not found." >&2
  exit 1
fi

REMOTE_DATA_DIR="${REMOTE_REPO}/alpaca_data/${DATASET_TAG}"
echo "⇪  Uploading '${LOCAL_PREP_DIR}'  →  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_DIR}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_DATA_DIR}'"
rsync -av --progress "${LOCAL_PREP_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_DIR}/"

echo "✅  Sync complete."
