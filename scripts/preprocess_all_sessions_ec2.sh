#!/usr/bin/env bash
# Preprocess every session's side-camera videos into an HDF5 file, meant to be
# run on an EC2 instance with local NVMe scratch space and the project's
# python environment (torchcodec, h5py, torchvision, ...) active.
#
# For each session found under s3://$SRC_BUCKET/$SRC_ROOT/:
#   1. Copies .../<session>/side/compressed/*.mp4 to $WORK_DIR/<session>/ on
#      local NVMe.
#   2. Runs preprocess_single_session_videos_to_h5.py with configs/ae_config.json
#      as the base config and configs/crop_configs/<session>.json as the
#      per-session crop override (skips the session if that file is missing).
#   3. Uploads the resulting HDF5 to s3://$DST_BUCKET/$DST_PREFIX/.
#   4. Deletes the local mp4s and the local HDF5, then moves to the next session.
#
# Usage:
#   ./preprocess_all_sessions_ec2.sh            # actually run
#   DRY_RUN=1 ./preprocess_all_sessions_ec2.sh  # just print what it would do
#   FORCE=1 ./preprocess_all_sessions_ec2.sh    # reprocess sessions whose output already exists on S3

set -uo pipefail

SRC_BUCKET="balint-fire-videos-autoencoder-2026"
SRC_ROOT="timwang/videos"
DST_BUCKET="balint-video-autoencoder-data-233060639700-us-west-2-an"
DST_PREFIX="preprocessed_videos"
WORK_DIR="/opt/dlami/nvme"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PREPROCESS_PY="$PROJECT_ROOT/scripts/preprocess_single_session_videos_to_h5.py"
BASE_CONFIG="$PROJECT_ROOT/configs/ae_config.json"
CROP_CONFIG_DIR="$PROJECT_ROOT/configs/crop_configs"

AWS="${AWS:-aws}"           # e.g. AWS="aws --profile myprofile"
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

run() { if [[ "$DRY_RUN" == "1" ]]; then echo "  [dry-run] $*"; else "$@"; fi; }

mkdir -p "$WORK_DIR"

# --- list sessions (common prefixes directly under $SRC_ROOT/) ---
mapfile -t SESSIONS < <($AWS s3 ls "s3://$SRC_BUCKET/$SRC_ROOT/" | awk '$1=="PRE"{sub(/\/$/,"",$2); print $2}')

if [[ ${#SESSIONS[@]} -eq 0 ]]; then
  echo "No sessions found under s3://$SRC_BUCKET/$SRC_ROOT/ -- check the prefix/credentials." >&2
  exit 1
fi
echo "Found ${#SESSIONS[@]} sessions."

ok=0; skipped=0; failed=0
for S in "${SESSIONS[@]}"; do
  echo "=== $S"

  SESSION_CONFIG="$CROP_CONFIG_DIR/${S}.json"
  if [[ ! -f "$SESSION_CONFIG" ]]; then
    echo "  !! no crop config at $SESSION_CONFIG, skipping"
    ((skipped++)); continue
  fi

  H5_FILENAME="${S}_side_crop.h5"
  DST_URI="s3://$DST_BUCKET/$DST_PREFIX/$H5_FILENAME"

  if [[ "$FORCE" != "1" ]] && $AWS s3 ls "$DST_URI" >/dev/null 2>&1; then
    echo "  already preprocessed at $DST_URI, skipping (set FORCE=1 to redo)"
    ((skipped++)); continue
  fi

  SESSION_DIR="$WORK_DIR/$S"
  LOCAL_H5="$WORK_DIR/$H5_FILENAME"
  SRC_URI="s3://$SRC_BUCKET/$SRC_ROOT/$S/side/compressed/"

  run mkdir -p "$SESSION_DIR"

  if ! run $AWS s3 cp "$SRC_URI" "$SESSION_DIR/" --recursive; then
    echo "  !! failed to copy videos for $S, skipping"
    ((failed++)); run rm -rf "$SESSION_DIR"; continue
  fi

  if ! run $PYTHON "$PREPROCESS_PY" \
        --config "$BASE_CONFIG" \
        --session-config "$SESSION_CONFIG" \
        --data-path "$SESSION_DIR" \
        --save-path "$WORK_DIR" \
        --h5-filename "$H5_FILENAME" \
        --overwrite; then
    echo "  !! preprocessing failed for $S, skipping"
    ((failed++)); run rm -rf "$SESSION_DIR"; continue
  fi

  if ! run $AWS s3 cp "$LOCAL_H5" "$DST_URI"; then
    echo "  !! failed to upload $LOCAL_H5 to $DST_URI, keeping local files for inspection"
    ((failed++)); continue
  fi

  run rm -rf "$SESSION_DIR"
  run rm -f "$LOCAL_H5"
  ((ok++))
done

echo
echo "Done: $ok sessions processed, $skipped skipped, $failed failed."
