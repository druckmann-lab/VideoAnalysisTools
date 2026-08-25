#!/usr/bin/env bash
# Copy one (mp4 + .ts.txt) trial pair from each behavioral session to a local folder.
#
# Usage:
#   ./grab_one_trial_per_session.sh            # actually copy
#   DRY_RUN=1 ./grab_one_trial_per_session.sh  # just print what it would do
#
# Layout assumed on S3:
#   s3://$BUCKET/$ROOT/<session>/side/compressed/<name>.mp4
#   s3://$BUCKET/$ROOT/<session>/side/<name>.ts.txt

set -uo pipefail

BUCKET="balint-fire-videos-autoencoder-2026"
ROOT="timwang/videos"
DEST="/mnt/m/Data/VideoAnalysisTools"
SKIP_HEAD=50          # avoid the first N trials of a session
SKIP_TAIL=10          # avoid the last N trials of a session
DRY_RUN="${DRY_RUN:-0}"
AWS="aws"             # e.g. AWS="aws --profile myprofile"

run() { if [[ "$DRY_RUN" == "1" ]]; then echo "  [dry-run] $*"; else "$@"; fi; }

mkdir -p "$DEST"

# --- list sessions (common prefixes directly under $ROOT/) ---
mapfile -t SESSIONS < <($AWS s3 ls "s3://$BUCKET/$ROOT/" | awk '$1=="PRE"{sub(/\/$/,"",$2); print $2}')

if [[ ${#SESSIONS[@]} -eq 0 ]]; then
  echo "No sessions found under s3://$BUCKET/$ROOT/ -- check the prefix/credentials." >&2
  exit 1
fi
echo "Found ${#SESSIONS[@]} sessions."

ok=0; skipped=0
for S in "${SESSIONS[@]}"; do
  echo "=== $S"
  SRC="s3://$BUCKET/$ROOT/$S/side/compressed/"

  # list mp4s, natural-sort so trial9 < trial10 < trial140
  mapfile -t MP4S < <($AWS s3 ls "$SRC" | awk '{print $4}' | grep -E '\.mp4$' | sort -V)
  n=${#MP4S[@]}
  if (( n == 0 )); then
    echo "  !! no .mp4 files found, skipping"; ((skipped++)); continue
  fi

  # pick the middle of the [SKIP_HEAD, n-SKIP_TAIL) window; fall back to plain middle
  if (( n > SKIP_HEAD + SKIP_TAIL )); then
    lo=$SKIP_HEAD; hi=$(( n - SKIP_TAIL - 1 ))
    idx=$(( (lo + hi) / 2 ))
  else
    idx=$(( n / 2 ))
    echo "  (only $n trials -- head/tail margins don't fit, taking the middle one)"
  fi

  MP4="${MP4S[$idx]}"
  BASE="${MP4%.mp4}"
  OUT="$DEST/$S"
  echo "  $n trials -> picking #$((idx+1)): $MP4"

  run mkdir -p "$OUT"
  run $AWS s3 cp "${SRC}${MP4}" "$OUT/"

  # the txt lives one level up, in side/, named <base>.ts.txt
  TXTDIR="s3://$BUCKET/$ROOT/$S/side/"
  if ! run $AWS s3 cp "${TXTDIR}${BASE}.ts.txt" "$OUT/" 2>/dev/null; then
    # fallback: some sessions may use <base>.txt instead
    if ! run $AWS s3 cp "${TXTDIR}${BASE}.txt" "$OUT/" 2>/dev/null; then
      echo "  !! no matching txt for $BASE (looked for .ts.txt and .txt in side/)"
    fi
  fi
  ((ok++))
done

echo
echo "Done: $ok sessions copied, $skipped skipped. Output in $DEST"