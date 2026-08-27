#!/usr/bin/env python3
"""
Precompute the per-session mean frame once, so the 20 training runs don't each
recompute it twice.

Currently every H5VideoDataset with mean_frame_path=null calls _build_mean_frame,
which fancy-indexes 50k frames out of the h5. The training script builds two
datasets per session, and both get the SAME trial_split_df, so they compute the
identical mean. Measured on kd104_twNew_20221124_104921: ~120s of the 185s
dataset construction, twice per session. Over 20 sessions that is ~80 minutes of
GPU-instance time spent recomputing the same arrays.

This runs ONE cheap CPU instance over all sessions sequentially and writes
  s3://<bucket>/mean_frames/<session>_mean_frame.npy

Outputs go to their own prefix so preprocessed_videos/ stays strictly
read-only for every role -- no carve-outs to reason about later.

Cost: ~20 sessions x ~3.5 min = ~75 min on c6id.2xlarge (~$0.40/hr) = under $1.

PREREQUISITES
  1. The instance role needs PutObject on mean_frames/*.
     Apply the updated policy first:
       aws iam put-role-policy --role-name VideoAutoencoderTrainingRole \
           --policy-name S3DataAccess \
           --policy-document file://instance_role_policy.json

  2. To USE the output, train_single_session_autoencoder_st.py needs a
     --mean_frame_path flag setting config['dataset']['mean_frame_path'],
     mirroring the existing --h5_path handling.

SCALING NOTE (important)
  dataset_st._build_mean_frame returns  torch.from_numpy(mean).float() / 255.0
  but the load path is just              np.load(mean_frame_path)
  with no division. So the saved file must be PRE-SCALED to [0,1] float32,
  which is what this script writes. Saving the raw 0-255 mean would silently
  subtract a mean 255x too large.

Usage:
  python precompute_mean_frames.py --dry-run
  python precompute_mean_frames.py
  python precompute_mean_frames.py --verify       # check results afterwards
"""

import argparse
import base64
import datetime
import sys

import boto3

REGION = "us-west-2"
BUCKET = "balint-video-autoencoder-data-233060639700-us-west-2-an"
AMI_ID = "ami-0bcccc2c1e9b9f874"
PYTHON = "/home/ubuntu/ml_env/bin/python"
REPO = "https://github.com/druckmann-lab/VideoAnalysisTools.git"
BRANCH = "balint-dev"
PROFILE = "VideoAutoencoderTrainingRole"
PREFIX = "preprocessed_videos/"      # source h5 files, read-only
OUT_PREFIX = "mean_frames/"           # outputs go here, never into the source prefix
SUFFIX = "_side_crop.h5"

# No GPU needed: this is h5 reads and a numpy mean. c6id gives 8 vCPU, good
# network, and local NVMe, at about a quarter of the g5.4xlarge rate.
INSTANCE_TYPE = "c6id.2xlarge"

# Whole-job ceiling. Expect ~80 min; this is a runaway guard, not a schedule.
JOB_TIMEOUT_HOURS = 4


# --------------------------------------------------------------------------
# Per-session worker. Replicates dataset_st._build_mean_frame exactly, but
# without constructing an H5VideoDataset (which would pull all 16 GB of frames
# into RAM just to compute a 120x112 mean).
# --------------------------------------------------------------------------
WORKER_PY = r'''
import json, os, sys, time
import numpy as np
import h5py

REPO = "/home/ubuntu/VideoAnalysisTools"
sys.path.insert(0, os.path.join(REPO, "src"))

H5, BPOD, ANIMAL, SESSION, OUT = sys.argv[1:6]


def update(d, u):
    for k, v in u.items():
        d[k] = update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


cfg = json.load(open(f"{REPO}/configs/ae_config.json"))
cfg = update(cfg, json.load(open(f"{REPO}/configs/aws_batch_config.json")))
cfg["metadata_config"]["bpod_path"] = BPOD
cfg["metadata_config"]["h5_path"] = H5

from behavioral_autoencoder.dataset_st import SessionMetadataHandler

# Same filtering the training run applies, so the mean is over the same trials.
t = time.time()
mh = SessionMetadataHandler(config=cfg["metadata_config"], mode="local",
                            animal=ANIMAL, session=SESSION)
df = mh.process_all()
print(f"  trials={len(df)} metadata_sec={time.time()-t:.0f}", flush=True)

# --- verbatim logic from _build_mean_frame ---
t = time.time()
with h5py.File(H5, "r") as h5f:
    h5_trial_ids = h5f["trial_ids"][:]
    indices = np.where(np.isin(h5_trial_ids, df["video_trial_id"].values))[0]
    n_inds = np.minimum(len(indices), 50000)
    subsample_indices = np.linspace(0, len(indices) - 1, num=n_inds, dtype=int)
    mean_frame = np.mean(h5f["frames"][indices[subsample_indices]], axis=0)
# --- end verbatim ---

# Pre-scale: the compute path divides by 255, the np.load path does not.
scaled = (mean_frame / 255.0).astype(np.float32)
np.save(OUT, scaled)

print(f"  frames_used={n_inds} of {len(indices)} valid  mean_sec={time.time()-t:.0f}",
      flush=True)
print(f"  shape={scaled.shape} dtype={scaled.dtype} "
      f"min={scaled.min():.4f} max={scaled.max():.4f} mean={scaled.mean():.4f}",
      flush=True)
if not (0.0 <= scaled.min() and scaled.max() <= 1.0):
    print("  WARNING: outside [0,1] -- scaling may be wrong", flush=True)
'''


USER_DATA_TEMPLATE = r'''#!/bin/bash
# Module-level, unindented: the shebang must be at byte 0 or the kernel falls
# back to /bin/sh and bash-only syntax dies silently.
LOG=/var/log/meanframes.log
exec >> $LOG 2>&1

S3="s3://@@BUCKET@@"
PREFIX="@@PREFIX@@"
OUT_PREFIX="@@OUT_PREFIX@@"
BOOT_TS=$(date +%s)

mark() { echo "MEAN: $*"; echo "MEAN: $*" > /dev/console; }

finish() {
    RC=$?
    mark "job exit=$RC total_min=$(( ($(date +%s) - BOOT_TS) / 60 ))"
    aws s3 cp $LOG "$S3/smoke/meanframes-@@RUN_ID@@/log.txt" --region @@REGION@@ || true
    tail -40 $LOG > /dev/console 2>/dev/null || true
    shutdown -h now
}
trap finish EXIT

mark "STARTED @@RUN_ID@@ $(date -Is)"

# --- scratch space ---
DATA_DIR=""
for i in $(seq 1 12); do
    if mountpoint -q /opt/dlami/nvme; then DATA_DIR=/opt/dlami/nvme; break; fi
    sleep 5
done
[ -z "$DATA_DIR" ] && DATA_DIR=/home/ubuntu/data && mkdir -p $DATA_DIR
mark "data_dir=$DATA_DIR free=$(df -h $DATA_DIR | awk 'NR==2 {print $4}')"

# --- code ---
cd /home/ubuntu && rm -rf VideoAnalysisTools
git clone --depth 1 --branch @@BRANCH@@ @@REPO@@ || { mark "clone FAILED"; exit 1; }
mark "clone=OK sha=$(cd VideoAnalysisTools && git rev-parse --short HEAD)"
cd /home/ubuntu/VideoAnalysisTools

echo "@@WORKER_B64@@" | base64 -d > /tmp/worker.py

SESSIONS="@@SESSIONS@@"
TOTAL=$(echo $SESSIONS | wc -w)
mark "processing $TOTAL sessions sequentially"

N=0
OK=0
for SESSION in $SESSIONS; do
    N=$((N + 1))
    ANIMAL=$(echo $SESSION | cut -d_ -f1)
    OUT_KEY="${OUT_PREFIX}${SESSION}_mean_frame.npy"
    mark "[$N/$TOTAL] $SESSION"

    # Idempotent: skip anything already done, so a rerun is cheap.
    if aws s3 ls "$S3/$OUT_KEY" --region @@REGION@@ > /dev/null 2>&1; then
        mark "  already exists, skipping"
        OK=$((OK + 1))
        continue
    fi

    T0=$(date +%s)
    aws s3 cp "$S3/${PREFIX}${SESSION}@@SUFFIX@@" "$DATA_DIR/" --region @@REGION@@ \
        || { mark "  h5 download FAILED"; continue; }
    aws s3 cp "$S3/bpod_files/$ANIMAL/${SESSION}.bpod.npy" "$DATA_DIR/" --region @@REGION@@ \
        || { mark "  bpod download FAILED"; rm -f "$DATA_DIR/${SESSION}@@SUFFIX@@"; continue; }
    mark "  staged in $(( $(date +%s) - T0 ))s"

    if timeout 30m @@PYTHON@@ -u /tmp/worker.py \
        "$DATA_DIR/${SESSION}@@SUFFIX@@" \
        "$DATA_DIR/${SESSION}.bpod.npy" \
        "$ANIMAL" "$SESSION" "$DATA_DIR/${SESSION}_mean_frame.npy"; then
        if aws s3 cp "$DATA_DIR/${SESSION}_mean_frame.npy" "$S3/$OUT_KEY" \
               --region @@REGION@@; then
            OK=$((OK + 1))
            mark "  uploaded, ${SESSION} done in $(( $(date +%s) - T0 ))s"
        else
            mark "  UPLOAD FAILED -- does the role allow PutObject on mean_frames/*?"
        fi
    else
        mark "  worker FAILED rc=$?"
    fi

    # Free the 14 GB before the next session.
    rm -f "$DATA_DIR/${SESSION}@@SUFFIX@@" "$DATA_DIR/${SESSION}.bpod.npy" \
          "$DATA_DIR/${SESSION}_mean_frame.npy"
done

mark "FINISHED $OK/$TOTAL succeeded"
'''


def list_sessions() -> list:
    """Session list comes from the h5 files, which are the master list."""
    s3 = boto3.client("s3", region_name=REGION)
    sessions, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": PREFIX}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            key = o["Key"]
            if key.endswith(SUFFIX):
                sessions.append(key[len(PREFIX):-len(SUFFIX)])
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return sorted(sessions)


def build_user_data(run_id: str, sessions: list) -> str:
    ud = USER_DATA_TEMPLATE
    for token, value in [
        ("@@RUN_ID@@", run_id),
        ("@@BUCKET@@", BUCKET),
        ("@@REGION@@", REGION),
        ("@@PREFIX@@", PREFIX),
        ("@@OUT_PREFIX@@", OUT_PREFIX),
        ("@@SUFFIX@@", SUFFIX),
        ("@@BRANCH@@", BRANCH),
        ("@@REPO@@", REPO),
        ("@@PYTHON@@", PYTHON),
        ("@@SESSIONS@@", " ".join(sessions)),
        ("@@WORKER_B64@@", base64.b64encode(WORKER_PY.encode()).decode()),
    ]:
        ud = ud.replace(token, value)
    assert ud.startswith("#!/bin/bash\n"), "shebang must be at byte 0"
    assert "@@" not in ud, "unsubstituted placeholder"
    assert ">(" not in ud, "no process substitution"
    assert len(ud) < 16384, f"user-data too large: {len(ud)}"
    return ud


def verify() -> None:
    s3 = boto3.client("s3", region_name=REGION)
    sessions = list_sessions()
    done = missing = 0
    for s in sessions:
        try:
            o = s3.head_object(Bucket=BUCKET, Key=OUT_PREFIX + s + "_mean_frame.npy")
            print("  OK      " + s + "  (" + str(o["ContentLength"]) + " B)")
            done += 1
        except Exception:
            print("  MISSING " + s)
            missing += 1
    print("\n" + str(done) + " done, " + str(missing) + " missing")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default=INSTANCE_TYPE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()

    if a.verify:
        verify()
        return

    sessions = list_sessions()
    print("found " + str(len(sessions)) + " sessions")
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ud = build_user_data(run_id, sessions)

    if a.dry_run:
        print(ud)
        print("\n# " + str(len(ud)) + " bytes (limit 16384)", file=sys.stderr)
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=a.instance_type,
        MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": PROFILE},
        UserData=ud,
        InstanceInitiatedShutdownBehavior="terminate",
        MetadataOptions={"HttpTokens": "optional"},
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": "meanframes-" + run_id},
                {"Key": "Project", "Value": "video-autoencoder"},
            ],
        }],
    )
    iid = r["Instances"][0]["InstanceId"]
    print("launched " + iid + " (" + a.instance_type + ")")
    print("\nwatch the serial console for 'MEAN:' lines; ~75 min for "
          + str(len(sessions)) + " sessions")
    print("\nafterwards:  python " + sys.argv[0] + " --verify")
    print("\nrunaway guard: terminate manually if still up after "
          + str(JOB_TIMEOUT_HOURS) + "h")
    print("  aws ec2 terminate-instances --instance-ids " + iid + " --region " + REGION)


if __name__ == "__main__":
    main()
