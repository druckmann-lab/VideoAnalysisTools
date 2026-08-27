#!/usr/bin/env python3
"""
Stage 2: full pre-flight probe for the per-session autoencoder sweep.

Builds on what the micro test established:
  awscli 2.34.63 preinstalled, instance profile works, S3 read+write OK,
  /opt/dlami/nvme mounts in ~5s with 412G, A10G with 23GB.

So this script skips those and answers the remaining questions:
  - how long does a real ~13 GB h5 download take?
  - do the repo imports resolve from a fresh clone?
  - PEAK RSS when both datasets are built -> g5.2xlarge (32 GiB) or
    g5.4xlarge (64 GiB)?  H5VideoDataset.__init__ loads the whole frames
    array into RAM and the training script instantiates it twice.

~15-20 min, ~$0.40.

TWO DESIGN RULES, both learned from failures:

1. NO textwrap.dedent. It strips only the COMMON leading whitespace, so a
   single column-0 line inside an embedded snippet leaves the shebang
   indented. A shebang not at byte 0 means the kernel picks /bin/sh (dash),
   and bash-only syntax dies instantly with output nowhere visible.
   The template below is module-level and unindented. Verified by assertion.

2. NO f-string templating of shell code. Doubling every brace for bash and
   awk is how the malformed awk line got in. Placeholders are @@TOKENS@@
   substituted with str.replace, so the shell text is exactly what you read.

Usage:
  python smoke_test.py --dry-run
  python smoke_test.py
  python smoke_test.py --fetch <run_id>
"""

import argparse
import base64
import datetime
import sys

import boto3

REGION = "us-west-2"
BUCKET = "balint-video-autoencoder-data-233060639700-us-west-2-an"
AMI_ID = "ami-0bcccc2c1e9b9f874"
INSTANCE_TYPE = "g5.4xlarge"   # 64 GiB; g5.2xlarge (32) OOMed at rc=137
PYTHON = "/home/ubuntu/ml_env/bin/python"
REPO = "https://github.com/druckmann-lab/VideoAnalysisTools.git"
BRANCH = "balint-dev"
PROFILE = "VideoAutoencoderTrainingRole"

# Both configs/aws_config.json and configs/aws_batch_config.json exist on
# balint-dev, and batch_size/epochs/checkpoint_interval are identical (2048/6000/100).
# aws_batch is chosen because:
#   - its checkpoint_dir is /opt/dlami/nvme/checkpoints/, whereas aws_config
#     points inside the repo dir on the root volume, which this script rm -rf's
#   - its loss_mask_exclude_regions is null, whereas aws_config's single region
#     has top=72 > bottom=65, an empty slice that masks nothing (a real bug --
#     probably meant top=65, bottom=72)
# Neither affects the RSS measurement.
ENV_NAME = "aws_batch"

# Largest session (12.9 GiB) so staging time and RSS are worst-case.
SESSION = "kd104_twNew_20221124_104921"
ANIMAL = "kd104"

PROBE_TIMEOUT = "25m"


# --------------------------------------------------------------------------
# Runs on the instance. Reports peak RSS after each dataset is built.
# --------------------------------------------------------------------------
PROBE_PY = r'''
import json, os, resource, sys, time

REPO = "/home/ubuntu/VideoAnalysisTools"
sys.path.insert(0, os.path.join(REPO, "src"))

H5, BPOD, ANIMAL, SESSION, ENV = sys.argv[1:6]


def update(d, u):
    for k, v in u.items():
        d[k] = update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


def rss_gb():
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return -1.0


cfg = json.load(open(f"{REPO}/configs/ae_config.json"))
env_path = f"{REPO}/configs/{ENV}_config.json"
if not os.path.exists(env_path):
    print(f"FATAL missing env config {env_path}")
    print("available:", os.listdir(f"{REPO}/configs"))
    sys.exit(2)
cfg = update(cfg, json.load(open(env_path)))

# Exactly what the launcher will pass via --h5_path / --bpod_path.
cfg["metadata_config"]["bpod_path"] = BPOD
cfg["metadata_config"]["h5_path"] = H5
cfg["dataset"]["dataset_path"] = H5

from behavioral_autoencoder.dataset_st import (
    SessionMetadataHandler, H5VideoDataset, build_loss_mask)

t = time.time()
mh = SessionMetadataHandler(config=cfg["metadata_config"], mode="local",
                            animal=ANIMAL, session=SESSION)
df = mh.process_all()
print(f"metadata=OK trials={len(df)} sec={time.time()-t:.0f} rss_gb={rss_gb():.1f} avail_gb={avail_gb():.1f}")

t = time.time()
train = H5VideoDataset(H5, df, split="train", config=cfg["dataset"])
print(f"train_ds=OK samples={len(train)} sec={time.time()-t:.0f} rss_gb={rss_gb():.1f} avail_gb={avail_gb():.1f}")

t = time.time()
val = H5VideoDataset(H5, df, split="test", config=cfg["dataset"])
print(f"val_ds=OK samples={len(val)} sec={time.time()-t:.0f} rss_gb={rss_gb():.1f} avail_gb={avail_gb():.1f}")

nbytes = train.frames.element_size() * train.frames.nelement()
print(f"frames dtype={train.frames.dtype} shape={tuple(train.frames.shape)}")
print(f"frames_gb_per_copy={nbytes/1e9:.1f}")
print(f"batch_size={cfg['training']['batch_size']} epochs={cfg['training']['epochs']} "
      f"ckpt_interval={cfg['training']['checkpoint_interval']}")
print(f"checkpoint_dir={cfg['training']['checkpoint_dir']}")
print(f"PEAK_RSS_GB={rss_gb():.1f}")
'''


# --------------------------------------------------------------------------
# Module level and unindented so the shebang lands at byte 0.
# @@TOKENS@@ are substituted with str.replace -- no brace escaping anywhere.
# --------------------------------------------------------------------------
USER_DATA_TEMPLATE = r'''#!/bin/bash
# No process substitution, no `set -e`. Every probe runs; the trap always fires.
LOG=/var/log/smoke.log
REPORT=/tmp/report.txt
exec >> $LOG 2>&1

S3_PREFIX="s3://@@BUCKET@@/smoke/@@RUN_ID@@"
BOOT_TS=$(date +%s)

# mark(): one-line status to the serial console (readable in the browser with
# no dependencies) AND to the log. note(): detail, report file only.
mark() { echo "SMOKE: $*"; echo "SMOKE: $*" > /dev/console; }
note() { echo "$*" >> $REPORT; }

finish() {
    RC=$?
    note "exit_status=$RC"
    note "total_seconds=$(( $(date +%s) - BOOT_TS ))"
    mark "uploading and shutting down (exit=$RC)"
    aws s3 cp $REPORT "$S3_PREFIX/report.txt" --region @@REGION@@ || true
    aws s3 cp $LOG    "$S3_PREFIX/smoke.log"  --region @@REGION@@ || true
    aws s3 cp /var/log/cloud-init-output.log "$S3_PREFIX/cloud-init.log" \
        --region @@REGION@@ || true
    # If S3 failed for any reason, the console still gets everything.
    tail -60 $LOG > /dev/console 2>/dev/null || true
    shutdown -h now
}
trap finish EXIT

mark "STARTED @@RUN_ID@@ $(date -Is)"
note "run_id=@@RUN_ID@@"
TOKEN=$(curl -sX PUT --max-time 5 http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 600")
note "instance_id=$(curl -s --max-time 5 -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)"
note "instance_type=@@INSTANCE_TYPE@@"

echo "alive $(date -Is)" > /tmp/alive.txt
aws s3 cp /tmp/alive.txt "$S3_PREFIX/alive.txt" --region @@REGION@@ && mark "s3=OK"

# --- 1. Inventory -------------------------------------------------------
note "--- free -m ---";  free -m  >> $REPORT
note "--- df -h ---";    df -h    >> $REPORT
nvidia-smi >> $REPORT 2>&1
note "ram_total_gb=$(free -g | awk '/^Mem:/ {print $2}')"
mark "ram=$(free -g | awk '/^Mem:/ {print $2}')GiB gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# --- 2. NVMe (mounts in ~5s per the micro test, but never assume) -------
DATA_DIR=""
for i in $(seq 1 12); do
    if mountpoint -q /opt/dlami/nvme; then
        DATA_DIR=/opt/dlami/nvme
        note "nvme_mount=OK size=$(df -h /opt/dlami/nvme | awk 'NR==2 {print $2}')"
        break
    fi
    sleep 5
done
if [ -z "$DATA_DIR" ]; then
    DATA_DIR=/home/ubuntu/data
    mkdir -p $DATA_DIR
    note "nvme_mount=ABSENT falling back to root volume"
fi
mark "data_dir=$DATA_DIR"

# --- 3. Fresh clone + import check --------------------------------------
cd /home/ubuntu
rm -rf VideoAnalysisTools
if git clone --depth 1 --branch @@BRANCH@@ @@REPO@@ 2>&1; then
    cd VideoAnalysisTools
    SHA=$(git rev-parse --short HEAD)
    note "clone=OK sha=$SHA"
    note "configs=$(ls configs/ | tr '\n' ' ')"
    mark "clone=OK sha=$SHA"
else
    note "clone=FAIL"
    mark "clone=FAIL"
fi

# --- 4. Real staging timing ---------------------------------------------
mark "downloading h5 (~13 GB)"
T0=$(date +%s)
if aws s3 cp "s3://@@BUCKET@@/preprocessed_videos/@@SESSION@@_side_crop.h5" \
       "$DATA_DIR/" --region @@REGION@@ 2>&1; then
    SECS=$(( $(date +%s) - T0 ))
    note "h5_download=OK seconds=$SECS"
    mark "h5 staged in ${SECS}s"
else
    note "h5_download=FAIL"
    mark "h5 download FAILED"
fi
aws s3 cp "s3://@@BUCKET@@/bpod_files/@@ANIMAL@@/@@SESSION@@.bpod.npy" \
    "$DATA_DIR/" --region @@REGION@@ 2>&1 \
    && note "bpod_download=OK" || note "bpod_download=FAIL"
note "--- data_dir ---"; ls -la $DATA_DIR >> $REPORT

# --- 5. Source prefixes must be read-only -------------------------------
echo x > /tmp/x
if aws s3 cp /tmp/x "s3://@@BUCKET@@/preprocessed_videos/_perm_probe" \
       --region @@REGION@@ 2>/dev/null; then
    note "source_prefix_writable=YES -- tighten the policy"
    mark "WARNING source prefix is writable"
else
    note "source_prefix_writable=no (correct)"
fi

# --- 6. Dataset construction: the memory answer -------------------------
mark "building datasets (this is the memory test)"
echo "@@PROBE_B64@@" | base64 -d > /tmp/probe.py
note "--- dataset probe, @@ENV_NAME@@ config ---"
timeout @@PROBE_TIMEOUT@@ @@PYTHON@@ -u /tmp/probe.py \
    "$DATA_DIR/@@SESSION@@_side_crop.h5" \
    "$DATA_DIR/@@SESSION@@.bpod.npy" \
    "@@ANIMAL@@" "@@SESSION@@" "@@ENV_NAME@@" 2>&1 | tee -a $REPORT /dev/console
PROBE_RC=${PIPESTATUS[0]}
note "dataset_probe_rc=$PROBE_RC"
# A kernel OOM kill leaves no Python traceback, so check dmesg too.
note "oom_kills=$(dmesg 2>/dev/null | grep -ci 'out of memory' || echo 0)"
note "mem_after=$(free -m | awk '/^Mem:/ {printf "%.1f GiB used", $3/1024}')"
mark "dataset probe rc=$PROBE_RC peak=$(grep -o 'PEAK_RSS_GB=.*' $REPORT | tail -1)"

note "PROBE COMPLETE"
mark "DONE"
'''


def build_user_data(run_id: str) -> str:
    probe_b64 = base64.b64encode(PROBE_PY.encode()).decode()
    ud = USER_DATA_TEMPLATE
    for token, value in [
        ("@@RUN_ID@@", run_id),
        ("@@BUCKET@@", BUCKET),
        ("@@REGION@@", REGION),
        ("@@INSTANCE_TYPE@@", INSTANCE_TYPE),
        ("@@BRANCH@@", BRANCH),
        ("@@REPO@@", REPO),
        ("@@SESSION@@", SESSION),
        ("@@ANIMAL@@", ANIMAL),
        ("@@ENV_NAME@@", ENV_NAME),
        ("@@PYTHON@@", PYTHON),
        ("@@PROBE_TIMEOUT@@", PROBE_TIMEOUT),
        ("@@PROBE_B64@@", probe_b64),
    ]:
        ud = ud.replace(token, value)

    # The bug that cost an instance launch. Never ship without these.
    assert ud.startswith("#!/bin/bash\n"), "shebang must be at byte 0"
    assert "@@" not in ud, "unsubstituted placeholder remains"
    assert ">(" not in ud, "process substitution is bash-only; avoid it"
    assert len(ud) < 16384, f"user-data too large: {len(ud)}"
    return ud


def launch(dry_run: bool) -> None:
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ud = build_user_data(run_id)

    if dry_run:
        print(ud)
        print(f"\n# {len(ud)} bytes (limit 16384)", file=sys.stderr)
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": PROFILE},
        UserData=ud,
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": "smoke-" + run_id},
                # Required: the IAM policy only permits terminating instances
                # carrying this exact tag.
                {"Key": "Project", "Value": "video-autoencoder"},
                {"Key": "Session", "Value": SESSION},
            ],
        }],
    )
    iid = r["Instances"][0]["InstanceId"]
    print("launched " + iid + "   run_id=" + run_id)
    print("\nprogress on the serial console (browser, grep 'SMOKE:'), or:")
    print("  aws s3 ls s3://" + BUCKET + "/smoke/" + run_id + "/")
    print("\nreport in ~15-20 min:")
    print("  python " + sys.argv[0] + " --fetch " + run_id)
    print("\nif still running after 40 min:")
    print("  aws ec2 terminate-instances --instance-ids " + iid + " --region " + REGION)


def fetch(run_id: str) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    prefix = "smoke/" + run_id + "/"
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])
    if not objs:
        print("nothing at s3://" + BUCKET + "/" + prefix + " yet")
        return
    for o in objs:
        print("  " + o["Key"] + "  (" + str(o["Size"]) + " B)")
    try:
        body = s3.get_object(Bucket=BUCKET, Key=prefix + "report.txt")["Body"].read()
        print("\n" + "=" * 60 + "\n" + body.decode())
    except s3.exceptions.NoSuchKey:
        print("\nno report.txt yet -- still running, or it died before the trap")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default=INSTANCE_TYPE,
                   help="default %(default)s; g5.2xlarge OOMs on the large sessions")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fetch", metavar="RUN_ID")
    a = p.parse_args()
    if a.instance_type:
        globals()["INSTANCE_TYPE"] = a.instance_type
    if a.fetch:
        fetch(a.fetch)
    else:
        launch(a.dry_run)