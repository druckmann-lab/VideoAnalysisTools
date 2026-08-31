#!/usr/bin/env python3
"""
Launch per-session autoencoder training on disposable EC2 instances.

Built for one session now; the same code runs the full sweep by passing more
sessions. Nothing about it is single-session specific. There is no concurrency
cap -- every session gets its own instance immediately, which is fine because
they are independent and 20 x g5.2xlarge is 160 of the 768 vCPU G-instance
quota. A session whose instance cannot be launched is reported at the end with
a retry command rather than aborting the rest.

  python launch_training.py --sessions kd104_twNew_20221124_104921
  python launch_training.py --all          # --max-concurrent is NOT IMPLEMENTED:
                                          # every session launches at once
  python launch_training.py --status <run_id>

  # scheduler A/B: two arms differing only in training.T_mult
  python launch_training.py --sessions kd104_twNew_20221124_104921 --env aws_batch
  python launch_training.py --sessions kd104_twNew_20221124_104921 --env aws_batch_fastcycle

Established by the smoke tests, so not re-litigated here:
  awscli 2.34.63 preinstalled; /opt/dlami/nvme mounts in ~5s with 412G;
  root volume 145G; staging a 13.9 GB h5 takes ~68s.

Instance sizing (measured, bench-20260827-134351 + the shared-tensor change):
  g5.2xlarge. The 32.7 GB RSS that once forced a 4xlarge predates sharing the
  frame tensor between splits; measured peak is now 19.6 GB, and once
  GpuTensorLoader releases the host tensor steady state is under 10 GB. Same
  A10G and same 23 GB of VRAM as the 4xlarge, for ~25% less. The 8 vCPUs are
  ample because the GPU-resident loader runs no worker processes at all.

What each instance does:
  1. clone the repo at a PINNED sha (not a branch tip, so all instances in a
     sweep run identical code and every checkpoint is reconstructible)
  2. stage h5 + bpod + mean_frame onto NVMe
  3. train under `timeout`, unbuffered, tee'd to a log
  4. background monitor: samples GPU/RAM every 60s, syncs checkpoints, log and
     status to S3 every 5 min
  5. on ANY exit path, final sync then `shutdown -h now` (instance terminates)

Known gaps, deliberately not handled:
  - VideoTrainer has no resume. A crash at hour 10 loses the run; the periodic
    sync means you keep the checkpoints up to that point. This is also what
    keeps the sweep on on-demand rather than spot.
  - Two sessions per GPU would roughly halve cost again (~9 GB of 23 used per
    session), but needs CUDA MPS to be worth anything -- without it the two
    processes time-slice instead of overlapping. Deliberately deferred.
"""

import argparse
import datetime
import sys

# Shared AWS plumbing. Imported by name (not `common.x`) so the names stay
# module-globals here -- tests monkeypatch e.g. launch_training.fetch_at_sha,
# and preflight must resolve the patched one.
import boto3  # noqa: F401  (tests patch launch_training.boto3.client)
from aws_launch_common import (BPOD_PREFIX, BRANCH, BUCKET, GH_REPO, H5_PREFIX,
                               H5_SUFFIX, MEAN_PREFIX, PYTHON, REGION, REPO_URL,
                               RUNS_PREFIX, fetch_at_sha, launch_instances,
                               missing_session_inputs, print_failure_summary,
                               resolve_config_chain, resolve_sha,
                               validate_user_data)

INSTANCE_TYPE = "g5.2xlarge"
TRAIN_SCRIPT = "scripts/train_single_session_autoencoder_st.py"
ENV_NAME = "aws_batch"          # checkpoint_dir=/opt/dlami/nvme/checkpoints/

# epochs=1500 at 5.43 s/epoch is ~2.3h expected. This is a runaway guard, not a
# schedule -- do not set it near the expected runtime or a slow-but-healthy run
# gets killed. If bf16 or cudnn.benchmark ever regress to the old 9.28 s/epoch, a
# guard set close to 2.3h would silently truncate every run in the sweep while
# still looking successful. Nothing is billed for headroom that is never used.
TIMEOUT = "12h"


USER_DATA_TEMPLATE = r'''#!/bin/bash
# Module-level and unindented: a shebang not at byte 0 means the kernel falls
# back to /bin/sh and bash-only syntax dies with output nowhere visible.
LOG=/var/log/wrapper.log
exec >> $LOG 2>&1

SESSION="@@SESSION@@"
ANIMAL="@@ANIMAL@@"
SHA="@@SHA@@"
S3="s3://@@BUCKET@@"
S3_RUN="$S3/@@RUNS_PREFIX@@@@RUN_ID@@/$SESSION"
TRAIN_LOG=/var/log/train.log
GPU_LOG=/var/log/gpu.log
STATUS=/tmp/status.txt
BOOT_TS=$(date +%s)
PHASE="boot"

mark() { echo "RUN[$SESSION]: $*"; echo "RUN[$SESSION]: $*" > /dev/console; }

write_status() {
    {
        echo "session=$SESSION"
        echo "animal=$ANIMAL"
        echo "sha=$SHA"
        echo "run_id=@@RUN_ID@@"
        echo "instance_id=$IID"
        echo "instance_type=@@INSTANCE_TYPE@@"
        echo "phase=$PHASE"
        echo "elapsed_min=$(( ($(date +%s) - BOOT_TS) / 60 ))"
        echo "last_epoch=$(grep -oE 'Epoch [0-9]+' $TRAIN_LOG 2>/dev/null | tail -1)"
        echo "updated=$(date -Is)"
    } > $STATUS
    aws s3 cp $STATUS "$S3_RUN/status.txt" --region @@REGION@@ --only-show-errors || true
}

sync_outputs() {
    aws s3 sync /opt/dlami/nvme/checkpoints/ "$S3_RUN/checkpoints/" \
        --region @@REGION@@ --only-show-errors || true
    aws s3 cp $TRAIN_LOG "$S3_RUN/train.log" --region @@REGION@@ --only-show-errors || true
    aws s3 cp $GPU_LOG   "$S3_RUN/gpu.log"   --region @@REGION@@ --only-show-errors || true
    aws s3 cp $LOG       "$S3_RUN/wrapper.log" --region @@REGION@@ --only-show-errors || true
}

# Always sync and always terminate, on every exit path including SIGTERM.
finish() {
    RC=$?
    PHASE="finished_rc$RC"
    [ -n "$MON_PID" ] && kill $MON_PID 2>/dev/null
    mark "finishing rc=$RC after $(( ($(date +%s) - BOOT_TS) / 60 )) min"
    echo "wrapper_exit=$RC" >> $STATUS
    sync_outputs
    write_status
    tail -30 $LOG > /dev/console 2>/dev/null || true
    shutdown -h now
}
trap finish EXIT

TOKEN=$(curl -sX PUT --max-time 5 http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
IID=$(curl -s --max-time 5 -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)
mark "STARTED sha=$SHA instance=$IID"
touch $TRAIN_LOG $GPU_LOG
PHASE="staging"; write_status

# --- scratch space -----------------------------------------------------
DATA_DIR=""
for i in $(seq 1 12); do
    if mountpoint -q /opt/dlami/nvme; then DATA_DIR=/opt/dlami/nvme; break; fi
    sleep 5
done
if [ -z "$DATA_DIR" ]; then
    mark "FATAL /opt/dlami/nvme never mounted"
    exit 1
fi
mkdir -p /opt/dlami/nvme/checkpoints

# --- code at a pinned sha ---------------------------------------------
cd /home/ubuntu && rm -rf @@GH_REPO@@
git clone --quiet @@REPO_URL@@ || { mark "FATAL clone failed"; exit 1; }
cd @@GH_REPO@@
git checkout --quiet "$SHA" || { mark "FATAL checkout $SHA failed"; exit 1; }
mark "code at $(git rev-parse --short HEAD)"

# --- stage inputs ------------------------------------------------------
T0=$(date +%s)
aws s3 cp "$S3/@@H5_PREFIX@@${SESSION}@@H5_SUFFIX@@" "$DATA_DIR/" \
    --region @@REGION@@ --only-show-errors || { mark "FATAL h5 download"; exit 1; }
aws s3 cp "$S3/@@BPOD_PREFIX@@$ANIMAL/${SESSION}.bpod.npy" "$DATA_DIR/" \
    --region @@REGION@@ --only-show-errors || { mark "FATAL bpod download"; exit 1; }
aws s3 cp "$S3/@@MEAN_PREFIX@@${SESSION}_mean_frame.npy" "$DATA_DIR/" \
    --region @@REGION@@ --only-show-errors || { mark "FATAL mean_frame download"; exit 1; }
mark "staged in $(( $(date +%s) - T0 ))s"

# --- provenance --------------------------------------------------------
cat > /tmp/meta.json <<META
{
  "session": "$SESSION",
  "animal": "$ANIMAL",
  "sha": "$SHA",
  "env": "@@ENV_NAME@@",
  "run_id": "@@RUN_ID@@",
  "instance_id": "$IID",
  "instance_type": "@@INSTANCE_TYPE@@",
  "started": "$(date -Is)",
  "timeout": "@@TIMEOUT@@"
}
META
aws s3 cp /tmp/meta.json "$S3_RUN/meta.json" --region @@REGION@@ --only-show-errors || true

# --- background monitor -----------------------------------------------
# Samples every 60s; syncs every 5 min. Also answers the open GPU-memory
# question -- gpu.log records utilisation and MiB used throughout the run.
monitor() {
    i=0
    while true; do
        {
            echo "-- $(date -Is) --"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                --format=csv,noheader 2>&1
            free -g | awk '/^Mem:/ {print "ram_used_gb="$3" avail_gb="$7}'
        } >> $GPU_LOG
        i=$((i + 1))
        if [ $((i % 5)) -eq 0 ]; then
            sync_outputs
            write_status
        fi
        if [ $((i % 10)) -eq 0 ]; then
            # Epoch rate from the timestamps: first and last epoch line.
            RATE=$(grep -E 'Epoch [0-9]+/' $TRAIN_LOG | awk '
                NR==1 { split($1,a,"T"); split(a[2],b,":");
                        t0=b[1]*3600+b[2]*60+b[3]; e0=$2 }
                      { split($1,a,"T"); split(a[2],b,":");
                        t1=b[1]*3600+b[2]*60+b[3]; e1=$2; n=NR }
                END   { if (n>1 && t1>t0) printf "%.1fs/epoch over %d epochs", (t1-t0)/(n-1), n }')
            mark "$(grep -oE 'Epoch [0-9]+/[0-9]+' $TRAIN_LOG | tail -1) | $RATE | \
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"
        fi
        sleep 60
    done
}
monitor & MON_PID=$!

# --- train -------------------------------------------------------------
PHASE="training"; write_status
mark "training started, timeout @@TIMEOUT@@"
timeout @@TIMEOUT@@ @@PYTHON@@ -u @@TRAIN_SCRIPT@@ \
    --env @@ENV_NAME@@ \
    --animal "$ANIMAL" \
    --session "$SESSION" \
    --h5_path "$DATA_DIR/${SESSION}@@H5_SUFFIX@@" \
    --bpod_path "$DATA_DIR/${SESSION}.bpod.npy" \
    --mean_frame_path "$DATA_DIR/${SESSION}_mean_frame.npy" \
    2>&1 | awk '{ printf "%s %s\n", strftime("%Y-%m-%dT%H:%M:%S"), $0; fflush() }' \
    | tee -a $TRAIN_LOG
# PIPESTATUS[0] is still the `timeout` command, not awk or tee.
TRAIN_RC=${PIPESTATUS[0]}

# 124 is `timeout`'s signal that it killed the command.
if [ $TRAIN_RC -eq 0 ]; then
    mark "training COMPLETED"
elif [ $TRAIN_RC -eq 124 ]; then
    mark "training TIMED OUT at @@TIMEOUT@@"
else
    mark "training FAILED rc=$TRAIN_RC"
    echo "oom_kills=$(dmesg 2>/dev/null | grep -ci 'out of memory' || echo 0)" >> $TRAIN_LOG
fi
echo "train_rc=$TRAIN_RC" >> $STATUS
exit $TRAIN_RC
'''


def preflight(sessions: list, sha: str, env_name: str) -> None:
    """Fail locally and for free rather than on twenty instances."""
    print(f"pre-flight against sha {sha[:7]}")

    # 1. Is the pushed code the code you think it is? The instance clones from
    #    GitHub, so uncommitted local edits do not exist as far as it is
    #    concerned -- and --mean_frame_path on an older script is argparse exit 2.
    try:
        src = fetch_at_sha(sha, TRAIN_SCRIPT)
    except Exception as e:
        sys.exit(f"  FAIL could not fetch {TRAIN_SCRIPT} at {sha[:7]}: {e}")
    for flag in ("--mean_frame_path", "--h5_path", "--bpod_path",
                 "--animal", "--session"):
        if flag not in src:
            sys.exit(f"  FAIL pushed {TRAIN_SCRIPT} has no {flag}\n"
                     f"       commit and push {BRANCH}, then rerun")
    print(f"  OK   pushed training script accepts all required flags")

    # 2. The env config must exist at this sha, including anything it extends.
    #    load_config only raises at runtime, by which point the instance has
    #    already staged 14 GB.
    chain = resolve_config_chain(sha, env_name, fetch=fetch_at_sha)
    print(f"  OK   env config chain present: {' -> '.join(chain)}")

    # 3. Every input object must exist before anything launches.
    missing = missing_session_inputs(sessions, need_mean_frame=True)
    if missing:
        print("  FAIL missing S3 objects:")
        for k in missing:
            print("       " + k)
        sys.exit(1)
    print(f"  OK   all inputs present for {len(sessions)} session(s)")


def list_all_sessions() -> list:
    s3 = boto3.client("s3", region_name=REGION)
    out, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": H5_PREFIX}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            if o["Key"].endswith(H5_SUFFIX):
                out.append(o["Key"][len(H5_PREFIX):-len(H5_SUFFIX)])
        if not r.get("IsTruncated"):
            break
        token = r["NextContinuationToken"]
    return sorted(out)


def build_user_data(session: str, sha: str, run_id: str, timeout: str,
                    env_name: str) -> str:
    ud = USER_DATA_TEMPLATE
    for tok, val in [
        ("@@SESSION@@", session),
        ("@@ANIMAL@@", session.split("_")[0]),
        ("@@SHA@@", sha),
        ("@@RUN_ID@@", run_id),
        ("@@BUCKET@@", BUCKET),
        ("@@REGION@@", REGION),
        ("@@RUNS_PREFIX@@", RUNS_PREFIX),
        ("@@H5_PREFIX@@", H5_PREFIX),
        ("@@H5_SUFFIX@@", H5_SUFFIX),
        ("@@BPOD_PREFIX@@", BPOD_PREFIX),
        ("@@MEAN_PREFIX@@", MEAN_PREFIX),
        ("@@REPO_URL@@", REPO_URL),
        ("@@GH_REPO@@", GH_REPO),
        ("@@TRAIN_SCRIPT@@", TRAIN_SCRIPT),
        ("@@ENV_NAME@@", env_name),
        ("@@PYTHON@@", PYTHON),
        ("@@INSTANCE_TYPE@@", INSTANCE_TYPE),
        ("@@TIMEOUT@@", timeout),
    ]:
        ud = ud.replace(tok, val)
    return validate_user_data(ud)


def launch(sessions: list, run_id: str, sha: str, itype: str,
           timeout: str, env_name: str, dry_run: bool) -> tuple:
    """One instance per session; returns (launched, failed)."""
    specs = ((sess,
              build_user_data(sess, sha, run_id, timeout, env_name),
              [{"Key": "Name", "Value": "train-" + sess},
               {"Key": "RunId", "Value": run_id},
               {"Key": "Session", "Value": sess},
               # So an A/B's two arms are distinguishable in the console.
               {"Key": "Env", "Value": env_name}])
             for sess in sessions)
    return launch_instances(specs, itype, dry_run)


def status(run_id: str) -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    print(f"run {run_id}\n")
    r = ec2.describe_instances(Filters=[
        {"Name": "tag:RunId", "Values": [run_id]}])
    live = 0
    for res in r["Reservations"]:
        for i in res["Instances"]:
            st = i["State"]["Name"]
            sess = next((t["Value"] for t in i.get("Tags", [])
                         if t["Key"] == "Session"), "?")
            if st in ("pending", "running"):
                live += 1
            print(f"  {i['InstanceId']}  {st:12s}  {sess}")
    print(f"\n{live} instance(s) still billing\n")

    pref = RUNS_PREFIX + run_id + "/"
    keys = s3.list_objects_v2(Bucket=BUCKET, Prefix=pref).get("Contents", [])
    for o in sorted(k["Key"] for k in keys):
        if o.endswith("status.txt"):
            body = s3.get_object(Bucket=BUCKET, Key=o)["Body"].read().decode()
            d = dict(l.split("=", 1) for l in body.strip().splitlines()
                     if "=" in l)
            print(f"  {d.get('session','?'):32s} {d.get('phase','?'):18s} "
                  f"{d.get('last_epoch','-'):16s} {d.get('elapsed_min','?')}min")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="+", help="session ids to train")
    p.add_argument("--all", action="store_true", help="every session in S3")
    p.add_argument("--sha", help="pin this commit (default: tip of %s)" % BRANCH)
    p.add_argument("--instance-type", default=INSTANCE_TYPE)
    p.add_argument("--run-id",
                   help="reuse an existing run id, for retrying failed sessions "
                        "so all outputs stay under one prefix")
    p.add_argument("--env", default=ENV_NAME,
                   help="config env: configs/<env>_config.json (default: %s)" % ENV_NAME)
    p.add_argument("--timeout", default=TIMEOUT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", metavar="RUN_ID")
    a = p.parse_args()

    if a.status:
        status(a.status)
        return
    if not (a.sessions or a.all):
        p.error("need --sessions, --all, or --status")

    sessions = list_all_sessions() if a.all else a.sessions
    sha = a.sha or resolve_sha(BRANCH)
    preflight(sessions, sha, a.env)

    run_id = a.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\nrun_id={run_id}  {len(sessions)} session(s) on {a.instance_type}"
          f"  env={a.env}")
    launched, failed = launch(sessions, run_id, sha, a.instance_type, a.timeout,
                              a.env, a.dry_run)
    if a.dry_run:
        return

    print(f"\n{len(launched)}/{len(sessions)} launched")

    print(f"\nwatch:   python {sys.argv[0]} --status {run_id}")
    print(f"outputs: s3://{BUCKET}/{RUNS_PREFIX}{run_id}/")
    print(f"kill:    aws ec2 terminate-instances --region {REGION} --instance-ids "
          f"$(aws ec2 describe-instances --region {REGION} "
          f"--filters Name=tag:RunId,Values={run_id} "
          f"Name=instance-state-name,Values=running,pending "
          f"--query 'Reservations[].Instances[].InstanceId' --output text)")

    if failed:
        retry = (f"  python {sys.argv[0]} --sha {sha} --run-id {run_id} \\\n"
                 f"      --env {a.env} --instance-type {a.instance_type} "
                 f"--timeout {a.timeout} \\\n"
                 f"      --sessions " + " ".join(s for s, _, _ in failed))
        print_failure_summary(failed, retry, noun="session")
        sys.exit(1)


if __name__ == "__main__":
    main()