#!/usr/bin/env python3
"""
Launch the epoch-time benchmark on a disposable EC2 instance.

Same skeleton as launch_training.py -- pinned sha, stage to NVMe, background
monitor, always-sync-always-terminate trap -- with three differences that follow
from this being a 15-minute measurement rather than a 6-hour run:

  1. TIMEOUT is 40m, not 16h. The benchmark is ~8 min of compute on top of ~2.5
     min of staging and h5 load. If it has not finished in 40 min something is
     wrong and you should not pay for the rest of the hour.
  2. No checkpoints. It syncs bench.json + bench.txt instead.
  3. --instance-type is a first-class knob. Comparing g5.4xlarge (A10G) against
     g6.4xlarge (L4) or a g5.2xlarge is now a $0.40 experiment, and the whole
     ladder runs on each, so you get per-instance kernel efficiency, not just a
     spec-sheet guess.

  python launch_benchmark.py --sessions kd104_twNew_20221124_104921
  python launch_benchmark.py --sessions kd104_... --instance-type g6.4xlarge
  python launch_benchmark.py --status  <run_id>
  python launch_benchmark.py --results <run_id>      # prints the table from S3

Sequencing note: run this BEFORE the next training sweep. The whole point is to
find out whether the 15 s/epoch is data or compute, and a 6-hour run started on
a guess costs ~40x what the answer costs.
"""

import argparse
import datetime
import fnmatch
import json
import os
import subprocess
import sys
import urllib.request

import boto3

REGION = "us-west-2"
BUCKET = "balint-video-autoencoder-data-233060639700-us-west-2-an"
AMI_ID = "ami-0bcccc2c1e9b9f874"
INSTANCE_TYPE = "g5.4xlarge"
PYTHON = "/home/ubuntu/ml_env/bin/python"
PROFILE = "VideoAutoencoderTrainingRole"

GH_OWNER = "druckmann-lab"
GH_REPO = "VideoAnalysisTools"
REPO_URL = f"https://github.com/{GH_OWNER}/{GH_REPO}.git"
BRANCH = "balint-dev"
BENCH_SCRIPT = "scripts/benchmark_epoch_time_st.py"

ENV_NAME = "aws_batch"          # same config the sweep uses: batch 2048, etc.
H5_PREFIX = "preprocessed_videos/"
H5_SUFFIX = "_side_crop.h5"
BPOD_PREFIX = "bpod_files/"
MEAN_PREFIX = "mean_frames/"

# MUST be a prefix the instance role can PutObject to. The role scopes writes to
# runs/*, mean_frames/* and smoke/* -- a fresh prefix like benchmarks/* gets
# AccessDenied on every sync, and because each sync ends in `|| true` it fails
# SILENTLY: the benchmark runs, the instance terminates, and nothing is saved.
# smoke/ is semantically right (this is a measurement, not a training run) and
# needs no IAM change. Override with --s3-prefix once benchmarks/* is in the role.
BENCH_PREFIX = "smoke/"

# Local copy of the instance role's inline policy, used by preflight below.
POLICY_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "instance_role_policy.json")

# Runaway guard, not a schedule. ~8 min compute + ~2.5 min setup expected.
TIMEOUT = "40m"


USER_DATA_TEMPLATE = r'''#!/bin/bash
# Shebang at byte 0 or the kernel falls back to /bin/sh and bash-only syntax
# dies with the output nowhere visible.
LOG=/var/log/wrapper.log
exec >> $LOG 2>&1

SESSION="@@SESSION@@"
ANIMAL="@@ANIMAL@@"
SHA="@@SHA@@"
S3="s3://@@BUCKET@@"
S3_RUN="$S3/@@BENCH_PREFIX@@@@RUN_ID@@/$SESSION"
BENCH_LOG=/var/log/bench.log
GPU_LOG=/var/log/gpu.log
BENCH_JSON=/tmp/bench.json
BENCH_TXT=/tmp/bench.txt
STATUS=/tmp/status.txt
BOOT_TS=$(date +%s)
PHASE="boot"

mark() { echo "BENCH[$SESSION]: $*"; echo "BENCH[$SESSION]: $*" > /dev/console; }

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
        echo "last_variant=$(grep -oE '^\[[a-z0-9_]+\]' $BENCH_LOG 2>/dev/null | tail -1)"
        echo "updated=$(date -Is)"
    } > $STATUS
    aws s3 cp $STATUS "$S3_RUN/status.txt" --region @@REGION@@ --only-show-errors || true
}

sync_outputs() {
    # bench.json is rewritten after every variant, so even a killed run leaves
    # the rungs that did complete.
    aws s3 cp $BENCH_JSON "$S3_RUN/bench.json"   --region @@REGION@@ --only-show-errors || true
    aws s3 cp $BENCH_TXT  "$S3_RUN/bench.txt"    --region @@REGION@@ --only-show-errors || true
    aws s3 cp $BENCH_LOG  "$S3_RUN/bench.log"    --region @@REGION@@ --only-show-errors || true
    aws s3 cp $GPU_LOG    "$S3_RUN/gpu.log"      --region @@REGION@@ --only-show-errors || true
    aws s3 cp $LOG        "$S3_RUN/wrapper.log"  --region @@REGION@@ --only-show-errors || true
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
    tail -60 $BENCH_LOG > /dev/console 2>/dev/null || true
    shutdown -h now
}
trap finish EXIT

TOKEN=$(curl -sX PUT --max-time 5 http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
IID=$(curl -s --max-time 5 -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)
mark "STARTED sha=$SHA instance=$IID type=@@INSTANCE_TYPE@@"
touch $BENCH_LOG $GPU_LOG

# --- S3 writability probe ----------------------------------------------
# The ONE S3 call that is not `|| true`. Every other sync tolerates failure so a
# transient blip cannot kill a run -- which also means a permanent permission
# error is swallowed and you pay for a benchmark whose results can never be
# saved. Probe once, before the 70s of staging, and die cheap if writes fail.
echo "probe $(date -Is) $IID" > /tmp/probe.txt
if ! aws s3 cp /tmp/probe.txt "$S3_RUN/probe.txt" \
        --region @@REGION@@ --only-show-errors; then
    mark "FATAL cannot PutObject to $S3_RUN"
    mark "FATAL the instance role has no write permission for this prefix;"
    mark "FATAL rerun with --s3-prefix pointing at an allowed prefix"
    exit 1
fi
mark "S3 writable at $S3_RUN"

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
  "kind": "epoch_time_benchmark",
  "session": "$SESSION",
  "animal": "$ANIMAL",
  "sha": "$SHA",
  "env": "@@ENV_NAME@@",
  "run_id": "@@RUN_ID@@",
  "instance_id": "$IID",
  "instance_type": "@@INSTANCE_TYPE@@",
  "bench_args": "@@BENCH_ARGS@@",
  "started": "$(date -Is)",
  "timeout": "@@TIMEOUT@@"
}
META
aws s3 cp /tmp/meta.json "$S3_RUN/meta.json" --region @@REGION@@ --only-show-errors || true

# --- background monitor -----------------------------------------------
# 30s sampling (not 60s) because variants only last ~40s each -- at 60s a whole
# rung could pass between samples and its GPU signature would be invisible.
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
        if [ $((i % 4)) -eq 0 ]; then
            sync_outputs
            write_status
        fi
        sleep 30
    done
}
monitor & MON_PID=$!

# --- benchmark ---------------------------------------------------------
PHASE="benchmarking"; write_status
mark "benchmark started, timeout @@TIMEOUT@@"
timeout @@TIMEOUT@@ @@PYTHON@@ -u @@BENCH_SCRIPT@@ \
    --env @@ENV_NAME@@ \
    --animal "$ANIMAL" \
    --session "$SESSION" \
    --h5_path "$DATA_DIR/${SESSION}@@H5_SUFFIX@@" \
    --bpod_path "$DATA_DIR/${SESSION}.bpod.npy" \
    --mean_frame_path "$DATA_DIR/${SESSION}_mean_frame.npy" \
    --instance-type "@@INSTANCE_TYPE@@" \
    --out $BENCH_JSON @@BENCH_ARGS@@ \
    2>&1 | awk '{ printf "%s %s\n", strftime("%Y-%m-%dT%H:%M:%S"), $0; fflush() }' \
    | tee -a $BENCH_LOG
# PIPESTATUS[0] is still `timeout`, not awk or tee.
BENCH_RC=${PIPESTATUS[0]}

if [ $BENCH_RC -eq 0 ]; then
    mark "benchmark COMPLETED"
elif [ $BENCH_RC -eq 124 ]; then
    mark "benchmark TIMED OUT at @@TIMEOUT@@ (partial bench.json still synced)"
else
    mark "benchmark FAILED rc=$BENCH_RC"
    echo "oom_kills=$(dmesg 2>/dev/null | grep -ci 'out of memory' || echo 0)" >> $BENCH_LOG
fi
echo "bench_rc=$BENCH_RC" >> $STATUS
exit $BENCH_RC
'''


def resolve_sha(ref: str) -> str:
    out = subprocess.run(["git", "ls-remote", REPO_URL, ref],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit(f"could not resolve '{ref}' on {REPO_URL}\n{out.stderr}")
    return out.stdout.split()[0]


def check_s3_write_prefix(prefix: str) -> None:
    """
    Verify the instance role can PutObject under `prefix`.

    This exists because getting it wrong is invisible and expensive: the role
    scopes writes to a few prefixes, every sync in the wrapper ends in `|| true`,
    so an AccessDenied is swallowed, the full benchmark runs, the instance
    terminates on schedule, and S3 is empty. You pay in full and learn nothing.

    Caveat: this reads the LOCAL copy of the policy, which can drift from what is
    actually attached to the role. The instance-side probe write is the backstop.
    """
    if not os.path.exists(POLICY_FILE):
        print(f"  WARN {os.path.basename(POLICY_FILE)} not found; cannot verify "
              f"PutObject under '{prefix}' -- relying on the instance probe")
        return

    with open(POLICY_FILE) as f:
        policy = json.load(f)

    arn_root = f"arn:aws:s3:::{BUCKET}/"
    allowed = []
    for st in policy.get("Statement", []):
        if st.get("Effect") != "Allow":
            continue
        actions = st.get("Action", [])
        actions = [actions] if isinstance(actions, str) else actions
        if not any(a in ("s3:PutObject", "s3:*", "*") for a in actions):
            continue
        resources = st.get("Resource", [])
        resources = [resources] if isinstance(resources, str) else resources
        allowed += [r[len(arn_root):] for r in resources if r.startswith(arn_root)]

    probe = prefix + "probe/probe.txt"
    if not any(fnmatch.fnmatch(probe, pat) for pat in allowed):
        sys.exit(
            f"  FAIL instance role cannot PutObject under '{prefix}'\n"
            f"       writable prefixes per {os.path.basename(POLICY_FILE)}: "
            f"{sorted(allowed)}\n"
            f"       fix by either:\n"
            f"         --s3-prefix smoke/          (no IAM change)\n"
            f"       or adding arn:aws:s3:::{BUCKET}/{prefix}* to the role's\n"
            f"       WriteOutputs statement and re-attaching the policy.")
    print(f"  OK   instance role can write under '{prefix}'")


def preflight(sessions: list, sha: str, prefix: str) -> None:
    """Fail locally and for free rather than on a booted instance."""
    print(f"pre-flight against sha {sha[:7]}")
    check_s3_write_prefix(prefix)

    # The instance clones from GitHub, so an unpushed benchmark script does not
    # exist as far as it is concerned -- and a missing file is a silent 5-minute
    # round trip to find out.
    url = (f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/"
           f"{sha}/{BENCH_SCRIPT}")
    try:
        src = urllib.request.urlopen(url, timeout=30).read().decode()
    except Exception as e:
        sys.exit(f"  FAIL could not fetch {BENCH_SCRIPT} at {sha[:7]}: {e}\n"
                 f"       commit and push {BRANCH}, then rerun")
    for flag in ("--mean_frame_path", "--h5_path", "--bpod_path", "--animal",
                 "--session", "--out", "--instance-type", "--variants"):
        if flag not in src:
            sys.exit(f"  FAIL pushed {BENCH_SCRIPT} has no {flag}")
    print("  OK   pushed benchmark script accepts all required flags")

    s3 = boto3.client("s3", region_name=REGION)
    missing = []
    for sess in sessions:
        animal = sess.split("_")[0]
        for key in (H5_PREFIX + sess + H5_SUFFIX,
                    BPOD_PREFIX + animal + "/" + sess + ".bpod.npy",
                    MEAN_PREFIX + sess + "_mean_frame.npy"):
            try:
                s3.head_object(Bucket=BUCKET, Key=key)
            except Exception:
                missing.append(key)
    if missing:
        print("  FAIL missing S3 objects:")
        for k in missing:
            print("       " + k)
        sys.exit(1)
    print(f"  OK   all inputs present for {len(sessions)} session(s)")


def build_user_data(session: str, sha: str, run_id: str, itype: str,
                    timeout: str, bench_args: str, prefix: str) -> str:
    ud = USER_DATA_TEMPLATE
    for tok, val in [
        ("@@SESSION@@", session),
        ("@@ANIMAL@@", session.split("_")[0]),
        ("@@SHA@@", sha),
        ("@@RUN_ID@@", run_id),
        ("@@BUCKET@@", BUCKET),
        ("@@REGION@@", REGION),
        ("@@BENCH_PREFIX@@", prefix),
        ("@@H5_PREFIX@@", H5_PREFIX),
        ("@@H5_SUFFIX@@", H5_SUFFIX),
        ("@@BPOD_PREFIX@@", BPOD_PREFIX),
        ("@@MEAN_PREFIX@@", MEAN_PREFIX),
        ("@@REPO_URL@@", REPO_URL),
        ("@@GH_REPO@@", GH_REPO),
        ("@@BENCH_SCRIPT@@", BENCH_SCRIPT),
        ("@@ENV_NAME@@", ENV_NAME),
        ("@@PYTHON@@", PYTHON),
        ("@@INSTANCE_TYPE@@", itype),
        ("@@TIMEOUT@@", timeout),
        ("@@BENCH_ARGS@@", bench_args),
    ]:
        ud = ud.replace(tok, val)
    assert ud.startswith("#!/bin/bash\n"), "shebang must be at byte 0"
    assert "@@" not in ud, "unsubstituted placeholder"
    assert ">(" not in ud, "no process substitution"
    assert len(ud) < 16384, f"user-data too large: {len(ud)}"
    return ud


def launch(sessions: list, run_id: str, sha: str, itype: str, timeout: str,
           bench_args: str, prefix: str, dry_run: bool) -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    for sess in sessions:
        ud = build_user_data(sess, sha, run_id, itype, timeout, bench_args,
                             prefix)
        if dry_run:
            print(ud)
            print(f"\n# {len(ud)} bytes for {sess}", file=sys.stderr)
            return
        r = ec2.run_instances(
            ImageId=AMI_ID,
            InstanceType=itype,
            MinCount=1, MaxCount=1,
            IamInstanceProfile={"Name": PROFILE},
            UserData=ud,
            InstanceInitiatedShutdownBehavior="terminate",
            MetadataOptions={"HttpTokens": "optional"},
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "bench-" + sess},
                    # The IAM policy only allows terminating instances carrying
                    # this tag -- omitting it means you cannot stop the run.
                    {"Key": "Project", "Value": "video-autoencoder"},
                    {"Key": "RunId", "Value": run_id},
                    {"Key": "Session", "Value": sess},
                    {"Key": "Kind", "Value": "benchmark"},
                ],
            }],
        )
        print(f"  {r['Instances'][0]['InstanceId']}  {sess}  {itype}")


def status(run_id: str, prefix: str) -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    print(f"benchmark run {run_id}\n")
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

    pref = prefix + run_id + "/"
    keys = s3.list_objects_v2(Bucket=BUCKET, Prefix=pref).get("Contents", [])
    for o in sorted(k["Key"] for k in keys):
        if o.endswith("status.txt"):
            body = s3.get_object(Bucket=BUCKET, Key=o)["Body"].read().decode()
            d = dict(l.split("=", 1) for l in body.strip().splitlines()
                     if "=" in l)
            print(f"  {d.get('session','?'):32s} {d.get('phase','?'):18s} "
                  f"{d.get('last_variant','-'):32s} {d.get('elapsed_min','?')}min")


def results(run_id: str, prefix: str) -> None:
    """Print bench.txt for every session in the run, straight from S3."""
    s3 = boto3.client("s3", region_name=REGION)
    pref = prefix + run_id + "/"
    keys = [k["Key"] for k in
            s3.list_objects_v2(Bucket=BUCKET, Prefix=pref).get("Contents", [])]
    txts = sorted(k for k in keys if k.endswith("bench.txt"))
    if not txts:
        print(f"no bench.txt under s3://{BUCKET}/{pref} yet.")
        print("the run writes it at the end; use --status to see progress, or")
        print("fetch the partial bench.json:")
        print(f"  aws s3 cp s3://{BUCKET}/{pref} . --recursive --region {REGION}")
        return
    for k in txts:
        print(f"\n### {k}\n")
        print(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read().decode())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="+", help="session ids to benchmark")
    p.add_argument("--sha", help="pin this commit (default: tip of %s)" % BRANCH)
    p.add_argument("--instance-type", default=INSTANCE_TYPE)
    p.add_argument("--s3-prefix", default=BENCH_PREFIX,
                   help="output prefix; must be writable by the instance role")
    p.add_argument("--timeout", default=TIMEOUT)
    p.add_argument("--variants", help="comma-separated subset of the ladder")
    p.add_argument("--epochs-per-variant", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--quick", action="store_true",
                   help="4-rung ladder, 1 timed epoch (~3 min of compute)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", metavar="RUN_ID")
    p.add_argument("--results", metavar="RUN_ID")
    a = p.parse_args()

    if a.status:
        status(a.status, a.s3_prefix)
        return
    if a.results:
        results(a.results, a.s3_prefix)
        return
    if not a.sessions:
        p.error("need --sessions, --status, or --results")

    # Everything here lands unquoted inside the user-data shell command, so it
    # has to be inert. Whitelist rather than escape.
    bench_args = []
    if a.variants:
        bench_args += ["--variants", a.variants]
    if a.epochs_per_variant is not None:
        bench_args += ["--epochs-per-variant", str(a.epochs_per_variant)]
    if a.warmup is not None:
        bench_args += ["--warmup", str(a.warmup)]
    if a.quick:
        bench_args += ["--quick"]
    bench_args = " ".join(bench_args)
    bad = set(bench_args) - set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_,. ")
    if bad:
        sys.exit(f"refusing to inject shell metacharacters into user-data: {bad}")

    sha = a.sha or resolve_sha(BRANCH)
    preflight(a.sessions, sha, a.s3_prefix)

    run_id = datetime.datetime.now().strftime("bench-%Y%m%d-%H%M%S")
    print(f"\nrun_id={run_id}  {len(a.sessions)} session(s) on {a.instance_type}")
    launch(a.sessions, run_id, sha, a.instance_type, a.timeout, bench_args,
           a.s3_prefix, a.dry_run)
    if a.dry_run:
        return

    print(f"\nwatch:   python {sys.argv[0]} --status {run_id}")
    print(f"read:    python {sys.argv[0]} --results {run_id}"
          + (f" --s3-prefix {a.s3_prefix}" if a.s3_prefix != BENCH_PREFIX else ""))
    print(f"outputs: s3://{BUCKET}/{a.s3_prefix}{run_id}/")
    print(f"kill:    aws ec2 terminate-instances --region {REGION} --instance-ids "
          f"$(aws ec2 describe-instances --region {REGION} "
          f"--filters Name=tag:RunId,Values={run_id} "
          f"Name=instance-state-name,Values=running,pending "
          f"--query 'Reservations[].Instances[].InstanceId' --output text)")


if __name__ == "__main__":
    main()
