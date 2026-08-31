#!/usr/bin/env python3
"""
Run inference on trained checkpoints from S3, on a disposable EC2 instance.

Same skeleton as launch_training.py -- pinned sha, stage to NVMe, background
monitor, always-sync-always-terminate trap. Built so you never need the 13.9 GB
h5 on your own machine: the instance pulls it from S3, runs every requested
checkpoint, uploads the latents, and terminates.

  # both A/B arms, three checkpoints each, one instance (h5 staged once)
  python launch_inference.py --runs 20260827-165607 20260827-165614 \
      --checkpoints 1499 3499 7499

  python launch_inference.py --runs 20260827-165607 --checkpoints final
  python launch_inference.py --list 20260827-165607     # what is available
  python launch_inference.py --status <inference_run_id>

Grouping: one instance per SESSION, handling every (run, checkpoint) pair asked
for. Both arms of an A/B are the same session, so they share one staging of the
h5 rather than paying ~70s + ~64s of load twice.

Config comes from the config.json that training saved beside the checkpoints, so
the model is rebuilt exactly as it was trained -- no reconstruction from env
configs, and no chance of a variant env resolving differently than it did during
training. The NVMe paths inside it are also correct here, because this wrapper
stages to the same /opt/dlami/nvme layout; --h5_path/--bpod_path are still passed
explicitly so the run does not depend on that coincidence.

Sizing (measured): staging 13.9 GB takes ~70s; each inference invocation reloads
the h5 (~64s) and then runs ~828k frames forward. Latents are ~53 MB per
checkpoint. Reconstructions are OFF by default: they buffer in RAM at ~1
byte/pixel, so a full session adds ~11 GB on top of the ~15 GB h5 -- 26 GB of a
32 GiB box. Pass --save-recons only with --instance-type g5.4xlarge.
"""

import argparse
import datetime
import sys

# Shared AWS plumbing; imported by name so tests can patch e.g.
# launch_inference.fetch_at_sha and have preflight resolve the patched one.
import boto3  # noqa: F401  (tests patch launch_inference.boto3.client)
from aws_launch_common import (BPOD_PREFIX, BRANCH, BUCKET, GH_REPO,
                               H5_PREFIX, H5_SUFFIX, MEAN_PREFIX, POLICY_FILE,
                               PYTHON, REGION, REPO_URL, RUNS_PREFIX,
                               check_s3_write_prefix, fetch_at_sha,
                               launch_instances, missing_session_inputs,
                               print_failure_summary, resolve_sha,
                               validate_user_data)

INSTANCE_TYPE = "g5.2xlarge"
INFER_SCRIPT = "scripts/single_session_inference.py"

# Runaway guard, not a schedule. Expect ~5 min setup + ~2.5 min per checkpoint.
TIMEOUT = "2h"


USER_DATA_TEMPLATE = r'''#!/bin/bash
# Shebang at byte 0 or the kernel falls back to /bin/sh and bash-only syntax
# dies with the output nowhere visible.
LOG=/var/log/wrapper.log
exec >> $LOG 2>&1

SESSION="@@SESSION@@"
ANIMAL="@@ANIMAL@@"
SHA="@@SHA@@"
S3="s3://@@BUCKET@@"
INFER_ID="@@INFER_ID@@"
INFER_LOG=/var/log/infer.log
GPU_LOG=/var/log/gpu.log
STATUS=/tmp/status.txt
MANIFEST=/tmp/manifest.txt
BOOT_TS=$(date +%s)
PHASE="boot"
CURRENT="-"

mark() { echo "INFER[$SESSION]: $*"; echo "INFER[$SESSION]: $*" > /dev/console; }

# Status and logs go under the FIRST run's prefix; per-checkpoint outputs go
# under their own run's prefix (see the loop below).
S3_STATUS="$S3/@@RUNS_PREFIX@@@@FIRST_RUN@@/$SESSION/inference/$INFER_ID"

write_status() {
    {
        echo "session=$SESSION"
        echo "animal=$ANIMAL"
        echo "sha=$SHA"
        echo "inference_id=$INFER_ID"
        echo "instance_id=$IID"
        echo "instance_type=@@INSTANCE_TYPE@@"
        echo "phase=$PHASE"
        echo "current=$CURRENT"
        echo "done_count=$(grep -c '^done ' $INFER_LOG 2>/dev/null || echo 0)"
        echo "total=@@N_JOBS@@"
        echo "elapsed_min=$(( ($(date +%s) - BOOT_TS) / 60 ))"
        echo "updated=$(date -Is)"
    } > $STATUS
    aws s3 cp $STATUS "$S3_STATUS/status.txt" --region @@REGION@@ --only-show-errors || true
}

sync_logs() {
    aws s3 cp $INFER_LOG "$S3_STATUS/infer.log"   --region @@REGION@@ --only-show-errors || true
    aws s3 cp $GPU_LOG   "$S3_STATUS/gpu.log"     --region @@REGION@@ --only-show-errors || true
    aws s3 cp $LOG       "$S3_STATUS/wrapper.log" --region @@REGION@@ --only-show-errors || true
}

finish() {
    RC=$?
    PHASE="finished_rc$RC"
    [ -n "$MON_PID" ] && kill $MON_PID 2>/dev/null
    mark "finishing rc=$RC after $(( ($(date +%s) - BOOT_TS) / 60 )) min"
    echo "wrapper_exit=$RC" >> $STATUS
    sync_logs
    write_status
    tail -40 $INFER_LOG > /dev/console 2>/dev/null || true
    shutdown -h now
}
trap finish EXIT

TOKEN=$(curl -sX PUT --max-time 5 http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
IID=$(curl -s --max-time 5 -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)
mark "STARTED sha=$SHA instance=$IID jobs=@@N_JOBS@@"
touch $INFER_LOG $GPU_LOG

# The one S3 call that is not `|| true`: a permission problem must kill the run
# cheaply rather than after all the compute, with nothing saved.
echo "probe $(date -Is) $IID" > /tmp/probe.txt
if ! aws s3 cp /tmp/probe.txt "$S3_STATUS/probe.txt" \
        --region @@REGION@@ --only-show-errors; then
    mark "FATAL cannot PutObject to $S3_STATUS"
    exit 1
fi
PHASE="staging"; write_status

# --- scratch space -----------------------------------------------------
DATA_DIR=""
for i in $(seq 1 12); do
    if mountpoint -q /opt/dlami/nvme; then DATA_DIR=/opt/dlami/nvme; break; fi
    sleep 5
done
if [ -z "$DATA_DIR" ]; then mark "FATAL /opt/dlami/nvme never mounted"; exit 1; fi
mkdir -p $DATA_DIR/ckpt $DATA_DIR/out

# --- code at a pinned sha ---------------------------------------------
cd /home/ubuntu && rm -rf @@GH_REPO@@
git clone --quiet @@REPO_URL@@ || { mark "FATAL clone failed"; exit 1; }
cd @@GH_REPO@@
git checkout --quiet "$SHA" || { mark "FATAL checkout $SHA failed"; exit 1; }
mark "code at $(git rev-parse --short HEAD)"

# --- stage session inputs (once, shared by every checkpoint) -----------
T0=$(date +%s)
aws s3 cp "$S3/@@H5_PREFIX@@${SESSION}@@H5_SUFFIX@@" "$DATA_DIR/" \
    --region @@REGION@@ --only-show-errors || { mark "FATAL h5 download"; exit 1; }
aws s3 cp "$S3/@@BPOD_PREFIX@@$ANIMAL/${SESSION}.bpod.npy" "$DATA_DIR/" \
    --region @@REGION@@ --only-show-errors || { mark "FATAL bpod download"; exit 1; }
aws s3 cp "$S3/@@MEAN_PREFIX@@${SESSION}_mean_frame.npy" "$DATA_DIR/" \
    --region @@REGION@@ --only-show-errors || true
mark "staged in $(( $(date +%s) - T0 ))s"

# --- job manifest: run_id|label|ckpt_key|config_key --------------------
cat > $MANIFEST <<'MANIFEST_EOF'
@@MANIFEST@@
MANIFEST_EOF

cat > /tmp/meta.json <<META
{
  "kind": "inference",
  "session": "$SESSION",
  "animal": "$ANIMAL",
  "sha": "$SHA",
  "inference_id": "$INFER_ID",
  "instance_id": "$IID",
  "instance_type": "@@INSTANCE_TYPE@@",
  "save_recons": "@@SAVE_RECONS@@",
  "n_jobs": "@@N_JOBS@@",
  "started": "$(date -Is)"
}
META
aws s3 cp /tmp/meta.json "$S3_STATUS/meta.json" --region @@REGION@@ --only-show-errors || true

# --- background monitor -----------------------------------------------
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
        if [ $((i % 2)) -eq 0 ]; then sync_logs; write_status; fi
        sleep 30
    done
}
monitor & MON_PID=$!

# --- one inference per manifest line ----------------------------------
PHASE="inference"; write_status
FAILED=0
while IFS='|' read -r RUN_ID LABEL CKPT_KEY CONFIG_KEY; do
    [ -z "$RUN_ID" ] && continue
    CURRENT="$RUN_ID/$LABEL"
    mark "=== $CURRENT"
    write_status

    # The checkpoint and its config.json must land in the SAME directory: the
    # inference script reads config.json from beside the checkpoint, which is
    # what makes the model identical to the one that was trained.
    CK_DIR="$DATA_DIR/ckpt/${RUN_ID}_${LABEL}"
    rm -rf "$CK_DIR"; mkdir -p "$CK_DIR"
    aws s3 cp "$S3/$CKPT_KEY"   "$CK_DIR/model.pt"    --region @@REGION@@ --only-show-errors \
        || { mark "FAIL download $CKPT_KEY"; FAILED=$((FAILED+1)); continue; }
    aws s3 cp "$S3/$CONFIG_KEY" "$CK_DIR/config.json" --region @@REGION@@ --only-show-errors \
        || { mark "FAIL download $CONFIG_KEY"; FAILED=$((FAILED+1)); continue; }

    OUT_DIR="$DATA_DIR/out/${RUN_ID}_${LABEL}"
    rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"

    @@PYTHON@@ -u @@INFER_SCRIPT@@ \
        --checkpoint "$CK_DIR/model.pt" \
        --animal "$ANIMAL" \
        --session "$SESSION" \
        --h5_path "$DATA_DIR/${SESSION}@@H5_SUFFIX@@" \
        --bpod_path "$DATA_DIR/${SESSION}.bpod.npy" \
        --output_dir "$OUT_DIR" \
        @@SAVE_RECONS_FLAG@@ \
        2>&1 | awk -v p="$CURRENT" '{ printf "%s [%s] %s\n", strftime("%Y-%m-%dT%H:%M:%S"), p, $0; fflush() }' \
        | tee -a $INFER_LOG
    RC=${PIPESTATUS[0]}

    if [ $RC -ne 0 ]; then
        mark "FAIL inference rc=$RC for $CURRENT"
        echo "oom_kills=$(dmesg 2>/dev/null | grep -ci 'out of memory' || echo 0)" >> $INFER_LOG
        FAILED=$((FAILED+1))
        continue
    fi

    # Outputs land beside the checkpoints they came from, under their own run.
    aws s3 sync "$OUT_DIR/" \
        "$S3/@@RUNS_PREFIX@@$RUN_ID/$SESSION/inference/$INFER_ID/$LABEL/" \
        --region @@REGION@@ --only-show-errors \
        || { mark "FAIL upload $CURRENT"; FAILED=$((FAILED+1)); continue; }

    echo "done $CURRENT" >> $INFER_LOG
    mark "done $CURRENT ($(du -sh $OUT_DIR | cut -f1))"
    # Free NVMe as we go; the h5 alone is 13.9 GB.
    rm -rf "$OUT_DIR" "$CK_DIR"
    write_status
done < $MANIFEST

mark "all jobs attempted, failures=$FAILED"
echo "failed=$FAILED" >> $STATUS
exit $FAILED
'''








def s3c():
    return boto3.client("s3", region_name=REGION)


def list_sessions(run_id: str) -> list:
    r = s3c().list_objects_v2(Bucket=BUCKET, Prefix=f"{RUNS_PREFIX}{run_id}/",
                              Delimiter="/")
    return sorted(p["Prefix"].rstrip("/").split("/")[-1]
                  for p in r.get("CommonPrefixes", []))


def find_ckpt_dir(run_id: str, session: str) -> str:
    """
    Locate the date-stamped checkpoint folder training created.

    runs/<run>/<session>/checkpoints/<animal>/<session>_<date>/
    """
    animal = session.split("_")[0]
    base = f"{RUNS_PREFIX}{run_id}/{session}/checkpoints/{animal}/"
    r = s3c().list_objects_v2(Bucket=BUCKET, Prefix=base, Delimiter="/")
    dirs = [p["Prefix"] for p in r.get("CommonPrefixes", [])]
    if not dirs:
        sys.exit(f"  FAIL no checkpoint folder under s3://{BUCKET}/{base}")
    if len(dirs) > 1:
        # One training run makes exactly one; more means the prefix was reused.
        sys.exit(f"  FAIL {len(dirs)} checkpoint folders under {base}; "
                 f"expected 1:\n       " + "\n       ".join(dirs))
    return dirs[0]


def ckpt_object(label: str) -> str:
    if label == "best":
        return "best_model.pt"
    if label == "final":
        return "final_model.pt"
    return f"checkpoint_epoch_{int(label)}.pt"


def list_available(run_id: str) -> None:
    for session in list_sessions(run_id):
        d = find_ckpt_dir(run_id, session)
        r = s3c().list_objects_v2(Bucket=BUCKET, Prefix=d)
        names = [o["Key"].split("/")[-1] for o in r.get("Contents", [])]
        eps = sorted(int(n.split("_")[-1][:-3]) for n in names
                     if n.startswith("checkpoint_epoch_"))
        print(f"\nrun {run_id}  session {session}")
        print(f"  folder    {d}")
        print(f"  config    {'yes' if 'config.json' in names else 'MISSING'}")
        print(f"  best/final {'best_model.pt' in names}/{'final_model.pt' in names}")
        print(f"  {len(eps)} periodic checkpoints: "
              f"{eps[:6]}{' ... ' + str(eps[-3:]) if len(eps) > 9 else ''}")


def build_jobs(runs: list, sessions_filter: list, labels: list) -> dict:
    """session -> list of (run_id, label, ckpt_key, config_key), all verified."""
    s3 = s3c()
    jobs = {}
    for run_id in runs:
        sessions = list_sessions(run_id)
        if not sessions:
            sys.exit(f"  FAIL run {run_id} has no sessions under "
                     f"s3://{BUCKET}/{RUNS_PREFIX}{run_id}/")
        if sessions_filter:
            missing = set(sessions_filter) - set(sessions)
            if missing:
                sys.exit(f"  FAIL run {run_id} has no session(s) {sorted(missing)}\n"
                         f"       available: {sessions}")
            sessions = [s for s in sessions if s in sessions_filter]
        for session in sessions:
            d = find_ckpt_dir(run_id, session)
            config_key = d + "config.json"
            try:
                s3.head_object(Bucket=BUCKET, Key=config_key)
            except Exception:
                sys.exit(f"  FAIL no config.json in {d}\n"
                         f"       inference rebuilds the model from it")
            for label in labels:
                key = d + ckpt_object(label)
                try:
                    s3.head_object(Bucket=BUCKET, Key=key)
                except Exception:
                    sys.exit(f"  FAIL checkpoint not in S3: {key}\n"
                             f"       try --list {run_id}")
                jobs.setdefault(session, []).append(
                    (run_id, label if label in ("best", "final") else f"epoch_{label}",
                     key, config_key))
    return jobs


def preflight(jobs: dict, sha: str) -> None:
    print(f"pre-flight against sha {sha[:7]}")
    check_s3_write_prefix(RUNS_PREFIX, POLICY_FILE)

    try:
        src = fetch_at_sha(sha, INFER_SCRIPT)
    except Exception as e:
        sys.exit(f"  FAIL could not fetch {INFER_SCRIPT} at {sha[:7]}: {e}\n"
                 f"       commit and push {BRANCH}, then rerun")
    for flag in ("--checkpoint", "--h5_path", "--bpod_path", "--output_dir",
                 "--animal", "--session", "save_recons"):
        if flag not in src:
            sys.exit(f"  FAIL pushed {INFER_SCRIPT} has no {flag}")
    if "BooleanOptionalAction" not in src:
        sys.exit(f"  FAIL pushed {INFER_SCRIPT} predates the two-way "
                 f"--save_recons flag; commit and push {BRANCH}")
    print("  OK   pushed inference script accepts all required flags")

    missing = missing_session_inputs(jobs, need_mean_frame=False)
    if missing:
        print("  FAIL missing session inputs:")
        for k in missing:
            print("       " + k)
        sys.exit(1)
    print(f"  OK   session inputs present for {len(jobs)} session(s)")
    print(f"  OK   {sum(len(v) for v in jobs.values())} checkpoint(s) resolved in S3")


def build_user_data(session: str, joblist: list, sha: str, infer_id: str,
                    itype: str, timeout: str, save_recons: bool) -> str:
    manifest = "\n".join(f"{r}|{l}|{c}|{cfg}" for r, l, c, cfg in joblist)
    ud = USER_DATA_TEMPLATE
    for tok, val in [
        ("@@SESSION@@", session),
        ("@@ANIMAL@@", session.split("_")[0]),
        ("@@SHA@@", sha),
        ("@@INFER_ID@@", infer_id),
        ("@@FIRST_RUN@@", joblist[0][0]),
        ("@@MANIFEST@@", manifest),
        ("@@N_JOBS@@", str(len(joblist))),
        ("@@BUCKET@@", BUCKET),
        ("@@REGION@@", REGION),
        ("@@RUNS_PREFIX@@", RUNS_PREFIX),
        ("@@H5_PREFIX@@", H5_PREFIX),
        ("@@H5_SUFFIX@@", H5_SUFFIX),
        ("@@BPOD_PREFIX@@", BPOD_PREFIX),
        ("@@MEAN_PREFIX@@", MEAN_PREFIX),
        ("@@REPO_URL@@", REPO_URL),
        ("@@GH_REPO@@", GH_REPO),
        ("@@INFER_SCRIPT@@", INFER_SCRIPT),
        ("@@PYTHON@@", PYTHON),
        ("@@INSTANCE_TYPE@@", itype),
        ("@@TIMEOUT@@", timeout),
        ("@@SAVE_RECONS@@", str(save_recons)),
        ("@@SAVE_RECONS_FLAG@@", "--save_recons" if save_recons else "--no-save_recons"),
    ]:
        ud = ud.replace(tok, val)
    return validate_user_data(ud)


def launch(jobs: dict, sha: str, infer_id: str, itype: str, timeout: str,
           save_recons: bool, dry_run: bool) -> tuple:
    """One instance per session, each running all of that session's jobs."""
    specs = ((session,
              build_user_data(session, joblist, sha, infer_id, itype, timeout,
                              save_recons),
              [{"Key": "Name", "Value": "infer-" + session},
               {"Key": "RunId", "Value": infer_id},
               {"Key": "Session", "Value": session},
               {"Key": "Kind", "Value": "inference"}])
             for session, joblist in jobs.items())
    return launch_instances(specs, itype, dry_run)


def status(infer_id: str) -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    print(f"inference run {infer_id}\n")
    r = ec2.describe_instances(Filters=[
        {"Name": "tag:RunId", "Values": [infer_id]}])
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

    s3 = s3c()
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=RUNS_PREFIX).get("Contents", []):
        if o["Key"].endswith(f"inference/{infer_id}/status.txt"):
            body = s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read().decode()
            d = dict(l.split("=", 1) for l in body.strip().splitlines() if "=" in l)
            print(f"  {d.get('session','?'):32s} {d.get('phase','?'):18s} "
                  f"{d.get('done_count','?')}/{d.get('total','?')} done  "
                  f"cur={d.get('current','-'):24s} {d.get('elapsed_min','?')}min")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", help="training run id(s) to take checkpoints from")
    p.add_argument("--sessions", nargs="+",
                   help="only these sessions (default: every session in each run)")
    p.add_argument("--checkpoints", nargs="+", default=["final"],
                   help="epoch numbers, and/or 'best'/'final' (default: final)")
    p.add_argument("--save-recons", action="store_true",
                   help="also write reconstructions (~11 GB RAM; use g5.4xlarge)")
    p.add_argument("--sha", help="pin this commit (default: tip of %s)" % BRANCH)
    p.add_argument("--instance-type", default=INSTANCE_TYPE)
    p.add_argument("--timeout", default=TIMEOUT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", metavar="RUN_ID", help="show available checkpoints")
    p.add_argument("--status", metavar="INFERENCE_ID")
    a = p.parse_args()

    if a.list:
        list_available(a.list)
        return
    if a.status:
        status(a.status)
        return
    if not a.runs:
        p.error("need --runs, --list, or --status")

    for label in a.checkpoints:
        if label not in ("best", "final") and not label.isdigit():
            p.error(f"--checkpoints takes epoch numbers or best/final, got {label!r}")

    if a.save_recons and a.instance_type == "g5.2xlarge":
        print("WARNING: --save-recons buffers ~11 GB of frames on top of the ~15 GB\n"
              "         h5, which will not fit a g5.2xlarge (32 GiB). Pass\n"
              "         --instance-type g5.4xlarge, or drop --save-recons.\n")

    sha = a.sha or resolve_sha(BRANCH)
    jobs = build_jobs(a.runs, a.sessions or [], a.checkpoints)
    preflight(jobs, sha)

    infer_id = datetime.datetime.now().strftime("infer-%Y%m%d-%H%M%S")
    print(f"\ninference_id={infer_id}  {len(jobs)} instance(s) on {a.instance_type}")
    for session, joblist in jobs.items():
        print(f"  {session}: " + ", ".join(f"{r}/{l}" for r, l, _, _ in joblist))
    launched, failed = launch(jobs, sha, infer_id, a.instance_type, a.timeout,
                              a.save_recons, a.dry_run)
    if a.dry_run:
        return

    print(f"\n{len(launched)}/{len(jobs)} instance(s) launched")

    print(f"\nwatch:   python {sys.argv[0]} --status {infer_id}")
    print(f"outputs: s3://{BUCKET}/{RUNS_PREFIX}<run_id>/<session>/inference/{infer_id}/<label>/")
    print(f"kill:    aws ec2 terminate-instances --region {REGION} --instance-ids "
          f"$(aws ec2 describe-instances --region {REGION} "
          f"--filters Name=tag:RunId,Values={infer_id} "
          f"Name=instance-state-name,Values=running,pending "
          f"--query 'Reservations[].Instances[].InstanceId' --output text)")

    if failed:
        retry = (f"  python {sys.argv[0]} --runs {' '.join(a.runs)} \\\n"
                 f"      --sha {sha} --checkpoints {' '.join(a.checkpoints)} "
                 f"--instance-type {a.instance_type} \\\n"
                 f"      --sessions " + " ".join(s for s, _, _ in failed))
        print_failure_summary(failed, retry, noun="session")
        sys.exit(1)


if __name__ == "__main__":
    main()
