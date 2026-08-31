"""
Shared AWS plumbing for the launcher scripts.

This exists because the same ~150 lines were copy-pasted across
launch_training.py, launch_inference.py and launch_benchmark.py, and a fix then
landed in only one of them: the per-session capacity handling was added to
training while inference kept aborting the rest of its run. Same failure the
"extends" mechanism and behavioral_autoencoder/config.py were introduced for,
one layer down.

What lives here is the AWS-API layer and the constants. What deliberately does
NOT is anything genuinely different per launcher:

  - USER_DATA_TEMPLATE and build_user_data -- these are different shell scripts
  - status()                               -- three different progress models
                                              (last_epoch / last_variant /
                                              done_count), not one abstraction
  - preflight()                            -- each checks different objects

launch_instances() always attaches Project=video-autoencoder itself rather than
trusting callers to remember: the IAM policy only permits terminating instances
carrying that tag, so omitting it makes a runaway instance unkillable. Putting it
here makes that bug structurally impossible for a new launcher to reintroduce.

2026.08.31. Balint w/ Claude
"""

import fnmatch
import json
import os
import subprocess
import sys
import urllib.request

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# --- account / environment -------------------------------------------------
REGION = "us-west-2"
BUCKET = "balint-video-autoencoder-data-233060639700-us-west-2-an"
AMI_ID = "ami-0bcccc2c1e9b9f874"
PYTHON = "/home/ubuntu/ml_env/bin/python"
PROFILE = "VideoAutoencoderTrainingRole"

# The IAM policy scopes ec2:TerminateInstances to instances with this tag.
PROJECT_TAG = "video-autoencoder"

# --- code source -----------------------------------------------------------
GH_OWNER = "druckmann-lab"
GH_REPO = "VideoAnalysisTools"
REPO_URL = f"https://github.com/{GH_OWNER}/{GH_REPO}.git"
BRANCH = "balint-dev"

# --- S3 layout -------------------------------------------------------------
H5_PREFIX = "preprocessed_videos/"
H5_SUFFIX = "_side_crop.h5"
BPOD_PREFIX = "bpod_files/"
MEAN_PREFIX = "mean_frames/"
RUNS_PREFIX = "runs/"

POLICY_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "instance_role_policy.json")

MAX_USER_DATA = 16384          # EC2 rejects anything larger


# --------------------------------------------------------------------------
# code resolution
# --------------------------------------------------------------------------

def resolve_sha(ref: str) -> str:
    """Resolve a branch name to the commit the instances will actually clone."""
    out = subprocess.run(["git", "ls-remote", REPO_URL, ref],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit(f"could not resolve '{ref}' on {REPO_URL}\n{out.stderr}")
    return out.stdout.split()[0]


def fetch_at_sha(sha: str, path: str) -> str:
    """Read a file from GitHub at a pinned sha -- what the instance will see."""
    url = f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{sha}/{path}"
    return urllib.request.urlopen(url, timeout=30).read().decode()


# --------------------------------------------------------------------------
# preflight helpers
# --------------------------------------------------------------------------

def check_s3_write_prefix(prefix: str, policy_file: str = None) -> None:
    """
    Verify the instance role can PutObject under `prefix`.

    Getting this wrong is invisible and expensive: the role scopes writes to a
    few prefixes, every sync in the wrappers ends in `|| true`, so an
    AccessDenied is swallowed, the work runs, the instance terminates on
    schedule, and S3 is empty. That happened once already.

    Reads the LOCAL copy of the policy, which can drift from what is actually
    attached -- the instance-side probe write is the backstop.
    """
    policy_file = policy_file or POLICY_FILE
    if not os.path.exists(policy_file):
        print(f"  WARN {os.path.basename(policy_file)} not found; cannot verify "
              f"PutObject under '{prefix}' -- relying on the instance probe")
        return

    with open(policy_file) as f:
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

    if not any(fnmatch.fnmatch(prefix + "probe/probe.txt", p) for p in allowed):
        sys.exit(
            f"  FAIL instance role cannot PutObject under '{prefix}'\n"
            f"       writable prefixes per {os.path.basename(policy_file)}: "
            f"{sorted(allowed)}\n"
            f"       either pass a prefix from that list, or add\n"
            f"       arn:aws:s3:::{BUCKET}/{prefix}* to the role's WriteOutputs\n"
            f"       statement and re-attach the policy.")
    print(f"  OK   instance role can write under '{prefix}'")


def missing_session_inputs(sessions, need_mean_frame: bool = True) -> list:
    """
    Which per-session S3 inputs are absent. Empty list means all present.

    Training needs the mean frame; inference takes it from the checkpoint, so it
    does not.
    """
    s3 = boto3.client("s3", region_name=REGION)
    missing = []
    for sess in sessions:
        animal = sess.split("_")[0]
        keys = [H5_PREFIX + sess + H5_SUFFIX,
                BPOD_PREFIX + animal + "/" + sess + ".bpod.npy"]
        if need_mean_frame:
            keys.append(MEAN_PREFIX + sess + "_mean_frame.npy")
        for key in keys:
            try:
                s3.head_object(Bucket=BUCKET, Key=key)
            except Exception:
                missing.append(key)
    return missing


def resolve_config_chain(sha: str, env_name: str, fetch=None,
                         max_depth: int = 6) -> list:
    """
    Every config in an "extends" chain must exist at the pinned sha.

    load_config only raises at runtime; by then the instance has already staged
    14 GB. Returns the chain, parent last.

    `fetch` is injectable so the network boundary stays where the caller put it
    -- a launcher passes its own module-level fetch_at_sha, which keeps that name
    patchable from tests instead of buried one import deeper.
    """
    fetch = fetch or fetch_at_sha
    name, chain = env_name, []
    while name and len(chain) < max_depth:
        cfg_path = f"configs/{name}_config.json"
        try:
            cfg = json.loads(fetch(sha, cfg_path))
        except Exception as e:
            sys.exit(f"  FAIL could not fetch {cfg_path} at {sha[:7]}: {e}\n"
                     f"       commit and push {BRANCH}, then rerun")
        chain.append(name)
        name = cfg.get("extends")
    return chain


def validate_user_data(ud: str) -> str:
    """The assertions are load-bearing: each one is a way to brick an instance."""
    assert ud.startswith("#!/bin/bash\n"), "shebang must be at byte 0"
    assert "@@" not in ud, "unsubstituted placeholder"
    assert ">(" not in ud, "no process substitution"
    assert len(ud) < MAX_USER_DATA, f"user-data too large: {len(ud)}"
    return ud


# --------------------------------------------------------------------------
# launching
# --------------------------------------------------------------------------

def launch_instances(specs, itype: str, dry_run: bool = False) -> tuple:
    """
    Launch one instance per spec; returns (launched, failed).

    `specs` is an iterable of (name, user_data, extra_tags), where extra_tags is
    a list of {"Key":..., "Value":...}. Project=video-autoencoder is added here,
    not by callers -- see the module docstring.

    MinCount=MaxCount=1 per call, so a capacity shortfall affects one instance
    rather than the whole batch. Each call is wrapped: an uncaught error would
    abandon every spec after it, leaving a half-launched batch and no summary.
    Only AWS-side failures are swallowed; a programming error still propagates.

      launched: [(instance_id, name), ...]
      failed:   [(name, error_code, message), ...]
    """
    ec2 = boto3.client("ec2", region_name=REGION)
    launched, failed = [], []

    for name, ud, extra_tags in specs:
        if dry_run:
            print(ud)
            print(f"\n# {len(ud)} bytes for {name}", file=sys.stderr)
            return [], []

        tags = [{"Key": "Project", "Value": PROJECT_TAG}]
        tags += [t for t in extra_tags if t["Key"] != "Project"]

        try:
            r = ec2.run_instances(
                ImageId=AMI_ID,
                InstanceType=itype,
                MinCount=1, MaxCount=1,
                IamInstanceProfile={"Name": PROFILE},
                UserData=ud,
                InstanceInitiatedShutdownBehavior="terminate",
                MetadataOptions={"HttpTokens": "optional"},
                TagSpecifications=[{"ResourceType": "instance", "Tags": tags}],
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "ClientError")
            msg = e.response.get("Error", {}).get("Message", str(e))
            print(f"  {'FAILED':19s}  {name}  {code}")
            failed.append((name, code, msg))
            continue
        except BotoCoreError as e:
            # Connection/endpoint trouble, not an AWS-side rejection.
            print(f"  {'FAILED':19s}  {name}  {type(e).__name__}")
            failed.append((name, type(e).__name__, str(e)))
            continue

        iid = r["Instances"][0]["InstanceId"]
        print(f"  {iid}  {name}")
        launched.append((iid, name))

    return launched, failed


def retry_command(script: str, flags: dict, sessions) -> str:
    """
    A paste-able command that relaunches exactly the sessions that failed.

    Every identity-bearing flag has to be pinned here, not just the obvious ones:
    the sha so the retry runs the same code as the sessions that did launch, and
    the run/inference id so its outputs land in the SAME S3 prefix. Without the
    id the retry mints a fresh one, and the results end up split across two
    prefixes that later analysis has to reconcile by hand -- which is exactly the
    kind of quiet mess that only shows up weeks later.
    """
    # Wrap between flags, never inside one: textwrap would happily put "--run-id"
    # at the end of a line and its value at the start of the next. Still valid
    # shell, but the flag stops reading as a unit and nothing can grep for it.
    lines, cur = [], ""
    for pair in (f"--{k} {v}" for k, v in flags.items()):
        if cur and len(cur) + 1 + len(pair) > 66:
            lines.append(cur)
            cur = pair
        else:
            cur = f"{cur} {pair}".strip()
    if cur:
        lines.append(cur)
    out = [f"  python {script} \\"]
    out += [f"      {l} \\" for l in lines]
    out.append("      --sessions " + " ".join(sessions))
    return "\n".join(out)


def print_failure_summary(failed, retry_command: str, noun: str = "session") -> None:
    """
    Report what did not launch, with a command to retry exactly those.

    The launched instances are already running and will self-terminate; only the
    missing ones need relaunching.
    """
    print(f"\n{'=' * 72}")
    print(f"{len(failed)} {noun}(s) did NOT launch:")
    for name, code, msg in failed:
        print(f"  {name:36s} {code}")
        print(f"      {msg[:110]}")
    print(f"\nRetry just those. --sha and --run-id are pinned so the retried")
    print(f"{noun}s run identical code and land under the same S3 prefix:\n")
    print(retry_command)
    print(f"{'=' * 72}")
