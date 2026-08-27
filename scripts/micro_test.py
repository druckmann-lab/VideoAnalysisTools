#!/usr/bin/env python3
"""
Stage 1: does the boot path work at all?

Deliberately minimal. Tests ONLY:
  - user-data executes
  - awscli can be installed (it is not on the AMI)
  - the instance profile can write to S3
  - /opt/dlami/nvme mounts
  - GPU + the pinned interpreter are usable
  - shutdown -h now actually terminates

No 13 GB download, no dataset construction. ~6 min, ~$0.12.

Design rule learned the hard way: EVERYTHING goes to /dev/console, so the
serial log in the browser tells the whole story even if S3 access is broken.
A debug script must not depend on the thing it is debugging.

Read the output at:
  EC2 -> Instances -> select -> Actions -> Monitor and troubleshoot
      -> Get system log        (look for lines prefixed MICRO:)

Usage:
  python micro_test.py --instance-profile VideoAutoencoderTrainingRole --dry-run
  python micro_test.py --instance-profile VideoAutoencoderTrainingRole
"""

import argparse
import datetime
import sys
import textwrap

import boto3

REGION = "us-west-2"
BUCKET = "balint-video-autoencoder-data-233060639700-us-west-2-an"
AMI_ID = "ami-0bcccc2c1e9b9f874"
INSTANCE_TYPE = "g5.2xlarge"
PYTHON = "/home/ubuntu/ml_env/bin/python"


def build_user_data(run_id: str) -> str:
    return textwrap.dedent(f"""\
        #!/bin/bash
        # No process substitution anywhere: it hid all output last time.
        # No `set -e`: every probe must run even if an earlier one fails.
        LOG=/var/log/micro.log
        exec >> $LOG 2>&1

        # mark() duplicates to the serial console, which is readable from the
        # browser with zero dependencies -- no awscli, no SSH, no S3.
        mark() {{ echo "MICRO: $*"; echo "MICRO: $*" > /dev/console; }}

        finish() {{
            RC=$?
            mark "exit_status=$RC"
            mark "---------- full log follows on console ----------"
            cat $LOG > /dev/console 2>/dev/null || true
            # Best-effort S3 upload; may well fail, that is fine.
            if command -v aws >/dev/null 2>&1; then
                aws s3 cp $LOG "s3://{BUCKET}/smoke/{run_id}/micro.log" \\
                    --region {REGION} > /dev/console 2>&1 || true
            fi
            mark "shutting down"
            shutdown -h now
        }}
        trap finish EXIT

        mark "user-data STARTED $(date -Is)"
        mark "whoami=$(whoami) shell=$BASH_VERSION"

        # --- 1. Is awscli present at all? --------------------------------
        if command -v aws >/dev/null 2>&1; then
            mark "aws_preinstalled=YES version=$(aws --version 2>&1)"
        else
            mark "aws_preinstalled=NO -- installing"
            T0=$(date +%s)
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq                  && mark "apt_update=OK"  || mark "apt_update=FAIL"
            apt-get install -y -qq unzip curl   && mark "apt_unzip=OK"   || mark "apt_unzip=FAIL"
            curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \\
                 -o /tmp/awscliv2.zip           && mark "download=OK"    || mark "download=FAIL"
            unzip -q -o /tmp/awscliv2.zip -d /tmp && mark "unzip=OK"     || mark "unzip=FAIL"
            /tmp/aws/install --update >/dev/null 2>&1 && mark "install=OK" || mark "install=FAIL"
            hash -r
            mark "aws_install_seconds=$(( $(date +%s) - T0 ))"
            mark "aws_now=$(command -v aws || echo MISSING)"
            mark "aws_version=$(aws --version 2>&1 || echo FAILED)"
        fi

        # --- 2. Can the instance profile reach S3? -----------------------
        mark "identity=$(aws sts get-caller-identity --query Arn --output text 2>&1)"
        echo "alive $(date -Is)" > /tmp/alive.txt
        aws s3 cp /tmp/alive.txt "s3://{BUCKET}/smoke/{run_id}/alive.txt" \\
            --region {REGION} 2>&1 && mark "s3_write=OK" || mark "s3_write=FAIL"
        aws s3 ls "s3://{BUCKET}/preprocessed_videos/" --region {REGION} 2>&1 \\
            | head -3 && mark "s3_list_source=OK" || mark "s3_list_source=FAIL"

        # --- 3. NVMe (the configs point at /opt/dlami/nvme) --------------
        systemctl is-active dlami-nvme.service > /dev/null 2>&1 \\
            && mark "dlami_nvme_service=active" || mark "dlami_nvme_service=inactive"
        for i in $(seq 1 12); do
            if mountpoint -q /opt/dlami/nvme; then
                mark "nvme_mounted=YES after $((i*5))s size=$(df -h /opt/dlami/nvme | awk 'NR==2{{print $2}}')"
                touch /opt/dlami/nvme/.wtest && mark "nvme_writable=YES" || mark "nvme_writable=NO"
                break
            fi
            [ $i -eq 12 ] && mark "nvme_mounted=NO after 60s"
            sleep 5
        done

        # --- 4. GPU + interpreter ---------------------------------------
        mark "nvidia_smi=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
        if [ -x {PYTHON} ]; then
            mark "torch=$({PYTHON} -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"
            mark "h5py=$({PYTHON} -c 'import h5py;print(h5py.__version__)' 2>&1 | tail -1)"
            mark "boto3=$({PYTHON} -c 'import boto3;print(boto3.__version__)' 2>&1 | tail -1)"
        else
            mark "interpreter=MISSING at {PYTHON}"
        fi

        # --- 5. Repo + resources ----------------------------------------
        cd /home/ubuntu && rm -rf VideoAnalysisTools
        git clone --depth 1 --branch balint-dev \\
            https://github.com/druckmann-lab/VideoAnalysisTools.git 2>&1 \\
            && mark "clone=OK configs=$(ls VideoAnalysisTools/configs/ | tr '\\n' ' ')" \\
            || mark "clone=FAIL"
        mark "ram=$(free -g | awk '/^Mem:/{{print $2\"GiB\"}}')  root_disk=$(df -h / | awk 'NR==2{{print $2\" avail \"$4}}')"

        mark "ALL PROBES DONE"
        """)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-profile", default="VideoAutoencoderTrainingRole")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ud = build_user_data(run_id)

    if a.dry_run:
        print(ud)
        print(f"\n# {len(ud)} bytes of user-data (EC2 limit 16384)", file=sys.stderr)
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": a.instance_profile},
        UserData=ud,
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"micro-{run_id}"},
                # Required: your IAM policy only allows terminating instances
                # carrying this exact tag.
                {"Key": "Project", "Value": "video-autoencoder"},
            ],
        }],
    )
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {iid}   run_id={run_id}")
    print("\nWait ~3 min, then read the serial log in the browser:")
    print("  EC2 -> Instances -> select -> Actions -> Monitor and troubleshoot")
    print("      -> Get system log, and grep for 'MICRO:'")
    print(f"\nOptional S3 check:")
    print(f"  aws s3 ls s3://{BUCKET}/smoke/{run_id}/")
    print(f"\nIf still running after 15 min:")
    print(f"  aws ec2 terminate-instances --instance-ids {iid} --region {REGION}")


if __name__ == "__main__":
    main()
