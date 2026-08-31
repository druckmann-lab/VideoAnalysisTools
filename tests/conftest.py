"""
Shared fixtures for the launcher tests.

Two things matter here beyond ordinary test hygiene:

1. NO TEST MAY LAUNCH A REAL INSTANCE. Every AWS call is served by
   botocore.stub.Stubber, which intercepts before the request leaves the
   process. Stubber (rather than moto) because it needs no extra dependency and
   validates every stubbed response against the real API model, so a test cannot
   assert on a response shape AWS would never actually return.

2. Real credentials are masked (see _fake_aws_credentials, autouse). If a test
   ever reaches an un-stubbed call, it fails on authentication instead of
   silently doing something to the live account.

The launchers live in scripts/ and are not an importable package, so scripts/ is
put on sys.path here rather than in every test module.
"""

import os
import sys

import boto3
import pytest
from botocore.stub import Stubber

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

REGION = "us-west-2"


@pytest.fixture(autouse=True)
def _fake_aws_credentials(monkeypatch):
    """
    Mask any real credentials for the whole test session.

    Autouse and unconditional: the cost of forgetting it once is a real instance
    launch against the live account.
    """
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                "AWS_SECURITY_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


class StubbedAWS:
    """Stubbed ec2 + s3 clients, and the boto3.client shim that hands them out."""

    def __init__(self):
        self.ec2 = boto3.client("ec2", region_name=REGION)
        self.s3 = boto3.client("s3", region_name=REGION)
        self.ec2_stub = Stubber(self.ec2)
        self.s3_stub = Stubber(self.s3)
        self.ec2_stub.activate()
        self.s3_stub.activate()
        self.calls = []          # (service, operation) in call order

    def client(self, service, **kwargs):
        self.calls.append(("client", service))
        if service == "ec2":
            return self.ec2
        if service == "s3":
            return self.s3
        raise AssertionError(f"test made an unexpected boto3 client: {service}")

    def install(self, monkeypatch, module):
        """Point a launcher module's boto3 at the stubbed clients."""
        monkeypatch.setattr(module.boto3, "client", self.client)

    def assert_no_pending(self):
        """Fail if a queued response went unused -- i.e. a call never happened."""
        self.ec2_stub.assert_no_pending_responses()
        self.s3_stub.assert_no_pending_responses()

    def close(self):
        self.ec2_stub.deactivate()
        self.s3_stub.deactivate()


@pytest.fixture
def aws():
    a = StubbedAWS()
    yield a
    a.close()


# --------------------------------------------------------------------------
# canned AWS responses
# --------------------------------------------------------------------------

def run_instances_response(instance_id="i-0123456789abcdef0"):
    return {"Instances": [{"InstanceId": instance_id}]}


def head_object_response(size=1234):
    return {"ContentLength": size}


def list_objects_prefixes(prefixes):
    return {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}


def list_objects_contents(keys):
    return {"Contents": [{"Key": k, "Size": 1} for k in keys]}


# --------------------------------------------------------------------------
# tests/test_dataset.py predates this work and imports cv2 (opencv-python),
# which is not installed in this venv. A collection ERROR aborts the entire
# pytest run, so it is skipped when cv2 is unavailable -- the launcher tests
# still run. It is not deleted or modified: install opencv-python to run it.
# --------------------------------------------------------------------------
collect_ignore = []
try:
    import cv2  # noqa: F401
except ImportError:
    collect_ignore.append("test_dataset.py")
