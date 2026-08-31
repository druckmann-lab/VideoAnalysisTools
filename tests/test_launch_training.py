"""
Tests for scripts/launch_training.py.

Written as a regression net for the shared-plumbing refactor: the AWS-API layer
is currently copy-pasted across three launchers, and a fix landing in only one of
them is what these are meant to catch.

The tests are grouped by what a failure would actually cost:

  TestBillingSafety   -- invariants where getting it wrong costs money or makes a
                         runaway instance unkillable. These matter most.
  TestUserData        -- the user-data contract (byte 0, size cap, substitution)
  TestFailureHandling -- one session failing must not abandon the others
  TestPreflight       -- fail locally and for free rather than on 20 instances

No test may reach real AWS; see tests/conftest.py.
"""

import json
import subprocess
import sys
from unittest import mock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import launch_training as LT
from conftest import head_object_response, run_instances_response

SESSIONS = ["kd104_twNew_20221124_104921", "kd102_twNew_20220929_133826"]
SHA = "b67d6040f985ebdd3a54d2e61254efba4ec4a6c3"
RUN_ID = "20260831-105222"


def _input_keys(session):
    """The three S3 objects preflight checks, in the order it checks them."""
    animal = session.split("_")[0]
    return [f"{LT.H5_PREFIX}{session}{LT.H5_SUFFIX}",
            f"{LT.BPOD_PREFIX}{animal}/{session}.bpod.npy",
            f"{LT.MEAN_PREFIX}{session}_mean_frame.npy"]


def _queue_inputs_present(aws, sessions):
    """
    Queue an exact-match head_object per expected key.

    expected_params is given in full rather than ANY, so the test also pins how
    the keys are built -- the bpod one nests under the animal, the others do not.
    """
    for sess in sessions:
        for key in _input_keys(sess):
            aws.s3_stub.add_response(
                "head_object", head_object_response(),
                expected_params={"Bucket": LT.BUCKET, "Key": key})


def _capacity_error():
    return ClientError(
        {"Error": {"Code": "InsufficientInstanceCapacity",
                   "Message": "Insufficient capacity."}},
        "RunInstances")


# --------------------------------------------------------------------------

class TestBillingSafety:
    """Invariants whose failure costs money or makes an instance unkillable."""

    def test_dry_run_makes_no_aws_calls_at_all(self, aws, monkeypatch, capsys):
        """
        The whole point of --dry-run is that nothing happens. The Stubber has no
        queued responses, so any API call raises.
        """
        aws.install(monkeypatch, LT)
        launched, failed = LT.launch(SESSIONS, RUN_ID, SHA, "g5.2xlarge", "12h",
                                     "aws_batch", dry_run=True)
        assert (launched, failed) == ([], []), \
            "dry-run must return the empty (launched, failed) contract, not None"
        assert not any(c for c in aws.calls if c[0] != "client")
        aws.assert_no_pending()

    def test_one_instance_per_session_never_a_bulk_request(self, aws, monkeypatch):
        """
        MinCount=MaxCount=1 per call is what makes a capacity shortfall affect one
        session instead of the whole sweep. A bulk request would be
        all-or-nothing, and MaxCount>1 would silently launch duplicates.
        """
        aws.install(monkeypatch, LT)
        seen = []

        def record(**kw):
            seen.append(kw)
            return run_instances_response(f"i-{len(seen):017x}")

        with mock.patch.object(aws.ec2, "run_instances", side_effect=record):
            launched, failed = LT.launch(SESSIONS, RUN_ID, SHA, "g5.2xlarge",
                                         "12h", "aws_batch", dry_run=False)

        assert len(seen) == len(SESSIONS)
        assert len(launched) == len(SESSIONS) and failed == []
        for kw in seen:
            assert kw["MinCount"] == 1 and kw["MaxCount"] == 1

    def test_every_instance_is_killable_and_self_terminating(self, aws, monkeypatch):
        """
        The IAM policy only permits terminating instances tagged
        Project=video-autoencoder. Without that tag a runaway instance cannot be
        stopped at all, which is the most expensive possible bug here.

        InstanceInitiatedShutdownBehavior=terminate is what turns the wrapper's
        `shutdown -h now` into a termination rather than a stop.
        """
        aws.install(monkeypatch, LT)
        seen = []

        def record(**kw):
            seen.append(kw)
            return run_instances_response(f"i-{len(seen):017x}")

        with mock.patch.object(aws.ec2, "run_instances", side_effect=record):
            LT.launch(SESSIONS, RUN_ID, SHA, "g5.2xlarge", "12h", "aws_batch",
                      dry_run=False)

        for kw, sess in zip(seen, SESSIONS):
            assert kw["InstanceInitiatedShutdownBehavior"] == "terminate"
            tags = {t["Key"]: t["Value"] for t in kw["TagSpecifications"][0]["Tags"]}
            assert tags["Project"] == "video-autoencoder", \
                "without this tag the IAM policy cannot terminate the instance"
            assert tags["RunId"] == RUN_ID      # --status and the kill command
            assert tags["Session"] == sess
            assert tags["Env"] == "aws_batch"

    def test_pinned_sha_and_env_reach_the_instance(self, aws, monkeypatch):
        """
        A sweep is only reconstructible if every instance checks out the same
        pinned sha, and only correct if it loads the intended env config.
        """
        ud = LT.build_user_data(SESSIONS[0], SHA, RUN_ID, "12h", "aws_batch")
        assert f'SHA="{SHA}"' in ud
        assert "--env aws_batch" in ud
        assert 'git checkout --quiet "$SHA"' in ud
        assert f'run_id={RUN_ID}' in ud


class TestUserData:
    """EC2 rejects or silently mis-runs user-data that breaks these."""

    @pytest.mark.parametrize("session", SESSIONS)
    def test_contract(self, session):
        ud = LT.build_user_data(session, SHA, RUN_ID, "12h", "aws_batch")
        # A shebang not at byte 0 means the kernel falls back to /bin/sh and
        # every bash-ism dies with the output nowhere visible.
        assert ud.startswith("#!/bin/bash\n")
        assert "@@" not in ud, "unsubstituted placeholder would reach the instance"
        assert ">(" not in ud, "process substitution is unavailable under /bin/sh"
        assert len(ud) < 16384, "EC2 rejects user-data above 16 KB"

    def test_animal_is_derived_from_the_session_name(self):
        ud = LT.build_user_data("kd104_twNew_20221124_104921", SHA, RUN_ID, "12h",
                                "aws_batch")
        assert 'ANIMAL="kd104"' in ud

    def test_timeout_is_applied_to_the_training_command(self):
        ud = LT.build_user_data(SESSIONS[0], SHA, RUN_ID, "7h", "aws_batch")
        assert "timeout 7h" in ud

    def test_unsubstituted_placeholder_is_caught_not_shipped(self, monkeypatch):
        """The assertions inside build_user_data are load-bearing, not decoration."""
        monkeypatch.setattr(LT, "USER_DATA_TEMPLATE",
                            LT.USER_DATA_TEMPLATE + "\n# @@NEVER_SUBSTITUTED@@\n")
        with pytest.raises(AssertionError, match="unsubstituted placeholder"):
            LT.build_user_data(SESSIONS[0], SHA, RUN_ID, "12h", "aws_batch")


class TestFailureHandling:
    """A capacity shortfall on one session must not abandon the rest."""

    def test_capacity_failure_does_not_abort_the_loop(self, aws, monkeypatch):
        sessions = [f"kd10{i}_sess" for i in range(5)]
        fail_on = {sessions[1], sessions[3]}
        aws.install(monkeypatch, LT)
        attempted = []

        def maybe_fail(**kw):
            sess = next(t["Value"] for t in kw["TagSpecifications"][0]["Tags"]
                        if t["Key"] == "Session")
            attempted.append(sess)
            if sess in fail_on:
                raise _capacity_error()
            return run_instances_response(f"i-{len(attempted):017x}")

        with mock.patch.object(aws.ec2, "run_instances", side_effect=maybe_fail):
            launched, failed = LT.launch(sessions, RUN_ID, SHA, "g5.2xlarge",
                                         "12h", "aws_batch", dry_run=False)

        assert attempted == sessions, "every session must be attempted"
        assert [s for _, s in launched] == [s for s in sessions if s not in fail_on]
        assert [s for s, _, _ in failed] == [s for s in sessions if s in fail_on]
        assert all(code == "InsufficientInstanceCapacity" for _, code, _ in failed)

    def test_connection_error_is_also_recorded_not_raised(self, aws, monkeypatch):
        """BotoCoreError is a different branch from ClientError; both must be caught."""
        aws.install(monkeypatch, LT)
        with mock.patch.object(
                aws.ec2, "run_instances",
                side_effect=EndpointConnectionError(endpoint_url="https://ec2")):
            launched, failed = LT.launch(SESSIONS[:1], RUN_ID, SHA, "g5.2xlarge",
                                         "12h", "aws_batch", dry_run=False)
        assert launched == [] and len(failed) == 1
        assert failed[0][1] == "EndpointConnectionError"

    def test_a_genuine_bug_still_propagates(self, aws, monkeypatch):
        """
        Only AWS-side failures are swallowed. A programming error must not be
        silently turned into "capacity trouble".
        """
        aws.install(monkeypatch, LT)
        with mock.patch.object(aws.ec2, "run_instances",
                               side_effect=TypeError("bad argument")):
            with pytest.raises(TypeError):
                LT.launch(SESSIONS[:1], RUN_ID, SHA, "g5.2xlarge", "12h",
                          "aws_batch", dry_run=False)


class TestPreflight:
    """Every check here exists to fail locally instead of on 20 instances."""

    def _patch_github(self, monkeypatch, body):
        monkeypatch.setattr(LT, "fetch_at_sha", lambda sha, path: body)

    GOOD_SCRIPT = ("--mean_frame_path --h5_path --bpod_path --animal --session")
    GOOD_CONFIG = json.dumps({"training": {"epochs": 1500}})

    def test_passes_when_everything_is_present(self, aws, monkeypatch, capsys):
        self._patch_github(monkeypatch, self.GOOD_SCRIPT)
        monkeypatch.setattr(LT, "fetch_at_sha",
                            lambda sha, p: self.GOOD_CONFIG
                            if p.startswith("configs/") else self.GOOD_SCRIPT)
        aws.install(monkeypatch, LT)
        _queue_inputs_present(aws, SESSIONS)
        LT.preflight(SESSIONS, SHA, "aws_batch")
        out = capsys.readouterr().out
        assert "all inputs present for 2 session(s)" in out
        aws.assert_no_pending()

    def test_rejects_a_training_script_missing_a_required_flag(self, monkeypatch):
        self._patch_github(monkeypatch, "--h5_path --animal --session")   # no --bpod_path
        with pytest.raises(SystemExit):
            LT.preflight(SESSIONS, SHA, "aws_batch")

    def test_rejects_a_missing_env_config(self, monkeypatch):
        def fetch(sha, path):
            if path.startswith("configs/"):
                raise RuntimeError("404")
            return self.GOOD_SCRIPT
        monkeypatch.setattr(LT, "fetch_at_sha", fetch)
        with pytest.raises(SystemExit, match="could not fetch configs/"):
            LT.preflight(SESSIONS, SHA, "aws_batch")

    def test_resolves_an_extends_chain(self, monkeypatch, aws, capsys):
        """A variant env inherits from its parent; both must exist at the sha."""
        seen = []

        def fetch(sha, path):
            seen.append(path)
            if path == "configs/aws_batch_fastcycle_config.json":
                return json.dumps({"extends": "aws_batch", "training": {"T_mult": 1}})
            if path == "configs/aws_batch_config.json":
                return self.GOOD_CONFIG
            return self.GOOD_SCRIPT
        monkeypatch.setattr(LT, "fetch_at_sha", fetch)
        aws.install(monkeypatch, LT)
        _queue_inputs_present(aws, SESSIONS[:1])
        LT.preflight(SESSIONS[:1], SHA, "aws_batch_fastcycle")
        assert "configs/aws_batch_config.json" in seen, "parent config not checked"
        assert "aws_batch_fastcycle -> aws_batch" in capsys.readouterr().out

    def test_rejects_a_missing_s3_input(self, aws, monkeypatch):
        monkeypatch.setattr(LT, "fetch_at_sha",
                            lambda sha, p: self.GOOD_CONFIG
                            if p.startswith("configs/") else self.GOOD_SCRIPT)
        aws.install(monkeypatch, LT)
        keys = _input_keys(SESSIONS[0])
        aws.s3_stub.add_response("head_object", head_object_response(),
                                 expected_params={"Bucket": LT.BUCKET,
                                                  "Key": keys[0]})
        # the bpod object is the one missing
        aws.s3_stub.add_client_error("head_object", service_error_code="404")
        aws.s3_stub.add_response("head_object", head_object_response(),
                                 expected_params={"Bucket": LT.BUCKET,
                                                  "Key": keys[2]})
        with pytest.raises(SystemExit):
            LT.preflight(SESSIONS[:1], SHA, "aws_batch")


class TestResolveSha:
    def test_returns_the_remote_tip(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: mock.Mock(returncode=0, stdout=f"{SHA}\trefs/heads/x\n"))
        assert LT.resolve_sha("balint-dev") == SHA

    def test_exits_when_the_ref_does_not_exist(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: mock.Mock(returncode=0, stdout="",
                                                      stderr=""))
        with pytest.raises(SystemExit):
            LT.resolve_sha("no-such-branch")


class TestSharedValidation:
    """
    validate_user_data lives in aws_launch_common and guards every launcher.

    These exercise the guard itself rather than a real user-data string: the
    generated ones are legitimately small, so weakening the size assertion would
    otherwise go unnoticed until a template grew past 16 KB in production.
    """

    def test_rejects_user_data_over_the_ec2_cap(self):
        import aws_launch_common as C
        oversized = "#!/bin/bash\n" + "x" * C.MAX_USER_DATA
        with pytest.raises(AssertionError, match="user-data too large"):
            C.validate_user_data(oversized)

    def test_accepts_user_data_just_under_the_cap(self):
        import aws_launch_common as C
        head = "#!/bin/bash\n"
        ok = head + "x" * (C.MAX_USER_DATA - len(head) - 1)
        assert C.validate_user_data(ok) is ok

    @pytest.mark.parametrize("bad,match", [
        ("# comment\n#!/bin/bash\n", "shebang must be at byte 0"),
        ("#!/bin/bash\n@@UNSET@@\n", "unsubstituted placeholder"),
        ("#!/bin/bash\ncat >(tee x)\n", "no process substitution"),
    ])
    def test_rejects_each_way_of_bricking_an_instance(self, bad, match):
        import aws_launch_common as C
        with pytest.raises(AssertionError, match=match):
            C.validate_user_data(bad)


class TestRetryCommandAndIdPinning:
    """
    Covers main(), which the rest of this file does not reach. The retry command
    is only useful if pasting it puts the retried sessions in the same sweep:
    same pinned sha, same run id, hence the same S3 prefix that --status reads.
    """

    def _run_main(self, monkeypatch, argv, failed, launched=None):
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(LT, "resolve_sha", lambda ref: SHA)
        monkeypatch.setattr(LT, "preflight", lambda sessions, sha, env: None)
        captured = {}

        def fake_launch(sessions, run_id, sha, itype, timeout, env, dry_run):
            captured.update(run_id=run_id, sha=sha, sessions=sessions)
            return (launched or []), failed

        monkeypatch.setattr(LT, "launch", fake_launch)
        return captured

    def test_a_generated_run_id_is_pinned_into_the_retry(self, monkeypatch, capsys):
        cap = self._run_main(
            monkeypatch,
            ["scripts/launch_training.py", "--sessions", "sess_a", "sess_b"],
            failed=[("sess_a", "InsufficientInstanceCapacity", "no capacity")])
        with pytest.raises(SystemExit):
            LT.main()
        out = capsys.readouterr().out
        assert f"--run-id {cap['run_id']}" in out, \
            "retry must reuse the same run id, not start a second sweep prefix"
        assert f"--sha {SHA}" in out

    def test_an_explicit_run_id_is_honoured(self, monkeypatch, capsys):
        cap = self._run_main(
            monkeypatch,
            ["scripts/launch_training.py", "--sessions", "sess_a",
             "--run-id", "20260831-PINNED"],
            failed=[("sess_a", "InsufficientInstanceCapacity", "no capacity")])
        with pytest.raises(SystemExit):
            LT.main()
        assert cap["run_id"] == "20260831-PINNED"
        assert "--run-id 20260831-PINNED" in capsys.readouterr().out

    def test_env_and_timeout_are_carried_into_the_retry(self, monkeypatch, capsys):
        """A retry under a different env would silently be a different experiment."""
        self._run_main(
            monkeypatch,
            ["scripts/launch_training.py", "--sessions", "sess_a",
             "--env", "aws_batch_fastcycle", "--timeout", "9h"],
            failed=[("sess_a", "InsufficientInstanceCapacity", "no capacity")])
        with pytest.raises(SystemExit):
            LT.main()
        out = capsys.readouterr().out
        assert "--env aws_batch_fastcycle" in out
        assert "--timeout 9h" in out

    def test_success_exits_cleanly_with_no_retry_block(self, monkeypatch, capsys):
        self._run_main(monkeypatch,
                       ["scripts/launch_training.py", "--sessions", "sess_a"],
                       launched=[("i-1", "sess_a")], failed=[])
        LT.main()
        assert "did NOT launch" not in capsys.readouterr().out


class TestRetryCommandFormatting:
    """The shared formatter -- both launchers depend on it being paste-able."""

    def test_is_a_single_shell_command_with_continuations(self):
        import aws_launch_common as C
        cmd = C.retry_command("scripts/x.py", {"sha": "abc", "run-id": "r1"},
                              ["s1", "s2"])
        lines = cmd.splitlines()
        # every line but the last must end in a backslash continuation
        assert all(l.rstrip().endswith("\\") for l in lines[:-1]), cmd
        assert not lines[-1].rstrip().endswith("\\")
        assert lines[-1].strip() == "--sessions s1 s2"
        assert "--sha abc" in cmd and "--run-id r1" in cmd

    def test_survives_many_sessions_and_flags(self):
        import aws_launch_common as C
        cmd = C.retry_command("scripts/x.py",
                              {f"flag{i}": f"value{i}" for i in range(8)},
                              [f"session_{i}" for i in range(20)])
        assert all(l.rstrip().endswith("\\") for l in cmd.splitlines()[:-1])
        for i in range(20):
            assert f"session_{i}" in cmd

    def test_never_splits_a_flag_from_its_value(self):
        """
        Regression: textwrap wrapped on whitespace, so a long flag list put
        "--run-id" on one line and its value on the next. Shell-valid, but the
        flag no longer reads as a unit and cannot be grepped for.
        """
        import aws_launch_common as C
        flags = {"sha": "b" * 40, "run-id": "20260831-105222",
                 "env": "aws_batch_fastcycle", "instance-type": "g5.2xlarge",
                 "timeout": "12h"}
        cmd = C.retry_command("scripts/x.py", flags, ["s1"])
        for k, v in flags.items():
            assert f"--{k} {v}" in cmd, f"--{k} was split from its value:\n{cmd}"
