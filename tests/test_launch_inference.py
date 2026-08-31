"""
Tests for scripts/launch_inference.py.

Same purpose as test_launch_training.py: a regression net for the shared-plumbing
refactor. This launcher has extra machinery of its own worth pinning -- it
resolves checkpoints out of S3, and it groups jobs by SESSION rather than by run
so that both arms of an A/B stage the 13.9 GB h5 once instead of twice.

test_capacity_failure_does_not_abort_the_remaining_sessions was xfail(strict=True)
until the shared-plumbing refactor: this launcher had no per-session handling,
which is the gap that prompted these tests. The refactor made it xpass, pytest
reported that as a failure, and the marker came off.
"""

import json
import sys
from unittest import mock

import pytest
from botocore.exceptions import ClientError

import launch_inference as LI
from conftest import head_object_response, list_objects_prefixes, run_instances_response

SESSION = "kd104_twNew_20221124_104921"
RUN_A = "20260827-165607"
RUN_B = "20260827-165614"
SHA = "b1bcf697d7dad556cb58af016b44dd284aeb82ff"
INFER_ID = "infer-20260828-120632"
CKPT_DIR_A = (f"runs/{RUN_A}/{SESSION}/checkpoints/kd104/"
              f"{SESSION}_2026-08-28_00-01-36/")
CKPT_DIR_B = (f"runs/{RUN_B}/{SESSION}/checkpoints/kd104/"
              f"{SESSION}_2026-08-28_00-01-36/")


def _queue_job_resolution(aws, runs_and_dirs, epochs):
    """
    Queue the S3 calls build_jobs makes per run:
      list_objects_v2 (sessions)  -> list_objects_v2 (ckpt dir) -> head_object
      (config.json) -> head_object per requested checkpoint
    """
    for run_id, ckpt_dir in runs_and_dirs:
        aws.s3_stub.add_response(
            "list_objects_v2", list_objects_prefixes([f"runs/{run_id}/{SESSION}/"]),
            expected_params={"Bucket": LI.BUCKET,
                             "Prefix": f"runs/{run_id}/", "Delimiter": "/"})
        aws.s3_stub.add_response(
            "list_objects_v2", list_objects_prefixes([ckpt_dir]),
            expected_params={"Bucket": LI.BUCKET,
                             "Prefix": f"runs/{run_id}/{SESSION}/checkpoints/kd104/",
                             "Delimiter": "/"})
        aws.s3_stub.add_response(
            "head_object", head_object_response(),
            expected_params={"Bucket": LI.BUCKET, "Key": ckpt_dir + "config.json"})
        for ep in epochs:
            aws.s3_stub.add_response(
                "head_object", head_object_response(),
                expected_params={"Bucket": LI.BUCKET,
                                 "Key": ckpt_dir + f"checkpoint_epoch_{ep}.pt"})


class TestCheckpointNaming:
    @pytest.mark.parametrize("label,expected", [
        ("1499", "checkpoint_epoch_1499.pt"),
        ("7499", "checkpoint_epoch_7499.pt"),
        ("best", "best_model.pt"),
        ("final", "final_model.pt"),
    ])
    def test_label_maps_to_the_right_object(self, label, expected):
        assert LI.ckpt_object(label) == expected

    def test_a_non_numeric_non_keyword_label_is_rejected(self):
        with pytest.raises(ValueError):
            LI.ckpt_object("penultimate")


class TestJobResolution:
    def test_two_runs_of_one_session_share_a_single_instance(self, aws, monkeypatch):
        """
        The point of grouping by session: both A/B arms are the same session, so
        six checkpoints must land on ONE instance and stage the h5 once.
        """
        aws.install(monkeypatch, LI)
        epochs = ["1499", "3499", "7499"]
        _queue_job_resolution(aws, [(RUN_A, CKPT_DIR_A), (RUN_B, CKPT_DIR_B)], epochs)

        jobs = LI.build_jobs([RUN_A, RUN_B], [], epochs)

        assert list(jobs) == [SESSION], "grouping must be per session, not per run"
        assert len(jobs[SESSION]) == 6
        assert [r for r, _, _, _ in jobs[SESSION]] == [RUN_A] * 3 + [RUN_B] * 3
        assert [l for _, l, _, _ in jobs[SESSION]] == \
            ["epoch_1499", "epoch_3499", "epoch_7499"] * 2
        aws.assert_no_pending()

    def test_each_checkpoint_is_paired_with_its_own_config(self, aws, monkeypatch):
        """
        Inference rebuilds the model from the config.json saved beside the
        checkpoint. A checkpoint paired with the wrong run's config would rebuild
        a differently-configured model.
        """
        aws.install(monkeypatch, LI)
        _queue_job_resolution(aws, [(RUN_A, CKPT_DIR_A), (RUN_B, CKPT_DIR_B)],
                              ["1499"])
        jobs = LI.build_jobs([RUN_A, RUN_B], [], ["1499"])
        for run_id, _, ckpt_key, config_key in jobs[SESSION]:
            assert ckpt_key.rsplit("/", 1)[0] == config_key.rsplit("/", 1)[0], \
                "checkpoint and config must come from the same folder"
            assert f"runs/{run_id}/" in ckpt_key

    def test_missing_checkpoint_fails_before_anything_launches(self, aws, monkeypatch):
        aws.install(monkeypatch, LI)
        aws.s3_stub.add_response(
            "list_objects_v2", list_objects_prefixes([f"runs/{RUN_A}/{SESSION}/"]),
            expected_params={"Bucket": LI.BUCKET, "Prefix": f"runs/{RUN_A}/",
                             "Delimiter": "/"})
        aws.s3_stub.add_response(
            "list_objects_v2", list_objects_prefixes([CKPT_DIR_A]),
            expected_params={"Bucket": LI.BUCKET,
                             "Prefix": f"runs/{RUN_A}/{SESSION}/checkpoints/kd104/",
                             "Delimiter": "/"})
        aws.s3_stub.add_response(
            "head_object", head_object_response(),
            expected_params={"Bucket": LI.BUCKET, "Key": CKPT_DIR_A + "config.json"})
        aws.s3_stub.add_client_error("head_object", service_error_code="404")
        with pytest.raises(SystemExit, match="checkpoint not in S3"):
            LI.build_jobs([RUN_A], [], ["9999"])

    def test_missing_config_json_fails(self, aws, monkeypatch):
        aws.install(monkeypatch, LI)
        aws.s3_stub.add_response(
            "list_objects_v2", list_objects_prefixes([f"runs/{RUN_A}/{SESSION}/"]),
            expected_params={"Bucket": LI.BUCKET, "Prefix": f"runs/{RUN_A}/",
                             "Delimiter": "/"})
        aws.s3_stub.add_response(
            "list_objects_v2", list_objects_prefixes([CKPT_DIR_A]),
            expected_params={"Bucket": LI.BUCKET,
                             "Prefix": f"runs/{RUN_A}/{SESSION}/checkpoints/kd104/",
                             "Delimiter": "/"})
        aws.s3_stub.add_client_error("head_object", service_error_code="404")
        with pytest.raises(SystemExit, match="no config.json"):
            LI.build_jobs([RUN_A], [], ["1499"])

    def test_ambiguous_checkpoint_folder_is_refused(self, aws, monkeypatch):
        """
        One training run creates exactly one date-stamped folder. Two means the
        prefix was reused, and silently picking one would be a coin flip.
        """
        aws.install(monkeypatch, LI)
        aws.s3_stub.add_response(
            "list_objects_v2",
            list_objects_prefixes([CKPT_DIR_A, CKPT_DIR_A.replace("00-01", "11-11")]),
            expected_params={"Bucket": LI.BUCKET,
                             "Prefix": f"runs/{RUN_A}/{SESSION}/checkpoints/kd104/",
                             "Delimiter": "/"})
        with pytest.raises(SystemExit, match="expected 1"):
            LI.find_ckpt_dir(RUN_A, SESSION)

    def test_session_filter_rejects_an_unknown_session(self, aws, monkeypatch):
        aws.install(monkeypatch, LI)
        aws.s3_stub.add_response(
            "list_objects_v2", list_objects_prefixes([f"runs/{RUN_A}/{SESSION}/"]),
            expected_params={"Bucket": LI.BUCKET, "Prefix": f"runs/{RUN_A}/",
                             "Delimiter": "/"})
        with pytest.raises(SystemExit, match="no session"):
            LI.build_jobs([RUN_A], ["kd999_not_a_session"], ["1499"])


class TestWritePrefixGuard:
    """
    This check exists because a run once wrote to a prefix the instance role had
    no PutObject grant for. Every sync in the wrapper past the probe ends in
    `|| true`, so it failed silently: the work ran, the instance terminated, and
    S3 was empty.
    """

    def test_allows_a_prefix_the_role_can_write(self, capsys):
        LI.check_s3_write_prefix("runs/")
        assert "can write under 'runs/'" in capsys.readouterr().out

    def test_rejects_a_prefix_the_role_cannot_write(self):
        with pytest.raises(SystemExit, match="cannot PutObject under 'benchmarks/'"):
            LI.check_s3_write_prefix("benchmarks/")

    def test_warns_rather_than_crashes_when_the_policy_file_is_absent(self, capsys):
        # preflight passes the launcher's POLICY_FILE explicitly, so the test
        # exercises the same call shape rather than patching a module global.
        LI.check_s3_write_prefix("anything/", "/nonexistent/policy.json")
        assert "WARN" in capsys.readouterr().out

    def test_preflight_checks_the_prefix_with_the_launchers_policy_file(
            self, monkeypatch):
        """Pins the wiring: preflight must pass POLICY_FILE, not rely on a default."""
        seen = {}
        monkeypatch.setattr(LI, "check_s3_write_prefix",
                            lambda prefix, pf=None: seen.update(prefix=prefix, pf=pf))
        monkeypatch.setattr(LI, "fetch_at_sha", lambda sha, p: TestPreflight.GOOD_SCRIPT)
        monkeypatch.setattr(LI, "missing_session_inputs", lambda s, **k: [])
        LI.preflight({SESSION: []}, SHA)
        assert seen["prefix"] == LI.RUNS_PREFIX
        assert seen["pf"] == LI.POLICY_FILE


class TestUserData:
    JOBS = [(RUN_A, "epoch_1499", CKPT_DIR_A + "checkpoint_epoch_1499.pt",
             CKPT_DIR_A + "config.json"),
            (RUN_B, "epoch_7499", CKPT_DIR_B + "checkpoint_epoch_7499.pt",
             CKPT_DIR_B + "config.json")]

    def _ud(self, save_recons=False, n_jobs=None):
        jobs = self.JOBS if n_jobs is None else self.JOBS * n_jobs
        return LI.build_user_data(SESSION, jobs, SHA, INFER_ID, "g5.2xlarge",
                                  "2h", save_recons)

    def test_contract(self):
        ud = self._ud()
        assert ud.startswith("#!/bin/bash\n")
        assert "@@" not in ud
        assert ">(" not in ud
        assert len(ud) < 16384

    def test_manifest_has_one_pipe_delimited_line_per_job(self):
        ud = self._ud()
        block = ud.split("cat > $MANIFEST <<'MANIFEST_EOF'\n")[1].split("MANIFEST_EOF")[0]
        lines = [l for l in block.strip().splitlines() if l]
        assert len(lines) == len(self.JOBS)
        for line, (run_id, label, ckpt, cfg) in zip(lines, self.JOBS):
            assert line.split("|") == [run_id, label, ckpt, cfg]

    def test_recons_flag_is_two_way(self):
        """
        The inference script buffers reconstructions in RAM (~11 GB for a full
        session), so the launcher must be able to turn them off -- and must not
        pass a bare flag that the script would read as always-on.
        """
        assert "--no-save_recons" in self._ud(save_recons=False)
        assert "--no-save_recons" not in self._ud(save_recons=True)
        assert "--save_recons" in self._ud(save_recons=True)

    def test_user_data_stays_under_the_cap_for_many_checkpoints(self):
        """A 20-checkpoint manifest must still fit EC2's 16 KB limit."""
        ud = self._ud(n_jobs=10)
        assert len(ud) < 16384, f"user-data grew to {len(ud)} bytes"

    def test_outputs_are_written_under_each_jobs_own_run(self):
        """
        Arm A's latents must not land under arm B's prefix. The sync target has
        to interpolate $RUN_ID, read per job from the manifest -- not the single
        FIRST_RUN value that the shared status/log path uses.
        """
        ud = self._ud()
        assert '"$S3/runs/$RUN_ID/$SESSION/inference/$INFER_ID/$LABEL/"' in ud
        # the status/log path is allowed to be pinned to the first run
        assert f'S3_STATUS="$S3/runs/{RUN_A}/$SESSION/inference/$INFER_ID"' in ud


class TestPreflight:
    GOOD_SCRIPT = ("--checkpoint --h5_path --bpod_path --output_dir --animal "
                   "--session save_recons BooleanOptionalAction")

    def test_rejects_a_script_predating_the_two_way_recons_flag(self, monkeypatch):
        """
        An older inference script would silently buffer ~11 GB and OOM, so the
        launcher refuses to run it rather than paying for the failure.
        """
        monkeypatch.setattr(LI, "fetch_at_sha", lambda sha, p:
                            self.GOOD_SCRIPT.replace("BooleanOptionalAction", ""))
        with pytest.raises(SystemExit, match="two-way"):
            LI.preflight({SESSION: []}, SHA)

    def test_rejects_a_script_missing_a_required_flag(self, monkeypatch):
        monkeypatch.setattr(LI, "fetch_at_sha", lambda sha, p:
                            self.GOOD_SCRIPT.replace("--bpod_path", ""))
        with pytest.raises(SystemExit, match="has no --bpod_path"):
            LI.preflight({SESSION: []}, SHA)

    def test_checks_the_session_inputs_exist(self, aws, monkeypatch, capsys):
        monkeypatch.setattr(LI, "fetch_at_sha", lambda sha, p: self.GOOD_SCRIPT)
        aws.install(monkeypatch, LI)
        animal = SESSION.split("_")[0]
        for key in (f"{LI.H5_PREFIX}{SESSION}{LI.H5_SUFFIX}",
                    f"{LI.BPOD_PREFIX}{animal}/{SESSION}.bpod.npy"):
            aws.s3_stub.add_response("head_object", head_object_response(),
                                     expected_params={"Bucket": LI.BUCKET,
                                                      "Key": key})
        LI.preflight({SESSION: [("r", "l", "c", "cfg")]}, SHA)
        out = capsys.readouterr().out
        assert "session inputs present for 1 session(s)" in out
        aws.assert_no_pending()


class TestBillingSafety:
    def test_dry_run_makes_no_aws_calls(self, aws, monkeypatch):
        aws.install(monkeypatch, LI)
        jobs = {SESSION: TestUserData.JOBS}
        LI.launch(jobs, SHA, INFER_ID, "g5.2xlarge", "2h", False, dry_run=True)
        aws.assert_no_pending()

    def test_every_instance_is_killable_and_self_terminating(self, aws, monkeypatch):
        aws.install(monkeypatch, LI)
        seen = []

        def record(**kw):
            seen.append(kw)
            return run_instances_response(f"i-{len(seen):017x}")

        with mock.patch.object(aws.ec2, "run_instances", side_effect=record):
            LI.launch({SESSION: TestUserData.JOBS}, SHA, INFER_ID, "g5.2xlarge",
                      "2h", False, dry_run=False)

        assert len(seen) == 1, "one instance per session"
        kw = seen[0]
        assert kw["MinCount"] == 1 and kw["MaxCount"] == 1
        assert kw["InstanceInitiatedShutdownBehavior"] == "terminate"
        tags = {t["Key"]: t["Value"] for t in kw["TagSpecifications"][0]["Tags"]}
        assert tags["Project"] == "video-autoencoder"
        assert tags["RunId"] == INFER_ID
        assert tags["Session"] == SESSION
        assert tags["Kind"] == "inference"


def test_capacity_failure_does_not_abort_the_remaining_sessions(aws, monkeypatch):
    """Was xfail until the shared-plumbing refactor gave this launcher the same
    per-session handling launch_training already had."""
    sessions = {"sess_a": TestUserData.JOBS, "sess_b": TestUserData.JOBS,
                "sess_c": TestUserData.JOBS}
    aws.install(monkeypatch, LI)
    attempted = []

    def maybe_fail(**kw):
        sess = next(t["Value"] for t in kw["TagSpecifications"][0]["Tags"]
                    if t["Key"] == "Session")
        attempted.append(sess)
        if sess == "sess_a":
            raise ClientError({"Error": {"Code": "InsufficientInstanceCapacity",
                                         "Message": "Insufficient capacity."}},
                              "RunInstances")
        return run_instances_response("i-0000000000000000a")

    with mock.patch.object(aws.ec2, "run_instances", side_effect=maybe_fail):
        launched, failed = LI.launch(sessions, SHA, INFER_ID, "g5.2xlarge", "2h",
                                     False, dry_run=False)

    assert attempted == list(sessions), "every session must be attempted"
    assert [n for _, n in launched] == ["sess_b", "sess_c"]
    assert [n for n, _, _ in failed] == ["sess_a"]
    assert failed[0][1] == "InsufficientInstanceCapacity"


class TestRetryCommandAndIdPinning:
    """
    This is the gap the earlier tests missed: everything below lives in main(),
    which nothing exercised. The retry command is only useful if it is genuinely
    paste-able -- pinning the sha so retried sessions run the same code, and the
    inference id so their outputs land in the SAME S3 prefix. Without the id a
    retry mints a fresh one and the latents end up split across two infer-*
    folders, which the notebook's "newest batch per run" lookup would silently
    read only half of.
    """

    def _run_main(self, monkeypatch, argv, failed, launched=None):
        """Drive main() with everything below launch() stubbed out."""
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(LI, "resolve_sha", lambda ref: SHA)
        monkeypatch.setattr(LI, "build_jobs",
                            lambda runs, sess, cks: {"sess_a": [], "sess_b": []})
        monkeypatch.setattr(LI, "preflight", lambda jobs, sha: None)
        captured = {}

        def fake_launch(jobs, sha, infer_id, itype, timeout, save_recons, dry_run):
            captured["infer_id"] = infer_id
            captured["sha"] = sha
            return (launched or []), failed

        monkeypatch.setattr(LI, "launch", fake_launch)
        return captured

    def test_a_generated_inference_id_is_pinned_into_the_retry(
            self, monkeypatch, capsys):
        cap = self._run_main(
            monkeypatch, ["scripts/launch_inference.py", "--runs", "R1",
                          "--checkpoints", "final"],
            failed=[("sess_a", "InsufficientInstanceCapacity", "no capacity")])
        with pytest.raises(SystemExit):
            LI.main()
        out = capsys.readouterr().out
        assert cap["infer_id"].startswith("infer-")
        assert f"--infer-id {cap['infer_id']}" in out, \
            "retry must reuse the same inference id, not mint a new one"
        assert f"--sha {SHA}" in out
        assert out.rstrip().endswith("=" * 72)

    def test_an_explicit_infer_id_is_honoured(self, monkeypatch, capsys):
        cap = self._run_main(
            monkeypatch,
            ["scripts/launch_inference.py", "--runs", "R1", "--checkpoints",
             "final", "--infer-id", "infer-PINNED"],
            failed=[("sess_a", "InsufficientInstanceCapacity", "no capacity")])
        with pytest.raises(SystemExit):
            LI.main()
        assert cap["infer_id"] == "infer-PINNED"
        assert "--infer-id infer-PINNED" in capsys.readouterr().out

    def test_retry_lists_only_the_failed_sessions(self, monkeypatch, capsys):
        self._run_main(
            monkeypatch, ["scripts/launch_inference.py", "--runs", "R1",
                          "--checkpoints", "final"],
            launched=[("i-1", "sess_b")],
            failed=[("sess_a", "InsufficientInstanceCapacity", "no capacity")])
        with pytest.raises(SystemExit):
            LI.main()
        out = capsys.readouterr().out
        sessions_line = [l for l in out.splitlines() if "--sessions" in l][0]
        assert "sess_a" in sessions_line
        assert "sess_b" not in sessions_line, "a launched session must not be retried"

    def test_no_failure_summary_and_no_exit_when_everything_launches(
            self, monkeypatch, capsys):
        self._run_main(monkeypatch,
                       ["scripts/launch_inference.py", "--runs", "R1",
                        "--checkpoints", "final"],
                       launched=[("i-1", "sess_a"), ("i-2", "sess_b")], failed=[])
        LI.main()          # must NOT raise SystemExit
        out = capsys.readouterr().out
        assert "did NOT launch" not in out
        assert "2/2 instance(s) launched" in out
