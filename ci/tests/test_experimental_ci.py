"""The merged-PR entry must gate on real experimental results, not host smoke."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import copy
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("experimental_ci", ROOT / "ci/run_experimental_ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class ExperimentalCI(unittest.TestCase):
    def test_formal_gdr_library_paths_do_not_leak_into_experimental_processes(self):
        paths = ["/cann/lib64", "/site/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib",
                 "/site/fla_npu/opp/vendors/fla_npu_overlay_test/op_api/lib", "/driver/lib64"]
        with patch.dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(paths)):
            env = ci.clean_environment({"mode": "stable", "scenario": "stable"}, {})
        self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep), [paths[0], paths[-1]])

    def execute(self, soc="ascend910b", fail=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, CI_SOC=soc, CI_MODE="quick"):
                runner = ci.Validation(root)
            calls = []

            def action(name):
                calls.append(name)
                if fail == name:
                    raise RuntimeError("injected missing/failed result")
                return {"checked": name}

            with patch.object(runner, "dependencies", lambda: action("dependencies")), \
                    patch.object(runner, "build", lambda: action("build")), \
                    patch.object(runner, "isolation", lambda: action("isolation")), \
                    patch.object(runner, "accuracy", lambda mode: action(mode)), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = runner.execute()
            return code, calls, json.loads((root / "summary.json").read_text())

    def test_a2_runs_both_launch_modes_after_isolation(self):
        code, calls, result = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["dependencies", "build", "isolation", "queue", "inline"])
        self.assertTrue(result["complete"])

    def test_a5_explicitly_does_not_claim_accuracy(self):
        code, calls, result = self.execute("ascend950")
        self.assertEqual((code, calls), (0, ["dependencies", "build"]))
        self.assertEqual(result["phases"]["accuracy-queue"]["status"], "skipped")

    def test_failure_prevents_following_checks_and_is_uploaded(self):
        for phase in ("dependencies", "build", "isolation", "queue", "inline"):
            with self.subTest(phase=phase):
                code, calls, result = self.execute(fail=phase)
                self.assertEqual(code, 1)
                self.assertEqual(calls[-1], phase)
                self.assertFalse(result["complete"])
                self.assertEqual(result["status"], "failure")

    def test_unknown_soc_cannot_silently_skip_npu_suite(self):
        code, calls, result = self.execute("unknown")
        self.assertEqual((code, calls, result["status"]), (1, [], "failure"))

    def test_missing_short_or_failed_reference_report_rejected(self):
        valid = {"complete": True, "results": [{"index": i, "status": "PASS"} for i in range(3)],
                 "case_indices": [0, 1, 2], "found": True, "all_pass": True, "total": 3, "check_pass": 3,
                 "exec_fail": 0, "check_fail": 0}
        self.assertEqual(ci.check_accuracy(valid, 3), valid)
        for change in ({"complete": False}, {"results": []}, {"case_indices": [0, 1, 1]},
                       {"results": [{"index": 0, "status": "ERROR"}] * 3},
                       {"found": False}, {"total": 0}, {"total": 2}, {"check_pass": 2},
                       {"exec_fail": 1}, {"check_fail": 1}, {"all_pass": False}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                ci.check_accuracy({**valid, **change}, 3)

    def test_formal_pipeline_has_no_experimental_call(self):
        self.assertNotIn("python3 ci/run_experimental_ci.py", (ROOT / "ci/run_checks.sh").read_text())


class IndependentGate(unittest.TestCase):
    def fixture(self, platform="a2"):
        runtime = platform == "a2"
        report = dict(schema="experimental-ci-v1", complete=True, status="success", build_only=not runtime,
                      head_sha="a" * 40, run_id="12", run_attempt="1", platform=platform,
                      soc="ascend910b" if runtime else "ascend950", mode="quick")
        report["phases"] = {name: {"status": "success" if runtime or i < 2 else "skipped"}
                            for i, name in enumerate(ci.PHASES)}
        report["phases"]["load-isolation"]["result"] = dict(complete=True, protected_unchanged=True,
            installation_rejections=3, planned=22, counts=dict(PASS=22, FAIL=0, BLOCKED=0),
            results=[dict(id=case["id"], status="PASS") for case in ci.ISOLATION.CASES])
        for launch in ("queue", "inline"):
            report["phases"]["accuracy-" + launch]["result"] = dict(
                schema="experimental-accuracy-v1", backend="cpu-reference", standard="mixed_tolerance_bm-v1-atk26.9.8",
                complete=True, all_pass=True, total=3, expected=3, check_pass=3, exec_fail=0, check_fail=0,
                case_indices=list(range(3)), results=[dict(index=i, status="PASS",
                    outputs={name: {"pass": True} for name in ("o", "s", "sa")}) for i in range(3)],
                execution_evidence=[dict(synchronized_after_real_call=True, launch_mode=launch)])
        return report

    def validate(self, report, expected=None):
        expected = expected or {key: report[key] for key in ("head_sha", "run_id", "run_attempt", "platform", "soc", "mode")}
        script = "const v=require('./.github/scripts/publish_experimental_ci_status.js').validateReport;" + \
                 "try { const x=JSON.parse(process.env.TEST_INPUT); v(x.report,x.expected); } " + \
                 "catch(e) { console.error(e.message); process.exit(1); }"
        return subprocess.run(["node", "-e", script], cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, "TEST_INPUT": json.dumps(dict(report=report, expected=expected))})

    def test_gate_accepts_only_matching_complete_reports(self):
        for platform in ("a2", "a5"):
            report = self.fixture(platform)
            self.assertEqual(self.validate(report).returncode, 0)
            for key in ("head_sha", "run_id", "run_attempt", "platform", "soc", "mode"):
                expected = {field: report[field] for field in ("head_sha", "run_id", "run_attempt", "platform", "soc", "mode")}
                expected[key] = "wrong"
                self.assertNotEqual(self.validate(report, expected).returncode, 0, key)

    def test_gate_rejects_zero_short_wrong_case_or_missing_execution_evidence(self):
        report = self.fixture()
        for change in ({"results": []}, {"case_indices": [0, 1, 1]}, {"total": 0},
                       {"execution_evidence": []}, {"check_fail": 1}, {"all_pass": False}):
            broken = copy.deepcopy(report)
            broken["phases"]["accuracy-queue"]["result"].update(change)
            self.assertNotEqual(self.validate(broken).returncode, 0, change)
        for phase in ci.PHASES:
            broken = copy.deepcopy(report)
            broken["phases"][phase]["status"] = "not_run"
            self.assertNotEqual(self.validate(broken).returncode, 0, phase)
        for key, value in (("installation_rejections", 0), ("protected_unchanged", False), ("results", [])):
            broken = copy.deepcopy(report)
            broken["phases"]["load-isolation"]["result"][key] = value
            self.assertNotEqual(self.validate(broken).returncode, 0, key)

    def test_jobs_and_status_publishers_are_independent(self):
        import yaml
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        jobs = workflow["jobs"]
        self.assertEqual(jobs["experimental-npu"]["needs"], "prepare")
        self.assertNotIn("ascend-npu", jobs["experimental-finalize"]["needs"])
        self.assertNotIn("experimental-npu", jobs["finalize"]["needs"])
        self.assertEqual(jobs["experimental-npu"]["env"]["CI_SUITE"], "experimental")
        self.assertEqual(jobs["ascend-npu"]["env"]["CI_SUITE"], "formal")
        self.assertEqual(jobs["ascend-npu"]["steps"], jobs["experimental-npu"]["steps"])
        self.assertNotEqual(jobs["ascend-npu"]["env"]["CI_HOST_TMPDIR"], jobs["experimental-npu"]["env"]["CI_HOST_TMPDIR"])
        self.assertIn("ops == ''", jobs["experimental-npu"]["if"])

    def test_publisher_fails_closed_and_writes_only_experimental_status(self):
        script = """
const publish = require(process.env.TEST_PUBLISHER);
const statuses = [];
const pr = { state:'open', draft:false, head:{sha:process.env.TEST_CURRENT_SHA}, base:{ref:'main'} };
const github = { rest:{pulls:{get:async()=>({data:pr})},repos:{createCommitStatus:async p=>statuses.push(p)}} };
const core = {warning:()=>{},setOutput:()=>{},setFailed:()=>{}};
publish({github,core,context:{repo:{owner:'fixture',repo:'repo'},serverUrl:'https://github.com',runId:12}})
 .then(()=>process.stdout.write(JSON.stringify(statuses))).catch(e=>{console.error(e);process.exit(1)});
"""
        for scenario in ("success", "missing-a5", "stale-head", "upload-failed"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for platform in ("a2", "a5"):
                    if scenario == "missing-a5" and platform == "a5":
                        continue
                    ci.write_json(root / f"experimental-reports/experimental-npu-ci-result-{platform}-12-1/experimental/summary.json",
                                  self.fixture(platform))
                result = subprocess.run(["node", "-e", script], cwd=root, capture_output=True, text=True,
                    env={**os.environ, "TEST_PUBLISHER": str(ROOT / ".github/scripts/publish_experimental_ci_status.js"),
                         "NPU_EXPECTED_HEAD_SHA": "a" * 40, "NPU_EXPECTED_BASE_REF": "main", "NPU_PR_NUMBER": "1",
                         "NPU_EXPECTED_RUN_ID": "12", "NPU_EXPECTED_RUN_ATTEMPT": "1", "NPU_CI_MODE": "quick",
                         "NPU_MATRIX_RESULT": "failure" if scenario == "upload-failed" else "success",
                         "TEST_CURRENT_SHA": ("b" if scenario == "stale-head" else "a") * 40})
                self.assertEqual(result.returncode, 0, result.stderr)
                statuses = json.loads(result.stdout)
                self.assertEqual(len(statuses), 1)
                self.assertEqual(statuses[0]["context"], "NPU CI / Experimental / A2 runtime + A5 build")
                self.assertEqual(statuses[0]["state"], "success" if scenario == "success" else "failure")


if __name__ == "__main__":
    unittest.main()
