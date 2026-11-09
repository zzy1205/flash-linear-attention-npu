"""The merged-PR entry must gate on real experimental results, not host smoke."""
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
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
            stage = root / "stage.json"
            original = {"schema": "npu-ci-stage-report-v1", "stages": {"gdr-example-st": {"status": "running"}}}
            ci.write_json(stage, original)
            with patch.dict(os.environ, CI_SOC=soc, CI_MODE="quick"):
                runner = ci.Validation(root, stage)
            calls = []

            def action(name):
                calls.append(name)
                if fail == name:
                    raise RuntimeError("injected missing/failed result")
                return {"checked": name}

            with patch.object(runner, "dependencies", lambda: action("dependencies")), \
                    patch.object(runner, "build", lambda: action("build")), \
                    patch.object(runner, "prepare", lambda: action("profiles")), \
                    patch.object(runner, "isolation", lambda: action("isolation")), \
                    patch.object(runner, "accuracy", lambda mode: action(mode)), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = runner.execute()
            payload = json.loads(stage.read_text())
            self.assertEqual(payload["stages"], original["stages"])
            self.assertEqual(payload["experimental"], json.loads((root / "summary.json").read_text()))
            return code, calls, payload["experimental"]

    def test_a2_runs_both_launch_modes_after_isolation(self):
        code, calls, result = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["dependencies", "build", "profiles", "isolation", "queue", "inline"])
        self.assertTrue(result["complete"])

    def test_a5_explicitly_does_not_claim_accuracy(self):
        code, calls, result = self.execute("ascend950")
        self.assertEqual((code, calls), (0, ["dependencies", "build"]))
        self.assertEqual(result["phases"]["accuracy-queue"]["status"], "skipped")

    def test_failure_prevents_following_checks_and_is_uploaded(self):
        for phase in ("dependencies", "build", "profiles", "isolation", "queue", "inline"):
            with self.subTest(phase=phase):
                code, calls, result = self.execute(fail=phase)
                self.assertEqual(code, 1)
                self.assertEqual(calls[-1], phase)
                self.assertFalse(result["complete"])
                self.assertEqual(result["status"], "failure")

    def test_unknown_soc_cannot_silently_skip_npu_suite(self):
        code, calls, result = self.execute("unknown")
        self.assertEqual((code, calls, result["status"]), (1, [], "failure"))

    def test_failed_isolation_observations_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = ci.Validation(root)
            result = {"complete": False, "results": [{"id": "L01", "status": "FAIL"}]}
            ci.write_json(root / "isolation/summary.json", result)
            with patch.object(runner, "run", side_effect=subprocess.CalledProcessError(1, "isolation")):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.isolation()
            self.assertEqual(runner.report["phases"]["load-isolation"]["result"], result)

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

    @unittest.skipUnless(shutil.which("bash"), "requires Linux bash")
    def test_existing_stage_wrapper_propagates_experimental_and_formal_failures(self):
        source = (ROOT / "ci/run_checks.sh").read_text()
        wrapper = re.search(r"run_gdr_and_experimental_st\(\) \{.*?\n\}", source, re.S).group()
        self.assertRegex(source, r'"\$gdr_example_enabled" \\\n    run_gdr_and_experimental_st')
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "order"
            script = '''set -euo pipefail
stage_report_file=report.json
formal_count=0
run_gdr_example_st() {
    formal_count=$((formal_count+1))
    echo "formal-$formal_count" >> "$ORDER"
    if [[ "$FAIL" == "formal-$formal_count" ]]; then return 31; fi
}
python3() {
    [[ "$1" == ci/run_experimental_ci.py && "$2" == --stage-report && "$3" == report.json ]]
    echo experimental >> "$ORDER"
    if [[ "$FAIL" == experimental ]]; then return 32; fi
}
'''
            for fail, code, order in (("", 0, ["formal-1", "experimental", "formal-2"]),
                                     ("formal-1", 31, ["formal-1"]),
                                     ("experimental", 32, ["formal-1", "experimental"]),
                                     ("formal-2", 31, ["formal-1", "experimental", "formal-2"])):
                log.unlink(missing_ok=True)
                result = subprocess.run(["bash", "-c", script + wrapper + "\nrun_gdr_and_experimental_st\n"],
                                        env={**os.environ, "ORDER": str(log), "FAIL": fail}, capture_output=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(log.read_text().splitlines(), order)


if __name__ == "__main__":
    unittest.main()
