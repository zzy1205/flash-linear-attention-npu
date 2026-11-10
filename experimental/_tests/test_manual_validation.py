"""Manual validation checks real results and stops on failures."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("experimental_validation", ROOT / "experimental/_tests/validate.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class ManualValidation(unittest.TestCase):
    def test_formal_gdr_library_paths_do_not_leak_into_experimental_processes(self):
        paths = ["/cann/lib64", "/site/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib",
                 "/site/fla_npu/opp/vendors/fla_npu_overlay_test/op_api/lib", "/driver/lib64"]
        with patch.dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(paths)):
            env = validation.clean_environment({"mode": "stable", "scenario": "stable"}, {})
        self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep), [paths[0], paths[-1]])

    def execute(self, soc="ascend910b", fail=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = validation.Validation(root, soc=soc)
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

    def test_failure_prevents_following_checks_and_is_recorded(self):
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
        self.assertEqual(validation.check_accuracy(valid, 3), valid)
        for change in ({"complete": False}, {"results": []}, {"case_indices": [0, 1, 1]},
                       {"results": [{"index": 0, "status": "ERROR"}] * 3},
                       {"found": False}, {"total": 0}, {"total": 2}, {"check_pass": 2},
                       {"exec_fail": 1}, {"check_fail": 1}, {"all_pass": False}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validation.check_accuracy({**valid, **change}, 3)



if __name__ == "__main__":
    unittest.main()
