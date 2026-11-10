"""Contract tests for the experimental operator discovery rules."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_build_ops = _load_module(
    "fla_check_build_ops", REPO_ROOT / "scripts" / "check_build_ops.py"
)


class ExperimentalBuildContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        operator = self.root / "experimental" / "rwkv_recurrent"
        operator.mkdir(parents=True)
        (operator / "CMakeLists.txt").write_text(
            "add_subdirectory(op_host)\n", encoding="utf-8"
        )
        (operator / "op_host").mkdir()
        (operator / "op_host" / "CMakeLists.txt").write_text(
            "# test operator\n", encoding="utf-8"
        )
        reserved = self.root / "experimental" / "_common"
        reserved.mkdir()
        (reserved / "CMakeLists.txt").write_text(
            "# reserved shared input\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_experimental_discovery_uses_flat_operator_directories(self):
        self.assertEqual(
            check_build_ops.discover_supported_ops(self.root, experimental=True),
            ["rwkv_recurrent"],
        )

    def test_default_discovery_stays_on_the_stable_tree(self):
        stable_host = self.root / "fla" / "ops" / "ascendc" / "gdn" / "stable_probe" / "op_host"
        stable_host.mkdir(parents=True)
        (stable_host / "CMakeLists.txt").write_text("# stable\n", encoding="utf-8")
        self.assertEqual(check_build_ops.discover_supported_ops(self.root), ["stable_probe"])

    def test_flat_name_containing_op_host_is_preserved(self):
        entry = self.root / "experimental" / "op_host_probe"
        entry.mkdir()
        (entry / "CMakeLists.txt").write_text("# operator\n", encoding="utf-8")
        self.assertIn("op_host_probe", check_build_ops.discover_supported_ops(self.root, True))

    @unittest.skipIf(os.name == "nt", "symlink creation requires Windows privileges")
    def test_discovery_rejects_operator_symlink_into_stable_tree(self):
        stable = self.root / "fla" / "stable_op"
        stable.mkdir(parents=True)
        (stable / "CMakeLists.txt").write_text("# stable\n", encoding="utf-8")
        (self.root / "experimental" / "linked_op").symlink_to(stable, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            check_build_ops.discover_supported_ops(self.root, True)


if __name__ == "__main__":
    unittest.main()
