"""Selected adapters must produce a nonempty, independently audited launcher."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.opp_abi_fixtures import write_elf_exports

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ExperimentalLauncherTest(unittest.TestCase):
    def test_shared_compiler_preserves_layer_isolation_and_explicit_units(self):
        compiler = load("shared_compiler", ROOT / "torch_custom/fla_npu/csrc/build_stable.py")
        with patch.object(compiler, "torch_lib_dir", return_value="/torch/lib"), \
                patch.object(compiler, "torch_include_dirs", return_value=[]), \
                patch.object(compiler.shutil, "which", return_value="g++"), \
                patch.object(compiler.subprocess, "run") as run:
            compiler.compile_launcher(["experimental_ops.cpp", "runtime.cpp"], "experimental.so", "stamp",
                                      extra_flags=["-DFLA_STABLE_EXPERIMENTAL"])
        command = run.call_args.args[0]
        self.assertIn("-fno-gnu-unique", command)
        self.assertIn("-DFLA_STABLE_NO_DEBUG_PROBE", command)
        self.assertIn("-DFLA_STABLE_EXPERIMENTAL", command)
        self.assertEqual([item for item in command if item.endswith(".cpp")],
                         ["experimental_ops.cpp", "runtime.cpp"])

    def setUp(self):
        self.builder = load("experimental_builder_test", ROOT / "experimental/_framework/build.py")
        self.audit = load("experimental_manifest_test", ROOT / "scripts/experimental_manifest.py")
        self.abi = load("experimental_abi_test", ROOT / "torch_custom/fla_npu/fla_npu/_op_abi.py")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = self.root / "ops.txt"
        self.names.write_text("fused_recurrent_rwkv8\n", encoding="utf-8")
        self.data = self.audit.create_manifest(ROOT, self.names, self.names, "ascend950", "arch35;arch22")
        self.manifest = self.root / "manifest.json"
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def vendor(self):
        vendor = self.root / "vendor"
        include = vendor / "op_api/include/aclnnop"
        include.mkdir(parents=True)
        for op in self.data["operators"]:
            for header in (ROOT / op["source"] / "op_host/op_api").glob("aclnn_*.h"):
                shutil.copy2(header, include)
        library = vendor / self.abi.LIBRARY
        exports = {s for op in self.data["operators"] for s in op["aclnn_symbols"]}
        write_elf_exports(library, exports)
        with patch.object(self.abi, "exported_symbols", return_value=exports):
            self.abi.write_manifest(vendor)
        return vendor

    def test_single_operator_has_one_registration_and_no_stable_adapter(self):
        data = self.builder.selection(self.manifest)
        unit = self.builder.registration_text(data)
        self.assertEqual(unit.count("m.def("), 1)
        self.assertEqual(unit.count("m.impl("), 1)
        self.assertIn("fla_npu_experimental_stable", unit)
        self.assertIn("fla_npu_experimental_source_hash", unit)
        self.assertNotIn("stable_ops.cpp", unit)

    def test_actual_rwkv8_adapter_has_one_checked_call(self):
        vendor = self.vendor()
        adapters = self.builder.prepare(self.data, self.root)
        self.builder.check(self.data, adapters, vendor)

    def test_empty_adapter_coverage_cannot_pass(self):
        vendor = self.vendor()
        adapters = self.root / "empty"
        adapters.mkdir()
        with self.assertRaisesRegex(ValueError, "Adapter coverage"):
            self.builder.check(self.data, adapters, vendor)

    def test_rejects_missing_and_duplicate_selections(self):
        op = self.data["operators"][0]
        for records in ([], [op, op], [{**op, "name": "missing"}]):
            self.data["operators"] = records
            self.write_manifest()
            with self.assertRaises(ValueError):
                self.builder.selection(self.manifest)

    def test_changed_adapter_snapshot_is_rejected(self):
        record = self.data["operators"][0]
        record["source_sha256"]["adapter/stable_fused_recurrent_rwkv8.cpp"] = "stale"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "OPP source snapshot"):
            self.builder.selection(self.manifest)

    def test_multiple_fixture_adapters_are_selected_and_registered(self):
        data = {"mode": "experimental", "vendor": self.builder.VENDOR, "operators": []}
        for name in ("fixture_alpha", "fixture_beta"):
            adapter = self.root / "experimental" / name / "adapter" / f"stable_{name}.cpp"
            adapter.parent.mkdir(parents=True)
            adapter.write_text(f"// {name} test fixture; never compiled\n", encoding="utf-8")
            data["operators"].append({"name": name, "source_sha256": {
                "adapter/" + adapter.name: hashlib.sha256(adapter.read_bytes()).hexdigest()}})
        self.manifest.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(self.builder, "ROOT", self.root):
            selected = self.builder.selection(self.manifest)
            adapters = self.builder.prepare(selected, self.root / "stage")
        self.assertEqual({p.name for p in adapters.iterdir()},
                         {"stable_fixture_alpha.cpp", "stable_fixture_beta.cpp"})
        unit = self.builder.registration_text(selected)
        self.assertEqual(unit.count("m.def("), 2)
        self.assertEqual(unit.count("m.impl("), 2)
        for name in ("fixture_alpha", "fixture_beta"):
            self.assertIn(f'm.impl("npu_{name}"', unit)

    def test_fingerprint_covers_generated_registration(self):
        first, _ = self.builder.fingerprint(self.data)
        unit = self.builder.registration_text(self.data)
        with patch.object(self.builder, "registration_text", return_value=unit + "\n// changed registration\n"):
            second, files = self.builder.fingerprint(self.data)
        self.assertNotEqual(first, second)
        self.assertIn("experimental/_framework/generated_ops.cpp", files)
        self.assertFalse(any("/csrc/src/stable_" in p for p in files))


if __name__ == "__main__":
    unittest.main()
