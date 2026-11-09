#!/usr/bin/env python3
"""Experimental checks inside the existing NPU CI test stage (fail closed)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OP = "fused_recurrent_rwkv8"
PHASES = ("dependencies", "build-host", "profiles", "load-isolation", "accuracy-queue", "accuracy-inline")


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


ISOLATION = module(ROOT / "experimental/_tests/a2_isolation/run.py")


def clean_environment(case, profile):
    env = ISOLATION.environment(case, profile)
    # The preceding formal GDR step exports its vendor library directory.
    # Keep toolkit/driver dependencies, but do not let ELF lookup reuse a FLA vendor.
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            path for path in env["LD_LIBRARY_PATH"].split(os.pathsep) if path and not any(
                part in {"fla_npu_transformer", "fla_npu_experimental_transformer"}
                or part.startswith("fla_npu_overlay_") for part in Path(path).parts))
    return env


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def check_accuracy(result, expected):
    if (expected <= 0 or not result.get("complete")
            or len(result.get("results", [])) != expected
            or result.get("case_indices") != list(range(expected))
            or any(item.get("index") != index or item.get("status") != "PASS"
                   for index, item in enumerate(result.get("results", [])))
            or not result.get("found") or not result.get("all_pass")
            or result.get("total") != expected or result.get("check_pass") != expected
            or result.get("exec_fail") != 0 or result.get("check_fail") != 0):
        raise RuntimeError(f"CPU-reference accuracy did not pass exactly {expected} planned cases")
    return result


class Validation:
    def __init__(self, output, stage_report=None):
        self.output = output
        self.stage_report = stage_report
        self.soc = os.environ.get("CI_SOC", "ascend910b")
        self.mode = os.environ.get("CI_MODE", "quick")
        self.env = clean_environment({"mode": "stable", "scenario": "stable"}, {})
        for key in ("ATK_ENV", "CANN_ENV", "FLA_NPU_ENV", "FLA_NPU_OPP_ENV", "FLA_NPU_STABLE_LAUNCH"):
            self.env.pop(key, None)
        self.report = {"schema": "experimental-ci-v1", "complete": False, "status": "running",
                       "soc": self.soc, "mode": self.mode, "output": str(output),
                       "device": os.environ.get("CI_CONTAINER_DEVICE", "0"),
                       "head_sha": os.environ.get("CI_ACCURACY_HEAD_SHA", ""),
                       "phases": {name: {"status": "not_run"} for name in PHASES}}
        self.profiles = {}
        self.log = None

    def save(self):
        write_json(self.output / "summary.json", self.report)
        if self.stage_report:
            payload = json.loads(self.stage_report.read_text(encoding="utf-8"))
            # Existing main workflow uploads this file; retain its v1 stage schema.
            payload["experimental"] = self.report
            write_json(self.stage_report, payload)

    def run(self, command, env=None, timeout=7200):
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env or self.env,
                       stdout=self.log, stderr=subprocess.STDOUT, check=True, timeout=timeout)

    def phase(self, name, action):
        item = self.report["phases"][name]
        item.update(status="running", log=str(self.output / (name + ".log")))
        self.save()
        started = time.monotonic()
        print(f"[experimental][CI] START {name}", flush=True)
        try:
            with Path(item["log"]).open("w", encoding="utf-8") as self.log:
                detail = action()
            item.update(status="success", result=detail)
        except Exception as exc:
            item.update(status="failure", reason=str(exc))
            item["log_tail"] = Path(item["log"]).read_text(encoding="utf-8", errors="replace")[-6000:]
            print(item["log_tail"], file=sys.stderr, flush=True)
            raise
        finally:
            item["seconds"] = round(time.monotonic() - started, 3)
            self.save()
            print(f"[experimental][CI] {item['status'].upper()} {name}", flush=True)

    def dependencies(self):
        output = self.output / "dependencies.json"
        try:
            self.run([sys.executable, ROOT / "ci/check_experimental_environment.py", "--output", output],
                     timeout=720)
        finally:
            if output.exists():
                self.report["phases"]["dependencies"]["result"] = json.loads(output.read_text())
        result = json.loads(output.read_text())
        for warning in result["warnings"]:
            print(f"[experimental][WARNING] {warning}", flush=True)
        return result

    def build(self):
        products = ROOT / "build_out/experimental-products.json"
        products.unlink(missing_ok=True)
        self.run(["bash", "ci/run_experimental_validation.sh"],
                 {**self.env, "CI_EXPERIMENTAL_OPS": "all", "CI_EXPERIMENTAL_RUNTIME": "false"})
        self.products = json.loads(products.read_text(encoding="utf-8"))
        result = {"products": self.products}
        for name in ("package", "load"):
            result[name] = json.loads((ROOT / f"build_out/experimental-{name}-audit.json").read_text())
        shutil.copy2(ROOT / "build/experimental-build-manifest.json", self.output)
        return result

    def profile(self, name, stable=False, experimental=False):
        env_root = self.output / "envs" / name
        self.run([sys.executable, "-m", "venv", "--without-pip", env_root])
        python = env_root / "bin/python"
        purelib = Path(subprocess.check_output([str(python), "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))"], env=self.env, text=True).strip())
        # Share dependencies, never the parent site's FLA packages or startup hooks.
        # --system-site-packages would invalidate the 'only one wheel' cases.
        for directory in site.getsitepackages():
            if not Path(directory).is_dir():
                continue
            for entry in Path(directory).iterdir():
                if (entry.name.startswith(("fla_npu", "flash_linear_attention", "__editable__"))
                        or entry.name in {"fla", "sitecustomize.py", "usercustomize.py", "__pycache__"}
                        or entry.suffix == ".pth" or (purelib / entry.name).exists()):
                    continue
                (purelib / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        profile = {"python": str(python)}
        if stable:
            # Reuse the exact installed formal artifact tested by the preceding stage.
            source = self.stable.parent
            for entry in source.glob("fla_npu*"):
                target = purelib / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, symlinks=True,
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                else:
                    shutil.copy2(entry, target)
            profile["stable_package"] = str(purelib / "fla_npu")
        if experimental:
            self.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                      "--target", purelib, self.products["wheel"]])
            profile["experimental_package"] = str(purelib / "fla_npu_experimental")
        self.profiles[name] = profile
        return env_root, purelib

    def prepare(self):
        spec = importlib.util.find_spec("fla_npu")
        if spec is None or not spec.origin:
            raise RuntimeError("The formal torch-adapter stage must install fla_npu first")
        self.stable = Path(spec.origin).parent.resolve()
        self.before = ISOLATION.snapshot([self.stable])
        for name, stable, experimental in (("stable_only", True, False), ("experimental_only", False, True),
                ("combined", True, True), ("overlay", True, True)):
            self.profile(name, stable, experimental)
        # There is currently exactly one experimental operator. Reject silently
        # treating a future multi-op wheel as a single-op selection test.
        build = json.loads((Path(self.profiles["experimental_only"]["experimental_package"]) / "_build.json").read_text())
        if [op["name"] for op in build["operators"]] != [OP]:
            raise RuntimeError("Update CI selection/accuracy coverage for the new experimental operator set")
        self.profiles["single"] = self.profiles["experimental_only"].copy()
        combined = self.profiles["combined"]
        combined["external_experimental_opp"] = self.products["opp"]
        self.run([combined["python"], "-m", "fla_npu_opp_manage", "--remove-overlay",
                  "--overlay-name", "fla_npu_overlay_unused_fixture"])
        overlay = self.profiles["overlay"]
        self.run([overlay["python"], "-m", "fla_npu_opp_manage", "--overlay", "--source",
                  self.stable / "opp/vendors/fla_npu_transformer", "--overlay-name", "fla_npu_overlay_control"])
        overlay["overlay_vendor"] = str(Path(overlay["stable_package"]) / "opp/vendors/fla_npu_overlay_control")
        config = {"device": int(os.environ.get("CI_CONTAINER_DEVICE", "0")), "timeout": 120,
                  "protected_roots": [str(self.stable)], "profiles": self.profiles}
        write_json(self.output / "profiles.json", config)
        return {"profiles": list(self.profiles), "formal_artifact": str(self.stable)}

    def isolation(self):
        summary = self.output / "isolation/summary.json"
        try:
            self.run([sys.executable, ROOT / "experimental/_tests/a2_isolation/run.py",
                      "--config", self.output / "profiles.json", "--output", self.output / "isolation"])
        finally:
            if summary.exists():
                # Keep negative-case diagnostics in the uploaded report even on failure.
                self.report["phases"]["load-isolation"]["result"] = json.loads(summary.read_text())
        result = json.loads(summary.read_text())
        if not result["complete"] or result["counts"] != {"PASS": 28, "FAIL": 0, "BLOCKED": 0}:
            raise RuntimeError("Incomplete A2 load isolation matrix")
        return result

    def accuracy(self, launch):
        profile = self.profiles["experimental_only"]
        env = clean_environment({"mode": "experimental", "scenario": "experimental"}, profile)
        env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   FLA_A2_TRACE_DIR=str(self.output / ("accuracy-" + launch + "-trace")))
        env.pop("FLA_NPU_STABLE_LAUNCH", None)
        if launch == "inline":
            env["FLA_NPU_STABLE_LAUNCH"] = "inline"
        expected = 3 if self.mode == "quick" else len(json.loads(
            (ROOT / f"tests/atk/{OP}/atk_{OP}.json").read_text()))
        result_file = self.output / ("accuracy-" + launch + ".json")
        try:
            self.run([profile["python"], ROOT / "ci/run_experimental_accuracy.py",
                      "--mode", self.mode, "--device", self.report["device"], "--output", result_file],
                     env, timeout=660 if self.mode == "quick" else 7260)
        finally:
            if result_file.exists():
                # Preserve per-case numerical diagnostics in the uploaded stage artifact.
                self.report["phases"]["accuracy-" + launch]["result"] = json.loads(result_file.read_text())
        result = json.loads(result_file.read_text())
        result["execution_evidence"] = [json.loads(path.read_text()) for path in
                                        Path(env["FLA_A2_TRACE_DIR"]).glob("experimental-*.json")]
        if not result["execution_evidence"]:
            raise RuntimeError("Missing evidence of real experimental NPU execution")
        return check_accuracy(result, expected)

    def execute(self):
        self.save()
        try:
            if self.soc not in {"ascend910b", "ascend950"} or self.mode not in {"quick", "full"}:
                raise ValueError(f"Unsupported experimental CI configuration: {self.soc}/{self.mode}")
            self.phase("dependencies", self.dependencies)
            self.phase("build-host", self.build)
            if self.soc == "ascend910b":
                self.phase("profiles", self.prepare)
                self.phase("load-isolation", self.isolation)
                for launch in ("queue", "inline"):
                    self.phase("accuracy-" + launch, lambda launch=launch: self.accuracy(launch))
            else:
                for name in PHASES[2:]:
                    self.report["phases"][name] = {"status": "skipped", "reason": "Runtime suite supports A2 only"}
            self.report.update(complete=True, status="success")
        except Exception as exc:
            self.report.update(status="failure", reason=str(exc))
            for item in self.report["phases"].values():
                if item["status"] == "not_run":
                    item.update(status="not_run", reason="Previous experimental check failed")
        finally:
            if hasattr(self, "before"):
                unchanged = self.before == ISOLATION.snapshot([self.stable])
                self.report["formal_payload_unchanged"] = unchanged
                if not unchanged:
                    self.report.update(complete=False, status="failure", reason="Formal installed payload changed")
            self.save()
        print(f"[experimental][CI] {self.report['status']}: {self.output / 'summary.json'}", flush=True)
        if self.report.get("reason"):
            print(self.report["reason"], file=sys.stderr, flush=True)
        return 0 if self.report["status"] == "success" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-report", type=Path)
    args = parser.parse_args()
    parent = Path(os.environ.get("CI_TMPDIR") or ROOT / ".ci-tmp").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="experimental-ci-", dir=parent))
    return Validation(output, args.stage_report).execute()


if __name__ == "__main__":
    raise SystemExit(main())
