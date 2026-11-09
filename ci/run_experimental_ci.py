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
PHASES = ("build-host", "profiles", "load-isolation", "accuracy-queue", "accuracy-inline")


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
    if (not result.get("found") or not result.get("all_pass")
            or result.get("total") != expected or result.get("check_pass") != expected
            or result.get("exec_fail") != 0 or result.get("check_fail") != 0):
        raise RuntimeError(f"ATK accuracy did not pass exactly {expected} cases: {result}")
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
        if "atk" not in self.profiles:
            self.atk_root, purelib = self.profile("atk", experimental=True)
            wheel = os.environ.get("CI_EXPERIMENTAL_ATK_WHEEL")
            if wheel:
                wheel = Path(wheel).resolve()
                if not wheel.is_file():
                    raise RuntimeError("CI_EXPERIMENTAL_ATK_WHEEL must be an existing wheel inside the container")
                dependencies = self.atk_root / "atk-dependencies"
                self.run([sys.executable, "-m", "pip", "install", "--target", dependencies,
                          "--constraint", ROOT / "ci/experimental-atk-constraints.txt", wheel])
                # Never replace links to the parent environment's dependencies.
                # Process this before the experimental bootstrap .pth below.
                (purelib / "00_atk_dependencies.pth").write_text(
                    f"import sys; sys.path.insert(0, {str(dependencies)!r})\n")
            # Resolve the entry point in this private environment, never a parent atk executable.
            entry = self.atk_root / "bin/atk"
            entry.write_text(f"#!{self.atk_root}/bin/python\nimport sys\nfrom importlib.metadata import distribution\n"
                             "sys.exit(next(e for e in distribution('ATK').entry_points if e.name == 'atk').load()())\n")
            entry.chmod(0o755)
            try:
                self.run([entry, "--version"], timeout=60)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("ATK is unavailable: preinstall ATK and dependencies in the CI image "
                                   "or set CI_EXPERIMENTAL_ATK_WHEEL to a container-accessible wheel") from exc
            self.run([self.atk_root / "bin/python", "-c", "from importlib.metadata import version; "
                      "assert version('kombu') == '5.6.2', 'Experimental CI requires kombu==5.6.2; "
                      "install using ci/experimental-atk-constraints.txt'"])
            (purelib / "00_experimental_atk_bootstrap.pth").write_text("import fla_npu_experimental\n")
        profile = self.profiles["atk"]
        env = clean_environment({"mode": "experimental", "scenario": "experimental"}, profile)
        for key in ("CASE_START", "CASE_END", "ACCURACY_START", "ACCURACY_END", "CANN_ENV", "FLA_NPU_OPP_ENV"):
            env.pop(key, None)
        expected = 3 if self.mode == "quick" else len(json.loads(
            (ROOT / f"tests/atk/{OP}/atk_{OP}.json").read_text()))
        env.update(ATK_ENV=str(self.atk_root), FLA_NPU_ENV=str(Path(profile["experimental_package"]) /
                   "opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash"),
                   ATK_OUTPUT_ROOT=str(self.output / ("atk-" + launch)),
                   ATK_TIMEOUT="600" if self.mode == "quick" else "7200",
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   CASE_START="0", CASE_END=str(expected))
        env.pop("FLA_NPU_STABLE_LAUNCH", None)
        if launch == "inline":
            env["FLA_NPU_STABLE_LAUNCH"] = "inline"
        self.run(["bash", "tests/atk/run_test_cpu.sh", f"-op={OP}", "-soc=ascend910b",
                  "-npu_device_id=" + os.environ.get("CI_CONTAINER_DEVICE", "0"), "-scope=accuracy"],
                 env, timeout=int(env["ATK_TIMEOUT"]) + 60)
        result_file = self.output / ("accuracy-" + launch + ".json")
        checker = ROOT / "tests/atk/common/check_atk_result.py"
        code = ("import importlib.util,json; from pathlib import Path; "
                f"s=importlib.util.spec_from_file_location('checker',{str(checker)!r}); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                f"Path({str(result_file)!r}).write_text(json.dumps(m.check_accuracy({env['ATK_OUTPUT_ROOT']!r},{OP!r})))")
        self.run([profile["python"], "-c", code], env, timeout=60)
        return check_accuracy(json.loads(result_file.read_text()), expected)

    def execute(self):
        self.save()
        try:
            if self.soc not in {"ascend910b", "ascend950"} or self.mode not in {"quick", "full"}:
                raise ValueError(f"Unsupported experimental CI configuration: {self.soc}/{self.mode}")
            self.phase("build-host", self.build)
            if self.soc == "ascend910b":
                self.phase("profiles", self.prepare)
                self.phase("load-isolation", self.isolation)
                for launch in ("queue", "inline"):
                    self.phase("accuracy-" + launch, lambda launch=launch: self.accuracy(launch))
            else:
                for name in PHASES[1:]:
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
