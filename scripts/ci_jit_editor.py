"""Build, package and exercise the actual Windows standalone JIT Editor.

macOS/Linux editor ports are separate work; catalog CI covers those platforms.
All tests below load this build's matching compiler payload.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/jit-editor")
    parser.add_argument("--skip-build", action="store_true", help="Qualify an existing matching local build")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("The standalone editor currently targets Windows x64; run catalog CI on other platforms")
    output = args.output.resolve()
    build = ROOT / "build/jit-editor"
    logs = ROOT / "build/ci/jit-editor"
    logs.mkdir(parents=True, exist_ok=True)

    def checked(name, command):
        print("[jit-ci] " + name, flush=True)
        log = logs / (name + ".log")
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(encoding="utf-8", errors="replace")[-12000:], file=sys.stderr)
            raise RuntimeError(f"{name} failed; complete log: {log}")

    if not args.skip_build:
        for name, source, destination in (
            ("frontend", "tools/native_compiler", "build/native-compiler"),
            ("preprocessor", "tools/jsfx_eel_pp", "build/tools/jsfx_eel_pp"),
            ("editor", "tools/jit_editor", "build/jit-editor"),
        ):
            checked("configure-" + name, ["cmake", "-S", source, "-B", destination,
                    "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++",
                    f"-DPython3_EXECUTABLE={sys.executable}"])
            targets = ["JITEditor_CLAP", "JITEditor_VST3", "jit_editor_check",
                       "jit_editor_clap_check", "jit_editor_vst3_check"] if name == "editor" else []
            command = ["cmake", "--build", destination, "--config", "Release", "--parallel", "2"]
            if targets:
                command += ["--target", *targets]
            checked("build-" + name, command)

    faust = os.environ.get("JSFX_FAUST_COMPILER") or shutil.which("faust")
    if not faust:
        raise RuntimeError("Faust must be installed before packaging")
    checked("package", [sys.executable, "tools/jit_editor/package_bundle.py",
            "--output", str(output), "--python", str(Path(sys.executable).parent),
            "--faust", str(Path(faust).resolve().parent.parent),
            "--staging", str(ROOT / "build/ci/jit-package")])
    package = ROOT / "build/ci/jit-package"
    runtime = package / "CLAP/JITEditor.runtime"
    # Test runner resolves its private payload beside its executable.
    shutil.copytree(runtime, build / "JITEditor.runtime", dirs_exist_ok=True)
    checked("frontend-contract", [str(ROOT / "build/native-compiler/jsfx_frontend_check.exe")])
    for frontend in ("python-reference", "cpp-frontend"):
        checked("compiler-" + frontend, [sys.executable, "tools/jit_editor/CompilerCheck.py",
                str(runtime), "--frontend", frontend])
    check = str(build / "jit_editor_check.exe")
    checked("runtime-standard", [check])
    checked("runtime-cpp", [check, "--cpp-frontend"])
    checked("controls", [check, "--control-defaults"])
    checked("interface-unicode", [check, "--interface-unicode", str(logs / "interface")])
    checked("examples", [check, "--examples", "standard", str(logs / "examples")])
    checked("packaged-clap", [str(build / "jit_editor_clap_check.exe"),
            str(package / "CLAP/ZorakAudio JIT Editor PoC.clap")])
    checked("packaged-vst3", [str(build / "jit_editor_vst3_check.exe"),
            str(package / "VST3/ZorakAudio JIT Editor PoC.vst3/Contents/x86_64-win/ZorakAudio JIT Editor PoC.vst3")])
    print("[jit-ci] All packaged compiler, runtime, interface and public plugin checks passed", flush=True)


if __name__ == "__main__":
    main()
