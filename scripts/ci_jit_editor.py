"""Build, package and exercise the actual standalone JIT Editor.

macOS editor support is separate work; Windows and Linux share this qualification.
All tests below load this build's matching compiler payload.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/jit-editor")
    parser.add_argument("--skip-build", action="store_true", help="Qualify an existing matching local build")
    args = parser.parse_args()
    if sys.platform not in ('win32', 'linux'):
        parser.error("The standalone editor currently targets Windows and Linux")
    exe = '.exe' if sys.platform == 'win32' else ''
    output = args.output.resolve()
    build = ROOT / "build/jit-editor"
    logs = ROOT / "build/ci/jit-editor"
    logs.mkdir(parents=True, exist_ok=True)

    def checked(name, command, isolated=False):
        print("[jit-ci] " + name, flush=True)
        log = logs / (name + ".log")
        with log.open("wb") as stream:
            environment = None
            if isolated:
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith('PYTHON') and key not in ('JSFX_FAUST_COMPILER', 'JSFX_EEL_PP')}
                environment['PATH'] = ''
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
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
            if name == 'editor' and sys.platform == 'linux': targets.append('za_compiler_launcher')
            command = ["cmake", "--build", destination, "--config", "Release", "--parallel", "2"]
            if targets:
                command += ["--target", *targets]
            checked("build-" + name, command)

    faust = os.environ.get("JSFX_FAUST_COMPILER") or shutil.which("faust")
    if not faust:
        raise RuntimeError("Faust must be installed before packaging")
    checked("package", [sys.executable, "tools/jit_editor/package_bundle.py",
            "--output", str(output), "--python", str(Path(sys.executable).parent if sys.platform == 'win32' else Path(sys.prefix)),
            "--faust", str(Path(faust).resolve().parent.parent),
            "--staging", str(ROOT / "build/ci/jit-package")])
    # Test what the user extracts, in a different location with spaces/Unicode.
    # Do not let staging paths or an installed developer compiler hide omissions.
    package = ROOT / "build/ci/jit-relocated Ω"
    if package.exists():
        if not package.resolve().is_relative_to((ROOT / 'build/ci').resolve()):
            raise ValueError('Refusing to clear qualification files outside build/ci')
        shutil.rmtree(package)
    for kind in ('CLAP', 'VST3'):
        destination = package / kind
        destination.mkdir(parents=True)
        label = 'Windows' if sys.platform == 'win32' else 'Linux'
        base = output / ('JIT-Editor-Shared-Runtime-' + label + '-' + kind)
        if sys.platform == 'win32':
            with zipfile.ZipFile(str(base) + '.zip') as archive: archive.extractall(destination)
        else:
            with tarfile.open(str(base) + '.tar.gz') as archive: archive.extractall(destination, filter='data')
    runtime = package / "CLAP/JITEditor.runtime"
    # Test runner resolves its private payload beside its executable.
    shutil.copytree(runtime, build / "JITEditor.runtime", dirs_exist_ok=True)
    checked("frontend-contract", [str(ROOT / ('build/native-compiler/jsfx_frontend_check' + exe))], isolated=True)
    if sys.platform == 'linux':
        checked('compiler-process-supervisor', [sys.executable, 'tests/build/test_linux_compiler_launcher.py',
                '--launcher', str(build / 'za_compiler_launcher')])
    for frontend in ("python-reference", "cpp-frontend"):
        checked("compiler-" + frontend, [sys.executable, "tools/jit_editor/CompilerCheck.py",
                str(runtime), "--frontend", frontend])
    check = str(build / ('jit_editor_check' + exe))
    checked("runtime-standard", [check], isolated=True)
    checked("runtime-cpp", [check, "--cpp-frontend"], isolated=True)
    checked("controls", [check, "--control-defaults"], isolated=True)
    for directory in ('interface', 'examples'):
        (logs / directory).mkdir(parents=True, exist_ok=True)
    checked("interface-unicode", [check, "--interface-unicode", str(logs / "interface")], isolated=True)
    checked("examples", [check, "--examples", "standard", str(logs / "examples")], isolated=True)
    for image in ('interface/unicode-gfx.png', 'interface/unicode-editor.png',
                  'examples/hybrid-gain.png', 'examples/studio-channel.png'):
        if not (logs / image).is_file() or not (logs / image).stat().st_size:
            raise RuntimeError('GUI qualification did not produce its diagnostic image: ' + image)
    checked("source-resources", [check, "--source-resources"], isolated=True)
    checked("oversampling", [check, "--oversampling"], isolated=True)
    for plugin in ('Sample', 'Corpus'):
        checked('loaded-bank-' + plugin, [check, '--loaded-bank',
                str(ROOT / 'plugins/Spectral' / plugin / 'src' / (plugin + '.jsfx'))], isolated=True)
    checked("packaged-clap", [str(build / ('jit_editor_clap_check' + exe)),
            str(package / "CLAP/ZorakAudio JIT Editor PoC.clap")], isolated=True)
    vst_binary = 'Contents/x86_64-win/ZorakAudio JIT Editor PoC.vst3' if sys.platform == 'win32' else 'Contents/x86_64-linux/ZorakAudio JIT Editor PoC.so'
    checked("packaged-vst3", [str(build / ('jit_editor_vst3_check' + exe)),
            str(package / 'VST3/ZorakAudio JIT Editor PoC.vst3' / vst_binary)], isolated=True)
    print("[jit-ci] All packaged compiler, runtime, interface and public plugin checks passed", flush=True)


if __name__ == "__main__":
    main()
