"""Stage an isolated compiler payload from explicitly supplied local installations.

No downloads or installation. Packaging a compiler also requires retaining its
licenses; public redistribution needs a separate license/source-offer review.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

REPO = Path(__file__).resolve().parents[2]


def stage(destination, python, faust):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    py = destination / "python"
    py.mkdir(exist_ok=True)
    for pattern in ("python.exe", "python3.dll", "python3*.dll", "vcruntime*.dll", "LICENSE.txt"):
        for source in python.glob(pattern):
            shutil.copy2(source, py / source.name)
    dlls = py / "DLLs"
    dlls.mkdir(exist_ok=True)
    for source in (python / "DLLs").iterdir():
        if source.suffix in (".dll", ".pyd"):
            shutil.copy2(source, dlls / source.name)
    excluded = {"site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter", "turtledemo", "ensurepip"}
    version = f"python{sys.version_info.major}{sys.version_info.minor}"
    with zipfile.ZipFile(py / (version + ".zip"), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sorted((python / "Lib").rglob("*.py")):
            relative = source.relative_to(python / "Lib")
            if not excluded.intersection(relative.parts):
                archive.write(source, str(relative))
    package = python / "Lib/site-packages/llvmlite"
    target = py / "Lib/site-packages/llvmlite"
    shutil.copytree(package, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "tests"))
    for source in (python / "Lib/site-packages/llvmlite.libs").glob("*.dll"):
        shutil.copy2(source, target / "binding" / source.name)
    # LLVM has C++ CRT imports; ship those adjacent to its DLL too.
    for source in (faust / "bin").glob("*.dll"):
        shutil.copy2(source, target / "binding" / source.name)
    (py / (version + "._pth")).write_text(f"{version}.zip\nDLLs\nLib/site-packages\n../compiler\n", encoding="utf-8")
    compiler = destination / "compiler"
    (compiler / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "dsp_jsfx_aot.py", compiler)
    shutil.copy2(REPO / "src/JsfxRuntimeExports.inc", compiler)
    shutil.copy2(Path(__file__).parent / "compiler_worker.py", compiler)
    shutil.copy2(Path(__file__).parent / "native_frontend.py", compiler)
    native_frontend = REPO / "build/native-compiler/jsfx_frontend.exe"
    if not native_frontend.is_file(): raise ValueError("Build the native frontend before packaging this editor version")
    shutil.copy2(native_frontend, compiler / "jsfx_frontend.exe")
    for name in ("jsfx_faust_compiler.py", "jsfx_tasks_compiler.py", "jsfx_source.py", "jsfx_preprocessor.py"):
        shutil.copy2(REPO / "scripts" / name, compiler / "scripts")
    preprocessor=REPO / "build/tools/jsfx_eel_pp/bin/jsfx_eel_pp.exe"
    if not preprocessor.exists():raise ValueError("Build the production JSFX preprocessor before packaging")
    shutil.copy2(preprocessor,compiler / "jsfx_eel_pp.exe")
    (compiler / "scripts/__init__.py").write_text("", encoding="utf-8")
    bin_path = destination / "faust/bin"
    bin_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(faust / "bin/faust.exe", bin_path)
    for source in (faust / "bin").glob("*.dll"):
        shutil.copy2(source, bin_path / source.name)
    for source in (faust / "share/faust").rglob("*.lib"):
        target = destination / "faust/share/faust" / source.relative_to(faust / "share/faust")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    licenses = destination / "licenses"
    licenses.mkdir(exist_ok=True)
    shutil.copy2(REPO / "LICENSE", licenses / "Repository-LICENSE")
    shutil.copy2(REPO / "libs/JUCE/LICENSE.md", licenses / "JUCE-LICENSE.md")
    shutil.copy2(REPO / "src/WDL/LICENSE.txt", licenses / "WDL-LICENSE.txt")
    for source in (REPO / "libs/clap-juce-extensions").glob("LICENSE*"):
        if source.is_file(): shutil.copy2(source, licenses / ("CLAP-JUCE-" + source.name))
    shutil.copy2(python / "LICENSE.txt", licenses / "Python-LICENSE.txt")
    for source in (python / "Lib/site-packages").glob("llvmlite-*.dist-info/licenses/*"):
        if source.is_file(): shutil.copy2(source, licenses / ("llvmlite-" + source.name))
    for source in (faust / "share/faust").glob("*COPYING*"):
        if source.is_file(): shutil.copy2(source, licenses / ("Faust-" + source.name))
    manifest = dict(python=sys.version, llvmlite=__import__("llvmlite").__version__,
                    platform="Windows x64", compilerBackend="LLVM ORC; standard or C++ frontend; production Python lowering/emission", files={})
    for category in ("python", "compiler", "faust", "licenses"):
        manifest["files"][category] = sum(p.stat().st_size for p in (destination / category).rglob("*") if p.is_file())
    (destination / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable).parent)
    parser.add_argument("--faust", type=Path, required=True)
    args = parser.parse_args()
    stage(args.destination, args.python, args.faust)
