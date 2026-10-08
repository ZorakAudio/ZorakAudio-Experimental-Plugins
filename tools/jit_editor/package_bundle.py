"""Package the standalone shared-runtime editor without installing plugins."""
import argparse
import json
import hashlib
from pathlib import Path
import shutil
import zipfile
from package_runtime import stage

REPO = Path(__file__).resolve().parents[2]


def bytes_in(folder):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def package(build, output, faust, python, staging=None):
    output.mkdir(parents=True, exist_ok=True)
    staging = staging or build / "package"
    if staging.exists():
        if not staging.resolve().is_relative_to((REPO / "build").resolve()):
            raise ValueError("Refusing to clear staging outside repository build directory")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    runtime = staging / "runtime/JITEditor.runtime"
    stage(runtime, python, faust)
    source_files=list((REPO/'src').glob('*'))+[REPO/'dsp_jsfx_aot.py']+list(Path(__file__).parent.glob('*.cpp'))+list(Path(__file__).parent.glob('*.h'))+list(Path(__file__).parent.glob('*.py'))
    source_files+=[REPO/'scripts'/name for name in ('jsfx_faust_compiler.py','jsfx_tasks_compiler.py','jsfx_source.py','jsfx_preprocessor.py')]
    source_files+=list((REPO/'tools/native_compiler').glob('*.cpp'))+list((REPO/'tools/native_compiler').glob('*.h'))
    source_files+=list(Path(__file__).with_name('examples').glob('*'))+[Path(__file__).with_name('CMakeLists.txt')]
    provenance={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files) if p.is_file()}
    result = {}
    for kind in ("CLAP", "VST3"):
        folder = staging / kind
        folder.mkdir()
        name = "ZorakAudio JIT Editor PoC." + kind.lower()
        source = build / "JITEditor_artefacts/Release" / kind / name
        if kind == "CLAP":
            shutil.copy2(source, folder / name)
            destination = folder / "JITEditor.runtime"
        else:
            shutil.copytree(source, folder / name, ignore=shutil.ignore_patterns("JITEditor.runtime"))
            destination = folder / name / "Contents/Resources/JITEditor.runtime"
        shutil.copytree(runtime, destination, ignore=shutil.ignore_patterns("__pycache__"))
        for document in ("README.md","CPP-MIGRATION.md","VALIDATION.md"):
            shutil.copy2(Path(__file__).parent / document, folder / document)
        shutil.copytree(Path(__file__).with_name('examples'),folder/'examples')
        (folder/'SOURCE-SHA256.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
        archive = output / ("JIT-Editor-Shared-Runtime-Windows-" + kind + ".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for file in sorted(folder.rglob("*")):
                if file.is_file(): zipped.write(file, str(file.relative_to(folder)))
        result[kind] = dict(zipBytes=archive.stat().st_size, installedBytes=bytes_in(folder),
                            compilerBytes=bytes_in(destination), binaryBytes=source.stat().st_size if kind == "CLAP" else bytes_in(source)-bytes_in(source / "Contents/Resources/JITEditor.runtime"))
    (output / "JIT-Editor-Shared-Runtime-sizes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=REPO / "build/jit-editor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--faust", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--staging",type=Path)
    args = parser.parse_args()
    package(args.build.resolve(),args.output.resolve(),args.faust.resolve(),args.python.resolve(),args.staging.resolve() if args.staging else None)
