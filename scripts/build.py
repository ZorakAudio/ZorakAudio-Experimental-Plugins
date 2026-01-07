import re
import argparse
import json
import os
import subprocess
import sys
import shutil
import zipfile
from pathlib import Path

def copy_bundle(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

def host_os() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"

def zip_path(src: Path, dst_zip: Path) -> None:
    dst_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if src.is_dir():
            base = src.parent
            for p in src.rglob("*"):
                if p.is_dir():
                    continue
                z.write(p, p.relative_to(base))
        else:
            z.write(src, src.name)

def find_single_dsp(plugin_dir: Path) -> Path:
    dsps = list(plugin_dir.glob("*.dsp"))
    if len(dsps) == 0:
        die(f"No .dsp found in {plugin_dir}")
    if len(dsps) > 1:
        die(f"Multiple .dsp files found in {plugin_dir}. Specify one by renaming or making it 1 file per plugin dir.")
    return dsps[0]


def cmake_safe_version(tag: str) -> str:
    # Accept things like v0.2.0, R0.1.1, 0.2.0, 0.2
    m = re.search(r'(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?', tag)
    if not m:
        return "0.0.0"
    parts = [m.group(1), m.group(2) or "0", m.group(3) or "0", m.group(4)]
    # CMake is fine with 3 or 4 numeric parts; keep 3 parts to be safe
    return ".".join(parts[:3])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Release")
    ap.add_argument("--tag", default="0.0.0")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "plugins.json"
    if not cfg_path.exists():
        die("plugins.json missing at repo root")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    manu = cfg["manufacturer"]
    plugins = cfg["plugins"]

    os_id = host_os()
    out_dir = (repo_root / args.out / args.tag / os_id)
    build_root = repo_root / "build" / os_id

    enable_clap = (os_id in ("windows", "linux"))  # matches your “time-optimal” plan

    bundle_root = out_dir / "bundle"
    vst3_dir = bundle_root / "VST3"
    clap_dir = bundle_root / "CLAP"


    for p in plugins:
        plugin_dir = repo_root / p["dir"]
        if not plugin_dir.exists():
            die(f"Plugin dir not found: {plugin_dir}")

        dsp = find_single_dsp(plugin_dir)
        slug = p["slug"]

        cmake_build = build_root / slug
        cmake_build.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "cmake",
            "-S", str(repo_root / "cmake" / "plugin"),
            "-B", str(cmake_build),
            f"-DZA_ROOT={repo_root}",
            f"-DPLUGIN_NAME={p['name']}",
            f"-DPLUGIN_SLUG={slug}",
            f"-DPLUGIN_CODE={p['pluginCode']}",
            f"-DMANUFACTURER_NAME={manu['name']}",
            f"-DMANUFACTURER_CODE={manu['code']}",
            f"-DBUNDLE_ID={p['bundleId']}",
            f"-DPLUGIN_VERSION={cmake_safe_version(args.tag)}",
            f"-DPLUGIN_DSP={dsp}",
        ]

        if enable_clap:
            feats = " ".join(p.get("clapFeatures", ["audio-effect"]))
            cmake_args += [
                "-DZA_ENABLE_CLAP=ON",
                f"-DCLAP_ID={p['clapId']}",
                f"-DCLAP_FEATURES={feats}",
            ]
        else:
            cmake_args += ["-DZA_ENABLE_CLAP=OFF"]

        if os_id == "windows":
            cmake_args += ["-G", "Visual Studio 17 2022", "-A", "x64"]
        else:
            cmake_args += ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"]

        run(cmake_args)
        run(["cmake", "--build", str(cmake_build), "--config", args.config])

        artefacts = cmake_build / f"{slug}_artefacts" / args.config
        if not artefacts.exists():
            die(f"Expected artefacts dir missing: {artefacts}")

        # Package VST3 / CLAP / AU (AU not enabled in CMake above, but kept here for later)
        # find artefacts
        vst3s = list(artefacts.rglob("*.vst3"))
        claps = list(artefacts.rglob("*.clap"))

        for a in vst3s:
            copy_bundle(a, vst3_dir)

        for a in claps:
            copy_bundle(a, clap_dir)

    if clap_dir.exists() and not any(clap_dir.iterdir()):
        clap_dir.rmdir()
    if vst3_dir.exists() and not any(vst3_dir.iterdir()):
        vst3_dir.rmdir()


    zip_name = f"ZorakAudio-Experimental-Plugins-{args.tag}-{os_id}.zip"
    zip_path(bundle_root, out_dir / zip_name)
    print(f"Packed: {zip_name}")


    print(f"Done. Output: {out_dir}")

if __name__ == "__main__":
    main()
