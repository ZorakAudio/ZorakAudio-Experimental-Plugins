import re
import argparse
import json
import os
import subprocess
import sys
import shutil
import zipfile
from pathlib import Path

def _run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace").strip()

def find_vs_installation_path() -> str | None:
    """Locate latest Visual Studio with VC tools using vswhere (Windows only)."""
    if os.name != "nt":
        return None

    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
              "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None

    try:
        return _run_text([
            str(vswhere),
            "-latest",
            "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "installationPath"
        ])
    except Exception:
        return None

def pick_cmake_vs_generator(vs_path: str | None) -> str:
    """Pick a CMake Visual Studio generator name from an installation path.

    VS 2022 -> 'Visual Studio 17 2022'
    VS 2026 -> 'Visual Studio 18 2026' (requires CMake that knows this generator)
    """
    if not vs_path:
        # Default to newest we intend to support.
        return "Visual Studio 18 2026"

    p = vs_path.lower().replace("/", "\\")
    if "\\2026\\" in p:
        return "Visual Studio 18 2026"
    if "\\2022\\" in p:
        return "Visual Studio 17 2022"

    # Unknown layout/version; default to newest.
    return "Visual Studio 18 2026"

def is_macos() -> bool:
    return sys.platform == "darwin"


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

def clean_build_dir(repo_root: Path, os_id: str) -> None:
    """
    Delete build/<os_id> entirely.

    CMake caches generator/instance/toolset in the binary dir; partial deletes are unreliable.
    """
    build_root = repo_root / "build" / os_id
    if build_root.exists():
        print(f"[clean] removing {build_root}")
        shutil.rmtree(build_root)
    else:
        print(f"[clean] nothing to remove ({build_root} does not exist)")



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

def find_single_jsfx(plugin_dir: Path) -> Path:
    jsfxs = list(plugin_dir.glob("*.jsfx"))
    if len(jsfxs) == 0:
        die(f"No .jsfx found in {plugin_dir}")
    if len(jsfxs) > 1:
        die(f"Multiple .jsfx files found in {plugin_dir}. Keep exactly 1 .jsfx per plugin dir.")
    return jsfxs[0]


def find_jsfx_aot_compiler(repo_root: Path) -> Path:
    p = repo_root / "dsp_jsfx_aot.py"
    if not p.exists():
        die(f"dsp_jsfx_aot.py missing at repo root: {p}")
    return p


def build_jsfx_aot(repo_root: Path, cmake_build: Path, slug: str, jsfx_path: Path) -> tuple[Path, Path, Path, Path]:
    """
    Produces:
      - JSFXDSP.o    (or .obj on Windows depending on your toolchain; we still name it .o here)
      - JSFXDSP.h
      - JSFXDSP_meta.json
      - JSFXDSP.ll
    inside the per-plugin build dir (cmake_build).
    Returns (out_obj, out_h, out_meta, out_ll).
    """
    out_obj = cmake_build / ("JSFXDSP.obj" if os.name == "nt" else "JSFXDSP.o")
    out_h    = cmake_build / "JSFXDSP.h"
    out_meta = cmake_build / "JSFXDSP_meta.json"
    out_ll   = cmake_build / "JSFXDSP.ll"

    # Emit JSFX source into a header so the plugin can parse slider declarations at compile-time.
    # This avoids runtime file IO and makes CI packaging sane.
    out_src_h = cmake_build / "JSFXSource.h"

    jsfx_text = jsfx_path.read_text(encoding="utf-8", errors="replace")

    # Use a raw-string delimiter that is extremely unlikely to appear in JSFX code.
    delim = "JSFXSRC"
    out_src_h.write_text(
        '// Auto-generated by build.py\n'
        '#pragma once\n'
        'static const char* kJsfxSourceText = R"' + delim + '(\n' +
        jsfx_text +
        '\n)' + delim + '";\n',
        encoding="utf-8"
    )


    comp = find_jsfx_aot_compiler(repo_root)
    cmd = [
        sys.executable, str(comp),
        str(jsfx_path),
        "--out-ll", str(out_ll),
        "--out-obj", str(out_obj),
        "--out-h", str(out_h),
        "--meta", str(out_meta),
        "--opt", "2",
    ]

    # Force COFF object on Windows. Your AOT script must support --target (see Patch B).
    if os.name == "nt":
        cmd += ["--target", "x86_64-pc-windows-msvc"]

    # Build object
    if sys.platform == "darwin":
        # If the build is universal2, we must produce a universal2 JSFXDSP.o as well.
        # Read arch intent from env (or default to universal2 in CI).
        archs = os.environ.get("CMAKE_OSX_ARCHITECTURES", "arm64;x86_64")
        arch_list = [a.strip() for a in archs.replace(",", ";").split(";") if a.strip()]

        # Build per-arch objects and lipo them if needed.
        objs = []
        for a in arch_list:
            if a == "arm64":
                triple = "arm64-apple-macos11.0"
                out_arch = cmake_build / "JSFXDSP_arm64.o"
            elif a == "x86_64":
                triple = "x86_64-apple-macos11.0"
                out_arch = cmake_build / "JSFXDSP_x86_64.o"
            else:
                die(f"Unsupported macOS arch in CMAKE_OSX_ARCHITECTURES: {a}")

            cmd_arch = cmd[:]  # clone
            # Ensure target triple and per-arch output obj
            cmd_arch += ["--target", triple]
            # Replace the --out-obj argument value (last one in cmd is safe in our usage)
            # Easiest: rebuild the arg list with correct out-obj.
            # We'll just append a second --out-obj that overrides the first if your argparse uses last-wins.
            cmd_arch += ["--out-obj", str(out_arch)]

            run(cmd_arch)
            objs.append(out_arch)

        if len(objs) == 1:
            # single-arch build
            pass
        else:
            # universal2 build
            run(["lipo", "-create", "-output", str(out_obj), *[str(p) for p in objs]])
    else:
        run(cmd)



    if not out_obj.exists():
        die(f"JSFX AOT did not produce object file: {out_obj}")
    if not out_h.exists():
        die(f"JSFX AOT did not produce header file: {out_h}")

    return out_obj, out_h, out_meta, out_ll


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
    ap.add_argument("--only", default="", help="Build only one plugin (match slug OR name OR dir). Case-insensitive.")
    ap.add_argument("--clean", action="store_true", help="Delete build directory for current platform before building")
    ap.add_argument("--clean-only", action="store_true", help="Delete build directory for current platform and exit")

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
    out_dir.mkdir(parents=True, exist_ok=True)
    build_root = repo_root / "build" / os_id
    # --clean / --clean-only: nuke build/<os_id> before generating DSP or running CMake.
    if args.clean or args.clean_only:
        clean_build_dir(repo_root, os_id)
        if args.clean_only:
            return


    enable_clap = (os_id in ("windows", "linux"))  # matches your “time-optimal” plan

    bundle_root = out_dir / "bundle"
    vst3_dir = bundle_root / "VST3"
    clap_dir = bundle_root / "CLAP"


    for p in plugins:
        slug = p["slug"]

        if args.only:
            needle = args.only.lower()
            if (needle not in slug.lower()
                and needle not in p["name"].lower()
                and needle not in p["dir"].lower()):
                continue

        plugin_dir = repo_root / p["dir"]

        if not plugin_dir.exists():
            die(f"Plugin dir not found: {plugin_dir}")

        plugin_type = p.get("pluginType", "faust").lower()
        if plugin_type not in ("faust", "jsfx"):
            die(f"Invalid pluginType for {slug}: {plugin_type} (expected 'faust' or 'jsfx')")


        cmake_build = build_root / slug
        cmake_build.mkdir(parents=True, exist_ok=True)

        # Decide build inputs per plugin type
        dsp = None
        jsfx_obj = None

        if plugin_type == "faust":
            dsp = find_single_dsp(plugin_dir)
        else:
            jsfx = find_single_jsfx(plugin_dir)
            jsfx_obj, _, _, _ = build_jsfx_aot(repo_root, cmake_build, slug, jsfx)

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
            f"-DPLUGIN_TYPE={plugin_type}",
            f"-DPLUGIN_DSP={dsp if dsp else ''}",
            f"-DPLUGIN_JSFX_OBJ={jsfx_obj if jsfx_obj else ''}",

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
            # Visual Studio generator selection:
            # - Allow override via env:
            #     ZA_CMAKE_GENERATOR="Visual Studio 18 2026"
            #     ZA_CMAKE_GENERATOR_INSTANCE="C:\\Path\\To\\VS"
            # - Otherwise, locate VS via vswhere and pick a generator based on its version.
            gen = os.environ.get("ZA_CMAKE_GENERATOR")
            inst = os.environ.get("ZA_CMAKE_GENERATOR_INSTANCE")

            if not gen:
                vs_path = find_vs_installation_path()
                gen = pick_cmake_vs_generator(vs_path)

                # If we found an install path and no explicit override was given,
                # point CMake at the correct instance so it doesn't try a hard-coded 2022 path.
                if vs_path and not inst:
                    inst = vs_path

            cmake_args += ["-G", gen, "-A", "x64"]

            # Only set instance if we actually have one. Never hardcode ".../2022/Community".
            if inst:
                cmake_args += [f"-DCMAKE_GENERATOR_INSTANCE={inst}"]
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

        if enable_clap:
            for a in claps:
                copy_bundle(a, clap_dir)


    # --- macOS: ad-hoc sign bundles BEFORE packaging ---
    if is_macos():
        print("Ad-hoc signing macOS bundles")

        if vst3_dir.exists():
            for b in vst3_dir.iterdir():
                if b.suffix == ".vst3":
                    run(["codesign", "--force", "--deep", "--sign", "-", str(b)])

        if clap_dir.exists():
            for c in clap_dir.iterdir():
                if c.suffix == ".clap":
                    run(["codesign", "--force", "--sign", "-", str(c)])

    if clap_dir.exists() and not any(clap_dir.iterdir()):
        clap_dir.rmdir()
    if vst3_dir.exists() and not any(vst3_dir.iterdir()):
        vst3_dir.rmdir()


    zip_name = f"ZorakAudio-Experimental-Plugins-{args.tag}-{os_id}.zip"
    zip_out = out_dir / zip_name

    if is_macos():
        print("Packaging macOS bundle with ditto")
        run([
            "ditto",
            "-c",
            "-k",
            "--sequesterRsrc",
            "--keepParent",
            str(bundle_root),
            str(zip_out),
        ])
    else:
        zip_path(bundle_root, zip_out)

    print(f"Packed: {zip_name}")


    print(f"Done. Output: {out_dir}")

if __name__ == "__main__":
    main()
