from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from pluginlib import PluginSpec, PluginDiscoveryError, discover_plugins, filter_plugins


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
            "-property", "installationPath",
        ])
    except Exception:
        return None


def pick_cmake_vs_generator(vs_path: str | None) -> str:
    """Pick a CMake Visual Studio generator name from an installation path."""
    if not vs_path:
        return "Visual Studio 18 2026"

    p = vs_path.lower().replace("/", "\\")
    if "\\2026\\" in p:
        return "Visual Studio 18 2026"
    if "\\2022\\" in p:
        return "Visual Studio 17 2022"
    return "Visual Studio 18 2026"


def is_macos() -> bool:
    return sys.platform == "darwin"


def copy_bundle(src: Path, dst_dir: Path) -> Path:
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
    return dst


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
    build_root = repo_root / "build" / os_id
    if build_root.exists():
        print(f"[clean] removing {build_root}")
        shutil.rmtree(build_root)
    else:
        print(f"[clean] nothing to remove ({build_root} does not exist)")


def zip_path(src: Path, dst_zip: Path) -> None:
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if src.is_dir():
            base = src.parent
            for p in src.rglob("*"):
                if p.is_dir():
                    continue
                zf.write(p, p.relative_to(base))
        else:
            zf.write(src, src.name)


def find_jsfx_aot_compiler(repo_root: Path) -> Path:
    p = repo_root / "dsp_jsfx_aot.py"
    if not p.exists():
        die(f"dsp_jsfx_aot.py missing at repo root: {p}")
    return p




# --- Joep/JSFX compatibility: section-aware textual import preprocessing for embedded source ---
_JSFX_IMPORT_RE = re.compile(
    r"^\s*import\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s;]+))\s*;?\s*(?://.*)?$"
)


def _merge_import_bundle(dst_preamble: list[str], dst_order: list[str], dst_sections: dict[str, list[str]],
                         src_preamble: list[str], src_order: list[str], src_sections: dict[str, list[str]]) -> None:
    dst_preamble.extend(src_preamble)
    for sec in src_order:
        if sec not in dst_sections:
            dst_sections[sec] = []
            dst_order.append(sec)
        dst_sections[sec].extend(src_sections.get(sec, []))


def preprocess_jsfx_imports_from_path(path: Path, _stack: tuple[Path, ...] = ()) -> str:
    path = path.resolve()
    if path in _stack:
        chain = " -> ".join(str(p) for p in (_stack + (path,)))
        raise RuntimeError(f"Cyclic JSFX import chain: {chain}")

    text = path.read_text(encoding="utf-8", errors="replace")
    preamble: list[str] = []
    order: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    current_lines: list[str] = []
    section_re = re.compile(r"^\s*@([A-Za-z_][A-Za-z0-9_]*)\b.*$")

    def flush_current() -> None:
        nonlocal current_lines
        if current is None:
            return
        if current not in sections:
            sections[current] = []
            order.append(current)
        sections[current].extend(current_lines)
        current_lines = []

    for raw_line in text.splitlines(True):
        m_imp = _JSFX_IMPORT_RE.match(raw_line)
        m_sec = section_re.match(raw_line)

        if m_imp:
            token = next((g for g in m_imp.groups() if g), "")
            if not token:
                if current is None:
                    preamble.append(raw_line)
                else:
                    current_lines.append(raw_line)
                continue

            inc_path = (path.parent / token).resolve()
            if not inc_path.exists():
                raise FileNotFoundError(
                    f"Unable to resolve JSFX import {token!r} from {path}"
                )

            child_text = preprocess_jsfx_imports_from_path(inc_path, _stack + (path,))
            child_preamble: list[str] = []
            child_order: list[str] = []
            child_sections: dict[str, list[str]] = {}
            child_current: str | None = None
            child_current_lines: list[str] = []

            def flush_child() -> None:
                nonlocal child_current_lines
                if child_current is None:
                    return
                if child_current not in child_sections:
                    child_sections[child_current] = []
                    child_order.append(child_current)
                child_sections[child_current].extend(child_current_lines)
                child_current_lines = []

            for child_line in child_text.splitlines(True):
                mc = section_re.match(child_line)
                if mc:
                    flush_child()
                    child_current = mc.group(1)
                    child_current_lines = []
                    continue
                if child_current is None:
                    child_preamble.append(child_line)
                else:
                    child_current_lines.append(child_line)
            flush_child()

            if current is None:
                _merge_import_bundle(preamble, order, sections, child_preamble, child_order, child_sections)
            else:
                current_lines.extend(child_preamble)
                for sec in child_order:
                    if sec == current:
                        current_lines.extend(child_sections.get(sec, []))
                    else:
                        if sec not in sections:
                            sections[sec] = []
                            order.append(sec)
                        sections[sec].extend(child_sections.get(sec, []))
            continue

        if m_sec:
            flush_current()
            current = m_sec.group(1)
            current_lines = []
            continue

        if current is None:
            preamble.append(raw_line)
        else:
            current_lines.append(raw_line)

    flush_current()
    out: list[str] = list(preamble)
    for sec in order:
        out.append(f"@{sec}\n")
        out.extend(sections.get(sec, []))
        if out and not out[-1].endswith("\n"):
            out.append("\n")
    return "".join(out)

def _c_escape_utf8_units(text: str) -> list[str]:
    units: list[str] = []
    for b in text.encode("utf-8"):
        if b == 0x5C:  # backslash
            units.append("\\\\")
        elif b == 0x22:  # quote
            units.append('\\"')
        elif b == 0x0A:  # newline
            units.append("\\n")
        elif b == 0x0D:  # carriage return
            continue
        elif 0x20 <= b <= 0x7E:
            units.append(chr(b))
        else:
            units.append("\\" + f"{b:03o}")
    return units

def write_embedded_text_header(*, text: str, variable_name: str, out_header: Path, banner: str) -> Path:
    chunk = 4000
    parts: list[str] = []
    current = ""

    for unit in _c_escape_utf8_units(text):
        if current and len(current) + len(unit) > chunk:
            parts.append(current)
            current = unit
        else:
            current += unit

    if current or not parts:
        parts.append(current)

    header_lines = [
        f"// Auto-generated by build.py ({banner})\n",
        "#pragma once\n",
        f"static const char {variable_name}[] =\n",
    ]
    for part in parts:
        header_lines.append(f'"{part}"\n')
    header_lines.append(";\n")
    out_header.write_text("".join(header_lines), encoding="utf-8")
    return out_header

def write_plugin_readme_header(cmake_build: Path, readme_path: Path) -> Path:
    readme_text = readme_path.read_text(encoding="utf-8", errors="replace")
    return write_embedded_text_header(
        text=readme_text,
        variable_name="kPluginReadmeMarkdownText",
        out_header=cmake_build / "PluginReadme.h",
        banner=f"embedded README from {readme_path.name}",
    )


def build_jsfx_aot(repo_root: Path, cmake_build: Path, slug: str, jsfx_path: Path) -> tuple[Path, Path, Path, Path]:
    """
    Produces:
      - JSFXDSP.o / JSFXDSP.obj
      - JSFXDSP.h
      - JSFXDSP_meta.json
      - JSFXDSP.ll
    inside the per-plugin build dir.
    """
    def env_truthy(name: str) -> bool:
        v = os.environ.get(name, "").strip().lower()
        return v not in ("", "0", "false", "no", "off")

    out_obj = cmake_build / ("JSFXDSP.obj" if os.name == "nt" else "JSFXDSP.o")
    out_h = cmake_build / "JSFXDSP.h"
    out_meta = cmake_build / "JSFXDSP_meta.json"
    out_ll = cmake_build / "JSFXDSP.ll"
    out_src_h = cmake_build / "JSFXSource.h"

    jsfx_text = preprocess_jsfx_imports_from_path(jsfx_path.resolve())

    write_embedded_text_header(
        text=jsfx_text,
        variable_name="kJsfxSourceText",
        out_header=out_src_h,
        banner=f"embedded JSFX source from {jsfx_path.name}",
    )

    opt_level = os.environ.get("JSFX_AOT_OPT_LEVEL", "2").strip() or "2"
    opt_dump_root = os.environ.get("JSFX_AOT_OPT_DUMP_DIR", "").strip()

    enable_custom_opt = env_truthy("JSFX_AOT_ENABLE_CUSTOM_OPT")
    enable_section_hoist = enable_custom_opt or env_truthy("JSFX_AOT_ENABLE_SECTION_HOIST")
    enable_loop_hoist = enable_custom_opt or env_truthy("JSFX_AOT_ENABLE_LOOP_HOIST")

    # Backward-compatible escape hatches if local scripts still export the old names.
    if env_truthy("JSFX_AOT_DISABLE_SECTION_HOIST"):
        enable_section_hoist = False
    if env_truthy("JSFX_AOT_DISABLE_LOOP_HOIST"):
        enable_loop_hoist = False

    comp = find_jsfx_aot_compiler(repo_root)
    cmd = [
        sys.executable, str(comp),
        str(jsfx_path),
        "--out-ll", str(out_ll),
        "--out-obj", str(out_obj),
        "--out-h", str(out_h),
        "--meta", str(out_meta),
        "--opt", opt_level,
    ]

    if opt_dump_root:
        cmd += ["--opt-dump-dir", str(Path(opt_dump_root) / slug)]
    if enable_section_hoist:
        cmd += ["--enable-section-hoist"]
    if enable_loop_hoist:
        cmd += ["--enable-loop-hoist"]
    if env_truthy("JSFX_AOT_DISABLE_SECTION_HOIST"):
        cmd += ["--no-section-hoist"]
    if env_truthy("JSFX_AOT_DISABLE_LOOP_HOIST"):
        cmd += ["--no-loop-hoist"]

    if os.name == "nt":
        cmd += ["--target", "x86_64-pc-windows-msvc"]

    if sys.platform == "darwin":
        archs = os.environ.get("CMAKE_OSX_ARCHITECTURES", "arm64;x86_64")
        arch_list = [a.strip() for a in archs.replace(",", ";").split(";") if a.strip()]

        objs: list[Path] = []
        for arch in arch_list:
            if arch == "arm64":
                triple = "arm64-apple-macos11.0"
                out_arch = cmake_build / "JSFXDSP_arm64.o"
            elif arch == "x86_64":
                triple = "x86_64-apple-macos11.0"
                out_arch = cmake_build / "JSFXDSP_x86_64.o"
            else:
                die(f"Unsupported macOS arch in CMAKE_OSX_ARCHITECTURES: {arch}")

            cmd_arch = cmd[:] + ["--target", triple, "--out-obj", str(out_arch)]
            run(cmd_arch)
            objs.append(out_arch)

        if len(objs) > 1:
            run(["lipo", "-create", "-output", str(out_obj), *[str(p) for p in objs]])
    else:
        run(cmd)

    if not out_obj.exists():
        die(f"JSFX AOT did not produce object file: {out_obj}")
    if not out_h.exists():
        die(f"JSFX AOT did not produce header file: {out_h}")

    return out_obj, out_h, out_meta, out_ll


def derive_jsfx_plugin_capabilities(meta: dict | None) -> dict[str, str]:
    meta = dict(meta or {})
    midi = dict(meta.get("midi") or {})
    plugin_kind = str(meta.get("plugin_kind") or "audio_effect").lower()

    accepts_midi = bool(midi.get("accepts_midi_input"))
    produces_midi = bool(midi.get("produces_midi_output"))

    is_synth = plugin_kind == "instrument"
    is_midi_effect = plugin_kind == "midi_effect"

    return {
        "PLUGIN_KIND": plugin_kind,
        "PLUGIN_IS_SYNTH": "ON" if is_synth else "OFF",
        "PLUGIN_NEEDS_MIDI_INPUT": "ON" if accepts_midi else "OFF",
        "PLUGIN_NEEDS_MIDI_OUTPUT": "ON" if produces_midi else "OFF",
        "PLUGIN_IS_MIDI_EFFECT": "ON" if is_midi_effect else "OFF",
    }


def cmake_safe_version(tag: str) -> str:
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?", tag)
    if not m:
        return "0.0.0"
    parts = [m.group(1), m.group(2) or "0", m.group(3) or "0", m.group(4)]
    return ".".join(parts[:3])


def list_plugins_for_humans(plugins: list[PluginSpec]) -> None:
    print("Discovered plugins:\n")

    categories: dict[str, list[PluginSpec]] = {}
    for spec in plugins:
        categories.setdefault(spec.category, []).append(spec)

    for category in sorted(categories):
        print(f"{category}:")
        for spec in sorted(categories[category], key=lambda s: (s.key.lower(), s.name.lower())):
            rel = spec.repo_rel_dir
            fmt = "JSFX" if spec.plugin_type == "jsfx" else "Faust"
            print(
                f"  - {spec.key:20} [{fmt:5}]  {spec.name}\n"
                f"      slug: {spec.slug}\n"
                f"      repo: {rel}\n"
                f"      package: {spec.install_display}\n"
            )


def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()


def write_install_guide(stage_root: Path, built_specs: list[PluginSpec]) -> None:
    lines = [
        "ZorakAudio Experimental Plugins",
        "",
        "Copy the category folders inside VST3/ into your VST3 plugin folder.",
        "Copy the category folders inside CLAP/ into your CLAP plugin folder.",
        "",
        "The release layout mirrors plugins/<Category>/<PluginKey>/ from the repository.",
        "There are no subcategory layers in the packaged install tree.",
        "Most hosts that support recursive scanning will preserve this organization in place.",
        "",
        "Plugins included in this package:",
    ]

    categories: dict[str, list[PluginSpec]] = {}
    for spec in built_specs:
        categories.setdefault(spec.category, []).append(spec)

    for category in sorted(categories):
        lines.append("")
        lines.append(f"[{category}]")
        for spec in sorted(categories[category], key=lambda s: (s.key.lower(), s.name.lower())):
            lines.append(f"- {spec.key} -> {spec.name} [{spec.plugin_type}]")

    (stage_root / "INSTALL.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_release_manifest(stage_root: Path, built_specs: list[PluginSpec]) -> None:
    manifest = {
        "schemaVersion": 2,
        "package": stage_root.name,
        "plugins": [
            {
                "category": spec.category,
                "folderKey": spec.key,
                "name": spec.name,
                "slug": spec.slug,
                "pluginType": spec.plugin_type,
                "entry": str(spec.entry_rel),
                "repositoryPath": str(spec.repo_rel_dir),
                "installPath": spec.install_display,
                "bundleId": spec.bundle_id,
                "clapId": spec.clap_id,
                "clapFeatures": list(spec.clap_features),
            }
            for spec in built_specs
        ],
    }
    (stage_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Release")
    ap.add_argument("--tag", default="0.0.0")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--only", default="", help="Build only one plugin (match category, key, slug, name, path, bundleId, or clapId).")
    ap.add_argument("--clean", action="store_true", help="Delete build directory for current platform before building")
    ap.add_argument("--clean-only", action="store_true", help="Delete build directory for current platform and exit")
    ap.add_argument("--correctness-check", action="store_true", help="Enable JSFX shadow EEL2 correctness monitor/instrumentation")
    ap.add_argument("--list", action="store_true", help="List discovered plugins and exit")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    try:
        plugins = discover_plugins(repo_root)
    except PluginDiscoveryError as exc:
        die(str(exc))

    if args.list:
        list_plugins_for_humans(plugins)
        return

    selected = filter_plugins(plugins, args.only)
    if args.only and not selected:
        die(f"No plugins matched --only={args.only!r}")

    os_id = host_os()
    out_dir = repo_root / args.out / args.tag / os_id
    out_dir.mkdir(parents=True, exist_ok=True)
    build_root = repo_root / "build" / os_id
    if args.clean or args.clean_only:
        clean_build_dir(repo_root, os_id)
        if args.clean_only:
            return

    enable_clap = os_id in ("windows", "linux")
    package_root_name = f"ZorakAudio-Experimental-Plugins-{args.tag}-{os_id}"
    stage_root = out_dir / package_root_name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    vst3_dir = stage_root / "VST3"
    clap_dir = stage_root / "CLAP"

    built_specs: list[PluginSpec] = []

    for spec in selected:
        slug = spec.slug
        print(f"\n=== Building {spec.name} ({slug}) ===")

        cmake_build = build_root / slug
        cmake_build.mkdir(parents=True, exist_ok=True)
        write_plugin_readme_header(cmake_build, spec.readme_path)

        dsp: Path | None = None
        jsfx_obj: Path | None = None
        jsfx_caps = {
            "PLUGIN_KIND": "audio_effect",
            "PLUGIN_IS_SYNTH": "OFF",
            "PLUGIN_NEEDS_MIDI_INPUT": "OFF",
            "PLUGIN_NEEDS_MIDI_OUTPUT": "OFF",
            "PLUGIN_IS_MIDI_EFFECT": "OFF",
        }

        if spec.plugin_type == "faust":
            dsp = spec.entry_path
        else:
            jsfx_obj, _, jsfx_meta_path, _ = build_jsfx_aot(repo_root, cmake_build, slug, spec.entry_path)
            jsfx_meta = json.loads(jsfx_meta_path.read_text(encoding="utf-8")) if jsfx_meta_path.exists() else {}
            jsfx_caps = derive_jsfx_plugin_capabilities(jsfx_meta)

        cmake_args = [
            "cmake",
            "-S", str(repo_root / "cmake" / "plugin"),
            "-B", str(cmake_build),
            f"-DZA_ROOT={repo_root}",
            f"-DPLUGIN_NAME={spec.name}",
            f"-DPLUGIN_SLUG={spec.slug}",
            f"-DPLUGIN_CODE={spec.plugin_code}",
            f"-DMANUFACTURER_NAME={spec.manufacturer_name}",
            f"-DMANUFACTURER_CODE={spec.manufacturer_code}",
            f"-DBUNDLE_ID={spec.bundle_id}",
            f"-DPLUGIN_VERSION={cmake_safe_version(args.tag)}",
            f"-DPLUGIN_TYPE={spec.plugin_type}",
            f"-DPLUGIN_DSP={dsp if dsp else ''}",
            f"-DPLUGIN_JSFX_OBJ={jsfx_obj if jsfx_obj else ''}",
            f"-DPLUGIN_KIND={jsfx_caps['PLUGIN_KIND']}",
            f"-DPLUGIN_IS_SYNTH={jsfx_caps['PLUGIN_IS_SYNTH']}",
            f"-DPLUGIN_NEEDS_MIDI_INPUT={jsfx_caps['PLUGIN_NEEDS_MIDI_INPUT']}",
            f"-DPLUGIN_NEEDS_MIDI_OUTPUT={jsfx_caps['PLUGIN_NEEDS_MIDI_OUTPUT']}",
            f"-DPLUGIN_IS_MIDI_EFFECT={jsfx_caps['PLUGIN_IS_MIDI_EFFECT']}",
            f"-DZA_JSFX_CORRECTNESS_CHECK={'ON' if args.correctness_check else 'OFF'}",
        ]

        fft_legacy_env = os.environ.get("ZA_JSFX_FFT_LEGACY_IN_ORDER")
        if fft_legacy_env is not None:
            fft_legacy_on = fft_legacy_env.strip().lower() in ("1", "true", "yes", "on")
            cmake_args.append(f"-DZA_JSFX_FFT_LEGACY_IN_ORDER={'ON' if fft_legacy_on else 'OFF'}")

        if enable_clap:
            feats = " ".join(spec.clap_features)
            cmake_args += [
                "-DZA_ENABLE_CLAP=ON",
                f"-DCLAP_ID={spec.clap_id}",
                f"-DCLAP_FEATURES={feats}",
            ]
        else:
            cmake_args += ["-DZA_ENABLE_CLAP=OFF"]

        if os_id == "windows":
            gen = os.environ.get("ZA_CMAKE_GENERATOR")
            inst = os.environ.get("ZA_CMAKE_GENERATOR_INSTANCE")

            if not gen:
                vs_path = find_vs_installation_path()
                gen = pick_cmake_vs_generator(vs_path)
                if vs_path and not inst:
                    inst = vs_path

            cmake_args += ["-G", gen, "-A", "x64"]
            if inst:
                cmake_args += [f"-DCMAKE_GENERATOR_INSTANCE={inst}"]
        else:
            cmake_args += ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"]

        run(cmake_args)
        run(["cmake", "--build", str(cmake_build), "--config", args.config])

        artefacts = cmake_build / f"{slug}_artefacts" / args.config
        if not artefacts.exists():
            die(f"Expected artefacts dir missing: {artefacts}")

        install_vst3_dir = vst3_dir / spec.install_rel_dir
        install_clap_dir = clap_dir / spec.install_rel_dir

        vst3s = [p for p in artefacts.rglob("*.vst3") if p.exists()]
        claps = [p for p in artefacts.rglob("*.clap") if p.exists()]

        for artifact in vst3s:
            copy_bundle(artifact, install_vst3_dir)

        if enable_clap:
            for artifact in claps:
                copy_bundle(artifact, install_clap_dir)

        built_specs.append(spec)

    if is_macos():
        print("Ad-hoc signing macOS bundles")
        for bundle in sorted((p for p in stage_root.rglob("*.vst3") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            run(["codesign", "--force", "--deep", "--sign", "-", str(bundle)])
        for binary in sorted(p for p in stage_root.rglob("*.clap") if p.is_file()):
            run(["codesign", "--force", "--sign", "-", str(binary)])

    prune_empty_dirs(stage_root)
    write_install_guide(stage_root, built_specs)
    write_release_manifest(stage_root, built_specs)

    zip_name = f"ZorakAudio-Experimental-Plugins-{args.tag}-{os_id}.zip"
    zip_out = out_dir / zip_name
    if zip_out.exists():
        zip_out.unlink()

    if is_macos():
        print("Packaging macOS bundle with ditto")
        run([
            "ditto",
            "-c",
            "-k",
            "--sequesterRsrc",
            "--keepParent",
            str(stage_root),
            str(zip_out),
        ])
    else:
        zip_path(stage_root, zip_out)

    print(f"Packed: {zip_name}")
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
