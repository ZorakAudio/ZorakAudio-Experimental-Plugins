#!/usr/bin/env python3
"""Unchanged full-native compile gate plus explicitly DSP-only WDL comparisons.

Requires llvmlite and the repository's regular source-expansion dependencies.
This is a headless numerical test, not a JUCE editor or REAPER qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import dsp_jsfx_aot as compiler
from jsfx_source import resolve_source
from pluginlib import discover_plugins

CHANGES = {
    "joep_bandjoiner": {},
    "joep_tanh_saturator_aa": {1: 12, 2: -6, 8: 2},
    "joep_ms_20": {1: 12, 13: .4, 14: .6, 60: 4},
    "joep_saike_morph": {1: 9, 12: 3, 13: .35, 15: .7, 60: 2},
    "joep_saike_stereo_bub_ii": {1: 17, 2: .75, 3: .7, 6: 1},
    "joep_saike_stereo_bub_iii": {1: 17, 2: .75},
    "joep_stereomanipulator": {1: 40, 2: 1800, 3: 170, 5: 3},
    "joep_tonestacks": {1: 4, 2: .2, 3: .8, 4: .3},
    "joep_saike_pitch_shift": {1: 7, 2: .8},
    "joep_saike_smooth": {1: 12, 2: -6, 3: .4, 4: 3},
    "joep_saike_abyss": {1: .35, 2: .65, 7: .35, 11: .8},
    "joep_bandsplitter": {1: 3, 2: .3},
}


def run(cmd, **kwargs):
    return subprocess.run([str(x) for x in cmd], text=True, capture_output=True, **kwargs)


def sliders(source):
    result = []
    for m in re.finditer(r"(?m)^\s*slider(\d+)\s*:\s*(?:([A-Za-z_]\w*)\s*=)?([^<\n]*)<", source):
        index, alias, token = int(m[1])-1, (m[2] or "").lower(), m[3].strip().rstrip(":").strip()
        try:
            value = float(token or "0")
        except ValueError:
            if re.fullmatch(r"[A-Za-z_]\w*", token):
                # EEL permits an expression/variable default; an undeclared
                # variable starts at zero, as in Nott's hidden advanced slider.
                value = 0.
            else:
                raise ValueError(f"non-numeric slider default {m[0]!r}")
        result.append((index, alias, value))
    return result


def config_header(source, slug):
    values = sliders(source)
    text = "#pragma once\n#include <array>\nstruct JoepSlider { int index; const char* alias; double value; };\n"
    text += 'static constexpr const char* joepPluginName = '+json.dumps(slug)+';\n'
    text += f"static const std::array<JoepSlider,{len(values)}> joepSliders = {{{{\n"
    text += "".join(f"  {{{i},{json.dumps(a)},{v!r}}},\n" for i, a, v in values) + "}};\n"
    changes = CHANGES.get(slug, {})
    text += f"static const std::array<JoepSlider,{len(changes)}> joepChanges = {{{{\n"
    text += "".join(f'  {{{i-1},"",{float(v)!r}}},\n' for i, v in changes.items()) + "}};\n"
    text += "inline double joepDefault(int index) { for (const auto& s : joepSliders) if (s.index == index) return s.value; return 0.; }\n"
    return text


def dsp_source(source):
    """Keep original audio bodies; supply a dummy GFX solely to exercise ABI4.

    The resulting test translation unit is never described as a full native
    plugin, and the WDL reference reads the original expanded source directly.
    """
    preamble = re.split(r"(?m)^\s*@\w+", source, maxsplit=1)[0]
    sections = compiler.extract_sections(source)
    return preamble + "\n" + "".join("@" + name + "\n" + sections[name][0] + "\n"
        for name in ("init", "slider", "block", "sample") if name in sections) + "@gfx\n0;\n"


def host_helpers(directory):
    """Extract production block automation/notification helpers verbatim."""
    processor = (REPO / "src/JSFXJuceProcessor.cpp").read_text()
    text = "// Extracted production helpers; do not edit.\n"
    start = processor.index('extern "C" double jsfx_slider_next_chg')
    end = processor.index('\n}', start) + 2
    text += processor[start:end] + "\n"
    start = processor.index("static jsfx_gfx::SliderMask jsfxSliderMask")
    end = processor.index('extern "C" double jsfx_slider_show', start)
    text += processor[start:end]
    (directory / "host_slider_runtime.inc").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wdl-build", type=Path, help="Existing showcase_eel library and numeric_runtime.inc directory")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--plugins", nargs="+", default=list(CHANGES))
    parser.add_argument("--rates", nargs="+", type=int, default=[44100, 48000, 96000])
    parser.add_argument("--reuse-full-matrix", action="store_true", help="Use existing full matrix only when all expanded source hashes match")
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    specs = [s for s in discover_plugins(REPO) if s.category == "JoepVanlier"]
    existing = {}
    matrix_file = out / "native-compile-matrix.json"
    if args.reuse_full_matrix and matrix_file.exists():
        existing = {r["plugin"]: r for r in json.loads(matrix_file.read_text())}
    matrix, sources, comparisons = [], {}, []
    compiler_hash = hashlib.sha256((REPO / "dsp_jsfx_aot.py").read_bytes()).hexdigest()
    for spec in specs:
        directory = out / spec.slug; directory.mkdir(exist_ok=True)
        row = {"plugin": spec.slug, "entry": str(spec.entry_path.relative_to(REPO)),
               "compiler_sha256": compiler_hash,
               "source_sha256": hashlib.sha256(spec.entry_path.read_bytes()).hexdigest()}
        try:
            resolved = resolve_source(spec.entry_path)
            source = resolved.text; sources[spec.slug] = source
            row["expanded_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            row["has_gfx"] = bool(compiler.extract_sections(source).get("gfx", ("",))[0].strip())
            (directory / "expanded.jsfx").write_text(source)
            (directory / "source-manifest.json").write_text(json.dumps(resolved.manifest(REPO), indent=2))
            old = existing.get(spec.slug, {})
            if old.get("expanded_sha256") == row["expanded_sha256"] and old.get("compiler_sha256") == compiler_hash:
                row.update({k: old[k] for k in ("status", "error") if k in old})
            else:
                compiler.compile_jsfx_to_ir(source, native_gfx_legacy=True)
                row["status"] = "PASS"
        except Exception as exc:
            row.update(status="BLOCKED", error=str(exc))
        matrix.append(row)
        matrix_file.write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"Full native: {sum(r['status']=='PASS' for r in matrix)}/{len(matrix)}", flush=True)

    wdl_build = args.wdl_build.resolve() if args.wdl_build else out / "wdl-build"
    initialized = bool(args.wdl_build)
    for slug in args.plugins:
        row = {"plugin": slug, "scope": "DSP only; original audio bodies, no GFX execution", "runs": []}
        directory = out / slug
        try:
            source = sources[slug]
            test_source = dsp_source(source)
            (directory / "dsp-only.jsfx").write_text(test_source)
            (directory / "JoepTestConfig.h").write_text(config_header(source, slug))
            host_helpers(directory)
            row["automation_changes"] = CHANGES[slug]
            mod, meta = compiler.compile_jsfx_to_ir(test_source, native_gfx_legacy=True)
            compiler._aot_opt_and_emit(mod_ir=mod, opt_level=3, emit_obj=str(directory / "JSFXDSP.o"), emit_asm=None, position_independent=True)
            (directory / "JSFXDSP.h").write_text(compiler._emit_header(meta))
            (directory / "JSFXDSP_meta.json").write_text(json.dumps(meta, indent=2))
            row["numeric_semantics"] = meta["numeric_semantics"]
            row["compiler_sha256"] = compiler_hash
            if not initialized:
                for cmd in ([args.cmake, "-S", REPO / "tests/jsfx_showcase", "-B", wdl_build, f"-DSHOWCASE_GENERATED={directory}", "-DCMAKE_BUILD_TYPE=Release"],
                            [args.cmake, "--build", wdl_build, "--target", "showcase_eel", "showcase_numeric_runtime", "-j2"]):
                    result = run(cmd, timeout=180)
                    if result.returncode: raise RuntimeError(result.stdout + result.stderr)
                initialized = True
            cmd = [args.cxx, "-std=c++20", "-O3", "-DNDEBUG", "-DEEL_TARGET_PORTABLE=1", "-DWDL_FFT_REALSIZE=8", "-I"+str(REPO / "src"), "-I"+str(directory), "-I"+str(wdl_build),
                   REPO / "tests/jsfx_showcase/joep_dsp_reference.cpp", directory / "JSFXDSP.o", wdl_build / "libshowcase_eel.a", "-lpthread", "-lm", "-o", directory / "dsp_reference"]
            result = run(cmd, timeout=180)
            (directory / "build.log").write_text(result.stdout + result.stderr)
            if result.returncode: raise RuntimeError("C++ harness/link failed; see build.log: " + result.stderr[-1600:])
            for rate in args.rates:
                for profile in ("defaults", "automation"):
                    for path in ("double", "float"):
                        result = run([directory / "dsp_reference", directory / "expanded.jsfx", rate, profile, path], timeout=120)
                        case = {"rate": rate, "profile": profile, "path": path, "exit_code": result.returncode, "stderr": result.stderr}
                        try: case.update(json.loads(result.stdout))
                        except ValueError: case["stdout"] = result.stdout
                        row["runs"].append(case)
                        (directory / "comparison.json").write_text(json.dumps(row, indent=2) + "\n")
                        print(slug, rate, profile, path, "PASS" if result.returncode == 0 else "FAIL", case.get("max_error", result.stderr[:120]), flush=True)
            row["status"] = "PASS" if all(r["exit_code"] == 0 for r in row["runs"]) else "FAIL"
        except Exception as exc:
            row.update(status="BLOCKED", error=str(exc))
            print(slug, "BLOCKED", str(exc)[:200], flush=True)
        comparisons.append(row)
        (out / "dsp-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    wdl_files = ["eel2/nseel-compiler.c", "eel2/nseel-eval.c", "eel2/nseel-yylex.c",
                 "eel2/nseel-ram.c", "eel2/nseel-cfunc.c", "fft.c", "eel2/eelscript.h",
                 "eel2/glue_port.h", "eel2/ns-eel.h", "eel2/eel_atomic.h", "eel2/eel_fft.h"]
    provenance = {"repository_head": run(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip(),
                  "wdl_source": "Vendored src/WDL portable engine, EEL_TARGET_PORTABLE=1, WDL_FFT_REALSIZE=8",
                  "wdl_file_sha256": {p: hashlib.sha256((REPO / "src/WDL" / p).read_bytes()).hexdigest() for p in wdl_files},
                  "compiler_sha256": compiler_hash,
                  "scope": "Headless portable WDL/EEL2 DSP oracle, original resolved audio bodies; no editor, @gfx, MIDI, serialization, concurrent scheduling or sub-block host automation qualification",
                  "threshold": 2e-7, "duration_seconds_per_case": 2, "block_sizes": [1,17,64,511,128,33]}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
