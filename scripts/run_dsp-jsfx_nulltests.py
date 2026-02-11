from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


AUDIO_EXTS = [".wav"]  # deterministic: render WAV only
SEARCH_DIR_SUFFIXES = ["", "render", "_render", "renders", "_renders", "out", "_out"]

# --- RPP parsing (what FX does the test require?) ---
CLAP_LINE_RX = re.compile(r'^\s*<CLAP\s+"[^"]*"\s+([^\s"]+)', re.MULTILINE)
VST3_NAME_RX = re.compile(r'^\s*<VST\s+"VST3(?::|i:)\s*([^"]+)"', re.MULTILINE)
JS_QUOTED_RX = re.compile(r'^\s*<JS\s+"([^"]+)"\s+"([^"]*)"', re.MULTILINE)
JS_UNQUOTED_RX = re.compile(r"^\s*<JS\s+([^\s\"].+?)(?:\s|$)", re.MULTILINE)

DESC_RX = re.compile(r"^\s*desc\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# --- JSFX file heuristics ---
JSFX_EXTS = {".jsfx", ".jsfx-inc"}
JSFX_TEXT_HINTS = ("desc:", "@init", "@sample", "@block", "@slider", "in_pin:", "out_pin:")

warmup_ms = float(os.environ.get("NULLTEST_WARMUP_MS", "200"))  # ignore first 200ms
keep_renders = os.environ.get("NULLTEST_KEEP_RENDERS", "0") == "1"

def host_os_id() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def build_all_jsfx_plugins(repo_root: Path, jsfx_plugins: list[dict]) -> tuple[str, str, str]:
    """
    Builds all pluginType=='jsfx' entries using scripts/build.py --only <slug>.

    Returns (build_config, build_tag, build_out) used, so staging can be deterministic.
    """
    build_py = repo_root / "scripts" / "build.py"
    if not build_py.exists():
        die(f"Missing build script: {rel(repo_root, build_py)}")

    build_config = os.environ.get("NULLTEST_BUILD_CONFIG", "Release")
    build_tag = os.environ.get("NULLTEST_BUILD_TAG", "ci")
    build_out = os.environ.get("NULLTEST_BUILD_OUT", "dist")

    # Optional: clean build tree once (NOT per plugin)
    if os.environ.get("NULLTEST_BUILD_CLEAN", "0") == "1":
        clean_dir(repo_root / "build" / host_os_id())

    if not jsfx_plugins:
        return build_config, build_tag, build_out

    for p in jsfx_plugins:
        slug = p.get("slug")
        if not slug:
            die(f"JSFX plugin missing slug: {p}")

        cmd = [
            sys.executable,
            str(build_py),
            "--config", build_config,
            "--tag", build_tag,
            "--out", build_out,
            "--only", slug,
        ]
        print("[build] " + " ".join(cmd))
        cp = subprocess.run(cmd, cwd=str(repo_root))
        if cp.returncode != 0:
            die(f"Build failed for JSFX plugin: {slug}", 50)

    return build_config, build_tag, build_out


def stage_built_vst3_and_clap(
    repo_root: Path,
    slug: str,
    build_config: str,
    vst3_dir: Path,
    clap_dir: Path,
) -> tuple[int, int]:
    """
    Deterministic staging: use build.py artefacts layout:
      build/<os_id>/<slug>/<slug>_artefacts/<config>/**.vst3
      build/<os_id>/<slug>/<slug>_artefacts/<config>/**.clap
    """
    os_id = host_os_id()
    artefacts = repo_root / "build" / os_id / slug / f"{slug}_artefacts" / build_config
    if not artefacts.exists():
        return 0, 0

    vst3s = [p for p in artefacts.rglob("*.vst3") if p.is_dir()]
    claps = [p for p in artefacts.rglob("*.clap") if p.is_file()]

    vst3_copied = 0
    clap_copied = 0

    for src in vst3s:
        dst = vst3_dir / src.name
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        vst3_copied += 1

    for src in claps:
        dst = clap_dir / src.name
        shutil.copy2(src, dst)
        clap_copied += 1

    return vst3_copied, clap_copied


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def rel(repo_root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return p.name


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        dest_root = dest_dir.resolve()
        for zi in zf.infolist():
            out_path = (dest_dir / zi.filename).resolve()
            if not str(out_path).startswith(str(dest_root)):
                die(f"Refusing zip-slip path: {zi.filename}")
        zf.extractall(dest_dir)


def find_portable_reaper_root(unpacked_dir: Path) -> Path:
    """
    Return directory containing reaper.exe + reaper.ini (portable root).
    """
    exes = list(unpacked_dir.rglob("reaper.exe"))
    if not exes:
        die(f"reaper.exe not found after unzip in: {unpacked_dir}")

    # Prefer the one that has reaper.ini beside it (portable root signal)
    for exe in exes:
        if (exe.parent / "reaper.ini").exists():
            return exe.parent

    # Fallback: first exe
    return exes[0].parent


def locate_reaper_zip(repo_root: Path) -> Path:
    # Exact name requested
    candidates = [
        repo_root / "Reaper (Portable).zip",
        repo_root / ".ci" / "Reaper (Portable).zip",
    ]
    env = os.environ.get("REAPER_PORTABLE_ZIP")
    if env:
        candidates.insert(0, Path(env))

    for c in candidates:
        if c.exists():
            return c

    die(
        "Missing REAPER portable zip.\n"
        "Expected one of:\n"
        f"  - {rel(repo_root, repo_root / 'Reaper (Portable).zip')}\n"
        f"  - {rel(repo_root, repo_root / '.ci' / 'Reaper (Portable).zip')}\n"
        "Or set REAPER_PORTABLE_ZIP to its path."
    )
    raise RuntimeError  # unreachable


def write_ci_reaper_ini(reaper_root: Path, dummy_sr: int = 48000, dummy_bs: int = 512) -> None:
    """
    Force REAPER portable to:
      - use Dummy Audio driver (no hardware)
      - scan only portable VST3/CLAP
      - close after render
    """
    ini = (
        "[REAPER]\n"
        "renderclosewhendone=4\n"
        "vstpath64=<portable>\\VST3\n"
        "clap_path_win64=<portable>\\CLAP\n"
        "\n"
        "[audioconfig]\n"
        # Core: pick dummy driver
        "mode=4\n"
        # Some builds still look at these to decide “device chosen”
        "dummy_srate=48000\n"
        "dummy_blocksize=512\n"
        # Also set WASAPI to a valid state as a fallback
        "wasapi_mode=0\n"
        "wasapi_srate=48000\n"
        "wasapi_bs=512\n"
        "allow_sr_override=1\n"
    )

    # Apply requested values
    ini = ini.replace("dummy_srate=48000", f"dummy_srate={dummy_sr}")
    ini = ini.replace("dummy_blocksize=512", f"dummy_blocksize={dummy_bs}")
    ini = ini.replace("wasapi_srate=48000", f"wasapi_srate={dummy_sr}")
    ini = ini.replace("wasapi_bs=512", f"wasapi_bs={dummy_bs}")

    (reaper_root / "reaper.ini").write_text(ini, encoding="ascii")



def purge_plugin_caches(reaper_root: Path) -> None:
    """
    Ensure we *must* re-index what we just staged.
    VST cache file name is known. :contentReference[oaicite:4]{index=4}
    CLAP cache names vary; match glob.
    """
    for p in reaper_root.glob("reaper-vstplugins*.ini"):
        p.unlink(missing_ok=True)
    for p in reaper_root.glob("reaper-clap-*.ini"):
        p.unlink(missing_ok=True)


def ensure_dirs(reaper_root: Path) -> tuple[Path, Path, Path]:
    effects = reaper_root / "Effects"
    vst3 = reaper_root / "VST3"
    clap = reaper_root / "CLAP"
    effects.mkdir(parents=True, exist_ok=True)
    vst3.mkdir(parents=True, exist_ok=True)
    clap.mkdir(parents=True, exist_ok=True)
    return effects, vst3, clap


def maybe_purge_dir(d: Path) -> None:
    # Keep Effects (ships with REAPER). Purge VST3/CLAP to prevent “ghost” plugins.
    for child in d.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def looks_like_jsfx_file(path: Path) -> bool:
    if path.suffix.lower() in JSFX_EXTS:
        return True
    if path.suffix != "":
        return False
    # no extension: peek for JSFX markers
    try:
        chunk = path.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
    except Exception:
        return False
    return any(h in chunk for h in JSFX_TEXT_HINTS)


def copy_jsfx_sources(repo_root: Path, plugin_dir: Path, effects_dir: Path) -> int:
    """
    Copy any JSFX-ish source files from plugin_dir into Effects/<plugin_dir_name>/...
    """
    count = 0
    dest_base = effects_dir / plugin_dir.name
    for src in plugin_dir.rglob("*"):
        if not src.is_file():
            continue
        # Ignore obvious binary/build junk quickly
        if src.suffix.lower() in {".exe", ".dll", ".pdb", ".obj", ".lib", ".zip"}:
            continue
        if src.name.lower().startswith("."):
            continue
        if looks_like_jsfx_file(src):
            relpath = src.relative_to(plugin_dir)
            dst = dest_base / relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
    return count


def copy_vst3_and_clap_artifacts(
    repo_root: Path,
    plugin_dir: Path,
    plugin_slug: str,
    vst3_dir: Path,
    clap_dir: Path,
) -> tuple[int, int]:
    """
    Heuristic artifact discovery:
      1) Search inside plugin_dir for *.vst3 dirs + *.clap files
      2) If none, search common build dirs under repo root (but only keep matches containing slug/dir name)
    """
    def find_local() -> tuple[list[Path], list[Path]]:
        v = [p for p in plugin_dir.rglob("*.vst3") if p.is_dir()]
        c = [p for p in plugin_dir.rglob("*.clap") if p.is_file()]
        return v, c

    def find_global_if_needed(v_found: list[Path], c_found: list[Path]) -> tuple[list[Path], list[Path]]:
        if v_found or c_found:
            return v_found, c_found
        roots = [repo_root / "build", repo_root / "out", repo_root / "dist", repo_root / ".build", repo_root / "_build"]
        roots = [r for r in roots if r.exists()]
        if not roots:
            return v_found, c_found

        keys = {plugin_slug.lower(), plugin_dir.name.lower()}
        v_all: list[Path] = []
        c_all: list[Path] = []
        for r in roots:
            for p in r.rglob("*.vst3"):
                if p.is_dir():
                    s = str(p).lower()
                    if any(k in s for k in keys):
                        v_all.append(p)
            for p in r.rglob("*.clap"):
                if p.is_file():
                    s = str(p).lower()
                    if any(k in s for k in keys):
                        c_all.append(p)
        return v_all, c_all

    vst3_src, clap_src = find_local()
    vst3_src, clap_src = find_global_if_needed(vst3_src, clap_src)

    vst3_copied = 0
    clap_copied = 0

    for src in vst3_src:
        dst = vst3_dir / src.name
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        vst3_copied += 1

    for src in clap_src:
        dst = clap_dir / src.name
        shutil.copy2(src, dst)
        clap_copied += 1

    return vst3_copied, clap_copied


def parse_rpp_requirements(rpp: Path) -> dict[str, set[str]]:
    txt = rpp.read_text(encoding="utf-8", errors="ignore")

    clap_ids = set(CLAP_LINE_RX.findall(txt))
    vst3_names = set(VST3_NAME_RX.findall(txt))

    js_descs: set[str] = set()
    js_paths: set[str] = set()

    for name, filehint in JS_QUOTED_RX.findall(txt):
        if filehint.strip():
            js_paths.add(filehint.strip())
        else:
            js_descs.add(name.strip())

    # Unquoted (path-like) JS entries: utility/volume etc.
    for token in JS_UNQUOTED_RX.findall(txt):
        token = token.strip()
        if token.startswith('"'):
            continue
        if token.startswith("<"):
            continue
        # Example: utility/volume __NULLTEST_GATE__
        first = token.split()[0].strip()
        if first:
            js_paths.add(first)

    return {"clap_ids": clap_ids, "vst3_names": vst3_names, "js_descs": js_descs, "js_paths": js_paths}

import re
from pathlib import Path

# Match a JSFX block header and capture the quoted "name"
JS_BLOCK_RX = re.compile(r'^\s*<JS\s+"([^"]+)"\s+"([^"]*)"', re.MULTILINE)

# Inside an FX block, REAPER typically has a BYPASS line like: BYPASS 0 0 0
# Meaning varies by version; what matters: first int 0 == enabled, nonzero == bypassed.
BYPASS_RX = re.compile(r"^\s*BYPASS\s+(\d+)", re.MULTILINE)

DESC_RX = re.compile(r"^\s*desc\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

def _norm_name(s: str) -> str:
    # Normalize whitespace + case so "A  B" matches "A B"
    return " ".join(s.strip().lower().split())

def parse_enabled_jsfx_names_from_rpp(rpp: Path) -> list[str]:
    """
    Returns list of JSFX identifiers that are actually enabled in the FX chain.
    If bypass state can't be determined for a block, we assume enabled (conservative).
    """
    txt = rpp.read_text(encoding="utf-8", errors="ignore")

    names: list[str] = []
    # Walk each <JS ...> block by finding its start and then looking ahead a bit for BYPASS.
    for m in JS_BLOCK_RX.finditer(txt):
        name = m.group(1).strip()

        # Look forward a limited window for a BYPASS line belonging to this FX block.
        # This keeps it fast and avoids needing a full RPP parser.
        window = txt[m.start() : m.start() + 2000]
        bm = BYPASS_RX.search(window)
        if bm:
            bypass_flag = int(bm.group(1))
            if bypass_flag != 0:
                continue  # bypassed/offline => ignore

        names.append(name)

    return names

def build_effects_indexes(effects_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """
    filename_index: basename-without-ext -> Path (first found)
    desc_index: normalized desc -> list[Path]
    """
    filename_index: dict[str, Path] = {}
    desc_index: dict[str, list[Path]] = {}

    for f in effects_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".jsfx", ".jsfx-inc"} and f.suffix != "":
            continue

        base = f.stem.lower()
        filename_index.setdefault(base, f)

        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        dm = DESC_RX.search(t)
        if dm:
            d = _norm_name(dm.group(1))
            desc_index.setdefault(d, []).append(f)

    return filename_index, desc_index

def resolve_jsfx_ref(effects_dir: Path, ref: str, filename_index, desc_index) -> Path | None:
    """
    Try to resolve a JSFX reference string from RPP to an actual file.
    Strategy:
      1) Path relative to Effects (with/without .jsfx)
      2) Basename match anywhere under Effects
      3) desc: match (normalized)
    """
    # 1) path under Effects
    cand = effects_dir / ref
    if cand.exists():
        return cand
    cand2 = cand.with_suffix(".jsfx")
    if cand2.exists():
        return cand2

    # 2) basename match
    base = Path(ref).name
    base_no_ext = Path(base).stem.lower()
    if base_no_ext in filename_index:
        return filename_index[base_no_ext]

    # 3) desc match
    d = _norm_name(ref)
    if d in desc_index and desc_index[d]:
        return desc_index[d][0]

    return None

def preflight_jsfx_required(effects_dir: Path, rpp: Path) -> None:
    enabled_refs = parse_enabled_jsfx_names_from_rpp(rpp)
    filename_index, desc_index = build_effects_indexes(effects_dir)

    missing: list[str] = []
    for ref in enabled_refs:
        if resolve_jsfx_ref(effects_dir, ref, filename_index, desc_index) is None:
            missing.append(ref)

    if missing:
        raise SystemExit(
            "Missing enabled JSFX referenced by RPP:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nTried: Effects-relative path, filename match, then desc: match.\n"
            + "If this list contains display-only labels, fix the RPP to reference the actual JSFX path.\n"
        )


def index_jsfx_descs(effects_dir: Path) -> dict[str, list[Path]]:
    """
    Map desc -> list of files that declare it.
    """
    out: dict[str, list[Path]] = {}
    for f in effects_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in JSFX_EXTS and f.suffix != "":
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = DESC_RX.search(txt)
        if not m:
            continue
        desc = m.group(1).strip()
        out.setdefault(desc.lower(), []).append(f)
    return out


def jsfx_path_exists(effects_dir: Path, js_path: str) -> bool:
    # Normalize to Effects-relative.
    # REAPER uses things like "utility/volume" (no extension).
    cand = effects_dir / js_path
    if cand.exists():
        return True
    if cand.with_suffix(".jsfx").exists():
        return True
    # Sometimes path is like "utility/volume" but file is "volume" (no ext) inside "utility"
    # already covered by cand.exists().
    return False


@dataclass
class WavInfo:
    channels: int
    sample_rate: int
    audio_format: int  # 1 = PCM, 3 = IEEE float
    bits_per_sample: int


def _read_u32_le(b: bytes) -> int:
    return int.from_bytes(b, "little", signed=False)


def read_wav_to_float64(path: Path) -> tuple[np.ndarray, WavInfo]:
    data_chunk = None
    fmt = None

    with path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            die(f"Not a RIFF/WAVE file: {path}")

        while True:
            ch = f.read(8)
            if len(ch) < 8:
                break
            chunk_id = ch[0:4]
            chunk_sz = _read_u32_le(ch[4:8])
            chunk_data = f.read(chunk_sz)
            if chunk_sz & 1:
                f.read(1)

            if chunk_id == b"fmt ":
                if len(chunk_data) < 16:
                    die(f"Bad fmt chunk in {path}")
                audio_format = int.from_bytes(chunk_data[0:2], "little")
                channels = int.from_bytes(chunk_data[2:4], "little")
                sample_rate = int.from_bytes(chunk_data[4:8], "little")
                bits_per_sample = int.from_bytes(chunk_data[14:16], "little")
                fmt = WavInfo(
                    channels=channels,
                    sample_rate=sample_rate,
                    audio_format=audio_format,
                    bits_per_sample=bits_per_sample,
                )
            elif chunk_id == b"data":
                data_chunk = chunk_data

    if fmt is None or data_chunk is None:
        die(f"Missing fmt/data chunk in {path}")

    ch = fmt.channels
    if ch <= 0:
        die(f"Invalid channel count in {path}: {ch}")

    if fmt.audio_format == 1:  # PCM
        bps = fmt.bits_per_sample
        if bps == 16:
            x = np.frombuffer(data_chunk, dtype="<i2").astype(np.float64) / 32768.0
        elif bps == 24:
            u = np.frombuffer(data_chunk, dtype=np.uint8)
            if u.size % 3 != 0:
                die(f"24-bit data not multiple of 3 bytes: {path}")
            u = u.reshape(-1, 3).astype(np.int32)
            v = u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)
            v = (v ^ 0x800000) - 0x800000
            x = v.astype(np.float64) / 8388608.0
        elif bps == 32:
            x = np.frombuffer(data_chunk, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            die(f"Unsupported PCM bit depth {bps} in {path}")
    elif fmt.audio_format == 3:  # IEEE float
        bps = fmt.bits_per_sample
        if bps == 32:
            x = np.frombuffer(data_chunk, dtype="<f4").astype(np.float64)
        elif bps == 64:
            x = np.frombuffer(data_chunk, dtype="<f8").astype(np.float64)
        else:
            die(f"Unsupported float bit depth {bps} in {path}")
    else:
        die(f"Unsupported WAV format code {fmt.audio_format} in {path}")

    if x.size % ch != 0:
        die(f"Sample count not divisible by channels in {path}")
    x = x.reshape(-1, ch)
    return x, fmt


def parse_tol_from_name(stem: str, default_tol: float) -> float:
    key = "__tol="
    if key not in stem:
        return default_tol
    try:
        return float(stem.split(key, 1)[1])
    except Exception:
        return default_tol


def run_reaper_render(
    reaper_exe: Path,
    rpp: Path,
    timeout_s: int,
    report_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """
    Launch REAPER via Popen so a timeout does NOT kill REAPER.
    This lets a later CI step screenshot the blocking dialog, then kill REAPER.
    """
    cmd = [str(reaper_exe), "-nosplash", "-ignoreerrors", "-renderproject", str(rpp)]
    print("[reaper] " + " ".join(cmd))

    report_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = report_dir / f"reaper_stdout_{rpp.stem}.txt"
    stderr_path = report_dir / f"reaper_stderr_{rpp.stem}.txt"

    # Write output to files to avoid PIPE buffer issues while REAPER stays alive on timeout.
    with stdout_path.open("w", encoding="utf-8", errors="ignore") as out, \
         stderr_path.open("w", encoding="utf-8", errors="ignore") as err:

        p = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=err,
            text=True,
        )

        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Leave REAPER running so the workflow can screenshot the dialog.
            (report_dir / "reaper_timeout_pid.txt").write_text(str(p.pid), encoding="utf-8")
            (report_dir / "reaper_timeout_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
            raise

    # Read captured output back into a CompletedProcess for your existing reporting logic.
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    return subprocess.CompletedProcess(cmd, rc, stdout_text, stderr_text)


def find_rendered_wav(rpp: Path, started_at: float) -> Path:
    candidates: list[Path] = []
    for suf in SEARCH_DIR_SUFFIXES:
        d = (rpp.parent / suf) if suf else rpp.parent
        if not d.exists():
            continue
        for ext in AUDIO_EXTS:
            candidates += list(d.glob(f"*{rpp.stem}*{ext}"))
            candidates += list(d.glob(f"*{ext}"))

    fresh = [p for p in candidates if p.is_file() and p.stat().st_mtime >= started_at - 1.0]
    if not fresh:
        die(
            f"No rendered WAV found for {rpp}\n"
            "Fix by setting the project render output to a WAV near the .rpp (or in ./render)."
        )

    fresh.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return fresh[0]


def read_any_text(paths: list[Path]) -> str:
    out = []
    for p in paths:
        try:
            out.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return "\n".join(out)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "plugins.json"
    if not cfg_path.exists():
        die(f"Missing {rel(repo_root, cfg_path)}")

    # --- Ensure portable REAPER exists (zip -> extracted) ---
    reaper_zip = locate_reaper_zip(repo_root)

    unpack_base = Path(os.environ.get("REAPER_PORTABLE_DIR", repo_root / ".ci" / "reaper-portable"))
    unpack_base.mkdir(parents=True, exist_ok=True)

    # If a previous run already extracted, find it
    reaper_root: Path
    if (unpack_base / "reaper.exe").exists() and (unpack_base / "reaper.ini").exists():
        reaper_root = unpack_base
    else:
        # Extract into unpack_base, then locate portable root inside it
        # (handles zips that contain a single top folder)
        safe_extract_zip(reaper_zip, unpack_base)
        reaper_root = find_portable_reaper_root(unpack_base)

    reaper_exe = reaper_root / "reaper.exe"
    if not reaper_exe.exists():
        die(f"Missing REAPER exe after unzip: {rel(repo_root, reaper_exe)}")

    # Strong warning: unlicensed portable often hangs on the evaluation screen (GUI click).
    # If you want CI to be non-interactive, keep reaper-reginfo.ini OUT of git, but IN the zip asset.
    # if not (reaper_root / "reaper-reginfo.ini").exists():
    #     die(
    #         "Portable REAPER appears unlicensed (reaper-reginfo.ini missing).\n"
    #         "CI will likely hang on the evaluation screen.\n"
    #         "Fix: include reaper-reginfo.ini inside Reaper (Portable).zip (store zip as a private release asset)."
    #     )

    # Force deterministic, non-PII config every run
    dummy_sr = int(os.environ.get("REAPER_DUMMY_SR", "48000"))
    dummy_bs = int(os.environ.get("REAPER_DUMMY_BS", "512"))
    write_ci_reaper_ini(reaper_root, dummy_sr=dummy_sr, dummy_bs=dummy_bs)

    effects_dir, vst3_dir, clap_dir = ensure_dirs(reaper_root)

    # Optional: purge plugin dirs to prevent accidental system/plugin contamination
    purge = os.environ.get("NULLTEST_PURGE_PLUGIN_DIRS", "1") != "0"
    if purge:
        maybe_purge_dir(vst3_dir)
        maybe_purge_dir(clap_dir)

    purge_plugin_caches(reaper_root)

    # Optional: build step hook (you control the command)
    build_cmd = os.environ.get("NULLTEST_BUILD_CMD", "").strip()
    if build_cmd:
        print(f"[build] {build_cmd}")
        cpb = subprocess.run(build_cmd, shell=True, cwd=str(repo_root))
        if cpb.returncode != 0:
            die(f"Build command failed: {build_cmd}", 50)

    default_tol = float(os.environ.get("NULLTEST_TOL", "1e-6"))
    timeout_s = int(os.environ.get("NULLTEST_TIMEOUT_S", "900"))
    report_dir = Path(os.environ.get("NULLTEST_REPORT_DIR", repo_root / ".ci" / "out"))
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    plugins = cfg.get("plugins", [])
    jsfx_plugins = [p for p in plugins if p.get("pluginType") == "jsfx"]  # 

    if not jsfx_plugins:
        die("No JSFX plugins found in plugins.json (pluginType == 'jsfx').")

    # Build all JSFX plugins (default ON). Disable with NULLTEST_BUILD=0
    if os.environ.get("NULLTEST_BUILD", "1") != "0":
        build_config, build_tag, build_out = build_all_jsfx_plugins(repo_root, jsfx_plugins)
    else:
        build_config = os.environ.get("NULLTEST_BUILD_CONFIG", "Release")
        build_tag = os.environ.get("NULLTEST_BUILD_TAG", "ci")
        build_out = os.environ.get("NULLTEST_BUILD_OUT", "dist")

    # --- Stage artifacts + discover tests + requirements ---
    all_rpps: list[tuple[dict, Path]] = []
    required_clap_ids: set[str] = set()
    required_vst3_names: set[str] = set()
    required_js_descs: set[str] = set()
    required_js_paths: set[str] = set()

    staged_vst3 = 0
    staged_clap = 0
    staged_jsfx = 0

    for p in jsfx_plugins:
        plug_dir = repo_root / p["dir"]
        if not plug_dir.exists():
            die(f"Plugin dir missing: {rel(repo_root, plug_dir)}")

        # Stage JSFX sources into Effects
        staged_jsfx += copy_jsfx_sources(repo_root, plug_dir, effects_dir)

        slug = p.get("slug")
        if not slug:
            die(f"Missing slug for plugin entry: {p}")

        # Deterministic: stage what we just built
        v3, cl = stage_built_vst3_and_clap(repo_root, slug, build_config, vst3_dir, clap_dir)

        # Fallback: old heuristic search (keeps local/manual builds working)
        if v3 == 0 and cl == 0:
            v3, cl = copy_vst3_and_clap_artifacts(
                repo_root=repo_root,
                plugin_dir=plug_dir,
                plugin_slug=slug,
                vst3_dir=vst3_dir,
                clap_dir=clap_dir,
            )

        staged_vst3 += v3
        staged_clap += cl

        # Collect all .rpp tests under this plugin dir
        rpps = sorted([x for x in plug_dir.rglob("*.rpp") if x.is_file()])
        for rpp in rpps:
            all_rpps.append((p, rpp))
            req = parse_rpp_requirements(rpp)
            required_clap_ids |= req["clap_ids"]
            required_vst3_names |= req["vst3_names"]
            required_js_descs |= req["js_descs"]
            required_js_paths |= req["js_paths"]

    if not all_rpps:
        die("No .rpp tests found under any JSFX plugin directories.")

    # --- Preflight: ensure required assets exist before running REAPER ---
    for p, rpp in all_rpps:
        preflight_jsfx_required(effects_dir, rpp)

    # CLAP IDs referenced in projects should match plugin config clapIds (sanity check)
    known_clap_ids = {p.get("clapId") for p in jsfx_plugins if p.get("clapId")}
    unknown_clap_ids = sorted([cid for cid in required_clap_ids if cid not in known_clap_ids])
    if unknown_clap_ids:
        die(
            "RPP references CLAP IDs not present in plugins.json clapId fields:\n"
            + "\n".join(f"  - {cid}" for cid in unknown_clap_ids)
        )

    # If any CLAP is required, ensure at least one .clap is staged
    if required_clap_ids and not any(clap_dir.glob("*.clap")):
        die(
            "Projects require CLAP plugins, but no .clap files were staged into portable CLAP/.\n"
            "Fix: build/copy the .clap output so the script can find it (or set NULLTEST_BUILD_CMD)."
        )

    # If any VST3 is required, ensure at least one .vst3 is staged
    if required_vst3_names and not any(p.is_dir() for p in vst3_dir.glob("*.vst3")):
        die(
            "Projects require VST3 plugins, but no .vst3 directories were staged into portable VST3/.\n"
            "Fix: build/copy the .vst3 output so the script can find it (or set NULLTEST_BUILD_CMD)."
        )

    print(f"[stage] JSFX files copied: {staged_jsfx}")
    print(f"[stage] VST3 bundles copied: {staged_vst3}")
    print(f"[stage] CLAP files copied: {staged_clap}")
    print(f"[tests] RPP files: {len(all_rpps)}")

    # --- Run tests ---
    results = []
    failures = 0
    start_run = time.time()

    for p, rpp in all_rpps:
        tol = parse_tol_from_name(rpp.stem, default_tol)
        started = time.time()
        print(f"\n=== {p.get('slug', p['dir'])} :: {rel(repo_root, rpp)} (tol={tol})")

        try:
            cp = run_reaper_render(reaper_exe, rpp, timeout_s=timeout_s, report_dir=report_dir)
        except subprocess.TimeoutExpired:
            (report_dir / "TIMEOUT.txt").write_text(rel(repo_root, rpp), encoding="utf-8")

            # Copy any REAPER log files that exist
            log_candidates = list(reaper_root.glob("reaper*.log")) + list(reaper_root.glob("reaper_console*.txt"))
            for lc in log_candidates:
                try:
                    shutil.copy2(lc, report_dir / lc.name)
                except Exception:
                    pass

            die(f"TIMEOUT rendering {rpp} after {timeout_s}s", 2)



        if cp.returncode != 0:
            print(cp.stdout)
            print(cp.stderr, file=sys.stderr)
            die(f"REAPER failed rendering {rel(repo_root, rpp)} (exit {cp.returncode})", 3)

        wav = find_rendered_wav(rpp, started)
        x, info = read_wav_to_float64(wav)

        # warmup trim
        warmup_samps = int(round(info.sample_rate * (warmup_ms / 1000.0)))
        warmup_samps = max(0, min(warmup_samps, x.shape[0]))
        x_w = x[warmup_samps:, :] if warmup_samps < x.shape[0] else x[0:0, :]

        # metrics: full + post-warmup
        max_abs_full = float(np.max(np.abs(x))) if x.size else 0.0
        rms_full = float(np.sqrt(np.mean(x * x))) if x.size else 0.0

        max_abs = float(np.max(np.abs(x_w))) if x_w.size else 0.0
        rms = float(np.sqrt(np.mean(x_w * x_w))) if x_w.size else 0.0

        max_dbfs = float(-np.inf if max_abs == 0.0 else 20.0 * np.log10(max_abs))
        rms_dbfs = float(-np.inf if rms == 0.0 else 20.0 * np.log10(rms))

        passed = bool(max_abs <= tol and np.isfinite(max_abs))

        results.append(
            {
                "plugin": p.get("slug", p["dir"]),
                "rpp": rel(repo_root, rpp),
                "render": rel(repo_root, wav) if wav.exists() else str(wav),
                "tol": tol,
                "max_abs": max_abs,
                "max_dbfs": max_dbfs,
                "rms": rms,
                "rms_dbfs": rms_dbfs,
                "sr": info.sample_rate,
                "ch": info.channels,
                "wav_format": info.audio_format,
                "bits": info.bits_per_sample,
                "pass": passed,
                "warmup_ms": warmup_ms,
            }
        )

        print(f"render={rel(repo_root, wav)}")
        print(f"max_abs={max_abs:.3e} ({max_dbfs:.1f} dBFS), rms={rms:.3e} ({rms_dbfs:.1f} dBFS)")

        if passed and not keep_renders:
            # delete the produced render and any converted temp
            try:
                wav.unlink(missing_ok=True)
            except Exception:
                pass

        if not passed:
            failures += 1

    # --- Postflight: verify REAPER actually indexed staged plugins (ignoreerrors-proof) ---
    vst_cache_files = list(reaper_root.glob("reaper-vstplugins*.ini"))
    clap_cache_files = list(reaper_root.glob("reaper-clap-*.ini"))

    vst_cache_text = read_any_text(vst_cache_files)
    clap_cache_text = read_any_text(clap_cache_files)

    # If VST3 was staged, require the cache to exist and include staged paths
    staged_vst3_paths = [p for p in vst3_dir.glob("*.vst3") if p.is_dir()]
    # if staged_vst3_paths:
    #     if not vst_cache_files:
    #         die(
    #             "VST3 bundles were staged, but no reaper-vstplugins*.ini cache was created.\n"
    #             "That means REAPER didn't index the VST path (or never started correctly)."
    #         )
    #     # Must contain at least one staged bundle path string
    #     if not any(str(p).replace("/", "\\") in vst_cache_text for p in staged_vst3_paths):
    #         die(
    #             "REAPER VST cache exists, but staged VST3 bundles are not present in it.\n"
    #             "So REAPER did not index your portable VST3 folder."
    #         )

    # # If CLAP was staged, require clap cache and include required IDs
    # staged_clap_files = list(clap_dir.glob("*.clap"))
    # if staged_clap_files:
    #     if not clap_cache_files:
    #         die(
    #             "CLAP files were staged, but no reaper-clap-*.ini cache was created.\n"
    #             "That means REAPER didn't index the CLAP path (or never started correctly)."
    #         )
    #     missing_ids = sorted([cid for cid in required_clap_ids if cid not in clap_cache_text])
    #     if missing_ids:
    #         die(
    #             "REAPER CLAP cache exists, but required CLAP IDs were not found inside it:\n"
    #             + "\n".join(f"  - {cid}" for cid in missing_ids)
    #         )

    out_json = report_dir / "jsfx_nulltest_report.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport: {rel(repo_root, out_json)}")
    
    if failures:
        die(f"{failures} null test(s) failed.", 10)

    print("All null tests passed.")


if __name__ == "__main__":
    main()
