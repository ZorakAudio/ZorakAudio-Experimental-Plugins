"""Build-time bridge to Cockos/WDL's EEL2 JSFX preprocessor.

The helper is compiled lazily only when a source file actually contains a
``<? ... ?>`` block.  This keeps ordinary JSFX builds on the existing fast path
while making preprocessing use the same WDL/EEL2 implementation as REAPER.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Iterable


class PreprocessorError(RuntimeError):
    pass


_cached_tool: Path | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tool_from_env() -> Path | None:
    raw = os.environ.get("JSFX_EEL_PP", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise PreprocessorError(f"JSFX_EEL_PP does not point to a file: {path}")
    return path


def ensure_preprocessor_tool() -> Path:
    global _cached_tool
    if _cached_tool is not None and _cached_tool.is_file():
        return _cached_tool

    override = _tool_from_env()
    if override is not None:
        _cached_tool = override
        return override

    root = _repo_root()
    source_dir = root / "tools" / "jsfx_eel_pp"
    build_dir = root / "build" / "tools" / "jsfx_eel_pp"
    exe = build_dir / "bin" / ("jsfx_eel_pp.exe" if os.name == "nt" else "jsfx_eel_pp")
    if not (source_dir / "CMakeLists.txt").is_file():
        raise PreprocessorError(f"JSFX EEL2 preprocessor source is missing: {source_dir}")

    # Avoid a CMake round-trip on every separate plug-in build once the helper
    # is current.  Any vendored EEL2/helper source update invalidates it.
    watched = [p for p in source_dir.rglob("*") if p.is_file()]
    wdl = root / "src" / "WDL"
    eel = wdl / "eel2"
    watched += [p for p in eel.rglob("*") if p.is_file()]
    watched += [wdl / "wdlstring.h", wdl / "ptrlist.h", wdl / "win32_utf8.h", wdl / "win32_utf8.c"]
    watched = [p for p in watched if p.is_file()]
    if exe.is_file() and watched and exe.stat().st_mtime >= max(p.stat().st_mtime for p in watched):
        _cached_tool = exe
        return exe

    configure = [
        "cmake", "-S", str(source_dir), "-B", str(build_dir),
        f"-DZA_ROOT={root}", "-DCMAKE_BUILD_TYPE=Release",
    ]
    build = ["cmake", "--build", str(build_dir), "--config", "Release", "--target", "jsfx_eel_pp"]
    print("[jsfx] preparing WDL/EEL2 preprocessor helper")
    try:
        for cmd in (configure, build):
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            if proc.returncode != 0:
                output = proc.stdout.decode("utf-8", errors="replace").strip()
                if len(output) > 6000:
                    output = "...\n" + output[-6000:]
                raise PreprocessorError(
                    f"Failed to build the WDL/EEL2 JSFX preprocessor helper (exit {proc.returncode})"
                    + (f"\n{output}" if output else "")
                )
    except FileNotFoundError as exc:
        raise PreprocessorError("CMake is required to build the WDL/EEL2 JSFX preprocessor helper") from exc

    if not exe.is_file():
        raise PreprocessorError(f"JSFX EEL2 preprocessor build completed but executable is missing: {exe}")
    _cached_tool = exe
    return exe


def preprocess_text(*, source_path: Path, text: str, include_roots: Iterable[Path] = (),
                    definitions: Mapping[str, float] | None = None) -> str:
    """Preprocess one JSFX source unit with WDL/EEL2.

    State persists across all ``<? ?>`` blocks in this *one* source unit, just
    as WDL's EEL2_PreProcessor specifies.  Imported JSFX files are processed by
    the resolver as separate units.  Compile-time ``config:`` values supplied
    by the root JSFX are injected into every unit's preprocessor context so
    dependency files can consume the plug-in configuration.
    """
    if "<?" not in text:
        return text

    tool = ensure_preprocessor_tool()
    source_path = Path(source_path).resolve()

    # eel_pproc.h resolves include() paths from last to first.  Put broad roots
    # first, then the current unit directory last so a sibling include wins.
    dirs: list[Path] = []
    for raw in (*tuple(include_roots), source_path.parent):
        p = Path(raw).resolve()
        if p not in dirs:
            dirs.append(p)

    cmd = [str(tool)]
    for p in dirs:
        cmd += ["--include", str(p)]
    for name, value in sorted((definitions or {}).items(), key=lambda kv: kv[0].casefold()):
        cmd += ["--define", f"{name}={float(value):.17g}"]

    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        if detail.startswith("Error: "):
            detail = detail[7:]
        raise PreprocessorError(f"{source_path}: {detail or 'EEL2 preprocessing failed'}")
    return proc.stdout.decode("utf-8", errors="strict")
