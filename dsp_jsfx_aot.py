#!/usr/bin/env python3
"""
Drop-in dsp_jsfx_aot.py replacement with the JSFX/EEL2 parser fix applied.

This wrapper loads the repository's committed/original dsp_jsfx_aot.py, patches
Parser.parse_expr in memory, and runs the patched compiler as __main__.

Fixed expression shape:

    wrapped
        || condition_a
        || condition_b

Newlines remain statement separators in general.  They are only swallowed when
the next token is an explicit infix/ternary continuation operator.  Unary-prefix
operators '+', '-', and '!' are deliberately excluded so separate statements are
not accidentally merged.

Normal use:
    replace repo-root dsp_jsfx_aot.py with this file and build as usual.

Optional materialization:
    set ZA_DSP_JSFX_AOT_DUMP_PATCHED=some/path.py to write the fully patched
    compiler source before executing it.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

__DSP_JSFX_AOT_SHIM__ = True
_PATCH_MARKER = "# --- JSFX AOT parser: newline-leading infix continuation support ---"

_CONTINUATION_METHODS = r'''

    # --- JSFX AOT parser: newline-leading infix continuation support ---
    def _is_line_continuation_op(self, tok: Tok, min_prec: int) -> bool:
        """True when a newline followed by tok must continue the current expr.

        JSFX/EEL2 permits expressions such as:
            wrapped
              || something

        Newlines still separate statements in general; we only join across a
        newline when the next token is an infix/ternary continuation operator
        that cannot safely start a standalone expression.  '+', '-', and '!'
        are intentionally excluded here because they are valid unary prefixes.
        """
        if tok.kind != "op":
            return False
        if tok.text == "?":
            return _TERNARY_PREC >= min_prec
        if tok.text == ":":
            return False
        if tok.text in ("+", "-", "!"):
            return False
        prec = _PRECEDENCE.get(tok.text)
        return prec is not None and prec >= min_prec

    def _skip_expr_continuation_eol(self, min_prec: int) -> None:
        # Skip blank lines while looking for an explicit continuation operator,
        # but do not swallow newlines before ordinary statement starts.
        while self.cur.kind == "eol" and (
            self.nxt.kind == "eol" or self._is_line_continuation_op(self.nxt, min_prec)
        ):
            self._adv()
'''


def _die(msg: str, code: int = 2) -> None:
    print(f"dsp_jsfx_aot.py shim error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _repo_root() -> pathlib.Path:
    env_root = os.environ.get("ZA_DSP_JSFX_AOT_REPO_ROOT", "").strip()
    if env_root:
        return pathlib.Path(env_root).resolve()
    return pathlib.Path(__file__).resolve().parent


def _run_git_show(repo: pathlib.Path, revspec: str) -> str | None:
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo), "show", revspec],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    text = cp.stdout
    if not text or "__DSP_JSFX_AOT_SHIM__" in text:
        return None
    return text


def _load_original_source(repo: pathlib.Path) -> str:
    env_path = os.environ.get("ZA_DSP_JSFX_AOT_ORIGINAL", "").strip()
    candidates: list[pathlib.Path] = []
    if env_path:
        candidates.append(pathlib.Path(env_path))
    candidates.extend(
        [
            repo / "dsp_jsfx_aot.py.orig",
            repo / "dsp_jsfx_aot.py.original",
            repo / "dsp_jsfx_aot.original.py",
            repo / "dsp_jsfx_aot.py.unpatched",
            repo / "dsp_jsfx_aot.unpatched.py",
            repo / "dsp_jsfx_aot.upstream.py",
            repo / "dsp_jsfx_aot_upstream.py",
            repo / "_dsp_jsfx_aot_unpatched.py",
        ]
    )

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                text = p.read_text(encoding="utf-8", errors="replace")
                if text and "__DSP_JSFX_AOT_SHIM__" not in text:
                    return text
        except Exception:
            pass

    # When this file replaces the working-tree compiler, Git still has the
    # original blob available in the index/HEAD.  Prefer the index first because
    # it preserves local staged content in the common "overwrite but don't stage"
    # workflow.
    for rev in (
        ":dsp_jsfx_aot.py",
        "HEAD:dsp_jsfx_aot.py",
        "origin/main:dsp_jsfx_aot.py",
        "upstream/main:dsp_jsfx_aot.py",
        "main:dsp_jsfx_aot.py",
        "HEAD~1:dsp_jsfx_aot.py",
    ):
        text = _run_git_show(repo, rev)
        if text:
            return text

    _die(
        "could not locate the original compiler source. Run this from a Git checkout, "
        "or set ZA_DSP_JSFX_AOT_ORIGINAL to the original dsp_jsfx_aot.py path."
    )
    raise AssertionError("unreachable")


def _patch_compiler_source(src: str) -> str:
    if "def parse_expr(self, min_prec: int)" not in src:
        _die("source does not look like the expected dsp_jsfx_aot.py parser")

    out = src

    if _PATCH_MARKER not in out and "def _skip_expr_continuation_eol" not in out:
        m = re.search(r"(?m)^    def parse_expr\(self, min_prec: int\)\s*(?:->\s*Node)?\s*:\n", out)
        if not m:
            _die("could not find Parser.parse_expr insertion point")
        out = out[: m.start()] + _CONTINUATION_METHODS + "\n" + out[m.start():]

    if "self._skip_expr_continuation_eol(min_prec)" not in out:
        pattern = re.compile(
            r"(?m)^(?P<indent> {8})while True:\n"
            r"(?P<comment> {12}# assignment / binary ops\n)"
        )
        m = pattern.search(out)
        if not m:
            _die("could not find parse_expr operator loop insertion point")
        out = (
            out[: m.end("indent")]
            + "while True:\n"
            + "            self._skip_expr_continuation_eol(min_prec)\n"
            + m.group("comment")
            + out[m.end():]
        )

    return out


def _maybe_dump_patched_source(repo: pathlib.Path, src: str) -> None:
    target = os.environ.get("ZA_DSP_JSFX_AOT_DUMP_PATCHED", "").strip()
    if not target:
        return
    p = pathlib.Path(target)
    if not p.is_absolute():
        p = repo / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding="utf-8", newline="")


def _exec_as_main(src: str, filename: pathlib.Path) -> None:
    g = globals()
    g.clear()
    g.update(
        {
            "__name__": "__main__",
            "__file__": str(filename),
            "__package__": None,
            "__cached__": None,
        }
    )
    code = compile(src, str(filename), "exec")
    exec(code, g, g)


def main() -> None:
    repo = _repo_root()
    original = _load_original_source(repo)
    patched = _patch_compiler_source(original)
    _maybe_dump_patched_source(repo, patched)
    _exec_as_main(patched, repo / "dsp_jsfx_aot.py")


if __name__ == "__main__":
    main()
