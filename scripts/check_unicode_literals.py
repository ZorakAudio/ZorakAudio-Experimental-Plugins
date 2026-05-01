#!/usr/bin/env python3
"""
Checks repository-authored C++ source for raw non-ASCII string literals that
can trip JUCE's debug assertion in juce::String(const char*).

Allowed C++ string-literal paths:
  - za::text::utf8("…")
  - juce::String::fromUTF8("…")
  - juce::CharPointer_UTF8("…")

Vendor WDL sources are skipped because they do not construct JUCE Strings from
those literals in this project.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"}
DEFAULT_SKIP_PARTS = {".git", "build", "out", "dist", "__pycache__"}
VENDOR_SKIP_PARTS = {"WDL"}
SAFE_MARKERS = (
    "za::text::utf8",
    "juce::String::fromUTF8",
    "juce::CharPointer_UTF8",
)


def iter_cpp_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in CPP_EXTENSIONS:
            continue
        parts = set(path.parts)
        if parts & DEFAULT_SKIP_PARTS:
            continue
        if parts & VENDOR_SKIP_PARTS:
            continue
        yield path


def strip_comments(line: str, in_block_comment: bool) -> tuple[str, bool]:
    out = []
    i = 0
    while i < len(line):
        if in_block_comment:
            end = line.find("*/", i)
            if end < 0:
                return "".join(out), True
            i = end + 2
            in_block_comment = False
            continue

        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            in_block_comment = True
            i += 2
            continue

        out.append(line[i])
        i += 1

    return "".join(out), in_block_comment


def has_non_ascii_string_literal(code: str) -> bool:
    """Conservative line-level scanner for ordinary C/C++ string literals."""
    i = 0
    while i < len(code):
        # Skip character literals.
        if code[i] == "'":
            i += 1
            while i < len(code):
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == "'":
                    i += 1
                    break
                i += 1
            continue

        # Handle common prefixes: "...", L"...", u"...", U"...", u8"...".
        start = i
        if code.startswith('u8"', i):
            i += 2
        elif i + 1 < len(code) and code[i] in "LuU" and code[i + 1] == '"':
            i += 1

        if i < len(code) and code[i] == '"':
            i += 1
            literal = []
            while i < len(code):
                ch = code[i]
                if ch == "\\":
                    if i + 1 < len(code):
                        literal.append(code[i:i + 2])
                        i += 2
                        continue
                    i += 1
                    break
                if ch == '"':
                    i += 1
                    break
                literal.append(ch)
                i += 1
            if any(ord(ch) > 127 for ch in "".join(literal)):
                return True
            continue

        i = start + 1

    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".", help="repository root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    failures: list[tuple[Path, int, str]] = []

    for path in iter_cpp_files(root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        in_block = False
        for line_no, line in enumerate(lines, 1):
            code, in_block = strip_comments(line, in_block)
            if not any(ord(ch) > 127 for ch in code):
                continue
            if not has_non_ascii_string_literal(code):
                continue
            if any(marker in code for marker in SAFE_MARKERS):
                continue
            failures.append((path.relative_to(root), line_no, line.rstrip()))

    if failures:
        print("Unsafe non-ASCII C++ string literals found:")
        for path, line_no, line in failures:
            print(f"{path}:{line_no}: {line}")
        return 1

    print("OK: no unsafe non-ASCII C++ string literals found outside skipped vendor paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
