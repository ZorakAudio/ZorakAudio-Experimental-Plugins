"""Shared, deterministic JSFX import expansion for the DSP and GFX frontends.

Imports are libraries, not C #includes: imported @init sections execute in
postorder; a main effect's other sections override imported fallback sections.
Search is confined to explicitly supplied package roots, never the entire repo.
Each source unit is preprocessed with Cockos/WDL EEL2 before metadata/import
parsing when it contains ``<? ... ?>`` blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import hashlib
import re
import shlex

_SECTION = re.compile(r"^\s*@([A-Za-z_][A-Za-z0-9_]*)\b", re.I)
_IMPORT = re.compile(r"^\s*import\b", re.I)
_TOKEN = re.compile(r'''import\s+(?:"([^"]+)"|'([^']+)'|([^\s;]+))''', re.I)
_CONFIG = re.compile(r"^\s*config:\s*(.*?)\s*$", re.I)
_CONFIG_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _config_number(path: Path, line: int, token: str) -> float:
    t = token.strip()
    try:
        low = t.casefold()
        if low.startswith("$x"):
            return float(int(t[2:], 16))
        if low.startswith("-$x"):
            return -float(int(t[3:], 16))
        if low.startswith("$~"):
            bits = int(t[2:], 10)
            if bits < 0 or bits > 53:
                raise ValueError
            return float((1 << bits) - 1)
        if low.startswith("-$~"):
            bits = int(t[3:], 10)
            if bits < 0 or bits > 53:
                raise ValueError
            return -float((1 << bits) - 1)
        if low.startswith(("0x", "+0x", "-0x")):
            return float(int(t, 0))
        return float(t)
    except (ValueError, OverflowError) as exc:
        raise SourceError(f"{path}:{line}: invalid config default {token!r}") from exc


def _config_defaults(path: Path, text: str) -> dict[str, float]:
    """Parse REAPER 7+ compile-time config defaults from the root preamble."""
    out: dict[str, float] = {}
    seen: set[str] = set()
    for line, raw in enumerate(text.splitlines(), 1):
        if _SECTION.match(raw):
            break
        m = _CONFIG.match(raw)
        if not m:
            continue
        try:
            # name, quoted description, default, then allowed numeric choices.
            parts = shlex.split(m.group(1), posix=True)
        except ValueError as exc:
            raise SourceError(f"{path}:{line}: invalid config directive: {exc}") from exc
        if len(parts) < 3 or not _CONFIG_NAME.match(parts[0]):
            raise SourceError(f"{path}:{line}: invalid config directive: {raw.strip()}")
        name = parts[0]
        folded = name.casefold()
        if folded in seen:
            raise SourceError(f"{path}:{line}: duplicate config variable {name!r}")
        seen.add(folded)
        out[name] = _config_number(path, line, parts[2])
    return out



class SourceError(ValueError):
    """A source/import error with an actionable filename/chain diagnostic."""


@dataclass
class _Unit:
    path: Path
    text: str
    preamble: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    sections: dict[str, list[str]] = field(default_factory=dict)
    imports: list[tuple[int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedSource:
    text: str
    # Postorder, including the main file last. A diamond dependency appears once.
    dependencies: tuple[Path, ...]
    section_sources: dict[str, tuple[Path, ...]]

    def manifest(self, root: Path) -> dict:
        def name(path: Path) -> str:
            try:
                return path.relative_to(root.resolve()).as_posix()
            except ValueError:
                return str(path)
        return {
            "format": 1,
            "expanded_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "files": [{"path": name(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in self.dependencies],
            "sections": {s: [name(p) for p in paths] for s, paths in self.section_sources.items()},
        }


def _code_lines(text: str):
    """Yield line+same-width code mask, ignoring EEL comments/strings in code.

    Everything before the first JSFX section is a REAPER header/directive
    preamble, not EEL program text.  In particular, metadata such as::

        provides:
          dependencies/*

    must never let the ``/*`` wildcard open an EEL block comment that hides
    later ``import`` directives.  We therefore keep pre-section text visible
    verbatim (so imports can be parsed) while still suppressing actual
    comment-only lines/blocks.  Once a real @section starts, normal EEL
    comment/string masking resumes.
    """
    block = False
    quote = ""
    code_started = False
    preamble_block = False
    metadata_continuation = False

    def blank(raw: str) -> str:
        # Keep line width/newlines stable for diagnostics and token offsets.
        return "".join("\n" if c == "\n" else "\r" if c == "\r" else " " for c in raw)

    for raw in text.splitlines(keepends=True):
        if not code_started:
            # REAPER's multiline metadata bodies are conventionally indented.
            # Do not interpret comment/string-looking text inside them as EEL.
            if metadata_continuation:
                if not raw.strip() or raw[:1].isspace():
                    yield raw, raw
                    continue
                metadata_continuation = False

            stripped = raw.lstrip()

            # Suppress genuine preamble block comments, but only when the
            # comment itself begins the logical line.  This distinction is what
            # keeps paths such as ``foo/*`` in provides: metadata harmless.
            if preamble_block:
                yield raw, blank(raw)
                if "*/" in raw:
                    preamble_block = False
                continue
            if stripped.startswith("/*"):
                yield raw, blank(raw)
                if "*/" not in stripped[2:]:
                    preamble_block = True
                continue
            if stripped.startswith("//"):
                yield raw, blank(raw)
                continue

            header = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*):", raw)
            if header:
                key = header.group(1).casefold()
                # These REAPER metadata fields own subsequent indented lines.
                if key in {"provides", "about"} and not raw[header.end():].strip():
                    metadata_continuation = True
                yield raw, raw
                continue

            if not _SECTION.match(raw):
                # Unknown/free-form preamble text is metadata too.  Keeping it
                # visible avoids apostrophes, URLs, and wildcard paths changing
                # parser state before a later import directive.
                yield raw, raw
                continue

            code_started = True

        mask = list(raw)
        i = 0
        while i < len(raw):
            c = raw[i]
            if block:
                mask[i] = " "
                if raw.startswith("*/", i):
                    mask[i:i+2] = [" ", " "]
                    i += 2
                    block = False
                else:
                    i += 1
            elif quote:
                mask[i] = " "
                if c == "\\" and i + 1 < len(raw):
                    mask[i+1] = " "
                    i += 2
                else:
                    if c == quote:
                        quote = ""
                    i += 1
            elif raw.startswith("//", i):
                mask[i:] = [" "] * (len(raw) - i)
                break
            elif raw.startswith("/*", i):
                mask[i:i+2] = [" ", " "]
                block = True
                i += 2
            elif c in "\"'":
                quote = c
                mask[i] = " "
                i += 1
            else:
                i += 1
        yield raw, "".join(mask)


def _parse(path: Path, text: str) -> _Unit:
    unit = _Unit(path, text)
    current = None
    for line, (raw, mask) in enumerate(_code_lines(text), 1):
        imp = _IMPORT.match(mask)
        sec = _SECTION.match(mask)
        if imp:
            start = mask.lower().index("import")
            token = _TOKEN.match(raw[start:])
            if token is None or mask[start + token.end():].strip(" \t\r\n;"):
                raise SourceError(f"{path}:{line}: invalid import directive: {raw.strip()}")
            unit.imports.append((line, next(g for g in token.groups() if g is not None)))
            # No directive is allowed to reach either compiler.
            (unit.preamble if current is None else unit.sections[current]).append("\n")
        elif sec:
            current = sec.group(1).lower()
            if current in unit.sections:
                raise SourceError(f"{path}:{line}: duplicate @{current} section")
            unit.headers[current] = raw if raw.endswith("\n") else raw + "\n"
            unit.sections[current] = []
        else:
            (unit.preamble if current is None else unit.sections[current]).append(raw)
    return unit


class SourceResolver:
    def __init__(self, package_root: Path, *, search_roots: tuple[Path, ...] = ()):
        self.roots = tuple(dict.fromkeys(p.resolve() for p in (package_root, *search_roots)))
        self._index: tuple[Path, ...] | None = None

    def _allowed(self, path: Path) -> bool:
        return any(path.is_relative_to(root) for root in self.roots)

    def _files(self) -> tuple[Path, ...]:
        if self._index is None:
            self._index = tuple(sorted({p.resolve() for root in self.roots for p in root.rglob("*")
                                        if p.is_file() and self._allowed(p.resolve())}, key=str))
        return self._index

    def find(self, token: str, importer: Path, line: int = 0) -> Path:
        token = token.replace("\\", "/")
        if not token or PurePosixPath(token).is_absolute() or re.match(r"^[A-Za-z]:", token):
            raise SourceError(f"{importer}:{line}: import must be a package-relative path: {token!r}")
        attempts = []
        for base in (importer.parent, *self.roots):
            candidate = (base / token).resolve()
            attempts.append(candidate)
            if self._allowed(candidate) and candidate.is_file():
                return candidate
        # A bare import may live in a provided dependency directory. Never take
        # the first arbitrary filesystem hit; ambiguous package names are errors.
        if ".." not in PurePosixPath(token).parts:
            suffix = "/" + token
            for insensitive in (False, True):
                key = suffix.casefold() if insensitive else suffix
                hits = [p for p in self._files()
                        if (p.as_posix().casefold() if insensitive else p.as_posix()).endswith(key)]
                if len(hits) == 1:
                    return hits[0]
                if hits:
                    raise SourceError(f"{importer}:{line}: ambiguous import {token!r}:\n  " +
                                      "\n  ".join(str(p) for p in hits))
        raise SourceError(f"{importer}:{line}: missing import {token!r}; searched direct paths:\n  " +
                          "\n  ".join(str(p) for p in attempts) +
                          "\nand dependency subdirectories of: " + ", ".join(map(str, self.roots)))

    def expand(self, entry: Path, *, text: str | None = None) -> ResolvedSource:
        entry = entry.resolve()
        units: dict[Path, _Unit] = {}
        ordered: list[_Unit] = []
        visiting: list[Path] = []

        entry_raw = text if text is not None else entry.read_text(encoding="utf-8-sig")
        config_defaults = _config_defaults(entry, entry_raw)

        def visit(path: Path, supplied: str | None = None) -> _Unit:
            if path in visiting:
                raise SourceError("Cyclic JSFX import: " + " -> ".join(map(str, [*visiting, path])))
            if path in units:
                return units[path]
            if len(visiting) >= 32:
                raise SourceError("JSFX import nesting exceeds 32: " + str(path))
            if not self._allowed(path):
                raise SourceError(f"Source is outside the package search roots: {path}")
            raw = supplied if supplied is not None else (entry_raw if path == entry else path.read_text(encoding="utf-8-sig"))
            if "<?" in raw:
                try:
                    if __package__:
                        from .jsfx_preprocessor import preprocess_text, PreprocessorError
                    else:
                        from jsfx_preprocessor import preprocess_text, PreprocessorError
                    raw = preprocess_text(source_path=path, text=raw, include_roots=self.roots,
                                          definitions=config_defaults)
                except PreprocessorError as exc:
                    raise SourceError(str(exc)) from exc
            unit = _parse(path, raw)
            visiting.append(path)
            for line, token in unit.imports:
                visit(self.find(token, path, line))
            visiting.pop()
            # Imported libraries must be JSFX sections, not arbitrary text macros.
            if path != entry and not unit.sections and any(m.strip() for _, m in _code_lines("".join(unit.preamble))):
                raise SourceError(f"{path}: sectionless import; put library functions in @init")
            units[path] = unit
            ordered.append(unit)
            return unit

        main = visit(entry, entry_raw)
        if not main.imports:
            # Preserve no-import/non-preprocessed plugins byte-for-text; this is not a reformatter.
            return ResolvedSource(main.text, (entry,), {s: (entry,) for s in main.sections})
        output = list(main.preamble)
        if output and not output[-1].endswith("\n"):
            output.append("\n")
        owners: dict[str, tuple[Path, ...]] = {}
        init_units = [u for u in ordered if "init" in u.sections]
        if init_units:
            output.append("@init\n")
            owners["init"] = tuple(u.path for u in init_units)
            for u in init_units:
                output.extend(u.sections["init"])
                output.append("\n")
        # Root takes priority even if its section body is empty. For an absent
        # root section, the first postorder imported definition is the fallback.
        names = dict.fromkeys(s for u in [main, *ordered[:-1]] for s in u.sections if s != "init")
        for section in names:
            owner = next(u for u in [main, *ordered[:-1]] if section in u.sections)
            owners[section] = (owner.path,)
            output.append(owner.headers[section])
            output.extend(owner.sections[section])
            output.append("\n")
        return ResolvedSource("".join(output), tuple(u.path for u in ordered), owners)


def resolve_source(path: Path, *, text: str | None = None,
                   package_root: Path | None = None) -> ResolvedSource:
    path = Path(path).resolve()
    return SourceResolver(package_root or path.parent).expand(path, text=text)


def apply_host_options(source: ResolvedSource, options: dict | None = None) -> ResolvedSource:
    """Apply wrapper-only compatibility settings without editing vendored JSFX.

    They live in plugin.json / jsfxCompatibility. Unknown options fail closed so
    a misspelled setting cannot silently revert this plugin to different DSP.
    """
    from dataclasses import replace
    if options is None:
        return source
    if not isinstance(options, dict):
        raise SourceError("jsfxCompatibility must be an object")
    unknown = options.keys() - {"eel2Stores", "gfxMemory"}
    if unknown:
        raise SourceError("Unknown jsfxCompatibility option(s): " + ", ".join(sorted(unknown)))
    settings = []
    if "eel2Stores" in options:
        if not isinstance(options["eel2Stores"], bool):
            raise SourceError("jsfxCompatibility.eel2Stores must be boolean")
        settings.append("options:za_eel2_stores=" + ("1" if options["eel2Stores"] else "0"))
    if "gfxMemory" in options:
        policy = options["gfxMemory"]
        if not isinstance(policy, str) or policy.lower() not in ("auto", "explicit"):
            raise SourceError("jsfxCompatibility.gfxMemory must be auto or explicit")
        settings.append("// @za:gfx_sync_policy " + policy.upper())
    if not settings:
        return source
    offset = 0
    for raw, mask in _code_lines(source.text):
        if _SECTION.match(mask):
            break
        offset += len(raw)
    header = "\n// Generated from plugin.json; not an upstream source edit.\n" + "\n".join(settings) + "\n"
    return replace(source, text=source.text[:offset] + header + source.text[offset:])
