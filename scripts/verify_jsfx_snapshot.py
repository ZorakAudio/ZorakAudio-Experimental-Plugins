#!/usr/bin/env python3
"""Verify a vendored JSFX package without changing it. Network is opt-in only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen

# Preserve string literals, operators and token boundaries. Only whitespace and
# comments are ignored. This is a source integrity check, not a semantic proof.
_TOKENS = re.compile(r'''//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_$][A-Za-z0-9_$]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|===|!==|==|!=|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|\^=|&&|\|\||[^\s]''', re.S)


def executable_tokens(text: str) -> tuple[str, ...]:
    # Header metadata can include unquoted apostrophes or wildcard /*. It is
    # excluded here and protected separately by the bundled-file hash.
    first = re.search(r"(?m)^\s*@(?:init|slider|block|sample|serialize|gfx)\b", text)
    if first is None:
        raise ValueError("No JSFX section found")
    return tuple(t for t in _TOKENS.findall(text[first.start():]) if not t.startswith(("//", "/*")))


def verify(package: Path, *, upstream: bool = False) -> list[str]:
    package = package.resolve()
    info = json.loads((package / 'upstream.json').read_text(encoding='utf-8'))
    if info.get('format') != 1 or not isinstance(info.get('files'), list):
        raise ValueError('Unsupported or malformed upstream.json')
    results = []
    for item in info['files']:
        path = (package / item['path']).resolve()
        if not path.is_relative_to(package):
            raise ValueError('Snapshot path escapes package')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f"Local snapshot changed: {item['path']} (expected {item['sha256']}, got {digest})")
        if upstream:
            # Reject moving branches or unrelated URL hosts in a tampered manifest.
            url = item['url']
            expected = f"https://raw.githubusercontent.com/JoepVanlier/JSFX/{info['commit']}/"
            if not url.startswith(expected) or not re.fullmatch(r'[0-9a-f]{40}', info['commit']):
                raise ValueError('Upstream verification requires a pinned JoepVanlier raw URL')
            with urlopen(url, timeout=30) as response:
                downloaded = response.read(8 * 1024 * 1024 + 1)
            if len(downloaded) > 8 * 1024 * 1024:
                raise ValueError('Upstream source exceeds verification limit')
            if executable_tokens(downloaded.decode('utf-8-sig')) != executable_tokens(raw.decode('utf-8-sig')):
                raise ValueError(f"Executable tokens differ from pinned upstream: {item['path']}")
        results.append(f"PASS {item['path']}: local SHA-256" + (' + pinned upstream code tokens' if upstream else ''))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('package', type=Path)
    ap.add_argument('--upstream', action='store_true', help='Fetch pinned originals and compare code tokens; never update files')
    args = ap.parse_args()
    try:
        for line in verify(args.package, upstream=args.upstream):
            print(line)
    except (OSError, ValueError, KeyError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
