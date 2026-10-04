"""Fingerprint actual guest, host support, and WDL inputs before reusing a pass."""
import hashlib
from pathlib import Path


def numeric_fingerprint(repo: Path, fixture: Path, wdl_build: Path, seconds: str):
    paths = [fixture/'JSFXDSP.h', fixture/'JSFXDSP.o', fixture/'expanded.jsfx',
             repo/'tests/jsfx_showcase/joep_dsp_reference.cpp',
             repo/'tests/jsfx_showcase/joep_legacy_qualification.py',
             Path(__file__), repo/'tests/jsfx_showcase/juce_contract_stub.h']
    paths += [p for p in sorted((repo/'src').glob('*')) if p.is_file()]
    paths += [wdl_build/'numeric_runtime.inc', wdl_build/'libshowcase_eel.a']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(seconds.encode())
    return digest.hexdigest()
