#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def expect_ok(cmd: list[str], label: str) -> None:
    cp = run(cmd)
    if cp.returncode != 0:
        print(f"[FAIL] {label}")
        print(cp.stdout)
        print(cp.stderr)
        raise SystemExit(1)
    print(f"[ OK ] {label}")


def expect_fail(cmd: list[str], label: str, needle: str) -> None:
    cp = run(cmd)
    if cp.returncode == 0:
        print(f"[FAIL] {label}: unexpectedly succeeded")
        raise SystemExit(1)
    merged = (cp.stdout or "") + "\n" + (cp.stderr or "")
    if needle not in merged:
        print(f"[FAIL] {label}: expected {needle!r}")
        print(merged)
        raise SystemExit(1)
    print(f"[ OK ] {label}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    aot = repo_root / "dsp_jsfx_aot.py"
    tests = repo_root / "tests" / "dsp-jsfx-sample-pool"
    out = repo_root / "build" / "samplepooltests"
    out.mkdir(parents=True, exist_ok=True)

    def base(jsfx: str) -> list[str]:
        stem = Path(jsfx).stem
        return [
            sys.executable,
            str(aot),
            str(tests / jsfx),
            "--out-ll", str(out / f"{stem}.ll"),
            "--out-h", str(out / f"{stem}.h"),
            "--meta", str(out / f"{stem}.json"),
        ]

    expect_ok(base("sample_pool_probe.jsfx"), "sample pool probe compiles")
    expect_fail(base("invalid_export_sample.jsfx"), "sample_export_mem rejected in @sample", "sample_export_mem() is only valid in @block")
