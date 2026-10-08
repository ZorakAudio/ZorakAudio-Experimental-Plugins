"""Idempotently prepare the shared AOT/JIT host wrappers on any build platform.

Only apply checked-in patches. Never reset a submodule or discard local edits.
Preflight every dependency before changing any; serialize concurrent builds.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PATCHES = Path(__file__).with_name("wrapper-patches")
DEPENDENCIES = (("juce", "libs/JUCE"), ("clap", "libs/clap-juce-extensions"))


@contextmanager
def build_lock(root: Path):
    lock = root / "build" / ".wrapper-patches.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 120
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for another build's wrapper patch check")
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def git_apply(root: Path, patch: Path, *flags: str):
    return subprocess.run(
        ["git", "-C", str(root), "apply", "--whitespace=nowarn",
         "--ignore-space-change", *flags, str(patch)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def ensure_patches(root: Path, *, check_only: bool = False, only: str = "all"):
    pending = []
    with build_lock(root):
        for name, relative in DEPENDENCIES:
            if only != "all" and only != name:
                continue
            dependency = root / relative
            patch = PATCHES / (name + ".patch")
            if not (dependency / "CMakeLists.txt").is_file():
                raise RuntimeError(
                    f"Dependency missing at {relative}. Run git submodule update --init --recursive."
                )
            if not patch.is_file():
                raise RuntimeError(f"Checked-in wrapper patch missing: {patch}")
            if git_apply(dependency, patch, "--reverse", "--check").returncode == 0:
                print(f"[wrappers] {name}: already applied")
                continue
            checked = git_apply(dependency, patch, "--check")
            if checked.returncode:
                raise RuntimeError(
                    f"Wrapper patch conflicts with {relative}; no patches applied. "
                    f"Preserve local edits and merge {patch} manually.\n{checked.stderr.strip()}"
                )
            pending.append((name, dependency, patch))
        if check_only and pending:
            raise RuntimeError(
                "Wrapper patches are missing: " + ", ".join(p[0] for p in pending)
                + ". Run this script without --check, or use the normal build."
            )
        applied = []
        try:
            for name, dependency, patch in pending:
                result = git_apply(dependency, patch)
                if result.returncode:
                    raise RuntimeError(f"Could not apply {name} wrapper patch: {result.stderr.strip()}")
                applied.append((name, dependency, patch))
                print(f"[wrappers] {name}: applied checked-in patch")
        except Exception:
            # Undo only our successful patches if a later apply unexpectedly fails.
            for name, dependency, patch in reversed(applied):
                rollback = git_apply(dependency, patch, "--reverse")
                if rollback.returncode:
                    print(f"[wrappers] Could not undo {name}; preserve local edits and inspect manually", file=sys.stderr)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Verify patches are present without applying them")
    parser.add_argument("--only", choices=("all", "juce", "clap"), default="all")
    args = parser.parse_args()
    try:
        ensure_patches(args.root.resolve(), check_only=args.check, only=args.only)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
