"""Fail before catalog compilation if required compiler/backends are missing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import llvmlite
from llvmlite import binding as llvm
from jsfx_faust_compiler import metadata, parse_bitcode


def check():
    tools = {}
    for name in ("git", "cmake", "ninja"):
        path = shutil.which(name)
        if not path:
            raise RuntimeError(f"Required build tool missing: {name}")
        tools[name] = path
    faust = os.environ.get("JSFX_FAUST_COMPILER") or shutil.which("faust")
    if not faust or not Path(faust).is_file():
        raise RuntimeError("Faust missing; install a compiler with both C++ and LLVM backends")
    version = subprocess.check_output([faust, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    # Test the actual binary payload, LLVM parser and layout metadata we consume.
    with tempfile.TemporaryDirectory(prefix="za-ci-backend-") as temporary:
        root = Path(temporary)
        source = root / "backend.dsp"
        source.write_text('import("stdfaust.lib");g=hslider("Gain",.5,0,1,.01);process=*(g),*(g);\n', encoding="utf-8")
        bitcode = root / "backend.bc"
        subprocess.run([faust, "-lang", "llvm", "-double", "-cn", "CiBackend", "-o", str(bitcode), str(source)], check=True)
        module = parse_bitcode(bitcode.read_bytes())
        module.verify()
        layout = metadata(module)
        if layout["inputs"] != 2 or layout["outputs"] != 2:
            raise RuntimeError("Faust backend reported unexpected audio topology")
        subprocess.run([faust, "-lang", "cpp", "-i", "-o", str(root / "backend.h"), str(source)], check=True)
    targets = [llvm.get_default_triple()]
    if sys.platform == "darwin":
        targets = ["arm64-apple-macos11.0", "x86_64-apple-macos11.0"]
    for triple in targets:
        machine = llvm.Target.from_triple(triple).create_target_machine(reloc="pic")
        module = llvm.parse_assembly("define double @probe(double %x) { ret double %x }")
        module.triple = triple
        module.data_layout = str(machine.target_data)
        if not machine.emit_object(module):
            raise RuntimeError("LLVM emitted an empty object for " + triple)
    return dict(passed=True, platform=platform.platform(), python=sys.version,
                llvmlite=llvmlite.__version__, llvm=llvm.llvm_version_info,
                faust=faust, faustVersion=version, targets=targets, tools=tools)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(check(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)
