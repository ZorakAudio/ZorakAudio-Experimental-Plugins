"""Instrument generated native DSP, not just its C++ test wrapper, with ASan."""
from pathlib import Path
import subprocess
import re


def instrument(source: Path, output: Path, clang: str) -> None:
    lines = source.read_text(encoding='utf-8').splitlines(keepends=True)
    count = 0
    for i, line in enumerate(lines):
        if line.startswith('define ') and '{' in line:
            # ASan's IR pass requires this attribute; merely passing -fsanitize
            # while compiling an existing .ll file does not add it for us.
            if 'sanitize_address' not in line:
                prefix, brace, suffix = line.rpartition('{')
                lines[i] = prefix + 'sanitize_address {' + suffix
            count += 1
    # LLVM 20 may add the optional samesign poison promise; Clang 17 cannot
    # parse it. Dropping this proof annotation does not change valid results.
    # This is test-only IR preparation, never the production object path.
    lines = [re.sub(r'^(\s*%[^=]+=\s*icmp\s+)samesign\s+', r'\1', line)
             for line in lines]
    if not count:
        raise ValueError('No generated LLVM function definitions to instrument')
    modified = source.with_suffix('.asan.ll')
    modified.write_text(''.join(lines), encoding='utf-8')
    result = subprocess.run([clang, '-x', 'ir', str(modified), '-c', '-O2', '-fsanitize=address',
                    '-fno-omit-frame-pointer', '-o', str(output)], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError("Cannot instrument native LLVM object:\n" + result.stdout + result.stderr)
