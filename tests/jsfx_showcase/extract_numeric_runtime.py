"""Compile the production FFT/memcpy implementation without the JUCE host TU.

This extracts an exact, delimited block; it does not reimplement FFT arithmetic.
The test provides only the allocation hook (preallocated bounded test memory).
"""
from pathlib import Path
import sys

def extract(source: Path, output: Path) -> None:
    text = source.read_text(encoding='utf-8')
    start = text.index('// ---- JSFX FFT runtime helpers ')
    end = text.index('extern "C" void jsfx_ensure_mem (DSPJSFX_State* st, int64_t needed)\n{', start)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('// Extracted from JSFXJuceProcessor.cpp; do not edit.\n' + text[start:end], encoding='utf-8')

if __name__ == '__main__':
    extract(Path(sys.argv[1]), Path(sys.argv[2]))
