# Saike BandJoiner

Sums five stereo band pairs back into one stereo output. It has no controls or custom canvas.

## Quick start

1. Route the five band pairs to inputs 1/2, 3/4, 5/6, 7/8 and 9/10.
2. Take the combined signal from output 1/2.
3. Keep band-processing latency aligned and adjust gain in the preceding processors.

## Controls and routing

Pair it with BandSplitter. It sums rather than averages: feeding the same full-range signal to every pair multiplies its level.

There are no declared slider parameters in this source. Only host audio routing is required.

Declared inputs: 1: left input 1, 2: right input 1, 3: left input 2, 4: right input 2, 5: left input 3, 6: right input 3, 7: left input 4, 8: right input 4, 9: left input 5, 10: right input 5 .

Declared outputs: 1: left output, 2: right output .

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.0346 | 0.0083 | 3.99× | 3.94–4.16× |
| 512 | 0.0323 | 0.0082 | 4.06× | 3.72–4.13× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: None. Original path: `Basics/BandJoiner.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
