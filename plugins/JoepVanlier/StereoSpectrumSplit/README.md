# Saike SideSpectrum Meter

A stereo side-spectrum meter with FFT size, floor, window and integration controls.

## Quick start

1. Feed stereo audio and open the canvas to view its analysis.
2. Set FFT size and Window, then adjust floor and integration time.
3. Enable phase display when that view is useful.

## Controls and routing

This is an analyzer: substantial FFT/display work occurs in @gfx. A no-GFX audio benchmark is not an analyzer performance comparison.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **FFT size** (`slider1`): default `10`; declared range/choices `0,9,1{16,32,64,128,256,512,1024,2048,4096,8192,16384,32768}`. Canvas / hidden.
- **floor** (`slider2`): default `-108`; declared range/choices `-450,-12,6`. Canvas / hidden.
- **show phase** (`slider3`): default `0`; declared range/choices `0,1,1{disabled,enabled}`. Canvas / hidden.
- **window** (`slider4`): default `2`; declared range/choices `0,3,1{rectangular,hamming,blackman-harris,blackman}`. Canvas / hidden.
- **integration time (ms)** (`slider5`): default `200`; declared range/choices `0,2500,1`. Canvas / hidden.
- **scaling** (`slider6`): default `1`; declared range/choices `1,6,.2`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.0358 | 0.0062 | 5.86× | 5.52–6.11× |
| 512 | 0.0353 | 0.0063 | 5.63× | 5.20–6.02× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

**Graphics-dependent workload:** this omits the analysis/display or simulation in `@gfx`. No complete analyzer or interactive-effect speedup is established.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Cockos, Joep Vanlier. Vendored version: 1.0. Original path: `SpectrumAnalyzer/StereoSpectrumSplit.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
