# ReaBee

An experimental stereo effect whose moving particle swarm controls multiple short buffer taps.

## Quick start

1. Feed stereo audio and start with the default bzzz depth and bee count.
2. Move the pointer on the swarm canvas and compare the resulting modulation.
3. Adjust buzz? spread and BUZZ time scale; reduce the bee count for a simpler swarm.

## Controls and routing

The swarm is advanced in @gfx and influences audio. DSP-only timings exclude this interactive simulation and cannot describe total open-editor cost.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **bzzz** (`slider2`): default `35`; declared range/choices `1,100,0.01`. Slider.
- **# bzzzzz!** (`slider3`): default `48`; declared range/choices `1, 64, 1`. Slider.
- **buzz?** (`slider4`): default `0`; declared range/choices `0,0.3,0.01`. Slider.
- **:(** (`slider5`): default `1`; declared range/choices `0.5,2.0>BUZZ`. Slider.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 2.9809 | 2.6355 | 1.13× | 1.12–1.14× |
| 512 | 2.9790 | 2.6527 | 1.12× | 1.09–1.13× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

**Graphics-dependent workload:** this omits the analysis/display or simulation in `@gfx`. No complete analyzer or interactive-effect speedup is established.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.02. Original path: `ReaBee/ReaBee.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
