# Saike MS-20 filter emulation

A nonlinear MS-20-style filter with lowpass, bandpass and highpass modes, drive and oversampling.

## Quick start

1. Insert on stereo audio, choose Type and lower Cutoff from its open position.
2. Raise Resonance and Drive gradually, using Post Boost to level-match.
3. Compare the integrator and diode nonlinearities; increase Oversampling when driven.

## Controls and routing

Inertia smooths parameter changes. Oversampling costs CPU and alters anti-alias behaviour; compare performance with the same setting.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Drive (dB)** (`slider1`): default `4`; declared range/choices `-6,24,1`. Slider.
- **Post Boost (dB)** (`slider2`): default `0`; declared range/choices `-6,24,1`. Slider.
- **Cutoff** (`slider13`): default `1`; declared range/choices `0,1,.0001`. Slider.
- **Resonance** (`slider14`): default `0`; declared range/choices `0,1,.0001`. Slider.
- **Type** (`slider5`): default `0`; declared range/choices `0,2,{LP,BP,HP}`. Slider.
- **Inertia** (`slider6`): default `1`; declared range/choices `0,1,1{Off,On}`. Slider.
- **Non-linearity integrator saturation (OTA)** (`slider7`): default `0`; declared range/choices `0,1,1{tanh,atan}`. Slider.
- **Non-linearity diode clipper** (`slider8`): default `0`; declared range/choices `0,1,1{clip,soft}`. Slider.
- **Oversampling** (`slider60`): default `2`; declared range/choices `1,8,1`. Slider.

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
| 64 | 1.5765 | 1.0491 | 1.50× | 1.48–1.51× |
| 512 | 1.5847 | 1.0563 | 1.51× | 1.48–1.53× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 1.06. Original path: `Basics/MS-20.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
