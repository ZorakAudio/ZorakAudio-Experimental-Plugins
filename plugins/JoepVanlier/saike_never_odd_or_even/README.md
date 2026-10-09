# Saike Never Odd or Even (Distortion)

A distortion processor that blends even and odd harmonics, with warmth, DC correction and gain compensation.

## Quick start

1. Feed audio and begin with Even and Odd low.
2. Raise Gain and the harmonic controls, then adjust Ceiling and Warmth.
3. Compare DC Correction and Dynamic Gain Compensation; level-match with bypass.

## Controls and routing

DC correction and compensation have state. Oversampling affects cost and anti-alias behaviour; preserve its setting in comparisons.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Gain (dB)** (`slider1`): default `0`; declared range/choices `-6,24,0.0001`. Slider.
- **Ceiling (dB)** (`slider2`): default `0`; declared range/choices `-36,0,0.0001`. Slider.
- **Even** (`slider3`): default `0`; declared range/choices `0,1,.0001`. Slider.
- **Odd** (`slider4`): default `0`; declared range/choices `0,1,.0001`. Slider.
- **Warmth (dB)** (`slider5`): default `0`; declared range/choices `-12,12,.001`. Slider.
- **DC Correction** (`slider7`): default `2`; declared range/choices `0,2,1{IIR (can induce phase distortion),OFF,Improved IIR (less phase distortion)}`. Slider.
- **Dynamic Gain Compensation** (`slider8`): default `0`; declared range/choices `0,2,1{OFF,active adjustment,fixed}`. Slider.
- **Oversampling factor** (`slider9`): default `1`; declared range/choices `1,4,1`. Slider.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.1497 | 0.0744 | 2.01× | 1.98–2.04× |
| 512 | 0.1454 | 0.0714 | 2.03× | 2.02–2.08× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.05. Original path: `Basics/saike_never_odd_or_even.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
