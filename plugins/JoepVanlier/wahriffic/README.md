# Saike Wahriffic

Wah models based on Weeping Demon and Crybaby circuits, with drive, position and grunge.

## Quick start

1. Feed stereo audio and choose Model.
2. Sweep Cutoff and compare Cutoff 2/Grunge while keeping Drive modest.
3. Use Post Gain to match levels and compare Inertia and Oversampling.

## Controls and routing

The Crybaby label explicitly calls out higher cost and recommends at least x2 oversampling at high drive. The source credits Chet Gnegy and the Holters et al. physical modelling work.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Model** (`slider1`): default `0`; declared range/choices `0,2,1{Weeping Demon Mode 1,Weeping Demon Mode 2,Crybaby (Warning: CPU intensive / high drive needs oversampling x2)`. Slider.
- **Drive (dB)** (`slider3`): default `0`; declared range/choices `-6,24,1`. Slider.
- **Post Gain (dB)** (`slider4`): default `0`; declared range/choices `-6,24,1`. Slider.
- **Cutoff** (`slider13`): default `1`; declared range/choices `0,1,.0001`. Slider.
- **Cutoff 2** (`slider14`): default `0`; declared range/choices `0,1,.0001`. Slider.
- **Grunge** (`slider15`): default `0`; declared range/choices `0,1,.0001`. Slider.
- **Inertia** (`slider17`): default `1`; declared range/choices `0,1,1{Off,On}`. Slider.
- **Oversampling** (`slider60`): default `2`; declared range/choices `1,3,1`. Slider.

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
| 64 | 0.3265 | 0.8144 | 0.40× | 0.40–0.41× |
| 512 | 0.3217 | 0.8143 | 0.39× | 0.38–0.40× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.04. Original path: `Basics/wahriffic.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
