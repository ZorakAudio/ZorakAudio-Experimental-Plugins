# Saike ToneStacks (BETA)

Classic guitar-amplifier tone-stack models with bass/mid/treble controls and gain normalization.

## Quick start

1. Feed audio and choose Type.
2. Adjust Bass, Mid and Treble while comparing the chosen stack's interactions.
3. Choose Normalize gain and balance Vol; keep correct frequency scaling enabled unless testing legacy behaviour.

## Controls and routing

These are interacting circuit-style tone controls, not independent shelving EQ bands. Some legacy model names/ranges in the source are irregular; the table preserves the declared values.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Type** (`slider1`): default `0`; declared range/choices `0,7,{Marshall,Fender,Vox,James,E-Series,Bench,Big Muff,Hiwatt,Crate,Dumble Rock (incorrect model),Dumble Jazz,Aria,BoneRay,Wah`. Slider.
- **Bass** (`slider2`): default `.5`; declared range/choices `0,1,.001`. Slider.
- **Mid** (`slider3`): default `.5`; declared range/choices `0,1,.001`. Slider.
- **Treble** (`slider4`): default `.5`; declared range/choices `0,1,.001`. Slider.
- **Vol** (`slider5`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Normalize gain** (`slider6`): default `1`; declared range/choices `0,2,1{No,Peak,RMS}`. Slider.
- **Magic factor** (`slider7`): default `.95`; declared range/choices `0.2,1,.00001`. Canvas / hidden.
- **Use the correct frequency scaling** (`slider8`): default `1`; declared range/choices `0,1,1{No,Yes}`. Canvas / hidden.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.0571 | 0.0186 | 3.05× | 3.00–3.24× |
| 512 | 0.0527 | 0.0162 | 3.25× | 3.14–3.37× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.07. Original path: `Basics/ToneStacks.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
