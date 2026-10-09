# Mod-izer (Saike)

A tracker-flavoured lo-fi processor with sample-rate reduction, pseudo bit depth, interpolation and an LED filter.

## Quick start

1. Feed stereo audio into the effect and lower samplerate or pseudo-bitrate gradually.
2. Compare S&H and linear Interpolation, then switch LED filtering.
3. Use the MIDI pitch reference and follow option if you want degradation to track notes.

## Controls and routing

The MIDI pitch reference changes the resampling behaviour. It is an effect on incoming audio, not a replacement sound source.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **samplerate** (`slider1`): default `28000`; declared range/choices `8192,48000,1`. Canvas / hidden.
- **midi pitch** (`slider2`): default `57`; declared range/choices `1,108,1`. Canvas / hidden.
- **pseudo-bitrate** (`slider3`): default `8`; declared range/choices `2,24,.1`. Canvas / hidden.
- **LED** (`slider4`): default `1`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Don't follow incoming MIDI pitch** (`slider5`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Interpolation** (`slider7`): default `0`; declared range/choices `0,1,{S&H,linear}`. Canvas / hidden.

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
| 64 | 0.0850 | 0.0433 | 2.01× | 1.96–2.14× |
| 512 | 0.0816 | 0.0407 | 2.01× | 1.97–2.05× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.06. Original path: `Modizer/modizer.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
