# Super Spreader

A spreading effect with wet/dry blend, optional attack reconstruction and selectable filtering.

## Quick start

1. Feed stereo audio and set Spread and Mix.
2. Choose Reconstruct Attack: Off, Audio, or MIDI; route notes for MIDI mode.
3. Choose a Filter target/order and cutoff, then check the resulting image in mono.

## Controls and routing

This source credits original work by lkjb. Mix extends beyond 100%, so it is not a simple clamped 0..1 wet/dry control.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Spread** (`slider1`): default `0.5`; declared range/choices `0, 1, 0.0001`. Slider.
- **Mix** (`slider2`): default `100`; declared range/choices `0, 200, 0.01`. Slider.
- **Reconstruct Attack** (`slider3`): default `0`; declared range/choices `0, 2, 1{Off,Audio,MIDI}`. Slider.
- **Filter** (`slider4`): default `0`; declared range/choices `0,4,1{Off,Highpass wet side,Highpass wet mid,Highpass wet,Highpass mix side}`. Slider.
- **Filter order** (`slider5`): default `0`; declared range/choices `0,1,1{4p,8p}`. Slider.
- **Filter frequency** (`slider6`): default `1`; declared range/choices `1,22050,0.01:log`. Slider.

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
| 64 | 0.4591 | 0.3698 | 1.26× | 1.23–1.40× |
| 512 | 0.4564 | 0.3521 | 1.30× | 1.29–1.42× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Original code by lkjb, basic port by Saike (Joep Vanlier). Vendored version: 0.09. Original path: `Basics/Saike SuperSpreaderClone.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
