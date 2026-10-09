# Saike Pitch Shifter

A stereo pitch shifter with independent semitone shift, playback speed and selectable transition region.

## Quick start

1. Feed audio with semitones at zero and playspeed at one.
2. Shift semitones, compare Phase matching and choose a Transition region.
3. Use snap to now to reposition and Snap semitones for chromatic steps when wanted.

## Controls and routing

Larger transition regions change latency/texture and cost. Default zero-shift performance is not a guarantee for extreme shifts or speeds.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **semitones** (`slider1`): default `0`; declared range/choices `-36,36,.0001`. Slider.
- **playspeed** (`slider2`): default `1`; declared range/choices `0,2,.0001`. Slider.
- **snap to now** (`slider3`): default `0`; declared range/choices `0,1`. Slider.
- **Phase matching** (`slider7`): default `1`; declared range/choices `0,1,1{Off,On}`. Slider.
- **Transition region** (`slider8`): default `5`; declared range/choices `0,10,1{32,64,128,256,512,1024,2048,4096,8192,16384,32768}`. Slider.
- **Snap semitones** (`slider9`): default `1`; declared range/choices `0,1,2{Off,Chromatic,}`. Slider.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.1510 | 0.1025 | 1.46× | 1.44–1.49× |
| 512 | 0.1556 | 0.1040 | 1.52× | 1.42–1.54× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.04. Original path: `Basics/Saike_Pitch_Shift.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
