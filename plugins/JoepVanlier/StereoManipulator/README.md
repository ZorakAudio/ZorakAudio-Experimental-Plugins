# Saike StereoManipulator

A two-band stereo-width processor with selectable crossover filters and channel audition.

## Quick start

1. Feed stereo audio with both Width controls at 100%.
2. Set Crossover and Filter Type, then adjust Width Low and Width High separately.
3. Use Listen Channel to audition the bands and Use Channel to compare source interpretation.

## Controls and routing

FIR and IIR choices have different latency/phase and processing costs. Keep Filter Type identical when comparing engines.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Width Low (%)** (`slider1`): default `100`; declared range/choices `0,200,1`. Slider.
- **Crossover (Hz)** (`slider2`): default `500`; declared range/choices `20,20000,1`. Slider.
- **Width High (%)** (`slider3`): default `100`; declared range/choices `0,200,1`. Slider.
- **Use Channel** (`slider4`): default `0`; declared range/choices `0,2,1{Mix,Left,Right}`. Slider.
- **Filter Type** (`slider5`): default `0`; declared range/choices `0,8,1{Cheapo,FIR32,FIR64,FIR128,FIR256,FIR512,FIR1024,IIR Butterworth 2 pole,IIR Butterworth 3 pole,IIR Butterworth 4 pole}`. Slider.
- **Listen Channel** (`slider6`): default `0`; declared range/choices `0,2,1{All,Low,High}`. Slider.

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
| 64 | 0.0490 | 0.0183 | 2.73× | 2.68–2.79× |
| 512 | 0.0509 | 0.0183 | 2.78× | 2.69–2.79× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Sai'ke. Vendored version: 1.0. Original path: `SpectrumAnalyzer/StereoManipulator.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
