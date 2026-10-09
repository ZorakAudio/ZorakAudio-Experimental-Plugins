# Saike Tanh Saturation with anti aliasing

An anti-aliased waveshaper with gain/ceiling, DC correction, alternate shapes and oversampling.

## Quick start

1. Feed stereo audio and raise Gain while lowering Ceiling if needed.
2. Compare Continuous antialias mode and the available Shaping function.
3. Set Oversampling, Fix DC and HF-loss correction; level-match against bypass.

## Controls and routing

Continuous anti-alias modes and oversampling are separate controls. Match both when comparing CPU or output.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Gain (dB)** (`slider1`): default `0`; declared range/choices `-6,24,1`. Slider.
- **Ceiling (dB)** (`slider2`): default `0`; declared range/choices `-18,0,1`. Slider.
- **Continuous antialias mode** (`slider3`): default `1`; declared range/choices `0,2,1{No,Constant,Linear (BETA)}`. Slider.
- **Fix DC?** (`slider4`): default `0`; declared range/choices `0,1,1`. Slider.
- **Inertia (ms)** (`slider5`): default `50`; declared range/choices `0,100,.001`. Slider.
- **Correct for HF loss** (`slider6`): default `1`; declared range/choices `0,1,1`. Slider.
- **Shaping function** (`slider7`): default `0`; declared range/choices `0,3,1{tanh,clip,sin}`. Slider.
- **Oversampling** (`slider8`): default `1`; declared range/choices `1,4,1`. Slider.

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
| 64 | 0.1394 | 0.0862 | 1.62× | 1.61–1.66× |
| 512 | 0.1306 | 0.0782 | 1.66× | 1.62–1.68× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier, Erich M. Burg. Vendored version: 1.16. Original path: `Basics/Tanh_Saturator_AA.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
