# Saike Multiband Ravager (BETA)

An extreme upward compressor that brings quiet material forward, optionally in several bands.

## Quick start

1. Feed stereo audio and start with restrained drive and ratio.
2. Use the crossover canvas to add or remove bands, then adjust their drive.
3. Set Release and limiting/clip behaviour while comparing output loudness.

## Controls and routing

This is deliberately aggressive upward processing. Raising quiet detail also raises existing noise; use the band controls to focus the effect.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Cuts** (`slider1`): default `1`; declared range/choices `0,4,1`. Canvas / hidden.
- **Frequency 1** (`slider2`): default `0.2`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 2** (`slider3`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 3** (`slider4`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 4** (`slider5`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Drive 1 (dB)** (`slider6`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Drive 2 (dB)** (`slider7`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Drive 3 (dB)** (`slider8`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Drive 4 (dB)** (`slider9`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Drive 5 (dB)** (`slider10`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Release** (`slider11`): default `0.735`; declared range/choices `0,1,.0001`. Slider.
- **Ratio** (`slider12`): default `2`; declared range/choices `0 ,2,.0001`. Slider.
- **Thresh** (`slider13`): default `0`; declared range/choices `-20,20,.0001`. Canvas / hidden.
- **Couple L/R?** (`slider14`): default `1`; declared range/choices `0,1,{False,True}`. Canvas / hidden.
- **DC offset filter** (`slider15`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Input gain** (`slider16`): default `0`; declared range/choices `-32,32,.001`. Slider.
- **Distortion gain** (`slider17`): default `-8`; declared range/choices `-32,32,0.0001`. Slider.
- **Output gain** (`slider18`): default `-12`; declared range/choices `-32,32,.001`. Slider.
- **Bidirectional** (`slider19`): default `0`; declared range/choices `0,1,1{No,Yes}`. Canvas / hidden.
- **Output limiting** (`slider20`): default `0`; declared range/choices `0,3,1{None,Clip,Soft,ClipAA}`. Canvas / hidden.
- **Gate level [dB]** (`slider21`): default `-340`; declared range/choices `0,-340,.0001`. Slider.
- **FIR mode** (`slider61`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Animate** (`slider62`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output 1, 2: right output 1.

## More background from the vendored source

### Multiband Audio Destroyer. Performs extreme upwards compression akin to DOOM compressor.
Compressor design based on: Giannoulis et al, "Digital Dynamic Range Compressor Design—A Tutorial and Analysis", Journal of the Audio Engineering Society 60(6)

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.4349 | 0.2869 | 1.53× | 1.51–1.53× |
| 512 | 0.4190 | 0.2668 | 1.56× | 1.53–1.58× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.14. Original path: `Ravager/Ravager_MB.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
