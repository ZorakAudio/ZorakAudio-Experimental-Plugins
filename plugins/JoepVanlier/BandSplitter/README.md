# Saike 4-pole BandSplitter

Splits stereo audio into up to five frequency bands with phase-matched crossovers.

## Quick start

1. Use a track with ten channels and expose all five output pairs.
2. Set Cuts and place the crossover frequencies in the canvas.
3. Process each band separately, then sum the pairs using BandJoiner.

## Controls and routing

IIR is the cheaper, zero-latency crossover mode. Linear-phase FIR introduces latency and pre-ringing; avoid automating FIR crossover frequencies during playback. Match any parallel dry path with the IIR phase matcher when required.

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
- **Fixed frequency range** (`slider58`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Absolute placement** (`slider59`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **FIR Quality** (`slider60`): default `0`; declared range/choices `0,2,{Normal,High,Ultra}`. Canvas / hidden.
- **FIR mode** (`slider61`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Master Gain** (`slider62`): default `0`; declared range/choices `-26,26,.1`. Canvas / hidden.
- **Band mode** (`slider63`): default `0`; declared range/choices `0,1,1{4p,2p}`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output 1, 2: right output 1, 3: left output 2, 4: right output 2, 5: left output 3, 6: right output 3, 7: left output 4, 8: right output 4, 9: left output 5, 10: right output 5.

## More background from the vendored source

### 4-pole Band Splitter
4-pole band splitter that preserves phase between the bands. It has a UI and uses much steeper crossover filters (24 dB/oct) than the default that ships with Reaper thereby providing sharper band transitions.
It also has an option for linear phase FIR crossovers instead of the default IIR filters. IIRs cost less CPU and introduce no preringing or latency. The linear phase FIRs however prevent phase distortion (which can be important in some mixing settings), but introduce latency compensation. Note that when using the linear phase filters, it is not recommended to modulate the crossover frequencies as this introduces crackles.
[Screenshot](https://i.imgur.com/nOhiaJB.png)
### Demos
You can find a tutorial of the plugin [here](https://www.youtube.com/watch?v=JU_7gIr5RTI).

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.1307 | 0.0612 | 2.15× | 2.10–2.19× |
| 512 | 0.1262 | 0.0588 | 2.15× | 2.08–2.27× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.23. Original path: `Basics/BandSplitter.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
