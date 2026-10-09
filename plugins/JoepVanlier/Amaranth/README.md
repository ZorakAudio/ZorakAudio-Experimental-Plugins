# Amaranth (Saike) [BETA]

A granular sampler that captures incoming audio or loads a sample, then plays overlapping grains from the selected region.

## Quick start

1. Feed audio into the plugin while Update Buffer is enabled, or drop a sample onto its waveform.
2. Disable Update Buffer to hold the recorded material; choose the start/end region and playback position.
3. Send MIDI notes to play it. Adjust Grain size, Overlap, Position Variance and Speed before adding filters or feedback.

## Controls and routing

Follow MIDI pitch, Reference semitone and Pitch Mode determine grain tuning. Store sample in preset controls whether sample data is serialized; enable it when the preset must be self-contained. The default captures input; a MIDI note without material is not a representative sampler workload.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Start point** (`slider1`): default `0`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **End point** (`slider2`): default `1`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **Position** (`slider3`): default `0`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **Grain size [ms]** (`slider4`): default `70`; declared range/choices `40,300,1`. Canvas / hidden.
- **Position Variance [%]** (`slider5`): default `80`; declared range/choices `0,1000,1`. Canvas / hidden.
- **Overlap (%)** (`slider6`): default `90`; declared range/choices `0,93,1`. Canvas / hidden.
- **Pan Spread** (`slider7`): default `1`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Speed** (`slider8`): default `1`; declared range/choices `.125,2,.0001`. Canvas / hidden.
- **Speed Spread** (`slider9`): default `.001`; declared range/choices `0,1,.000001`. Canvas / hidden.
- **Follow MIDI pitch** (`slider10`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Reference semitone** (`slider11`): default `0`; declared range/choices `-48,48,1`. Canvas / hidden.
- **Pitch Mode** (`slider12`): default `0`; declared range/choices `0,1,1{Normal,RandomPitch}`. Canvas / hidden.
- **Octave Min** (`slider13`): default `0`; declared range/choices `-4,4,1`. Canvas / hidden.
- **Octave Max** (`slider14`): default `0`; declared range/choices `-4,4,1`. Canvas / hidden.
- **Reverse Probability** (`slider15`): default `0`; declared range/choices `0,1,.000001`. Canvas / hidden.
- **Feedback** (`slider16`): default `0`; declared range/choices `0,1,.000001`. Canvas / hidden.
- **Invariance** (`slider17`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **F1 Type** (`slider18`): default `0`; declared range/choices `0,11,1`. Canvas / hidden.
- **F1 Freq** (`slider19`): default `0`; declared range/choices `0,1,.0000000000000001`. Canvas / hidden.
- **F1 Resonance** (`slider20`): default `0`; declared range/choices `0,1,.0000000000000001`. Canvas / hidden.
- **F2 Type** (`slider21`): default `0`; declared range/choices `0,11,1`. Canvas / hidden.
- **F2 Freq** (`slider22`): default `0`; declared range/choices `0,1,.0000000000000001`. Canvas / hidden.
- **F2 Resonance** (`slider23`): default `0`; declared range/choices `0,1,.0000000000000001`. Canvas / hidden.
- **Amp Attack** (`slider24`): default `0.1`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Amp Decay** (`slider25`): default `0.56`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Amp Release** (`slider26`): default `0.56`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Sustain level** (`slider27`): default `0.6`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Env1 Loop Start (experimental)** (`slider28`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Env1 Loop End (experimental)** (`slider29`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Store sample in preset** (`slider62`): default `0`; declared range/choices `0,1,0`. Canvas / hidden.
- **Update Buffer** (`slider63`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **preGain** (`slider61`): default `0`; declared range/choices `-32,32,.00001`. Canvas / hidden.
- **postGain** (`slider64`): default `0`; declared range/choices `-32,32,.00001`. Canvas / hidden.
- **maximum grains** (`slider65`): default `0`; declared range/choices `0,3,1{15,30,45,60}`. Canvas / hidden.
- **window type** (`slider66`): default `0`; declared range/choices `0,5,.000001`. Canvas / hidden.
- **stochasticity** (`slider67`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.4549 | 0.4511 | 1.01× | 0.96–1.05× |
| 512 | 0.3984 | 0.3908 | 1.03× | 0.96–1.07× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.32. Original path: `Amaranth/Amaranth.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
