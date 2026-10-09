# Saike Protosynth

A polyphonic synthesizer with eight oscillators, nonlinear filters and a configurable signal-combination graph.

## Quick start

1. Route MIDI notes to the plugin and enable an oscillator in the canvas.
2. Set its waveform and amplitude envelope, then select a simple mix path.
3. Add oscillators, filter modules and phase/ring/convolution combinations incrementally.

## Controls and routing

Voice count, active oscillators and mix algorithms dominate workload. A short default-note benchmark cannot represent a dense patch with all graph nodes active.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Mix mode 1** (`slider128`): default `0`; declared range/choices `0,9,1`. Canvas / hidden.
- **Mix mode 2** (`slider129`): default `0`; declared range/choices `0,9,1`. Canvas / hidden.
- **Mix mode 3** (`slider130`): default `0`; declared range/choices `0,9,1`. Canvas / hidden.
- **Mix mode 4** (`slider131`): default `0`; declared range/choices `0,9,1`. Canvas / hidden.
- **Mix mode 21** (`slider132`): default `0`; declared range/choices `0,9,1`. Canvas / hidden.
- **Mix mode 22** (`slider133`): default `0`; declared range/choices `0,9,1`. Canvas / hidden.
- **Mix mode 31** (`slider134`): default `0`; declared range/choices `0,9,1`. Canvas / hidden.
- **Cutoff Mix 21** (`slider135`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Resonance Mix 21** (`slider136`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Morph Mix 21** (`slider137`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Cutoff Mix 22** (`slider138`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Resonance Mix 22** (`slider139`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Morph Mix 22** (`slider140`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Filter Type** (`slider141`): default `0`; declared range/choices `0,16,1`. Canvas / hidden.
- **Drive Mix** (`slider142`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Cutoff Mix** (`slider143`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Resonance Mix** (`slider144`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Morph Mix** (`slider145`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **F1 Cutoff** (`slider146`): default `1`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F1 Resonance** (`slider147`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F1 Morph** (`slider148`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F2 Cutoff** (`slider149`): default `1`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F2 Resonance** (`slider150`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F2 Morph** (`slider151`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F3 Filter Cutoff** (`slider152`): default `1`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F3 Filter Resonance** (`slider153`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F3 Filter Morph** (`slider154`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **F3 Type** (`slider155`): default `0`; declared range/choices `0,6,1{LIN,MS,L2,L4,SHRK,PL,SHRK2`. Canvas / hidden.
- **F3 Drive** (`slider156`): default `0`; declared range/choices `-6,48,0.00001`. Canvas / hidden.
- **Key follow** (`slider157`): default `0`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Output Filter Cutoff** (`slider158`): default `1`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Output Filter Resonance** (`slider159`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Output Filter Morph** (`slider160`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Output Filter Type** (`slider161`): default `0`; declared range/choices `0,6,1{LIN,MS,L2,L4,SHRK,PL,SHRK2`. Canvas / hidden.
- **Output Drive** (`slider162`): default `0`; declared range/choices `-6,24,0.00001`. Canvas / hidden.
- **Output Key follow** (`slider163`): default `0`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Spread** (`slider231`): default `0.5`; declared range/choices `0, 1, 0.0001`. Canvas / hidden.
- **Mix** (`slider232`): default `100`; declared range/choices `0, 200, 0.01`. Canvas / hidden.
- **Filter frequency** (`slider233`): default `20`; declared range/choices `20,22050,0.01:log`. Canvas / hidden.
- **AP frequency** (`slider234`): default `241`; declared range/choices `20, 22000, 0.0001:log`. Canvas / hidden.
- **AP Feedback** (`slider235`): default `0.0`; declared range/choices `-0.999, 0.999, 0.001`. Canvas / hidden.
- **AP saturation** (`slider236`): default `0.01`; declared range/choices `0.01, 0.25, 0.001`. Canvas / hidden.
- **Pitch bend range** (`slider237`): default `2`; declared range/choices `0,24,0.001`. Canvas / hidden.
- **Global tuning** (`slider238`): default `0`; declared range/choices `-24,24,0.001`. Canvas / hidden.
- **fb1** (`slider239`): default `0`; declared range/choices `-0.999,0.999,.0001`. Canvas / hidden.
- **fb2** (`slider240`): default `0`; declared range/choices `-0.999,0.999,.0001`. Canvas / hidden.
- **fb3** (`slider241`): default `0`; declared range/choices `-0.999,0.999,.0001`. Canvas / hidden.
- **fb4** (`slider242`): default `0`; declared range/choices `-0.999,0.999,.0001`. Canvas / hidden.
- **fb21** (`slider243`): default `0`; declared range/choices `-0.999,0.999,.0001`. Canvas / hidden.
- **fb22** (`slider244`): default `0`; declared range/choices `-0.999,0.999,.0001`. Canvas / hidden.
- **fb31** (`slider245`): default `0`; declared range/choices `-0.999,0.999,.0001`. Canvas / hidden.
- **glide time** (`slider246`): default `100`; declared range/choices `3,3000,0.001:log`. Canvas / hidden.
- **poisson_freq** (`slider247`): default `500`; declared range/choices `20, 10000, 0.01:log`. Canvas / hidden.
- **Verb Time [ms]** (`slider248`): default `900`; declared range/choices `5,2800,1`. Canvas / hidden.
- **Verb Mix** (`slider249`): default `0.4`; declared range/choices `0, 1, 0.01`. Canvas / hidden.
- **Side low cut frequency** (`slider250`): default `120`; declared range/choices `20,22050,0.001:log`. Canvas / hidden.
- **Low cut frequency** (`slider251`): default `20`; declared range/choices `20,22050,0.001:log`. Canvas / hidden.
- **High cut frequency** (`slider252`): default `22050`; declared range/choices `20,22050,0.001:log`. Canvas / hidden.
- **Damping frequency** (`slider253`): default `22050`; declared range/choices `20,22050,0.001:log`. Canvas / hidden.
- **Output Gain** (`slider254`): default `-12`; declared range/choices `-24,6,0,0.0001`. Canvas / hidden.
- **Chorus amount** (`slider255`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Noise** (`slider256`): default `0`; declared range/choices `0.0001,0.4,0.0001:log`. Canvas / hidden.

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
| 64 | 0.8026 | 0.5391 | 1.48× | 1.47–1.51× |
| 512 | 0.2544 | 0.1464 | 1.73× | 1.70–1.77× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.60. Original path: `protosynth/saike_protosynth.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
