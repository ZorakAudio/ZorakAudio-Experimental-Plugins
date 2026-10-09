# Dusk Verb (Saike) (beta)

An atmospheric multi-effect combining reverb, granular resampling, shimmer and frequency shifting.

## Quick start

1. Feed stereo audio and set Verb Mix and Verb Time.
2. Adjust the Verb, Grain, Shimmer and Haunt X/Y controls in the canvas.
3. Compare algorithms and add one transformation at a time.

## Controls and routing

The X/Y coordinates are saved automation controls. Reverb and granular histories continue across blocks; dry/wet and feedback settings materially change processing cost.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Verb Time [ms]** (`slider1`): default `900`; declared range/choices `5,2800,1`. Canvas / hidden.
- **Random Seed** (`slider2`): default `9860959.345567`; declared range/choices `0,12311323,1`. Canvas / hidden.
- **Reverb Mode** (`slider3`): default `0`; declared range/choices `0,2,1`. Canvas / hidden.
- **Upward Shimmer (Shimmer Y)** (`slider4`): default `0.4`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Downward Shimmer (Shimmer X)** (`slider5`): default `0`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Frequency Amount (Haunt Y)** (`slider6`): default `0`; declared range/choices `0,1.5.0,0.00001`. Canvas / hidden.
- **Frequency Shift (Haunt X)** (`slider7`): default `60`; declared range/choices `60,880,0.00001`. Canvas / hidden.
- **Grain Mix (Grain Y)** (`slider8`): default `0.2`; declared range/choices `0, 2, 0.01`. Canvas / hidden.
- **Verb Mix (Verb Y)** (`slider9`): default `0.4`; declared range/choices `0, 1, 0.01`. Canvas / hidden.
- **Shimmer Mode** (`slider11`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Haunt Algorithm** (`slider20`): default `0`; declared range/choices `0,3,1`. Canvas / hidden.
- **Grain Algorithm** (`slider30`): default `0`; declared range/choices `0,4,1`. Canvas / hidden.
- **Grain Length (Grain X)** (`slider31`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Grain Frequency** (`slider32`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Brightness (Verb X)** (`slider40`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Side low cut frequency** (`slider41`): default `120`; declared range/choices `20,22050,0.001:log`. Canvas / hidden.
- **Low cut frequency** (`slider42`): default `20`; declared range/choices `20,22050,0.001:log`. Canvas / hidden.
- **High cut frequency** (`slider43`): default `22050`; declared range/choices `20,22050,0.001:log`. Canvas / hidden.
- **Damping frequency** (`slider44`): default `22050`; declared range/choices `20,22050,0.001:log`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## More background from the vendored source

### A multi-effect plugin intended to enhance atmospheric arpeggios
[Screenshot](https://user-images.githubusercontent.com/19836026/221384927-db1d9f3e-df04-4676-a4d4-aa508ad1ade6.gif)
### Features:
- 3 Reverberation algorithms.
- Granular resampler.
- Frequency shifter / pitch shifter. 
- Several audio shimmer modes.
- X/Y controls for automation.
- Classic adventure game look.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.9671 | 0.9161 | 1.06× | 1.04–1.13× |
| 512 | 0.9061 | 0.7855 | 1.15× | 1.09–1.22× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.17. Original path: `DuskVerb/saike_duskverb.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
