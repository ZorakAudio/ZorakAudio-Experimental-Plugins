# Satan verb (Saike)

An FFT reverb with spectral bleed, pitch drop/shift, envelopes and nonlinear spectral shaping.

## Quick start

1. Feed audio and begin with a modest Dry/Wet and Maximum based mode.
2. Adjust Spectral Bleed, Pitch drop and return filtering.
3. Add shifted signal or envelope shaping only after checking the basic tail.

## Controls and routing

Growth/Decay mode carries an explicit source-level volume caution. FFT size/latency and tails matter; a short DSP test does not cover every long-tail configuration.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Spectral Bleed** (`slider1`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Pitch drop** (`slider2`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Mode** (`slider3`): default `0`; declared range/choices `0,1,1{Maximum based,Growth/Decay (watch the volume!)}`. Canvas / hidden.
- **Spectral ceiling** (`slider4`): default `3`; declared range/choices `0,5,.001`. Canvas / hidden.
- **Dry/Wet** (`slider5`): default `0.5`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Width** (`slider6`): default `1.0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Unshifted Amount** (`slider7`): default `1`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Spectral Shift Amount** (`slider8`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Spectral Shift** (`slider9`): default `0`; declared range/choices `-1.5,5,.0001`. Canvas / hidden.
- **Lowpass** (`slider10`): default `1`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Highpass** (`slider11`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Input Distortion** (`slider12`): default `0`; declared range/choices `-3,1,.001`. Canvas / hidden.
- **No optimization** (`slider64`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Compensate delay** (`slider29`): default `1`; declared range/choices `0,1,1{No,Yes}`. Canvas / hidden.
- **Use Envelope** (`slider30`): default `0`; declared range/choices `0,1,1{No,Yes}`. Canvas / hidden.
- **Attack** (`slider31`): default `0.5`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Decay** (`slider32`): default `.5`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Relative Envelope?** (`slider33`): default `1`; declared range/choices `0,1,1{No,Yes}`. Canvas / hidden.

## More background from the vendored source

### Satan Verb
Satan verb is a reverberation unit mostly meant for diffuse and gated style reverberation. It can either be used without an envelope, to generate large ambient spaces, or be modulated by an envelope based on the input sound to give a sound more body while not adding too much noise to the dead time.
[Screenshot 1](https://i.imgur.com/JLXFrOH.png), [Screenshot 2](https://i.imgur.com/EclxtWp.gif)
### Demos
You can find a demo of the plugin [here](https://www.youtube.com/watch?v=4aI-Gg8ETAM)
### Features:
- FFT based reverberation algorithm.
- Optional downward spectral smearing for creepy effects.
- Optional spectrally shifted copy can be mixed in.
- Steep IIR LPF/HPF filters for the verb.
- Optional delay compensation.
- Envelopes based on the input envelope.
- Input non-linearity (dist), spectrum non-linearity (ceiling).
- Dry/Wet controls.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.8605 | 0.7097 | 1.19× | 1.17–1.23× |
| 512 | 0.8449 | 0.7038 | 1.20× | 1.18–1.23× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.12. Original path: `SatanVerb/SatanVerb.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
