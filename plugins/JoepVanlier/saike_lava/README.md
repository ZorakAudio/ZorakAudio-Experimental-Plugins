# Lava Reverb (Saike) [ALPHA]

A large-space shimmer reverb with several algorithms, pitch shifting, saturation and Ice effects.

## Quick start

1. Feed stereo audio and set dry/wet and verb_decay.
2. Choose Algorithm in the canvas and compare the reverb before adding shimmer.
3. Set modulation, EQ Curve and drop effects to shape the return.

## Controls and routing

Different algorithms have different CPU costs, and tails persist after input stops. The fluid canvas also has a cost that an audio-only benchmark does not include.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **diffusion** (`slider1`): default `0.6`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **verb_decay** (`slider2`): default `0.8`; declared range/choices `0,.95,0.00001`. Canvas / hidden.
- **mod_depth** (`slider3`): default `0.1`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **mod_rate** (`slider4`): default `0.3`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **lowpass** (`slider5`): default `1`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **highpass** (`slider6`): default `0`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **shimmer** (`slider7`): default `0.2`; declared range/choices `0,0.4,.001`. Canvas / hidden.
- **drop mode** (`slider8`): default `2`; declared range/choices `0,2,1{Off,Water,Ice}`. Canvas / hidden.
- **drop frequency** (`slider9`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **non-linearity** (`slider10`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **dry/wet** (`slider11`): default `0.25`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Algorithm** (`slider12`): default `0`; declared range/choices `0,1,4{Abyss,Lava1,Cold,Ethereal,fft_verb}`. Canvas / hidden.
- **Curve** (`slider13`): default `0`; declared range/choices `0,14,1`. Canvas / hidden.
- **Pitch** (`slider14`): default `-1`; declared range/choices `-2,0,0.001`. Canvas / hidden.
- **STFT modulation** (`slider15`): default `0`; declared range/choices `0,400,0.001`. Canvas / hidden.
- **Economy** (`slider16`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Frequency Shift** (`slider17`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Eco graphics** (`slider18`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**No validated speedup claim: output differs from native WDL.** The raw timings below are diagnostic only; they do not demonstrate an equivalent faster implementation. Maximum absolute float-output error: 0.0825767; maximum relative RMS error: 0.219565; MIDI differences: 0. The cause is not resolved by this documentation/timing audit.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.7953 | 0.6865 | 1.16× | 1.16–1.17× |
| 512 | 0.7478 | 0.6274 | 1.20× | 1.15–1.22× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.28. Original path: `lavaverb/saike_lava.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
