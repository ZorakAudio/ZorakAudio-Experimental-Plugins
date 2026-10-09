# Abyss Reverb (Saike) [BETA]

A modulated stereo reverb with pitch-shifted shimmer, nonlinearity and Water/Ice drop effects.

## Quick start

1. Feed stereo audio and begin with the default 25% dry/wet blend.
2. Adjust verb_decay and diffusion, then mod_depth/mod_rate.
3. Add shimmer or drop mode and use lowpass/highpass to shape the return.

## Controls and routing

The reverb and pitch shifters retain state and produce tails. A quiet block does not imply an empty reverb. Parameter ranges in the reference are the stored source values.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **diffusion** (`slider1`): default `0.6`; declared range/choices `0,1,0.00001`. Slider.
- **verb_decay** (`slider2`): default `0.8`; declared range/choices `0,.95,0.00001`. Slider.
- **mod_depth** (`slider3`): default `0.1`; declared range/choices `0,1,0.00001`. Slider.
- **mod_rate** (`slider4`): default `0.3`; declared range/choices `0,1,0.00001`. Slider.
- **lowpass** (`slider5`): default `1`; declared range/choices `0,1,0.00001`. Slider.
- **highpass** (`slider6`): default `0`; declared range/choices `0,1,0.00001`. Slider.
- **shimmer** (`slider7`): default `0.2`; declared range/choices `0,0.4,.001`. Slider.
- **drop mode** (`slider8`): default `0`; declared range/choices `0,2,1{Off,Water,Ice}`. Slider.
- **drop frequency** (`slider9`): default `0`; declared range/choices `0,1,.001`. Slider.
- **non-linearity** (`slider10`): default `0`; declared range/choices `0,1,.001`. Slider.
- **dry/wet** (`slider11`): default `0.25`; declared range/choices `0,1,.00001`. Slider.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**No validated speedup claim: output differs from native WDL.** The raw timings below are diagnostic only; they do not demonstrate an equivalent faster implementation. Maximum absolute float-output error: 0.129075; maximum relative RMS error: 0.292086; MIDI differences: 0. The cause is not resolved by this documentation/timing audit.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.8053 | 0.6879 | 1.17× | 1.16–1.25× |
| 512 | 0.7777 | 0.6511 | 1.21× | 1.19–1.29× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.06. Original path: `Abyss/saike_abyss.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
