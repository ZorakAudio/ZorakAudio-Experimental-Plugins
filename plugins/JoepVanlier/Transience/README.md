# Saike Transience (transient shaper)

A transient/decay modifier comparing the input envelope with a shaped target envelope in logarithmic space.

## Quick start

1. Feed audio and start with the attack/decay enhancement amounts at zero.
2. Set Attack and Decay, then enhance or reduce the corresponding portions.
3. Use Gain compensation, Gain smoothing and Oversampling to refine the result.

## Controls and routing

Its imported upsampling library is included in this leaf: the compiled plugin does not require installing a second Tight Compressor binary.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Attack** (`slider1`): default `0.5`; declared range/choices `0,1,.000000001`. Slider.
- **Decay** (`slider2`): default `0.5`; declared range/choices `0,1,.000000001`. Slider.
- **Reduce / Enhance Attack** (`slider3`): default `0`; declared range/choices `-1,1,.000000001`. Slider.
- **Reduce / Enhance Decay** (`slider4`): default `0`; declared range/choices `-1,1,.000000001`. Slider.
- **Gain compensation** (`slider5`): default `0`; declared range/choices `-20,20,0.0005`. Slider.
- **Peak/Squared** (`slider6`): default `0`; declared range/choices `0,1,1`. Slider.
- **Oversampling** (`slider7`): default `1`; declared range/choices `1,4,1`. Slider.
- **Gain smoothing** (`slider8`): default `0`; declared range/choices `0,1,0.00001`. Slider.

## More background from the vendored source

### Transience
Transience is a plugin for enhancing or reducing transients. It works by using two envelopes. One is an envelope follower (short attack, longer decay; roughly follows the peaks of the sound), the other is a user specified envelope (with attack/decay). You can then shape the sound according to the difference between the two, making attacks or decays longer or shorter. The plugin operates in logarithmic space.
[Screenshot](https://i.imgur.com/TgC7n2B.png)

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.0986 | 0.0600 | 1.63× | 1.61–1.72× |
| 512 | 0.0905 | 0.0566 | 1.61× | 1.59–1.69× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.03. Original path: `Basics/Transience.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
