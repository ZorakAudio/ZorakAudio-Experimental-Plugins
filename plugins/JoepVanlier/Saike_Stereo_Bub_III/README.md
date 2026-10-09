# Saike Stereo Bub III Stereoizer

A Stereo Bub variant with vibrato and nonlinear colouring in addition to widening.

## Quick start

1. Set Delay, Strength and Crossover as in the simpler Stereo Bub.
2. Add Vibrato Amount/Speed or Non-linearity gradually.
3. Check Mono after changing the modulation and side blend.

## Controls and routing

This has additional motion compared with Stereo Bub II. An audio-only benchmark excludes the custom canvas.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Delay [ms / 2]** (`slider1`): default `8`; declared range/choices `1,25,1`. Canvas / hidden.
- **Strength [-]** (`slider2`): default `0.3`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Crossover [log(w)]** (`slider3`): default `0.4`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Blend old side level [-]** (`slider4`): default `1`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Check Mono?** (`slider5`): default `0`; declared range/choices `0,1,1{No, Yes}`. Canvas / hidden.
- **Pass side through HPF?** (`slider6`): default `0`; declared range/choices `0,1,1{No,Yes}`. Canvas / hidden.
- **Vibrato Speed [-]** (`slider7`): default `15`; declared range/choices `0,30,.1`. Canvas / hidden.
- **Vibrato Amount [-]** (`slider8`): default `0`; declared range/choices `0,40,.1`. Canvas / hidden.
- **Non-linearity [-]** (`slider9`): default `0`; declared range/choices `0,1,0.0000001`. Canvas / hidden.

## More background from the vendored source

### A basic stereo widener
Similar to Stereo Bub II (see II for a description), but adds vibrato and non-linearity options.
### Features:
- Add stereo to mono audio.
- Control existing stereo in audio.
- Use steep 12-pole crossover filter to keep bass mono.
- Vibrato.
- Non-linearity.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.1727 | 0.0931 | 1.86× | 1.85–1.88× |
| 512 | 0.1704 | 0.0913 | 1.87× | 1.84–1.90× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.08. Original path: `Basics/Saike Stereo Bub III.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
