# Saike / Nostalgizer / Lo-Fi (BETA)

A lo-fi effect combining a lowpass gate, pitch instability, compander noise and saturation.

## Quick start

1. Feed stereo audio and choose Gate, Follower or Shallow mode.
2. Adjust lpf, slowness and depth to set the gate and detune movement.
3. Add flutter/noise and saturation gradually, then match output level.

## Controls and routing

Noise and free movement can produce activity on quiet input. The source remains marked BETA.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Operating mode** (`slider1`): default `2`; declared range/choices `0,2,1{Gate,Follower,Shallow}`. Canvas / hidden.
- **lpf** (`slider2`): default `0.701`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **slowness [ms]** (`slider3`): default `591.6`; declared range/choices `75,1000,.1`. Canvas / hidden.
- **depth** (`slider4`): default `179.9`; declared range/choices `0,350,.0001`. Canvas / hidden.
- **flutter** (`slider5`): default `0.0`; declared range/choices `0,.05,0.00001`. Canvas / hidden.
- **flutter rate** (`slider6`): default `1.0`; declared range/choices `0,50,.01`. Canvas / hidden.
- **Enable high pass** (`slider7`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **L/R LPG Linked** (`slider8`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **L/R Mod Linked** (`slider9`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Bypass LPG** (`slider10`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **dynamic saturation (0=off)** (`slider11`): default `0.1003`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **saturation asymmetry** (`slider12`): default `0.663367`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **dimension expander (0=off)** (`slider14`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **feedback** (`slider15`): default `0`; declared range/choices `0,.5,.00001`. Canvas / hidden.
- **Input boost** (`slider16`): default `0`; declared range/choices `0,18,.1`. Canvas / hidden.
- **Dry/Wet ratio** (`slider17`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **resonance** (`slider20`): default `0`; declared range/choices `0,.99,.01`. Canvas / hidden.
- **Model noise (0=off)** (`slider30`): default `0.0057`; declared range/choices `0,.05,.0001`. Canvas / hidden.
- **Attack [ms]** (`slider32`): default `9.22`; declared range/choices `5,80,.001`. Canvas / hidden.
- **Decay [ms]** (`slider33`): default `110.666667`; declared range/choices `5,500,.001`. Canvas / hidden.
- **lpg minimum** (`slider34`): default `0.15`; declared range/choices `0,1,.00011`. Canvas / hidden.
- **Use sidechain for LPG** (`slider35`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Show advanced** (`slider60`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input, 3: left lpg sidechain, 4: right lpg sidechain.

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
| 64 | 0.2885 | 0.2163 | 1.34× | 1.31–1.35× |
| 512 | 0.2928 | 0.2155 | 1.35× | 1.32–1.41× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.15. Original path: `Nostalgizer/saike_nostalgizer.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
