# Final Boss (Saike)

A distortion device with allpass feedback, upward dynamics, octaving, modulation and cabinet shaping.

## Quick start

1. Insert on stereo audio and start with low Saturation, Feedback and Oct amp.
2. Choose a cabinet and EQ voicing, then increase one effect section at a time.
3. Adjust modulation frequency/depth and output level while comparing bypass.

## Controls and routing

This combines several nonlinear and stateful stages. Disabling a visible return is not automatically equivalent to removing all internal work.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Saturation** (`slider2`): default `0`; declared range/choices `0,0.25,0.0001`. Canvas / hidden.
- **Freq** (`slider3`): default `5`; declared range/choices `5, 22000, 0.1:log`. Canvas / hidden.
- **Mod Freq** (`slider4`): default `20`; declared range/choices `0.1, 22000, 0.1:log`. Canvas / hidden.
- **Mod Depth Freq** (`slider5`): default `0`; declared range/choices `0, 1, 0.001:log`. Canvas / hidden.
- **Strength** (`slider6`): default `1`; declared range/choices `1,16,1`. Canvas / hidden.
- **Feedback** (`slider7`): default `0`; declared range/choices `-0.99,0.99, 0.00001`. Canvas / hidden.
- **Feedback Mod** (`slider8`): default `0`; declared range/choices `0, 0.5, 0.00001`. Canvas / hidden.
- **Oct LP** (`slider9`): default `1`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Equ** (`slider10`): default `0`; declared range/choices `0,2,1`. Canvas / hidden.
- **Cabinet** (`slider11`): default `0`; declared range/choices `0,14,1`. Canvas / hidden.
- **Oct amp** (`slider12`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Rebuild env** (`slider13`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Octave mode** (`slider14`): default `0`; declared range/choices `0,5,1`. Canvas / hidden.
- **Enable distortion 1** (`slider15`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Enable shifter** (`slider16`): default `0`; declared range/choices `0,3,1{Off,Low,High,Both}`. Canvas / hidden.
- **Frequency Shift** (`slider17`): default `5`; declared range/choices `0.1,10,0.00001`. Canvas / hidden.
- **Enable distortion 2** (`slider18`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Cab split** (`slider19`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Cab drywet** (`slider20`): default `1`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Input gain** (`slider21`): default `0`; declared range/choices `-32,32,0.0001`. Canvas / hidden.
- **Allpass Freq 2** (`slider22`): default `5`; declared range/choices `5, 22000, 0.1:log`. Canvas / hidden.
- **Allpass decoupled** (`slider23`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Output gain** (`slider24`): default `0`; declared range/choices `-32,32,0.0001`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## More background from the vendored source

### A small distortion effect unit for grungy distortion effects
### Features:
- Allpass stack with feedback.
- Upwards compression.
- Octaver.
- Pitch shifting chorus.
- Cabinet filters.
- Frequency shifter based spectral movement.
- A big skull looking mad at you.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.5351 | 0.3919 | 1.38× | 1.37–1.40× |
| 512 | 0.4941 | 0.3506 | 1.41× | 1.38–1.45× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.13. Original path: `FinalBoss/saike_final_boss.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
