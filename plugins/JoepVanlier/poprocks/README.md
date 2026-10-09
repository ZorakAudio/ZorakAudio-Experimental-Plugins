# Pop rocks (Saike)

A stochastic crackle/noise generator with timing distributions and optional vowel colouring or input multiplication.

## Quick start

1. Start with a low Postgain and adjust Rate and Slow down for event density.
2. Use the Uniform and Power law time constants to change the event texture.
3. Enable Vowelize and change Vowel position/resonance, or compare Multiply mode with incoming audio.

## Controls and routing

It can generate noise independently of input. Multiply changes how that generated texture interacts with the input; do not treat an empty-input test as ordinary idle audio.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Rate** (`slider1`): default `1.27`; declared range/choices `0,5,.0001`. Slider.
- **Slow down** (`slider2`): default `0.57`; declared range/choices `0,3`. Slider.
- **Uniform time constant** (`slider3`): default `4.55`; declared range/choices `0,5`. Slider.
- **Power law time constant** (`slider4`): default `0.55`; declared range/choices `0,5`. Slider.
- **Multiply?** (`slider5`): default `1`; declared range/choices `0,1,{No,Yes}`. Slider.
- **Vowelize?** (`slider6`): default `1`; declared range/choices `0,1,1{No,Yes}`. Slider.
- **Vowel position** (`slider7`): default `0.45`; declared range/choices `0,1,.0001`. Slider.
- **Vowel resonance** (`slider8`): default `0.77`; declared range/choices `0,1,.0001`. Slider.
- **Pregain [dB]** (`slider9`): default `-4.11`; declared range/choices `-12,6,.00001`. Slider.
- **Postgain [dB]** (`slider10`): default `0`; declared range/choices `-12,6,.00001`. Slider.
- **Inertia (ms)** (`slider11`): default `0`; declared range/choices `0,500,.001`. Slider.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.3613 | 0.2279 | 1.59× | 1.56–1.60× |
| 512 | 0.3481 | 0.2269 | 1.56× | 1.53–1.57× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.02. Original path: `Poprocks/poprocks.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
