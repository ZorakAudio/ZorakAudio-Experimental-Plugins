# Saike Nuker (EARLY ALPHA - DO NOT USE)

An early experimental distortion tool combining even/odd shaping, octaving, dimension expansion and EQ.

## Quick start

1. Treat this as a preserved development prototype: the source title explicitly says EARLY ALPHA - DO NOT USE.
2. If investigating it locally, begin by understanding Ceiling and Postgain before enabling octaving or expansion.
3. Use the reference below to identify the experimental parameters.

## Controls and routing

Packaging and benchmarking do not promote this prototype to a supported production processor.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **ceiling (dB)** (`slider1`): default `-25`; declared range/choices `-36,36,1`. Slider.
- **postgain (dB)** (`slider2`): default `13`; declared range/choices `-36,36,1`. Slider.
- **Even** (`slider3`): default `0.66`; declared range/choices `0,1,.0001`. Slider.
- **Odd** (`slider4`): default `0.66`; declared range/choices `0,1,.0001`. Slider.
- **Fix bug** (`slider9`): default `0`; declared range/choices `0,1,1`. Slider.
- **Octaver** (`slider10`): default `.3604`; declared range/choices `0,1,.0001`. Slider.
- **dimension expander** (`slider20`): default `0.00601`; declared range/choices `0.001,.02,.000001`. Slider.
- **warmth (dB)** (`slider30`): default `6.2`; declared range/choices `-12, 12, .000001`. Slider.
- **Dip (dB)** (`slider40`): default `0`; declared range/choices `-36,0,.001`. Slider.
- **Attack** (`slider41`): default `0.5`; declared range/choices `0,1,.0001`. Slider.
- **Release** (`slider42`): default `0.5`; declared range/choices `0,1,.00001`. Slider.
- **Eq Curve** (`slider50`): default `0`; declared range/choices `0,14,1`. Slider.

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
| 64 | 0.4616 | 0.3058 | 1.50× | 1.49–1.52× |
| 512 | 0.4455 | 0.2975 | 1.51× | 1.48–1.53× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.04. Original path: `Nuker/saike_nuker.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
