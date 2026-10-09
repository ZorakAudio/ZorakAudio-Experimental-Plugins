# Saike Tight Compressor

A dynamics compressor with detector topology, knee, stereo linking, sidechain and oversampling.

## Quick start

1. Feed stereo audio to inputs 1/2; use inputs 3/4 if an external sidechain is wanted.
2. Set Threshold and Ratio, then shape Attack, Release and Knee.
3. Compare topology, Link Stereo and Dry/Wet; use Post-Gain or auto make-up to match level.

## Controls and routing

Attack/Release and Ratio use normalized source mappings rather than conventional millisecond/ratio labels. Oversampling changes processing cost.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Threshold (dB)** (`slider1`): default `-20`; declared range/choices `-40,0,.1`. Slider.
- **Ratio (-)** (`slider2`): default `.85`; declared range/choices `0,2,.0001`. Slider.
- **Attack (-)** (`slider3`): default `0.1`; declared range/choices `0,1,.00001`. Slider.
- **Release (-)** (`slider4`): default `0.1`; declared range/choices `0,1,.00001`. Slider.
- **Knee (dB)** (`slider5`): default `0`; declared range/choices `0,30,0.1`. Slider.
- **Auto make-up gain** (`slider6`): default `0`; declared range/choices `0,1,1{No,Yes}`. Slider.
- **Pre-Gain** (`slider7`): default `0`; declared range/choices `-24,12,0.01`. Slider.
- **Post-Gain** (`slider8`): default `0`; declared range/choices `-24,12,0.01`. Slider.
- **Scrolls?** (`slider9`): default `0`; declared range/choices `0,1,1{No,Yes}`. Slider.
- **Topology** (`slider10`): default `0`; declared range/choices `0,4,1{LogDetect (Smoother),LinDetect (Harsher),LogDetect Non-smooth,LinDetect Non-smooth}`. Slider.
- **Dry/Wet** (`slider11`): default `1`; declared range/choices `0,1,.00001`. Slider.
- **Oversampling factor** (`slider12`): default `1`; declared range/choices `1,8,1`. Slider.
- **Link Stereo** (`slider13`): default `1`; declared range/choices `0,1,1{No,Yes}`. Slider.
- **Operating Mode** (`slider14`): default `0`; declared range/choices `0,1,1{Normal (1-2),Sidechain (3-4)}`. Slider.

Declared inputs: 1: Input L, 2: Input R, 3: Sidechain L, 4: Sidechain R.

Declared outputs: 1: Output L, 2: Output R.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.1389 | 0.0952 | 1.45× | 1.43–1.47× |
| 512 | 0.1337 | 0.0931 | 1.44× | 1.40–1.45× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.23. Original path: `Basics/Tight_Compressor.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
