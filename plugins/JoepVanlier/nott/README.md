# Saike Not OTT (ALPHA)

An experimental multiband compressor with per-band timing and upward/downward dynamics controls.

## Quick start

1. Feed stereo audio and set Cuts and crossover positions in the canvas.
2. Adjust one band at a time, starting with Attack and Release.
3. Compare the processed level against bypass before increasing compression intensity.

## Controls and routing

This source is marked ALPHA. Hidden advanced controls and normalized timing values belong to its custom interface; do not interpret normalized slider values as milliseconds.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Cuts** (`slider1`): default `2`; declared range/choices `0,4,1`. Canvas / hidden.
- **Frequency 1** (`slider2`): default `0.213`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 2** (`slider3`): default `0.609`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 3** (`slider4`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 4** (`slider5`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Attack 1** (`slider6`): default `0.5672`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Attack 2** (`slider7`): default `0.6619`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Attack 3** (`slider8`): default `0.8111`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Attack 4** (`slider9`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Attack 5** (`slider10`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Release 1** (`slider11`): default `0.747`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Release 2** (`slider12`): default `0.9203`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Release 3** (`slider13`): default `0.9203`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Release 4** (`slider14`): default `0.9203`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Release 5** (`slider15`): default `0.9203`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Lower Ratio 1** (`slider16`): default `2`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Lower Ratio 2** (`slider17`): default `1.1396`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Lower Ratio 3** (`slider18`): default `1.1396`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Lower Ratio 4** (`slider19`): default `1.1396`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Lower Ratio 5** (`slider20`): default `1.1396`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Upper Ratio 1** (`slider21`): default `2`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Upper Ratio 2** (`slider22`): default `1.9543`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Upper Ratio 3** (`slider23`): default `1.9543`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Upper Ratio 4** (`slider24`): default `1.9543`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Upper Ratio 5** (`slider25`): default `1.9543`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Lower Threshold 1** (`slider26`): default `-40.8`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Lower Threshold 2** (`slider27`): default `-41.8`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Lower Threshold 3** (`slider28`): default `-40.8`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Lower Threshold 4** (`slider29`): default `-40.8`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Lower Threshold 5** (`slider30`): default `-40.8`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Upper Threshold 1** (`slider31`): default `-35.5`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Upper Threshold 2** (`slider32`): default `-30.2`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Upper Threshold 3** (`slider33`): default `-33.8`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Upper Threshold 4** (`slider34`): default `-33.8`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Upper Threshold 5** (`slider35`): default `-33.8`; declared range/choices `-96,0,.0001`. Canvas / hidden.
- **Gain 1** (`slider36`): default `0`; declared range/choices `-48,48,0.00001`. Canvas / hidden.
- **Gain 2** (`slider37`): default `0`; declared range/choices `-48,48,0.00001`. Canvas / hidden.
- **Gain 3** (`slider38`): default `0`; declared range/choices `-48,48,0.00001`. Canvas / hidden.
- **Gain 4** (`slider39`): default `0`; declared range/choices `-48,48,0.00001`. Canvas / hidden.
- **Gain 5** (`slider40`): default `0`; declared range/choices `-48,48,0.00001`. Canvas / hidden.
- **Couple L/R?** (`slider41`): default `1`; declared range/choices `0,1,{False,True}`. Canvas / hidden.
- **DC offset filter** (`slider42`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Input gain** (`slider43`): default `0`; declared range/choices `-32,32,.001`. Slider.
- **Output gain** (`slider44`): default `0`; declared range/choices `-32,32,.001`. Slider.
- **Dry/Wet** (`slider45`): default `1`; declared range/choices `-1,1,.00001`. Slider.
- **Output limiting** (`slider46`): default `0`; declared range/choices `0,3,1{None,Clip,Soft,ClipAA}`. Canvas / hidden.
- **Gate level [dB]** (`slider47`): default `-280`; declared range/choices `0,-340,.0001`. Canvas / hidden.
- **Epsilon** (`slider60`): default `-4`; declared range/choices `-10,-2,.0001`. Canvas / hidden.
- **FIR mode** (`slider61`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Advanced** (`slider62`): default `advanced`; declared range/choices `0,1,1`. Canvas / hidden.
- **FIR Quality** (`slider63`): default `0`; declared range/choices `0,2,{Normal,High,Ultra}`. Canvas / hidden.
- **Gain Compensation (experimental)** (`slider64`): default `0`; declared range/choices `0,2,1{OFF,active adjustment,fixed}`. Slider.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output 1, 2: right output 1.

## More background from the vendored source

### Multiband Compressor.
Compressor design based on: Giannoulis et al, "Digital Dynamic Range Compressor Design - A Tutorial and Analysis", Journal of the Audio Engineering Society 60(6)
Filters are basic Linkwitz Riley filters using phase matching to maintain phase coherence for unfiltered bands.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.4281 | 0.2705 | 1.58× | 1.51–1.59× |
| 512 | 0.4069 | 0.2476 | 1.64× | 1.62–1.68× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.16. Original path: `NOTT/nott.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
