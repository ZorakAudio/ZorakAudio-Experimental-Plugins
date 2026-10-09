# Squashman (Saike)

A multiband distortion processor with selectable shapers, envelopes and LFO modulation.

## Quick start

1. Feed stereo audio and set the band count and crossovers in the canvas.
2. Choose one shaper per band and adjust Drive and Gain.
3. Add modulation and compare oversampling while checking output level.

## Controls and routing

Oversampling, active bands and shaper choice materially change work. Crossovers are 24 dB/oct Linkwitz-Riley; the custom canvas edits many hidden automation controls.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Cuts** (`slider1`): default `1`; declared range/choices `0,4,1`. Canvas / hidden.
- **Frequency 1** (`slider2`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 2** (`slider3`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 3** (`slider4`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Frequency 4** (`slider5`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Drive 1 (dB)** (`slider6`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Drive 2 (dB)** (`slider7`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Drive 3 (dB)** (`slider8`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Drive 4 (dB)** (`slider9`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Drive 5 (dB)** (`slider10`): default `0`; declared range/choices `-40,60,.1`. Canvas / hidden.
- **Gain 1 (dB)** (`slider11`): default `0`; declared range/choices `-60,40,.1`. Canvas / hidden.
- **Gain 2 (dB)** (`slider12`): default `0`; declared range/choices `-60,40,.1`. Canvas / hidden.
- **Gain 3 (dB)** (`slider13`): default `0`; declared range/choices `-60,40,.1`. Canvas / hidden.
- **Gain 4 (dB)** (`slider14`): default `0`; declared range/choices `-60,40,.1`. Canvas / hidden.
- **Gain 5 (dB)** (`slider15`): default `0`; declared range/choices `-60,40,.1`. Canvas / hidden.
- **Mode 1** (`slider16`): default `0`; declared range/choices `0,24,1`. Canvas / hidden.
- **Mode 2** (`slider17`): default `0`; declared range/choices `0,24,1`. Canvas / hidden.
- **Mode 3** (`slider18`): default `0`; declared range/choices `0,24,1`. Canvas / hidden.
- **Mode 4** (`slider19`): default `0`; declared range/choices `0,24,1`. Canvas / hidden.
- **Mode 5** (`slider20`): default `0`; declared range/choices `0,24,1`. Canvas / hidden.
- **Modifier1** (`slider21`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Modifier2** (`slider22`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Modifier3** (`slider23`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Modifier4** (`slider24`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Modifier5** (`slider25`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Delay1** (`slider26`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Delay2** (`slider27`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Delay3** (`slider28`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Delay4** (`slider29`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Delay5** (`slider30`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Feedback1** (`slider31`): default `0`; declared range/choices `0,.99,.001`. Canvas / hidden.
- **Feedback2** (`slider32`): default `0`; declared range/choices `0,.99,.001`. Canvas / hidden.
- **Feedback3** (`slider33`): default `0`; declared range/choices `0,.99,.001`. Canvas / hidden.
- **Feedback4** (`slider34`): default `0`; declared range/choices `0,.99,.001`. Canvas / hidden.
- **Feedback5** (`slider35`): default `0`; declared range/choices `0,.99,.001`. Canvas / hidden.
- **Dry/Wet 1** (`slider36`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Dry/Wet 2** (`slider37`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Dry/Wet 3** (`slider38`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Dry/Wet 4** (`slider39`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Dry/Wet 5** (`slider40`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Widen 1** (`slider41`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Widen 2** (`slider42`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Widen 3** (`slider43`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Widen 4** (`slider44`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Widen 5** (`slider45`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Verb 1** (`slider46`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Verb 2** (`slider47`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Verb 3** (`slider48`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Verb 4** (`slider49`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Verb 5** (`slider50`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Gain compensation** (`slider51`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **LFO1 Amount** (`slider53`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **LFO2 Amount** (`slider54`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **LFO3 Amount** (`slider55`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **LFO4 Amount** (`slider56`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **LFO1 Frequency** (`slider57`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **LFO2 Frequency** (`slider58`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **LFO3 Frequency** (`slider59`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **LFO4 Frequency** (`slider60`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Inertia** (`slider61`): default `1`; declared range/choices `0,1,1{No,Yes}`. Canvas / hidden.
- **Master Gain** (`slider62`): default `0`; declared range/choices `-26,26,.1`. Canvas / hidden.
- **Oversampling** (`slider63`): default `1`; declared range/choices `1,4,1`. Canvas / hidden.
- **Fix DC options** (`slider64`): default `1`; declared range/choices `0,2,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## More background from the vendored source

### Squashman
Squashman is a multi-band saturation / distortion plugin that allows modulation of several of its parameters.
[Screenshot](https://i.imgur.com/egp00QC.png)
### Demos
You can find a demo of the plugin [here](https://www.youtube.com/watch?v=mK0xAhq4pK4)
### Features:
- Flexible band count, up to five bands can be used to manipulate sound
- 24 db/oct Linkwitz Riley crossover filters
- Graphical user interface
- Optional high quality oversampling  
- 25 modulatable waveshapers and 4 fixed ones.
- Several modulation sources (4 LFOs, 2 MIDI triggered and/or loopable envelopes).

Thanks to tviler / samuele pizzi / RCJacH / BethHarmon (inflator waveshaper curves).

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.5330 | 0.4804 | 1.10× | 1.08–1.12× |
| 512 | 0.4438 | 0.3958 | 1.12× | 1.11–1.14× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.86. Original path: `Squashman/Squashman.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
