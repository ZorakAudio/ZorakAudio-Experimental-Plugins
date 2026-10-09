# Saike Reflectosaurus (beta)

A node-based creative delay/reverb network with per-node feedback, filtering and modulation.

## Quick start

1. Feed audio and start with a small number of enabled delay nodes.
2. Move a node horizontally for time and vertically for level; adjust its radius for feedback.
3. Connect nodes, shape their filter arcs and introduce tempo sync or a reverb node.

## Controls and routing

Mute unused nodes to reduce work. The wet default is 100%; choose the dry/wet balance appropriate to an insert or return. Node positions and routes are edited in the canvas; the upstream PDF contains the full workflow.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Dry/Wet** (`slider1`): default `1.0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Mode** (`slider2`): default `0`; declared range/choices `0,1,6{Off,Fourth,Third,Fifth,Eight,Sixth,Tonal`. Canvas / hidden.
- **Snap** (`slider3`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Global LP Cutoff** (`slider4`): default `1`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Global HP Cutoff** (`slider5`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Global LP Resonance** (`slider6`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **X1** (`slider10`): default `0.09`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y1** (`slider11`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X2** (`slider12`): default `0.18`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y2** (`slider13`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X3** (`slider14`): default `0.27`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y3** (`slider15`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X4** (`slider16`): default `0.36`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y4** (`slider17`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X5** (`slider18`): default `0.45`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y5** (`slider19`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X6** (`slider20`): default `0.55`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y6** (`slider21`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X7** (`slider22`): default `0.63`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y7** (`slider23`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X8** (`slider24`): default `0.73`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y8** (`slider25`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X9** (`slider26`): default `0.82`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y9** (`slider27`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X10** (`slider28`): default `.91`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y10** (`slider29`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **X11** (`slider30`): default `1.0`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Y11** (`slider31`): default `0.6`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Feedback 1** (`slider32`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 2** (`slider33`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 3** (`slider34`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 4** (`slider35`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 5** (`slider36`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 6** (`slider37`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 7** (`slider38`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 8** (`slider39`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 9** (`slider40`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 10** (`slider41`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Feedback 11** (`slider42`): default `0`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **shine** (`slider43`): default `0`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **Pan 1** (`slider44`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 2** (`slider45`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 3** (`slider46`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 4** (`slider47`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 5** (`slider48`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 6** (`slider49`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 7** (`slider50`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 8** (`slider51`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 9** (`slider52`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 10** (`slider53`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Pan 11** (`slider54`): default `0.5`; declared range/choices `0,1,.00000001`. Canvas / hidden.
- **Glide time** (`slider61`): default `0`; declared range/choices `0,100,.001`. Canvas / hidden.
- **Force Enabled (overrides automute)** (`slider62`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Inertia** (`slider63`): default `50`; declared range/choices `0,500,.001`. Canvas / hidden.
- **Gain** (`slider64`): default `0`; declared range/choices `-32,32,.001`. Canvas / hidden.

## More background from the vendored source

### A flexible delay plugin for setting up complex delays and reverbs.
[Screenshot](https://raw.githubusercontent.com/JoepVanlier/JSFX/master/Reflectosaurus_Manual/Overview.png)
### Manual
A full manual can be found here: [manual](https://github.com/JoepVanlier/JSFX/raw/master/Reflectosaurus_Manual/Reflectosaurus_Manual.pdf)
### Demos
You can find demos of the plugin [here](https://www.youtube.com/watch?v=47L9bysgIiA) and [here](https://www.youtube.com/watch?v=pUu3h21yARY).
### Features:
- Up to 10 node delay.
- Positive, negative and allpass delay.
- Delay filtering (LPF w/ resonance, HPF).
- Various delay saturation algorithms.
- Delay sends.
- Two reverberation algorithms (FFT-based and allpass).
- Delay time pitch tracking based on MIDI input.
- Granular resynthesis.
- Pitch shifting.
- Side chain compressing the delays.
- A decent selection of presets.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 1.9891 | 1.8889 | 1.05× | 1.02–1.07× |
| 512 | 1.6174 | 1.5690 | 1.02× | 1.00–1.07× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.106. Original path: `Reflectosaurus/Reflectosaurus.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
