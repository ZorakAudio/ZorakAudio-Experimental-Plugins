# Saike FM Filter

An audio-rate frequency-modulated filter, with MIDI-pitched modulation and external audio modulation modes.

## Quick start

1. Insert it after an audio source; set Filter type, Cutoff and Resonance.
2. For note-following operation, route the same MIDI to the source and this effect.
3. For external audio modulation, route a modulator to inputs 3/4 and choose the corresponding audio mode.

## Controls and routing

Drive and Post Boost affect level, while Inertia smooths changes. External modulation inputs are optional; ordinary stereo input is 1/2.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Drive (dB)** (`slider1`): default `0`; declared range/choices `-6,48,1`. Canvas / hidden.
- **Post Boost (dB)** (`slider2`): default `0`; declared range/choices `-6,48,1`. Canvas / hidden.
- **Autokill** (`slider3`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Inertia [ms]** (`slider6`): default `60`; declared range/choices `0,200,.001`. Canvas / hidden.
- **Filter type** (`slider12`): default `1`; declared range/choices `0,5,1{Linear,MS-20,Linear x2,Moog,Ladder,303}`. Canvas / hidden.
- **Cutoff** (`slider13`): default `.6`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Resonance** (`slider14`): default `0.7`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Morph** (`slider15`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Bleed** (`slider16`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Morph LFO amount** (`slider17`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Morph LFO speed [Hz]** (`slider18`): default `0`; declared range/choices `0,20,.001`. Canvas / hidden.
- **Morph LFO phase [radian]** (`slider19`): default `0`; declared range/choices `0,30,.001`. Canvas / hidden.
- **Cutoff LFO amount** (`slider20`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Cutoff LFO speed [Hz]** (`slider21`): default `0`; declared range/choices `0,20,.001`. Canvas / hidden.
- **Cutoff LFO phase [radian]** (`slider22`): default `0`; declared range/choices `0,36,.001`. Canvas / hidden.
- **FM mode** (`slider23`): default `0`; declared range/choices `0,5,1{MIDI sin,MIDI square,Self,Self Abs,Audio Stereo 3/4,Audio Mono 3/4}`. Canvas / hidden.
- **FM level** (`slider24`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **FM rate factor** (`slider25`): default ``; declared range/choices `-8,8,1`. Canvas / hidden.
- **FM spread** (`slider26`): default ``; declared range/choices `0,1,.001`. Canvas / hidden.
- **Key Follow** (`slider27`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **FM Cutoff** (`slider28`): default `1`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Envelope Amount** (`slider29`): default `0`; declared range/choices `-1,1,.0001`. Canvas / hidden.
- **Decay [ms]** (`slider30`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Distortion level [dB]** (`slider31`): default `0`; declared range/choices `0,48,1`. Canvas / hidden.
- **Warmth** (`slider32`): default `0`; declared range/choices `-12,12,1`. Canvas / hidden.
- **Oversampling** (`slider60`): default `1`; declared range/choices `1,8,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input, 3: secondary_input_left, 4: secondary_input_right.

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
| 64 | 1.1174 | 0.7674 | 1.46× | 1.45–1.47× |
| 512 | 1.1027 | 0.7426 | 1.48× | 1.47–1.49× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.21. Original path: `FMFilter/FM Filter.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
