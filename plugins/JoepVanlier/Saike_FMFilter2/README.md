# Saike FM Filter 2

A filter/modulation effect built around Yutani-style nonlinear models, envelopes, LFOs and distortion.

## Quick start

1. Feed stereo audio and choose a filter model in the custom canvas.
2. Set cutoff, resonance and drive; choose audio- or MIDI-driven envelopes.
3. Route MIDI when using MIDI envelope modes and add LFO modulation gradually.

## Controls and routing

Deprecated and placeholder sliders retain compatibility identities; they are not useful controls. MIDI modes need note events even though this is an audio effect.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Amplitude Envelope Mode** (`slider10`): default `0`; declared range/choices `0,3,1{Off,MIDI Legato,Threshold,MIDI Triggered,Proportional}`. Canvas / hidden.
- **Filter Envelope Mode** (`slider11`): default `0`; declared range/choices `0,4,1{Off,MIDI Legato,Threshold,MIDI Triggered,Proportional}`. Canvas / hidden.
- **reset_fm_on_release** (`slider12`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Filter type** (`slider25`): default `1`; declared range/choices `0,28,1{Linear,MS-20,Linear x2,Moog,Ladder,303,MS-20 asym,DblRes,DualPeak,TriplePeak,svf nl 2p,svf nl 4p,svf nl 2p inc,svf nl 4p inc,rectified resonance,Steiner,SteinerA,Muck,Pill2p,Pill4p,Pill2p Aggro,Pill4p Aggro,Pill2p Stacc,Pill4p Stacc,Ladder3,Ladder6,HLadder,SVF2,SVF4}`. Canvas / hidden.
- **Filter Drive (dB)** (`slider26`): default `0`; declared range/choices `-32,48,1`. Canvas / hidden.
- **Post Boost (dB)** (`slider27`): default `0`; declared range/choices `-6,48,1`. Canvas / hidden.
- **Cutoff** (`slider28`): default `.6`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Resonance** (`slider29`): default `0.7`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Morph** (`slider30`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Morph LFO amount** (`slider31`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Morph LFO speed [-]** (`slider32`): default `0`; declared range/choices `0,20,.001`. Canvas / hidden.
- **Morph LFO phase** (`slider33`): default `0`; declared range/choices `-1,1,.001`. Canvas / hidden.
- **Cutoff LFO amount** (`slider34`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Cutoff LFO speed [-]** (`slider35`): default `0`; declared range/choices `0,20,.001`. Canvas / hidden.
- **Cutoff LFO phase** (`slider36`): default `0`; declared range/choices `-1,1,.001`. Canvas / hidden.
- **FM mode** (`slider37`): default `0`; declared range/choices `0,5,1{MIDI sin,MIDI square,Self,Self Abs,Audio Stereo 3/4,Audio Mono 3/4}`. Canvas / hidden.
- **FM level** (`slider38`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **FM rate factor** (`slider39`): default ``; declared range/choices `-8,8,1`. Canvas / hidden.
- **FM spread** (`slider40`): default ``; declared range/choices `0,1,.001`. Canvas / hidden.
- **Key Follow** (`slider41`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **FM Cutoff** (`slider42`): default `1`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Envelope Amount** (`slider43`): default `0`; declared range/choices `-1,1,.0001`. Canvas / hidden.
- **Cutoff Attack** (`slider44`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Cutoff Decay** (`slider45`): default `0.5`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Cutoff Sustain** (`slider46`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Cutoff lower threshold (RMS mode only)** (`slider47`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Distortion level [dB]** (`slider48`): default `0`; declared range/choices `0,48,1`. Canvas / hidden.
- **Warmth** (`slider49`): default `0`; declared range/choices `-12,12,1`. Canvas / hidden.
- **Amplitude lower threshold (RMS mode only)** (`slider50`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Feedback** (`slider53`): default `0`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **Amplitude Envelope Attack** (`slider55`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Amplitude Envelopep Decay** (`slider56`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Amplitude Envelope Sustain** (`slider57`): default `0`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Pitch bend range** (`slider58`): default `0`; declared range/choices `0,12,1`. Canvas / hidden.
- **Free LFO amount** (`slider59`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Free LFO speed [-]** (`slider60`): default `0`; declared range/choices `0,20,.001`. Canvas / hidden.
- **fine tune FM** (`slider61`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Fix DC** (`slider62`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Filter Inertia [ms]** (`slider63`): default `60`; declared range/choices `0,200,.001`. Canvas / hidden.
- **Oversampling** (`slider64`): default `1`; declared range/choices `1,8,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input, 3: left input, 4: right input.

Declared outputs: 1: left output, 2: right output, 3: left output, 4: right output.

## More background from the vendored source

### An FM filter plugin
[Screenshot](https://user-images.githubusercontent.com/19836026/110242715-998d7900-7f57-11eb-8c6e-48b825b8f47e.gif)
### Features:
- Anti-aliased oscillators.
- 15 filters, from well behaved linear models, to gnarly analog modelled nastiness.
- Audio and MIDI controllable filters.
- Audio and MIDI controllable gate.
- Three LFOs.
- Modwheel and MIDI velocity support.
- Stereo widening effect.
- Distortion module.

Attribution: Moog filter implementation was based on the paper:
S. D'Angelo and V. Vaelimaeki, "Generalized Moog Ladder Filter: Part II - Explicit Non linear Model through a Novel Delay-Free
Loop Implementation Method". IEEE Trans. Audio,Speech, and Lang. Process., vol. 22, no. 12, pp. 1873-1883, December 2014.
303 emulation is Copyright (c) 2012 Dominique Wurtz (www.blaukraut.info)
minBLEP methodology Eli Brandt, "Hard Sync Without Aliasing"

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.8296 | 0.5726 | 1.46× | 1.45–1.46× |
| 512 | 0.7945 | 0.5408 | 1.47× | 1.46–1.48× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.23. Original path: `Yutani/Saike_FMFilter2.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
