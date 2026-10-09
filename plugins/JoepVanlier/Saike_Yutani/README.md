# Yutani Mono Bass Synth [Saike] (BETA)

A mono/paraphonic bass synthesizer with anti-aliased oscillators, nonlinear filters, envelopes and modulation.

## Quick start

1. Route MIDI notes to the plugin and start with a simple oscillator pair.
2. Set amplitude envelope and filter model/cutoff, then adjust glide.
3. Add modulation, distortion, widening or a wavetable once the basic patch plays.

## Controls and routing

The canvas exposes more useful meanings than the normalized automation values. Active voices, oscillator models and driven filters alter CPU cost.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Oscillator 1 Gain** (`slider1`): default `-9`; declared range/choices `-48, 0, .000001`. Canvas / hidden.
- **Oscillator 1 Semitone** (`slider2`): default `0`; declared range/choices `-36, 36, .000001`. Canvas / hidden.
- **Reset Osc on note** (`slider3`): default `0`; declared range/choices `0, 1, 1`. Canvas / hidden.
- **Oscillator 1 Shape** (`slider4`): default `0`; declared range/choices `0,9,1{Saw,Square,Triangle,Fin,PWM,Comb Sa,Comb Sq,SSaw,Glot,WT}`. Canvas / hidden.
- **Oscillator 2 Gain** (`slider5`): default `-9`; declared range/choices `-48, 0, .000001`. Canvas / hidden.
- **Oscillator 2 Semitone** (`slider6`): default `12`; declared range/choices `-48, 48, .000001`. Canvas / hidden.
- **Hard Sync** (`slider7`): default `1`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Oscillator 2 Shape** (`slider8`): default `0`; declared range/choices `0,9,1{Saw,Square,Triangle,Fin,PWM,Comb Sa,Comb Sq,SSaw,Glot,WT}`. Canvas / hidden.
- **Glide time** (`slider9`): default `.4`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Amp Accent** (`slider10`): default `0`; declared range/choices `0, 1, 1`. Canvas / hidden.
- **Amp Attack** (`slider11`): default `0`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **Amp Decay** (`slider12`): default `0.2`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **Sustain level** (`slider13`): default `1`; declared range/choices `0, 1, .00001`. Canvas / hidden.
- **Pitch envelope level** (`slider14`): default `0`; declared range/choices `-12, 12, .000001`. Canvas / hidden.
- **Pitch Attack** (`slider15`): default `0`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **Pitch Decay** (`slider16`): default `1`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **Fm level** (`slider17`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Active Oscs** (`slider18`): default `1`; declared range/choices `1, 4, 1`. Canvas / hidden.
- **Voice 2 [%]** (`slider19`): default `0`; declared range/choices `0, 1,.00001`. Canvas / hidden.
- **Detune [semitones]** (`slider20`): default `0`; declared range/choices `-12, 12, .00001`. Canvas / hidden.
- **Voice 3 [%]** (`slider21`): default `0`; declared range/choices `0, 1,.00001`. Canvas / hidden.
- **Detune [semitones]** (`slider22`): default `0`; declared range/choices `-12, 12, .00001`. Canvas / hidden.
- **Voice 4 [%]** (`slider23`): default `0`; declared range/choices `0, 1,.00001`. Canvas / hidden.
- **Detune [semitones]** (`slider24`): default `0`; declared range/choices `-12, 12, .00001`. Canvas / hidden.
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
- **SubOsc Shape** (`slider47`): default `0`; declared range/choices `0,4,1`. Canvas / hidden.
- **Distortion level [dB]** (`slider48`): default `0`; declared range/choices `0,48,1`. Canvas / hidden.
- **Warmth** (`slider49`): default `0`; declared range/choices `-12,12,1`. Canvas / hidden.
- **PWM phase** (`slider50`): default `0`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **PWM depth** (`slider51`): default `0.4`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **PWM rate** (`slider52`): default `0.85`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **Feedback** (`slider53`): default `0`; declared range/choices `0, 1, .000001`. Canvas / hidden.
- **Vibrato Amount** (`slider54`): default `0`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **Vibrato frequency** (`slider55`): default `0`; declared range/choices `0,10,.00001`. Canvas / hidden.
- **Sub oscillator Gain** (`slider56`): default `-9`; declared range/choices `-48, 12, .000001`. Canvas / hidden.
- **Sub oscillator Semi** (`slider57`): default `-12`; declared range/choices `-36, 12, .000001`. Canvas / hidden.
- **Noise type** (`slider58`): default `0`; declared range/choices `0,4,1`. Canvas / hidden.
- **Free LFO amount** (`slider59`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Free LFO speed [-]** (`slider60`): default `0`; declared range/choices `0,20,.001`. Canvas / hidden.
- **fine tune FM** (`slider61`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Fix DC** (`slider62`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Filter Inertia [ms]** (`slider63`): default `60`; declared range/choices `0,200,.001`. Canvas / hidden.
- **Oversampling** (`slider64`): default `1`; declared range/choices `1,8,1`. Canvas / hidden.
- **Cutoff env shape** (`slider65`): default `0`; declared range/choices `-1.0, 1.0, 0.001`. Canvas / hidden.
- **Cutoff env shape** (`slider66`): default `0`; declared range/choices `-2.0, 2.0, 0.001`. Canvas / hidden.
- **AP frequency** (`slider71`): default `0.4431`; declared range/choices `0, 1, 0.0001`. Canvas / hidden.
- **AP Feedback** (`slider72`): default `0.17`; declared range/choices `-1, 1, 0.001`. Canvas / hidden.
- **AP saturation** (`slider73`): default `0.25`; declared range/choices `0.01, 0.25, 0.001`. Canvas / hidden.
- **WT1 position** (`slider80`): default `0`; declared range/choices `0,7,0.001`. Canvas / hidden.
- **WT2 position** (`slider81`): default `0`; declared range/choices `0,7,0.001`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input, 3: left input, 4: right input.

Declared outputs: 1: left output, 2: right output, 3: left output, 4: right output.

## More background from the vendored source

### A mono-synth plugin with some analog-emulated filters and modulation options
[Screenshot](https://user-images.githubusercontent.com/19836026/110242823-0739a500-7f58-11eb-9473-8cd214746b13.gif)
### Features:
- Anti-aliased oscillators.
- 14 Filters of which 9 non-linear analog modelled ones, all with their own unique tone. Try driving them!
- Audio-rate modulation options on the filter.
- Velocity, modulation wheel and LFO modulation options.
- Stereo widening effect.
- Noise.
- Distortion module.
- Glide.
- Modwheel, MIDI velocity and pitch bend support.

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
| 64 | 0.9138 | 0.6720 | 1.36× | 1.32–1.36× |
| 512 | 0.8345 | 0.6020 | 1.39× | 1.36–1.41× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.103. Original path: `Yutani/Saike_Yutani.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
