# Filther (Saike)

A dynamic filter and distortion processor with a drawn waveshaper, two filter stages, modulation and feedback.

## Quick start

1. Feed stereo audio to inputs 1/2 and begin with modest drive and feedback.
2. Choose the filter models, cutoff/resonance and routing in the canvas.
3. Shape the transfer curve and introduce envelope or LFO modulation one stage at a time.

## Controls and routing

Inputs 3/4 provide an external sidechain. Many parameters are stored as hidden slider coordinates and edited through the canvas; those coordinates are not physical units. The upstream Filther manual explains the full filter/model matrix.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Nodes negative** (`slider1`): default `2`; declared range/choices `2,9,1`. Canvas / hidden.
- **Nodes positive** (`slider2`): default `2`; declared range/choices `2,9,1`. Canvas / hidden.
- **Pos1x** (`slider3`): default `0.15`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos1y** (`slider4`): default `1.0`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos2x** (`slider5`): default `0.25`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos2y** (`slider6`): default `0.25`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos3x** (`slider7`): default `0.35`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos3y** (`slider8`): default `0.35`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos4x** (`slider9`): default `0.5`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos4y** (`slider10`): default `0.5`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos5x** (`slider11`): default `0.6`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos5y** (`slider12`): default `0.6`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos6x** (`slider13`): default `0.7`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos6y** (`slider14`): default `0.7`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos7x or FB amnt** (`slider15`): default `0.0`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Pos7y or FB time** (`slider16`): default `0.8`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Post-Gain Mod %** (`slider17`): default `0`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **Pos8y or Morph value** (`slider18`): default `0.9`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg1x** (`slider19`): default `0.15`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg1y** (`slider20`): default `1.0`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg2x** (`slider21`): default `0.25`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg2y** (`slider22`): default `0.25`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg3x** (`slider23`): default `0.35`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg3y** (`slider24`): default `0.35`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg4x** (`slider25`): default `0.5`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg4y** (`slider26`): default `0.5`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg5x** (`slider27`): default `0.6`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg5y** (`slider28`): default `0.6`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg6x** (`slider29`): default `0.7`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg6y** (`slider30`): default `0.7`; declared range/choices `0,1,.01`. Canvas / hidden.
- **Neg7x or FB amnt mod** (`slider31`): default `0.8`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **Neg7y or FB time mod** (`slider32`): default `0.8`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **Key follow amount** (`slider33`): default `1.0`; declared range/choices `0,2,.01`. Canvas / hidden.
- **Neg8y or Morph mod %** (`slider34`): default `0.9`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **Toggles (DO NOT AUTOMATE)** (`slider35`): default `16384`; declared range/choices `0,32767,1`. Canvas / hidden.
- **LFO type** (`slider36`): default `0`; declared range/choices `0,26,1{OFF,Cosine,Sine,Cos^2,Sin^2,Ramp up,Ramp down,Exponential,Exp + Atk,1-Exponential,Random,Random Exps,Rand Exps + Atk,Single Exp,Single Exp + Atk,Sixteenth pulse,Eighth pulse,Quarter pulse,Half pulse,Triplet,Sine Pulse,Polyrhythm,Polyrhythm ][,Polyrhythm ]/[,Triangle,Two harmonics,Three harmonics}`. Canvas / hidden.
- **LFO freq** (`slider37`): default `0`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Reset LFO** (`slider38`): default `0`; declared range/choices `0,7,1{No reset,Reset,No reset + temposync,Reset + temposync,No reset + centered,Reset + centered,No reset + temposync + centered,Reset + temposync + centered}`. Canvas / hidden.
- **Modulation range** (`slider39`): default `1`; declared range/choices `0,4,.00001`. Canvas / hidden.
- **Filter2 Type** (`slider40`): default ``; declared range/choices `0,90,1{OFF,LP RC-C,Diode Ladder,Vowel,Karlsen,Karlsen S,WS LP,WS HP,WS BP,Moog (ZDF),Ch. Moog (unstable),Notch,Narsty,Modulator,Phaser (OTA),Phaser (FET),Delay Feedbok,Phase Mangler,MS20 LP lin (ZDF),MS20 BP lin(ZDF),MS20 HP lin (ZDF),MS20 LP NL (ZDF),MS20 BP NL (ZDF),MS20 HP NL (ZDF),Experimental,Rezzy (ZDF),SSM LP NL (ZDF),ch. SSM LP NL (Approx),CEM LP NL (ZDF),SSM LP lin (ZDF),CEM LP lin (ZDF),Sine,FM FB,FM-ish,Broken conn,Broken FB (ZDF),Waspey Lin (ZDF),Waspey LP NL(ZDF),Waspey BP NL (ZDF),SVF LP (ZDF),SVF BP (ZDF),SVF HP (ZDF),SVF Notch (ZDF),SVF Peak (ZDF),Saw (ZDF),SVF w/WS Res (ZDF),Voodoo,Junk (ZDF),Comb,Combres (LP),Combedres (BP),MS20x LP,MS20x BP,MS20x HP,WahDemon,PWM LP,PWM BP,Bitred,Muck,WahDemon2,Crybaby,CrybabyH,CrybabyL,VowelSVF,Monstro,KingOfTone,Modulon,OctaverDown,OctaverUp,Metallic,Frazzle,Phone,T.Modulon,Modulatrix,Vibrato,Spin,Wavefold,Multi-WF,Serge WF,Metallic diff,Sproing,Worp,Crunch,Athena,Resonant 1,Resonant 2,Resonant 3,Resonant 4,Harmonizer x2,Harmonizer x4,Harmonizer x1.5}`. Canvas / hidden.
- **Filter2 Cutoff** (`slider41`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Filter2 Reso** (`slider42`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Filter2 Cutoff Mod %** (`slider43`): default `0`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **Filter2 Reso Mod %** (`slider44`): default `0`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **Dynamics mode** (`slider45`): default `0`; declared range/choices `0,11,1{Dynamic RMS Post Drive,Direct RMS Post Drive,MIDI poly,MIDI legato,MIDI poly vel,MIDI legato vel,Dynamic RMS Pre Drive,Direct RMS Pre Drive,Dynamic RMS Sidechain,Direct RMS Sidechain,ModWheel`. Canvas / hidden.
- **Filter Operation Mode** (`slider46`): default ``; declared range/choices `0,9,1{Stereo,Mono2dangerous,M1S2dangerous,M2S1dangerous,Side,Mid,Stereo Boost,Stereoize,Subtle Stereo,Inverted}`. Canvas / hidden.
- **Filter Type** (`slider47`): default ``; declared range/choices `0,90,1{OFF,LP RC-C,Diode Ladder,Vowel,Karlsen,Karlsen S,WS LP,WS HP,WS BP,Moog (ZDF),Ch. Moog (unstable),Notch,Narsty,Modulator,Phaser (OTA),Phaser (FET),Delay Feedbok,Phase Mangler,MS20 LP lin (ZDF),MS20 BP lin(ZDF),MS20 HP lin (ZDF),MS20 LP NL (ZDF),MS20 BP NL (ZDF),MS20 HP NL (ZDF),Experimental,Rezzy (ZDF),SSM LP NL (ZDF),ch. SSM LP NL (Approx),CEM LP NL (ZDF),SSM LP lin (ZDF),CEM LP lin (ZDF),Sine,FM FB,FM-ish,Broken conn,Broken FB (ZDF),Waspey Lin (ZDF),Waspey LP NL(ZDF),Waspey BP NL (ZDF),SVF LP (ZDF),SVF BP (ZDF),SVF HP (ZDF),SVF Notch (ZDF),SVF Peak (ZDF),Saw (ZDF),SVF w/WS Res (ZDF),Voodoo,Junk (ZDF),Comb,Combres (LP),Combedres (BP),MS20x LP,MS20x BP,MS20x HP,WahDemon,PWM LP,PWM BP,Bitred,Muck,WahDemon2,Crybaby,CrybabyH,CrybabyL,VowelSVF,Monstro,KingOfTone,Modulon,OctaverDown,OctaverUp,Metallic,Frazzle,Phone,T.Modulon,Modulatrix,Vibrato,Spin,Wavefold,Multi-WF,Serge WF,Metallic diff,Sproing,Worp,Crunch,Athena,Resonant 1,Resonant 2,Resonant 3,Resonant 4,Harmonizer x2,Harmonizer x4,Harmonizer x1.5}`. Canvas / hidden.
- **Cutoff** (`slider48`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Reso** (`slider49`): default `0`; declared range/choices `0,1,.001`. Canvas / hidden.
- **FIR resampling and linking mode** (`slider50`): default `0`; declared range/choices `0,16,1{Serial DualDist,Serial DualDist (FIR),Serial,Serial (FIR),Parallel DualDist,Parallel DualDist (FIR),Parallel,Parallel (FIR),Morph DualDist,Morph DualDist (FIR),Morph,Morph (FIR)}`. Canvas / hidden.
- **Pre-Gain/Drive Mod %** (`slider51`): default `0`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **PreGain** (`slider52`): default `0`; declared range/choices `-40,40,.01`. Canvas / hidden.
- **PostGain** (`slider53`): default `0`; declared range/choices `-40,40,.01`. Canvas / hidden.
- **Oversampling** (`slider54`): default `1`; declared range/choices `1,8,1`. Canvas / hidden.
- **Clipping and Inertia** (`slider55`): default `0`; declared range/choices `0,7,1{No clipping,Input clipping,Output clipping,Input/output clipping,No clipping + Inertia,Input clipping + Inertia,Output clipping + Inertia,Input/output clipping + Inertia}`. Canvas / hidden.
- **Waveshaping Mode** (`slider56`): default `0`; declared range/choices `0,5,1{Spline,Tanh,Fast Tanh,None,Sine,Tanh}`. Canvas / hidden.
- **Multipliers** (`slider57`): default `0`; declared range/choices `0,63,1{Attack x1 Decay x1 RMS x1,Attack x8 Decay x1 RMS x1,Attack x4 Decay x1 RMS x1,Attack x32 Decay x1 RMS x1,Attack x1 Decay x8 RMS x1,Attack x8 Decay x8 RMS x1,Attack x4 Decay x8 RMS x1,Attack x32 Decay x8 RMS x1,Attack x1 Decay x4 RMS x1,Attack x8 Decay x4 RMS x1,Attack x4 Decay x4 RMS x1,Attack x32 Decay x4 RMS x1,Attack x1 Decay x32 RMS x1,Attack x8 Decay x32 RMS x1,Attack x4 Decay x32 RMS x1,Attack x32 Decay x32 RMS x1,Attack x1 Decay x1 RMS x8,Attack x8 Decay x1 RMS x8,Attack x4 Decay x1 RMS x8,Attack x32 Decay x1 RMS x8,Attack x1 Decay x8 RMS x8,Attack x8 Decay x8 RMS x8,Attack x4 Decay x8 RMS x8,Attackx32 Decay x8 RMS x8,Attack x1 Decay x4 RMS x8,Attack x8 Decay x4 RMS x8,Attack x4 Decay x4 RMS x8,Attack x32 Decay x4 RMS x8,Attack x1 Decay x32 RMS x8,Attack x8 Decay x32 RMS x8,Attack x4 Decay x32 RMS x8,Attack x32 Decay x32 RMS x8,Attack x1 Decay x1 RMS x4,Attack x8 Decay x1 RMS x4,Attack x4 Decay x1 RMS x4,Attack x32 Decay x1 RMS x4,Attack x1 Decay x8 RMS x4,Attack x8 Decay x8 RMS x4,Attack x4 Decay x8 RMS x4,Attack x32 Decay x8 RMS x4,Attack x1 Decay x4 RMS x4,Attack x8 Decay x4 RMS x4,Attack x4 Decay x4 RMS x4,Attack x32 Decay x4 RMS x4,Attack x1 Decay x32 RMS x4,Attack x8 Decay x32 RMS x4,Attack x4 Decay x32 RMS x4,Attack x32 Decay x32 RMS x4,Attack x1 Decay x1 RMS x32,Attack x8 Decay x1 RMS x32,Attack x4 Decay x1 RMS x32,Attack x32 Decay x1 RMS x32,Attack x1 Decay x8 RMS x32,Attack x8 Decay x8 RMS x32,Attack x4 Decay x8 RMS x32,Attack x32 Decay x8 RMS x32,Attack x1 Decay x4 RMS x32,Attack x8 Decay x4 RMS x32,Attack x4 Decay x4 RMS x32,Attack x32 Decay x4 RMS x32,Attack x1 Decay x32 RMS x32,Attack x8 Decay x32 RMS x32,Attack x4 Decay x32 RMS x32,Attack x32 Decay x32 RMS x3}`. Canvas / hidden.
- **Dynamics** (`slider58`): default `0`; declared range/choices `0,4095,1`. Canvas / hidden.
- **Thresh** (`slider59`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Attack** (`slider60`): default `5`; declared range/choices `0,50,.1`. Canvas / hidden.
- **Decay** (`slider61`): default `5`; declared range/choices `0.1,50,.1`. Canvas / hidden.
- **Filt Cutoff Mod %** (`slider62`): default `0`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **Filt Reso Mod %** (`slider63`): default `0`; declared range/choices `-1,1,.01`. Canvas / hidden.
- **RMS Integration time** (`slider64`): default `.34`; declared range/choices `0.02,40,0.001`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input, 3: left sidechain, 4: right sidechain.

Declared outputs: 1: left output, 2: right output.

## More background from the vendored source

### Filther
Filther is a waveshaping / filterbank plugin that allows for some dynamic processing as well.
[Screenshot](https://imgur.com/GPk7WmN.png)
### Manual
A manual can be found here: [manual](https://joepvanlier.github.io/FiltherManual/)
### Demos
You can find demos of the plugin [soundcloud](https://soundcloud.com/saike/ohnoesitsaboss2/s-zYCOt) and [youtube](https://www.youtube.com/watch?v=-VUckbkJ3EY).
Small tutorial here: [here](https://www.youtube.com/watch?v=jtc8kp57xpI).
### Features:
- Spline waveshaping curve based on placing nodes. Can draw asymmetric curves as well.
- Two non-linear filter modules which can be automated by dynamics from the input signal or a side chain, LFO or envelopes.
- Waveshaping amount can be modulated by input dynamics, LFOs or envelopes.
- Modulators can optionally be triggered by MIDI notes.
- Huge array of filter types (linear filters, analog models, FM, AM filters, reverbs, distortions).
- Feedback section.
- Automatic Gain Control to protect your ears somewhat
Copyright (C) 2019 Joep Vanlier

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.3143 | 0.2459 | 1.28× | 1.27–1.36× |
| 512 | 0.2308 | 0.1606 | 1.42× | 1.38–1.47× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 3.21. Original path: `Filther/Filther.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
