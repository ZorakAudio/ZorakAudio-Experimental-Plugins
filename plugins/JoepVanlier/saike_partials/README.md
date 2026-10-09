# Partials (Saike)

A modal resonator effect that turns audio impulses or loaded samples into pitched material.

## Quick start

1. Feed audio and enable a few pitches on the canvas keyboard, or enable MIDI mode and play notes.
2. Choose a physical/material model and adjust decay and partial balance.
3. Use sampling mode only after loading samples onto the pads; add spin or envelopes later.

## Controls and routing

Frequency-domain and time-domain modes have very different cost and behaviour. The latter permits feedback. MIDI and sample-driven modes require the corresponding input; default continuous audio is only one workload.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **model** (`slider1`): default `0`; declared range/choices `0,12,1{Metal,Tube,Beating,Beam open,Beam clamped,Membrane,Marimba,Pan,Voice male,Voice female,Custom,Custom_Mem,Custom_Mem2`. Canvas / hidden.
- **Inverse Brightness** (`slider2`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Relative position** (`slider3`): default `0.1`; declared range/choices `0.0001,0.999,0.001`. Canvas / hidden.
- **Damping** (`slider4`): default `0.1`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Frequency Dependent Damping** (`slider5`): default `-3.5`; declared range/choices `-6,-1,0.0001`. Canvas / hidden.
- **Inharmonic** (`slider6`): default `-3.5`; declared range/choices `-4, 0, 0.0001`. Canvas / hidden.
- **Stiffness** (`slider7`): default `4.6`; declared range/choices `2,6,0.0001`. Canvas / hidden.
- **Stiffness Exponent** (`slider8`): default `2.3`; declared range/choices `1,3.0,.0001`. Canvas / hidden.
- **Base note** (`slider10`): default `0`; declared range/choices `-12,12,1`. Canvas / hidden.
- **Forced feedback (TD only)** (`slider11`): default `1`; declared range/choices `0,15,0.0001`. Canvas / hidden.
- **Partials** (`slider12`): default `32`; declared range/choices `16,64,16`. Canvas / hidden.
- **Stereo-ize** (`slider13`): default `1`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Position velocity sensitivity** (`slider14`): default `0`; declared range/choices `-1,1,0.000001`. Canvas / hidden.
- **Damping velocity sensitivity** (`slider15`): default `0`; declared range/choices `-4,4,0.000001`. Canvas / hidden.
- **Frequency dependent damping velocity sensitivity** (`slider16`): default `-0.75`; declared range/choices `-5,5,0.000001`. Canvas / hidden.
- **Inharmonicity velocity sensitivity** (`slider17`): default `0`; declared range/choices `-4,4,0.000001`. Canvas / hidden.
- **Midi note 1** (`slider19`): default `45`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 2** (`slider20`): default `52`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 3** (`slider21`): default `60`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 4** (`slider22`): default `64`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 5** (`slider23`): default `48`; declared range/choices `0,127,1`. Canvas / hidden.
- **fft size** (`slider24`): default `1`; declared range/choices `0,4,1`. Canvas / hidden.
- **Spin freq** (`slider25`): default `2`; declared range/choices `0,20,0.01`. Canvas / hidden.
- **Spin depth** (`slider26`): default `0.1`; declared range/choices `0,4,0.01`. Canvas / hidden.
- **Filter Cutoff** (`slider30`): default `0.4`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Filter Cutoff Velocity Sensitivity** (`slider31`): default `0.25`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Filter Envelope** (`slider34`): default `0.5`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Filter Envelope Velocity Sensitivity** (`slider35`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Filter Attack** (`slider36`): default `0`; declared range/choices `0,2,0.0001`. Canvas / hidden.
- **Filter Decay** (`slider37`): default `1`; declared range/choices `0,3,0.0001`. Canvas / hidden.
- **Filter Release** (`slider38`): default `1.5`; declared range/choices `0,3,0.0001`. Canvas / hidden.
- **Filter Sustain** (`slider39`): default `0.3`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Damping release mod** (`slider40`): default `0`; declared range/choices `-4, 4, 0.000001`. Canvas / hidden.
- **Frequency dependent damping release mod** (`slider41`): default `0`; declared range/choices `-5, 5, 0.000001`. Canvas / hidden.
- **Inharmonicity release mod** (`slider42`): default `0`; declared range/choices `-4,4,0.00000001`. Canvas / hidden.
- **Forced feedback Velocity Sensitivity (USE AT YOUR OWN RISK)** (`slider43`): default `1`; declared range/choices `-15,15,0.0001`. Canvas / hidden.
- **Modulation** (`slider44`): default `0`; declared range/choices `0,4,0.01`. Canvas / hidden.
- **Modulation velocity sensitivity** (`slider45`): default `0`; declared range/choices `-4,4,0.01`. Canvas / hidden.
- **Modulation 2** (`slider46`): default `0`; declared range/choices `0,4,0.01`. Canvas / hidden.
- **Modulation 2 velocity sensitivity** (`slider47`): default `0`; declared range/choices `-4,4,0.01`. Canvas / hidden.
- **Follow note** (`slider51`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Glide [10-1000ms]** (`slider52`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Legacy mode** (`slider53`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Brightness vel** (`slider54`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Impulse mode** (`slider55`): default `3`; declared range/choices `0,5.99,1`. Canvas / hidden.
- **Attack** (`slider56`): default `0`; declared range/choices `0,2,0.0001`. Canvas / hidden.
- **Decay** (`slider57`): default `1`; declared range/choices `0,3,0.0001`. Canvas / hidden.
- **Release** (`slider58`): default `1.5`; declared range/choices `0,3,0.0001`. Canvas / hidden.
- **Sustain** (`slider59`): default `0.3`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Use Envelopes** (`slider60`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Large pitch bend** (`slider61`): default `0`; declared range/choices `-24,24,0.000001`. Canvas / hidden.
- **Display type** (`slider62`): default `0`; declared range/choices `0,1,1{Linear,Logarithmic}`. Canvas / hidden.
- **STFT** (`slider63`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Midi note 6** (`slider64`): default `45`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 7** (`slider65`): default `52`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 8** (`slider66`): default `60`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 9** (`slider67`): default `60`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 10** (`slider68`): default `64`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 11** (`slider69`): default `48`; declared range/choices `0,127,1`. Canvas / hidden.
- **Midi note 12** (`slider70`): default `48`; declared range/choices `0,127,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## More background from the vendored source

### An effect which simulates different materials
This effect takes both audio and MIDI input. Based on the model selected the incoming audio will excite
a number of resonators that produce particular sounds. Up to 4 note polyphony is supported.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.6290 | 0.5411 | 1.18× | 1.11–1.19× |
| 512 | 0.5193 | 0.4180 | 1.24× | 1.23–1.26× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.68. Original path: `partials/saike_partials.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
