# Saike MIDI ARP (beta)

A pattern-based MIDI arpeggiator with polyphony, octave extension, velocity/CC lanes and randomisation.

## Quick start

1. Place it before a synth and route its MIDI output to the synth.
2. Draw or load a pattern; an empty pattern does not produce the intended sequence.
3. Hold a MIDI chord, start the host and choose speed, polyphony and sync behaviour.

## Controls and routing

This primarily transforms MIDI rather than generating audio. The benchmark programs the same four active steps into both guests so note generation is exercised; it does not pretend the empty default bank is active.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Current speed** (`slider1`): default `4`; declared range/choices `-6,16,1`. Canvas / hidden.
- **Current pattern** (`slider2`): default `0`; declared range/choices `0,63,1`. Canvas / hidden.
- **Max Polyphony** (`slider3`): default `5`; declared range/choices `1,12,1`. Canvas / hidden.
- **Poly Mode** (`slider4`): default `0`; declared range/choices `0,5,1,{No extend,Repeat,Back_Forth}`. Canvas / hidden.
- **Extra octaves** (`slider5`): default `0`; declared range/choices `0,2,1`. Canvas / hidden.
- **Minimum Velocity** (`slider6`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Velocity** (`slider7`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Minimum Modwheel** (`slider8`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Modwheel** (`slider9`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Assignable CC1** (`slider10`): default `0`; declared range/choices `0,119,1`. Canvas / hidden.
- **Minimum Assignable CC1** (`slider11`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Assignable CC1** (`slider12`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Assignable CC2** (`slider13`): default `0`; declared range/choices `0,119,1`. Canvas / hidden.
- **Minimum Assignable CC2** (`slider14`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Assignable CC2** (`slider15`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Assignable CC3** (`slider16`): default `0`; declared range/choices `0,119,1`. Canvas / hidden.
- **Minimum Assignable CC3** (`slider17`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Assignable CC3** (`slider18`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Assignable CC4** (`slider19`): default `0`; declared range/choices `0,119,1`. Canvas / hidden.
- **Minimum Assignable CC4** (`slider20`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Assignable CC4** (`slider21`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Dummy for undo** (`slider22`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Swing** (`slider23`): default `0`; declared range/choices `-50,50,1`. Canvas / hidden.
- **In channel** (`slider24`): default `0`; declared range/choices `0,16,1`. Canvas / hidden.
- **Out channel** (`slider25`): default `1`; declared range/choices `1,16,1`. Canvas / hidden.
- **Assignable CC5** (`slider26`): default `0`; declared range/choices `0,119,1`. Canvas / hidden.
- **Minimum Assignable CC5** (`slider27`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Assignable CC5** (`slider28`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Assignable CC6** (`slider29`): default `0`; declared range/choices `0,119,1`. Canvas / hidden.
- **Minimum Assignable CC6** (`slider30`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Assignable CC6** (`slider31`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Assignable CC7** (`slider32`): default `0`; declared range/choices `0,119,1`. Canvas / hidden.
- **Minimum Assignable CC7** (`slider33`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Assignable CC7** (`slider34`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Assignable CC8** (`slider35`): default `0`; declared range/choices `0,119,1`. Canvas / hidden.
- **Minimum Assignable CC8** (`slider36`): default `1`; declared range/choices `0,127,1`. Canvas / hidden.
- **Maximum Assignable CC8** (`slider37`): default `127`; declared range/choices `0,128,1`. Canvas / hidden.
- **Enable speed override** (`slider38`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Loop length** (`slider39`): default `32`; declared range/choices `2,64,1`. Canvas / hidden.
- **CC which resets MIDI position** (`slider40`): default `102`; declared range/choices `0,255,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## More background from the vendored source

### A small utility JSFX to arpeggiate midi chords.
Program patterns and play chords. The JSFX will then play the notes according to that note pattern.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.0813 | 0.0741 | 1.08× | 1.06–1.12× |
| 512 | 0.0761 | 0.0701 | 1.09× | 1.06–1.14× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.44. Original path: `saike_midi_arp/saike_midi_arp.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
