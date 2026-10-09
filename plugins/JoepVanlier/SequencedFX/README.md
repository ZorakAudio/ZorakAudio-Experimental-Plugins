# Saike SEQS (Sequenced FX) (beta)

SEQS: a graphical effect sequencer for stutters, slowdowns, modulation, filtering and other transformations.

## Quick start

1. Feed audio, create effect blocks in a pattern and enable Playback.
2. Choose speed and host/free/MIDI synchronization.
3. Adjust a block's effect parameters and add macro modulation; use additional patterns for variation.

## Controls and routing

The default pattern/playback state may leave the transformations inactive. An empty pattern measures the baseline path, not an active fifteen-effect sequence.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Current speed** (`slider1`): default `4`; declared range/choices `-6,16,1`. Canvas / hidden.
- **Current pattern** (`slider2`): default `0`; declared range/choices `0,63,1`. Canvas / hidden.
- **Shuffle amount** (`slider6`): default `0`; declared range/choices `-50,50,1`. Canvas / hidden.
- **Playback enabled** (`slider7`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Record enabled** (`slider8`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Chorus enabled** (`slider9`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Reset enabled** (`slider10`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Slowdown enabled** (`slider11`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Dynamic slowdown enabled** (`slider12`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Retrigger enabled** (`slider13`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Reverse enabled** (`slider14`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Gate enabled** (`slider15`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Filter enabled** (`slider16`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Reverb enabled** (`slider17`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Degrade enabled** (`slider18`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Tapestop enabled** (`slider19`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Karplus enabled** (`slider20`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Pitch shifter enabled** (`slider21`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Modulation enabled** (`slider22`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Filter enabled** (`slider23`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Delay enabled** (`slider24`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Filter type** (`slider25`): default `1`; declared range/choices `0,26,1{Linear,MS-20,Linear x2,Moog,Ladder,303,MS-20 asym,DblRes,DualPeak,TriplePeak,svf nl 2p,svf nl 4p,svf nl 2p inc,svf nl 4p inc,rectified resonance,Steiner,SteinerA,Muck,Pill2p,Pill4p,Pill2p Aggro,Pill4p Aggro,Pill2p Stacc,Pill4p Stacc,Ladder3,Ladder6,HLadder}`. Canvas / hidden.
- **Filter Drive (dB)** (`slider26`): default `0`; declared range/choices `-6,48,1`. Canvas / hidden.
- **Cutoff Start** (`slider27`): default `.6`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Cutoff Finish** (`slider28`): default `.6`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Resonance** (`slider29`): default `0.7`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Morph** (`slider30`): default `0`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Envelope Rise** (`slider31`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Envelope Decay** (`slider32`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Envelope Sustain** (`slider33`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Pitch Shift** (`slider34`): default `0`; declared range/choices `-24,24,.0001`. Canvas / hidden.
- **Frequency shift** (`slider35`): default `0.5`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Frequency Shifter Enabled** (`slider36`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Sample playback offset** (`slider37`): default `0`; declared range/choices `0,1,0.000011`. Canvas / hidden.
- **Filter Inertia [ms]** (`slider63`): default `60`; declared range/choices `0,200,.001`. Canvas / hidden.
- **First Pattern** (`slider64`): default `120`; declared range/choices `0,127,1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## More background from the vendored source

### SEQS: A small GUI-based effect sequencer for stutters, slowdowns and various audio effects.
[drag_drop](https://user-images.githubusercontent.com/19836026/115153701-a2ee2300-a077-11eb-86bc-8eab6f13450d.gif)
[modulators_new](https://user-images.githubusercontent.com/19836026/115153706-a681aa00-a077-11eb-8105-ec78bf7133e1.gif)
### Demos
You can find demos of the plugin [here](https://www.youtube.com/watch?v=0cF9u7FiwuM) and [here](https://www.youtube.com/watch?v=VHcXz9xgGqo)
### Features
- Choose from 14 effects, with lots of parameters inside each effect.
- Modulate all of the effect parameters by linking them up to the two macro modulator controls.
- Drag and drop to reorder the effects that do not control the playhead.
- Synchronize the patterns to the host, free or MIDI.
- See exactly what audio is coming in, right above the pattern, making it easier to place the blocks in the correct places.
- Build up to 64 patterns.
- Select pattern by incoming MIDI note.
- Choose to set times in the plugin by time or beats.
- Randomize tracks.
- Choose from a large number of effects:
  - Effects that modify the playhead: Slowdown, Tape stop, Retrigger, Reverse.
  - Chorus / Phaser / Flaser module.
  - Pitch shifter.
  - Degradation effects (sample rate and bitrate reduction).
  - Two non-linear envelope controlled multimode filters (choose from 15 filter types, with several non-linear ones).
  - Volume envelope.
  - Reverb.
  - Pitched Delay (delay with delay length such that it produces tonal sounds).
  - Amplitude / Ring modulation module.
  - Tempo synchronized delay.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.2214 | 0.1580 | 1.40× | 1.38–1.43× |
| 512 | 0.1883 | 0.1331 | 1.43× | 1.41–1.45× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

**Workload limit:** Default playback/pattern state; this does not measure an active chain of sequenced effects.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.126. Original path: `SequencedFX/SequencedFX.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
