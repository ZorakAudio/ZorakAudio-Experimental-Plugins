# bric-a-brac (Saike)

A four-slot texture processor that triggers or follows incoming audio with loaded samples.

## Quick start

1. Drop a sample onto a slot in the canvas, then feed stereo audio.
2. Choose a trigger/envelope mode and set the sample threshold, attack and hold/decay.
3. Balance the slot gain, pitch, filter and LFO; add more slots when the first behaves as intended.

## Controls and routing

Sample slots need material; empty defaults omit the main sample playback work. Extra output pairs allow separate slot routing according to the canvas output mode.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Sample 1 Attack** (`slider1`): default `0`; declared range/choices `0,1,0.0000001`. Canvas / hidden.
- **Sample 1 Decay** (`slider2`): default `0.3`; declared range/choices `0,1,0.00000001`. Canvas / hidden.
- **Sample 1 Hold Level (%)** (`slider3`): default `100`; declared range/choices `0,100,0.000001`. Canvas / hidden.
- **Sample 1 Hold Time** (`slider4`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Sample 1 Minimum / Threshold** (`slider5`): default `-36`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 1 Maximum** (`slider6`): default `3`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 1 Output gain** (`slider7`): default `0`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 1 Pitch** (`slider8`): default `0`; declared range/choices `-24,24,0.0001`. Canvas / hidden.
- **Sample 1 HPF** (`slider9`): default `0`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Sample 1 LPF** (`slider10`): default `1`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **LFO 1 Amount** (`slider11`): default `1`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **LFO 1 Frequency** (`slider12`): default `0.5`; declared range/choices `0,10,.0000001`. Canvas / hidden.
- **Pan 1** (`slider13`): default `0`; declared range/choices `-1,1,0.000000001`. Canvas / hidden.
- **Spacer 6** (`slider14`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Spacer 7** (`slider15`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Sample 2 Attack** (`slider16`): default `0`; declared range/choices `0,1,0.0000001`. Canvas / hidden.
- **Sample 2 Decay** (`slider17`): default `0.3`; declared range/choices `0,1,0.00000001`. Canvas / hidden.
- **Sample 2 Hold Level (%)** (`slider18`): default `100`; declared range/choices `0,100,0.000001`. Canvas / hidden.
- **Sample 2 Hold Time** (`slider19`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Sample 2 Minimum / Threshold** (`slider20`): default `-36`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 2 Maximum** (`slider21`): default `3`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 2 Output gain** (`slider22`): default `0`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 2 Pitch** (`slider23`): default `0`; declared range/choices `-24,24,0.0001`. Canvas / hidden.
- **Sample 2 HPF** (`slider24`): default `0`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Sample 2 LPF** (`slider25`): default `1`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **LFO 2 Amount** (`slider26`): default `1`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **LFO 2 Frequency** (`slider27`): default `0.5`; declared range/choices `0,10,.0000001`. Canvas / hidden.
- **Pan 2** (`slider28`): default `0`; declared range/choices `-1,1,0.000000001`. Canvas / hidden.
- **Spacer 13** (`slider29`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Spacer 14** (`slider30`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Sample 3 Attack** (`slider31`): default `0`; declared range/choices `0,1,0.0000001`. Canvas / hidden.
- **Sample 3 Decay** (`slider32`): default `0.3`; declared range/choices `0,1,0.00000001`. Canvas / hidden.
- **Sample 3 Hold Level (%)** (`slider33`): default `100`; declared range/choices `0,100,0.000001`. Canvas / hidden.
- **Sample 3 Hold Time** (`slider34`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Sample 3 Minimum / Threshold** (`slider35`): default `-36`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 3 Maximum** (`slider36`): default `3`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 3 Output gain** (`slider37`): default `0`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 3 Pitch** (`slider38`): default `0`; declared range/choices `-24,24,0.0001`. Canvas / hidden.
- **Sample 3 HPF** (`slider39`): default `0`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Sample 3 LPF** (`slider40`): default `1`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **LFO 3 Amount** (`slider41`): default `1`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **LFO 3 Frequency** (`slider42`): default `0.5`; declared range/choices `0,10,.0000001`. Canvas / hidden.
- **Pan 3** (`slider43`): default `0`; declared range/choices `-1,1,0.000000001`. Canvas / hidden.
- **Spacer 20** (`slider44`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Spacer 21** (`slider45`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Sample 4 Attack** (`slider46`): default `0`; declared range/choices `0,1,0.0000001`. Canvas / hidden.
- **Sample 4 Decay** (`slider47`): default `0.3`; declared range/choices `0,1,0.00000001`. Canvas / hidden.
- **Sample 4 Hold Level (%)** (`slider48`): default `100`; declared range/choices `0,100,0.000001`. Canvas / hidden.
- **Sample 4 Hold Time** (`slider49`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **Sample 4 Minimum / Threshold** (`slider50`): default `-36`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 4 Maximum** (`slider51`): default `3`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 4 Output gain** (`slider52`): default `0`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Sample 4 Pitch** (`slider53`): default `0`; declared range/choices `-24,24,0.0001`. Canvas / hidden.
- **Sample 4 HPF** (`slider54`): default `0`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **Sample 4 LPF** (`slider55`): default `1`; declared range/choices `0,1,.000000001`. Canvas / hidden.
- **LFO 4 Amount** (`slider56`): default `1`; declared range/choices `0,1,.0000001`. Canvas / hidden.
- **LFO 4 Frequency** (`slider57`): default `0.5`; declared range/choices `0,10,.0000001`. Canvas / hidden.
- **Pan 4** (`slider58`): default `0`; declared range/choices `-1,1,0.000000001`. Canvas / hidden.
- **Dry/Wet** (`slider63`): default `0`; declared range/choices `-1,1,.00001`. Canvas / hidden.
- **palette** (`slider64`): default `0`; declared range/choices `0,10,1`. Slider.
- **Use only a single set of thresholds** (`slider65`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Global Minimum / Threshold 1** (`slider66`): default `-36`; declared range/choices `-48,3,0.0001`. Canvas / hidden.
- **Global Maximum 1** (`slider67`): default `3`; declared range/choices `-48,3,0.0001`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output, 3: left output 2, 4: right output 2, 5: left output 3, 6: right output 3, 7: left output 4, 8: right output 4.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.2823 | 0.2206 | 1.29× | 1.26–1.33× |
| 512 | 0.2537 | 0.1863 | 1.36× | 1.32–1.42× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

**Workload limit:** No sample files were loaded; this measures the empty-slot baseline, not loaded texture playback.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.48. Original path: `bric-a-brac/saike_bric_a_brac.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
