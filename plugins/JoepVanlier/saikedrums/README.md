# Saike Dum Drums (DD-101)

The DD-101 synthesis drum machine, with twelve instrument sounds, MIDI mappings and optional separate outputs.

## Quick start

1. Route MIDI drum notes to the plugin and use the canvas to verify the note map.
2. Select a sound and adjust its pitch, envelope, noise and gain.
3. Use the stereo mix first; enable multi-output mode only after routing the separate pairs.

## Controls and routing

The declared output pairs cover mix/kick, snare, clap, ride, hat, three toms, rim, cowbell, shaker and optionally uncoupled open hat. The short benchmark note pattern is not a dense twelve-part drum arrangement.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Kick Type** (`slider1`): default `0`; declared range/choices `0,3,1`. Canvas / hidden.
- **Kick Pitch Decay** (`slider2`): default `0.7`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Kick Minimum Pitch** (`slider3`): default `0.15`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Kick Amp Decay** (`slider4`): default `0.3`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Kick Pitch Envelope** (`slider5`): default `0.3`; declared range/choices `0,0.5,.00001`. Canvas / hidden.
- **Noise** (`slider6`): default `0.78`; declared range/choices `0,1,.000001`. Canvas / hidden.
- **Kick decay** (`slider7`): default `0`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Kick gain** (`slider8`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Kick panning** (`slider9`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Snare type** (`slider10`): default `1`; declared range/choices `0,4,1`. Canvas / hidden.
- **Snare Pitch Decay** (`slider11`): default `0.67`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Snare Minimum Pitch** (`slider12`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Snare amplitude decay** (`slider13`): default `0.3`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Snare Pitch Envelope** (`slider14`): default `0.3`; declared range/choices `0,0.5,.00001`. Canvas / hidden.
- **Snare Noise Decay** (`slider15`): default `0.2`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Snare gain** (`slider16`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Snare panning** (`slider17`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Clap type** (`slider20`): default `0`; declared range/choices `0,2,1`. Canvas / hidden.
- **Clap Attack** (`slider21`): default `0.5`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Clap Decay** (`slider22`): default `0.25`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **Clap gain** (`slider23`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Clap panning** (`slider24`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Rim type** (`slider25`): default `0`; declared range/choices `0,2,1`. Canvas / hidden.
- **Rim decay** (`slider26`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Rim tune** (`slider27`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Rim gain** (`slider28`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Rim panning** (`slider29`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Hat type** (`slider30`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Hat attack** (`slider31`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Hat decay** (`slider32`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Hat tone** (`slider33`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Hat body** (`slider34`): default `0.5`; declared range/choices `0, 1.2, 0.0001`. Canvas / hidden.
- **Hat gain** (`slider35`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Hat panning** (`slider36`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Cowbell type** (`slider37`): default `0`; declared range/choices `0,3,1`. Canvas / hidden.
- **Cowbell tune** (`slider38`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Cowbell decay** (`slider39`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Cowbell gain** (`slider40`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Cowbell panning** (`slider41`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Ride type** (`slider42`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ride attack** (`slider43`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Ride decay** (`slider44`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Ride tone** (`slider45`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Ride duty cycle** (`slider46`): default `0.4798`; declared range/choices `0.2,0.8,0.00001`. Canvas / hidden.
- **Ride gain** (`slider47`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Ride panning** (`slider48`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Shaker type** (`slider49`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Shaker tune** (`slider50`): default `0.5`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Shaker decay** (`slider51`): default `0.5`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Shaker gain** (`slider52`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Shaker panning** (`slider53`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Tom type** (`slider54`): default `0`; declared range/choices `0,2,1`. Canvas / hidden.
- **Low tom tune** (`slider55`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Low tom decay** (`slider56`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Low tom gain** (`slider57`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Low tom panning** (`slider58`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Mid tom tune** (`slider59`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Mid tom decay** (`slider60`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Mid tom gain** (`slider61`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Mid tom panning** (`slider62`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **High tom tune** (`slider63`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **High tom decay** (`slider64`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **High tom gain** (`slider65`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **High tom panning** (`slider66`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Kick Pitch Decay Velocity** (`slider102`): default `0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Kick Minimum Pitch Velocity** (`slider103`): default `0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Kick Amp Decay Velocity** (`slider104`): default `0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Kick Pitch Envelope Velocity** (`slider105`): default `0`; declared range/choices `-1, 1,.00001`. Canvas / hidden.
- **Noise Velocity** (`slider106`): default `0`; declared range/choices `-2,2,.000001`. Canvas / hidden.
- **Kick decay Velocity** (`slider107`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Kick gain Velocity** (`slider108`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Kick panning Velocity** (`slider109`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Snare Pitch Decay Velocity** (`slider111`): default `0.0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Snare Minimum Pitch Velocity** (`slider112`): default `0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Snare amplitude decay Velocity** (`slider113`): default `0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Snare Pitch Envelope Velocity** (`slider114`): default `0`; declared range/choices `-1.0,1.0,.00001`. Canvas / hidden.
- **Snare Noise Decay Velocity** (`slider115`): default `0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Snare gain Velocity** (`slider116`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Snare panning Velocity** (`slider117`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Clap Attack Velocity** (`slider121`): default `0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Clap Decay Velocity** (`slider122`): default `0`; declared range/choices `-2,2,.00001`. Canvas / hidden.
- **Clap gain Velocity** (`slider123`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Clap panning Velocity** (`slider124`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Rim decay Velocity** (`slider126`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Rim tune Velocity** (`slider127`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Rim gain Velocity** (`slider128`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Rim panning Velocity** (`slider129`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Hat attack Velocity** (`slider131`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Hat decay Velocity** (`slider132`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Hat tone Velocity** (`slider133`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Hat body Velocity** (`slider134`): default `0`; declared range/choices `-2.4, 2.4, 0.0001`. Canvas / hidden.
- **Hat gain Velocity** (`slider135`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Hat panning Velocity** (`slider136`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Cowbell tune Velocity** (`slider138`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Cowbell decay Velocity** (`slider139`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Cowbell gain Velocity** (`slider140`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Cowbell panning Velocity** (`slider141`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Ride attack Velocity** (`slider143`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Ride decay Velocity** (`slider144`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Ride tone Velocity** (`slider145`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Ride duty cycle Velocity** (`slider146`): default `0`; declared range/choices `-1.6,1.6,0.00001`. Canvas / hidden.
- **Ride gain Velocity** (`slider147`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Ride panning Velocity** (`slider148`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Shaker tune Velocity** (`slider150`): default `0`; declared range/choices `-2,2,0.001`. Canvas / hidden.
- **Shaker decay Velocity** (`slider151`): default `0`; declared range/choices `-2,2,0.001`. Canvas / hidden.
- **Shaker gain Velocity** (`slider152`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Shaker panning Velocity** (`slider153`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Low tom tune Velocity** (`slider155`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Low tom decay Velocity** (`slider156`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Low tom gain Velocity** (`slider157`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Low tom panning Velocity** (`slider158`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Mid tom tune Velocity** (`slider159`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Mid tom decay Velocity** (`slider160`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Mid tom gain Velocity** (`slider161`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Mid tom panning Velocity** (`slider162`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **High tom tune Velocity** (`slider163`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **High tom decay Velocity** (`slider164`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **High tom gain Velocity** (`slider165`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **High tom panning Velocity** (`slider166`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.
- **Muted channels (bitmask)** (`slider170`): default `0`; declared range/choices `0,8191,1`. Canvas / hidden.
- **Solod channels (bitmask)** (`slider171`): default `0`; declared range/choices `0,8191,1`. Canvas / hidden.
- **Hat coupling** (`slider172`): default `0`; declared range/choices `0,1,{Coupled,Uncoupled}`. Canvas / hidden.
- **Open Hat type** (`slider173`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Open Hat attack** (`slider174`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Open Hat decay** (`slider175`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Open Hat tone** (`slider176`): default `0.5`; declared range/choices `0,1,0.00001`. Canvas / hidden.
- **Open Hat body** (`slider177`): default `0.5`; declared range/choices `0, 1.2, 0.0001`. Canvas / hidden.
- **Open Hat gain** (`slider178`): default `0`; declared range/choices `-24,6,0.0001`. Canvas / hidden.
- **Open Hat panning** (`slider179`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **Open Hat attack Velocity** (`slider182`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Open Hat decay Velocity** (`slider183`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Open Hat tone Velocity** (`slider184`): default `0`; declared range/choices `-2,2,0.00001`. Canvas / hidden.
- **Open Hat body Velocity** (`slider185`): default `0`; declared range/choices `-2.4, 2.4, 0.0001`. Canvas / hidden.
- **Open Hat gain Velocity** (`slider186`): default `0`; declared range/choices `-36,36,0.0001`. Canvas / hidden.
- **Open Hat panning Velocity** (`slider187`): default `0`; declared range/choices `-2,2,0.0001`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left mix or kick, 2: right mix or kick, 3: left snare, 4: right snare, 5: left clap, 6: right clap, 7: left ride, 8: right ride, 9: left hat, 10: right hat, 11: left low tom, 12: right low tom, 13: left mid tom, 14: right mid tom, 15: left hi tom, 16: right hi tom, 17: left rim, 18: right rim, 19: left cowbell, 20: right cowbell, 21: left shaker, 22: right shaker, 23: left open hat (when decoupled), 24: right open hat (when decoupled).

## More background from the vendored source

### A small drum computer with synthed drums.
### Features:
- Different synthesis algorithms for drum kit elements
- Remappable MIDI
- Pixel-based UI

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.2551 | 0.1662 | 1.55× | 1.49–1.75× |
| 512 | 0.2125 | 0.1346 | 1.58× | 1.57–1.60× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.18. Original path: `saikedrums/saikedrums.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
