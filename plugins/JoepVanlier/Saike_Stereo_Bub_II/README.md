# Saike Stereo Bub II Stereoizer

A stereo widener designed around a delayed side contribution with a protected low-frequency region.

## Quick start

1. Feed audio and start with a small Strength and default Delay.
2. Move Crossover to keep low material centered; balance the old side contribution.
3. Use Check Mono and SideHP to inspect compatibility.

## Controls and routing

Delay is labelled ms/2 in this source. The canvas maps normalized crossover values; do not assume the raw value is hertz.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **Delay [ms/2]** (`slider1`): default `8`; declared range/choices `1,25,1`. Canvas / hidden.
- **Strength [-]** (`slider2`): default `0.3`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Crossover [log(w)]** (`slider3`): default `0.4`; declared range/choices `0,1,.0001`. Canvas / hidden.
- **Blend old side level [-]** (`slider4`): default `1`; declared range/choices `0,2,.0001`. Canvas / hidden.
- **Check Mono?** (`slider5`): default `0`; declared range/choices `0,1,1{No, Yes}`. Canvas / hidden.
- **Pass side through HPF?** (`slider6`): default `0`; declared range/choices `0,1,1{No,Yes}`. Canvas / hidden.

## More background from the vendored source

### A basic stereo widener
A fairly basic stereo widening tool. Widens the sound, but makes sure that the mono-mix stays unaffected (unlike Haas). The crossover is basically a 12 pole HPF that cuts the bass of the widening to avoid widening the bass too much. The last slider allows you to mix in the original side channel (which can optionally also be run through the 12-pole highpass).
You can either add stereo sound from nothing, using the Strength slider. This adds a comb filtered version of the average signal with opposite polarity to the different channels. Be careful not to overdo it, or you get a flangey sound (unless that is what you want).
You can manipulate the existing side channel that's in the input. The gain of the original side channel is scaled by the old "Old side" knob. Depending on the button "HP original side" this signal route will be highpassed (mono-izing the low frequencies).
[Screenshot](https://i.imgur.com/a09HF51.png)
### Demos
You can find demos of the plugin [here](https://www.youtube.com/watch?v=47L9bysgIiA) and [here](https://www.youtube.com/watch?v=pUu3h21yARY).
### Features:
- Add stereo to mono audio.
- Control existing stereo in audio.
- Use steep 12-pole crossover filter to keep bass mono.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.1517 | 0.0782 | 1.94× | 1.93–1.96× |
| 512 | 0.1504 | 0.0777 | 1.95× | 1.70–1.97× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.06. Original path: `Basics/Saike Stereo Bub II.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
