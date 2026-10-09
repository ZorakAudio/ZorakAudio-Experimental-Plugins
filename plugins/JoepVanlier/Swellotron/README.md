# Saike Swellotron

A spectral soundscape effect that combines frequency content into a slowly drained resynthesis buffer.

## Quick start

1. Route the first stereo signal to inputs 1/2 and the second to 3/4. Begin with modest output and default spectral settings.
2. Set Adapt and Diffusion, then add Shine, Aether or Ice.
3. Use Scorch/Ruin for input/output saturation and adjust Dry/Wet for an insert.

## Controls and routing

The two stereo inputs are combined spectrally, with stereo output on 1/2. The spectral energy history persists; buffer/input mode determines how it accumulates. A continuous synthetic default workload is only a baseline.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **FFT size** (`slider1`): default `7`; declared range/choices `0,7,1{256,512,1024,2048,4096,8192,16384,32768`. Slider.
- **Adapt** (`slider2`): default `-2`; declared range/choices `-2,0,.01`. Canvas / hidden.
- **Shine** (`slider3`): default `0.3`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Ruin** (`slider4`): default `0.2`; declared range/choices `0,1,0.0000001`. Canvas / hidden.
- **Diffusion** (`slider5`): default `0`; declared range/choices `0,0.5,.001`. Canvas / hidden.
- **Scorch** (`slider6`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Aether** (`slider7`): default `0`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Pitch** (`slider8`): default `0`; declared range/choices `-1,1,0.0001`. Canvas / hidden.
- **NonLinear** (`slider9`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ice** (`slider10`): default `0`; declared range/choices `0,1,.00001`. Canvas / hidden.
- **IcePanned** (`slider11`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Diffusion Mode** (`slider12`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Noise mode** (`slider13`): default `0`; declared range/choices `0,5,.001`. Canvas / hidden.
- **Width** (`slider14`): default `1`; declared range/choices `0,1,.001`. Canvas / hidden.
- **Theta** (`slider15`): default `0`; declared range/choices `-3.14,3.14,.001`. Canvas / hidden.
- **Input Gain** (`slider16`): default `0`; declared range/choices `-12,0,12`. Canvas / hidden.
- **Output Gain** (`slider17`): default `0`; declared range/choices `-12,0,12`. Canvas / hidden.
- **Buffer mode** (`slider18`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.

## More background from the vendored source

### Swellotron
Swellotron computes the spectrum of both signals (using the STFT), multiplies the magnitudes in the spectral domain and puts the result of that in an energy buffer. This energy buffer is drained proportionally to its contents. The energy buffer is then used to resynthesize the sound, but this time with a random phase.
In plain terms, it behaves almost like a reverb, where frequencies that both sounds have in common are emphasized and frequencies where the sounds differ are attenuated. This will almost always lead to something that sounds pretty harmonic.
[Screenshot](https://i.imgur.com/ikizwwk.gif)
### Demos
You can find demos of the plugin [here](https://www.youtube.com/watch?v=PSaL8BvYdKk) and [here](https://www.youtube.com/watch?v=Ggojmb9wd5U).
### Features:
- FFT Reverberation
- Shimmer: Copies energy to twice the frequency (leading to iterative octave doubling).
- Aether: Same as shimmer but for fifths.
- Scorch: Input saturation.
- Ruin: Output saturation.
- Diffusion: Spectral blur.
- Ice: Chops small bandwidth bits from the energy at random, and copies them to a higher frequency (at 1x or 2x the frequency), thereby giving narrowband high frequency sounds (sounding very cold).

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
| 64 | 1.0854 | 0.8735 | 1.26× | 1.22–1.28× |
| 512 | 1.0250 | 0.8309 | 1.23× | 1.21–1.29× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.10. Original path: `Swellotron/Swellotron.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
