# Saike Swellotron

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Swellotron/Swellotron.jsfx`
- Author: Joep Vanlier
- Version: 0.10
- Tags: ambient, soundscape, long, reverb, convolution, stft
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# Swellotron
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

## Attribution / license

See `LICENSE.upstream` and the original source header.
