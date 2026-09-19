# Yutani Mono Bass Synth [Saike] (BETA)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Yutani/Saike_Yutani.jsfx`
- Author: Joep Vanlier
- Version: 0.103
- Tags: synth bass, instrument
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A mono-synth plugin with some analog-emulated filters and modulation options
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

## Attribution / license

See `LICENSE.upstream` and the original source header.
