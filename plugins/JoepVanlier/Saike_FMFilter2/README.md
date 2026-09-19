# Saike FM Filter 2

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Yutani/Saike_FMFilter2.jsfx`
- Author: Joep Vanlier
- Version: 0.23
- Tags: audio-rate filters
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# An FM filter plugin
[Screenshot](https://user-images.githubusercontent.com/19836026/110242715-998d7900-7f57-11eb-8c6e-48b825b8f47e.gif)
### Features:
- Anti-aliased oscillators.
- 15 filters, from well behaved linear models, to gnarly analog modelled nastiness.
- Audio and MIDI controllable filters.
- Audio and MIDI controllable gate.
- Three LFOs.
- Modwheel and MIDI velocity support.
- Stereo widening effect.
- Distortion module.

Attribution: Moog filter implementation was based on the paper:
S. D'Angelo and V. Vaelimaeki, "Generalized Moog Ladder Filter: Part II - Explicit Non linear Model through a Novel Delay-Free
Loop Implementation Method". IEEE Trans. Audio,Speech, and Lang. Process., vol. 22, no. 12, pp. 1873-1883, December 2014.
303 emulation is Copyright (c) 2012 Dominique Wurtz (www.blaukraut.info)
minBLEP methodology Eli Brandt, "Hard Sync Without Aliasing"

## Attribution / license

See `LICENSE.upstream` and the original source header.
