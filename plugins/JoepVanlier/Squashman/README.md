# Squashman (Saike)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Squashman/Squashman.jsfx`
- Author: Joep Vanlier
- Version: 0.86
- Tags: multiband saturation plugin
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# Squashman
Squashman is a multi-band saturation / distortion plugin that allows modulation of several of its parameters.
[Screenshot](https://i.imgur.com/egp00QC.png)
### Demos
You can find a demo of the plugin [here](https://www.youtube.com/watch?v=mK0xAhq4pK4)
### Features:
- Flexible band count, up to five bands can be used to manipulate sound
- 24 db/oct Linkwitz Riley crossover filters
- Graphical user interface
- Optional high quality oversampling  
- 25 modulatable waveshapers and 4 fixed ones.
- Several modulation sources (4 LFOs, 2 MIDI triggered and/or loopable envelopes).

Thanks to tviler / samuele pizzi / RCJacH / BethHarmon (inflator waveshaper curves).

## Attribution / license

See `LICENSE.upstream` and the original source header.
