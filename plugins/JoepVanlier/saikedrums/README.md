# Saike Dum Drums (DD-101)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `saikedrums/saikedrums.jsfx`
- Author: Joep Vanlier
- Version: 0.18
- Tags: drum, drum machine, drumkit, drums, drum synth, instrument
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A small drum computer with synthed drums.
### Features:
- Different synthesis algorithms for drum kit elements
- Remappable MIDI
- Pixel-based UI

## Attribution / license

See `LICENSE.upstream` and the original source header.
