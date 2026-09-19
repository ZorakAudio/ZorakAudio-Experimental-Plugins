# Amaranth (Saike) [BETA]

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Amaranth/Amaranth.jsfx`
- Author: Joep Vanlier
- Version: 0.32
- Tags: amaranth granular synth graintable grains
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Attribution / license

See `LICENSE.upstream` and the original source header.
