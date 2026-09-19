# Saike Pitch Shifter

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Basics/Saike_Pitch_Shift.jsfx`
- Author: Joep Vanlier
- Version: 0.04
- Tags: Saike - Basic pitch shifter
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Attribution / license

See `LICENSE.upstream` and the original source header.
