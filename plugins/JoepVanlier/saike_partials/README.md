# Partials (Saike)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `partials/saike_partials.jsfx`
- Author: Joep Vanlier
- Version: 0.68
- Tags: modal effect, instrument
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# An effect which simulates different materials
This effect takes both audio and MIDI input. Based on the model selected the incoming audio will excite
a number of resonators that produce particular sounds. Up to 4 note polyphony is supported.

## Attribution / license

See `LICENSE.upstream` and the original source header.
