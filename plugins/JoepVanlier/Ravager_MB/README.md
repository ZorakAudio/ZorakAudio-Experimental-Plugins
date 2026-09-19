# Saike Multiband Ravager (BETA)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Ravager/Ravager_MB.jsfx`
- Author: Joep Vanlier
- Version: 0.14
- Tags: Ravager
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# Multiband Audio Destroyer. Performs extreme upwards compression akin to DOOM compressor.
Compressor design based on: Giannoulis et al, "Digital Dynamic Range Compressor Design—A Tutorial and Analysis", Journal of the Audio Engineering Society 60(6)

## Attribution / license

See `LICENSE.upstream` and the original source header.
