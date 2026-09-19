# Saike Spectral Analyzer (beta)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `SpectrumAnalyzer/SaikeMultiSpectralAnalyzer.jsfx`
- Author: Joep Vanlier, Trond-Viggo Melssen, Cockos, Feed The Cat
- Version: 5.0.42
- Tags: analysis FFT meter spectrum
- Upstream license declaration: LGPL - http://www.gnu.org/licensses/lgpl.html
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Attribution / license

See `LICENSE.upstream` and the original source header.
