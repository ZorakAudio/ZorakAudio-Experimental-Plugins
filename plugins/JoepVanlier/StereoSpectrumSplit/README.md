# Saike SideSpectrum Meter

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `SpectrumAnalyzer/StereoSpectrumSplit.jsfx`
- Author: Cockos, Joep Vanlier
- Version: 1.0
- Tags: analysis FFT meter spectrum
- Upstream license declaration: LGPL - http://www.gnu.org/licenses/lgpl.html
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Attribution / license

See `LICENSE.upstream` and the original source header.
