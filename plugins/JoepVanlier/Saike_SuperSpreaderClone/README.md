# Super Spreader

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Basics/Saike SuperSpreaderClone.jsfx`
- Author: Original code by lkjb, basic port by Saike (Joep Vanlier)
- Version: 0.09
- Tags: chorus pitch shifting supersaw
- Upstream license declaration: MIT (repository-level)
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Attribution / license

See `LICENSE.upstream` and the original source header.
