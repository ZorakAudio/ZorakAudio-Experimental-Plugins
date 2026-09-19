# Saike Stereo Bub III Stereoizer

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Basics/Saike Stereo Bub III.jsfx`
- Author: Joep Vanlier
- Version: 0.08
- Tags: comb stereoizer stereo
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A basic stereo widener
Similar to Stereo Bub II (see II for a description), but adds vibrato and non-linearity options.
### Features:
- Add stereo to mono audio.
- Control existing stereo in audio.
- Use steep 12-pole crossover filter to keep bass mono.
- Vibrato.
- Non-linearity.

## Attribution / license

See `LICENSE.upstream` and the original source header.
