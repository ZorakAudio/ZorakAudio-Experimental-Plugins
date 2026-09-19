# Saike Transience (transient shaper)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Basics/Transience.jsfx`
- Author: Joep Vanlier
- Version: 0.03
- Tags: Transient Shaper Saike
- Upstream license declaration: MIT (repository-level)
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# Transience
Transience is a plugin for enhancing or reducing transients. It works by using two envelopes. One is an envelope follower (short attack, longer decay; roughly follows the peaks of the sound), the other is a user specified envelope (with attack/decay). You can then shape the sound according to the difference between the two, making attacks or decays longer or shorter. The plugin operates in logarithmic space.
[Screenshot](https://i.imgur.com/TgC7n2B.png)

## Attribution / license

See `LICENSE.upstream` and the original source header.
