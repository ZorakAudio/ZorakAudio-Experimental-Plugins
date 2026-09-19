# Final Boss (Saike)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `FinalBoss/saike_final_boss.jsfx`
- Author: Joep Vanlier
- Version: 0.13
- Tags: distortion, multi-effect, mangler, grunge
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A small distortion effect unit for grungy distortion effects
### Features:
- Allpass stack with feedback.
- Upwards compression.
- Octaver.
- Pitch shifting chorus.
- Cabinet filters.
- Frequency shifter based spectral movement.
- A big skull looking mad at you.

## Attribution / license

See `LICENSE.upstream` and the original source header.
