# Dusk Verb (Saike) (beta)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `DuskVerb/saike_duskverb.jsfx`
- Author: Joep Vanlier
- Version: 0.17
- Tags: effect, reverb, atmosphere, granular, long
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A multi-effect plugin intended to enhance atmospheric arpeggios
[Screenshot](https://user-images.githubusercontent.com/19836026/221384927-db1d9f3e-df04-4676-a4d4-aa508ad1ade6.gif)
### Features:
- 3 Reverberation algorithms.
- Granular resampler.
- Frequency shifter / pitch shifter. 
- Several audio shimmer modes.
- X/Y controls for automation.
- Classic adventure game look.

## Attribution / license

See `LICENSE.upstream` and the original source header.
