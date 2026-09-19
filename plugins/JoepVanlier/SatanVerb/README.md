# Satan verb (Saike)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `SatanVerb/SatanVerb.jsfx`
- Author: Joep Vanlier
- Version: 0.12
- Tags: Satan verb (work in progress)
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# Satan Verb
Satan verb is a reverberation unit mostly meant for diffuse and gated style reverberation. It can either be used without an envelope, to generate large ambient spaces, or be modulated by an envelope based on the input sound to give a sound more body while not adding too much noise to the dead time.
[Screenshot 1](https://i.imgur.com/JLXFrOH.png), [Screenshot 2](https://i.imgur.com/EclxtWp.gif)
### Demos
You can find a demo of the plugin [here](https://www.youtube.com/watch?v=4aI-Gg8ETAM)
### Features:
- FFT based reverberation algorithm.
- Optional downward spectral smearing for creepy effects.
- Optional spectrally shifted copy can be mixed in.
- Steep IIR LPF/HPF filters for the verb.
- Optional delay compensation.
- Envelopes based on the input envelope.
- Input non-linearity (dist), spectrum non-linearity (ceiling).
- Dry/Wet controls.

## Attribution / license

See `LICENSE.upstream` and the original source header.
