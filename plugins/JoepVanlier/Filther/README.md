# Filther (Saike)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Filther/Filther.jsfx`
- Author: Joep Vanlier
- Version: 3.21
- Tags: Filther
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# Filther
Filther is a waveshaping / filterbank plugin that allows for some dynamic processing as well.
[Screenshot](https://imgur.com/GPk7WmN.png)
### Manual
A manual can be found here: [manual](https://joepvanlier.github.io/FiltherManual/)
### Demos
You can find demos of the plugin [soundcloud](https://soundcloud.com/saike/ohnoesitsaboss2/s-zYCOt) and [youtube](https://www.youtube.com/watch?v=-VUckbkJ3EY).
Small tutorial here: [here](https://www.youtube.com/watch?v=jtc8kp57xpI).
### Features:
- Spline waveshaping curve based on placing nodes. Can draw asymmetric curves as well.
- Two non-linear filter modules which can be automated by dynamics from the input signal or a side chain, LFO or envelopes.
- Waveshaping amount can be modulated by input dynamics, LFOs or envelopes.
- Modulators can optionally be triggered by MIDI notes.
- Huge array of filter types (linear filters, analog models, FM, AM filters, reverbs, distortions).
- Feedback section.
- Automatic Gain Control to protect your ears somewhat
Copyright (C) 2019 Joep Vanlier

## Attribution / license

See `LICENSE.upstream` and the original source header.
