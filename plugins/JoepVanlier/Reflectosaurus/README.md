# Saike Reflectosaurus (beta)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Reflectosaurus/Reflectosaurus.jsfx`
- Author: Joep Vanlier
- Version: 0.106
- Tags: multi-tap delay plugin
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A flexible delay plugin for setting up complex delays and reverbs.
[Screenshot](https://raw.githubusercontent.com/JoepVanlier/JSFX/master/Reflectosaurus_Manual/Overview.png)
### Manual
A full manual can be found here: [manual](https://github.com/JoepVanlier/JSFX/raw/master/Reflectosaurus_Manual/Reflectosaurus_Manual.pdf)
### Demos
You can find demos of the plugin [here](https://www.youtube.com/watch?v=47L9bysgIiA) and [here](https://www.youtube.com/watch?v=pUu3h21yARY).
### Features:
- Up to 10 node delay.
- Positive, negative and allpass delay.
- Delay filtering (LPF w/ resonance, HPF).
- Various delay saturation algorithms.
- Delay sends.
- Two reverberation algorithms (FFT-based and allpass).
- Delay time pitch tracking based on MIDI input.
- Granular resynthesis.
- Pitch shifting.
- Side chain compressing the delays.
- A decent selection of presets.

## Attribution / license

See `LICENSE.upstream` and the original source header.
