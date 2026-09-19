# Saike Stereo Bub II Stereoizer

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Basics/Saike Stereo Bub II.jsfx`
- Author: Joep Vanlier
- Version: 0.06
- Tags: comb stereoizer stereo
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A basic stereo widener
A fairly basic stereo widening tool. Widens the sound, but makes sure that the mono-mix stays unaffected (unlike Haas). The crossover is basically a 12 pole HPF that cuts the bass of the widening to avoid widening the bass too much. The last slider allows you to mix in the original side channel (which can optionally also be run through the 12-pole highpass).
You can either add stereo sound from nothing, using the Strength slider. This adds a comb filtered version of the average signal with opposite polarity to the different channels. Be careful not to overdo it, or you get a flangey sound (unless that is what you want).
You can manipulate the existing side channel that's in the input. The gain of the original side channel is scaled by the old "Old side" knob. Depending on the button "HP original side" this signal route will be highpassed (mono-izing the low frequencies).
[Screenshot](https://i.imgur.com/a09HF51.png)
### Demos
You can find demos of the plugin [here](https://www.youtube.com/watch?v=47L9bysgIiA) and [here](https://www.youtube.com/watch?v=pUu3h21yARY).
### Features:
- Add stereo to mono audio.
- Control existing stereo in audio.
- Use steep 12-pole crossover filter to keep bass mono.

## Attribution / license

See `LICENSE.upstream` and the original source header.
