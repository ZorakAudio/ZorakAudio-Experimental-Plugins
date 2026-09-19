# Saike SEQS (Sequenced FX) (beta)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `SequencedFX/SequencedFX.jsfx`
- Author: Joep Vanlier
- Version: 0.126
- Tags: time-based effect
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# SEQS: A small GUI-based effect sequencer for stutters, slowdowns and various audio effects.
[drag_drop](https://user-images.githubusercontent.com/19836026/115153701-a2ee2300-a077-11eb-86bc-8eab6f13450d.gif)
[modulators_new](https://user-images.githubusercontent.com/19836026/115153706-a681aa00-a077-11eb-8105-ec78bf7133e1.gif)
### Demos
You can find demos of the plugin [here](https://www.youtube.com/watch?v=0cF9u7FiwuM) and [here](https://www.youtube.com/watch?v=VHcXz9xgGqo)
### Features
- Choose from 14 effects, with lots of parameters inside each effect.
- Modulate all of the effect parameters by linking them up to the two macro modulator controls.
- Drag and drop to reorder the effects that do not control the playhead.
- Synchronize the patterns to the host, free or MIDI.
- See exactly what audio is coming in, right above the pattern, making it easier to place the blocks in the correct places.
- Build up to 64 patterns.
- Select pattern by incoming MIDI note.
- Choose to set times in the plugin by time or beats.
- Randomize tracks.
- Choose from a large number of effects:
  - Effects that modify the playhead: Slowdown, Tape stop, Retrigger, Reverse.
  - Chorus / Phaser / Flaser module.
  - Pitch shifter.
  - Degradation effects (sample rate and bitrate reduction).
  - Two non-linear envelope controlled multimode filters (choose from 15 filter types, with several non-linear ones).
  - Volume envelope.
  - Reverb.
  - Pitched Delay (delay with delay length such that it produces tonal sounds).
  - Amplitude / Ring modulation module.
  - Tempo synchronized delay.

## Attribution / license

See `LICENSE.upstream` and the original source header.
