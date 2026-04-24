# Texture

## What it is
Texture is the current **Performance Texture Engine (PTE)**.

Load a texture into **slot 0**, then let the engine inject, gate, slice, and perform that material based on live input, sidechain input, or MIDI. This is one of the most capable engines in the repo, and the current source is far beyond the older placeholder README.

---

## Why use it
Use Texture when you want to turn a loaded material into a **played texture system** instead of just a static sample playback effect.

It can behave like:

- texture injection
- gated granular layer
- phrase-following texture performer
- MIDI-driven gran synth
- sidechain-driven support layer

---

## Quick start
1. Load the texture file into **slot 0**.
2. Choose a **Playback Mode**: One-Shot, Random, Granular, or Gran Synth.
3. Set **Trigger Source** to Auto, Input, Sidechain, or MIDI.
4. Dial in **Mix**, **Texture Gain**, **Sensitivity**, and **Gate Threshold** first.
5. Then shape the engine with **Flow**, **Focus**, **Grounding**, **Jitter**, **Slice**, **Support**, and the envelope controls.

---

## Main controls
### Blend / detection
Mix and Texture Gain set the obvious blend. Sensitivity, Gate Threshold, Follower Attack, Follower Release, and Gate Hysteresis decide how the engine listens and opens.

### Motion / feel
Flow, Focus, Grounding, and Jitter are the core feel macros. They decide how smooth, sharp, stable, or unruly the injected texture behaves.

### Playback mode
One-Shot, Random, Granular, and Gran Synth are genuinely different workflows. Gran Synth in the current source is a MIDI-note-driven texture synth, not just ordinary grain spray.

### Slicing
Slice sets the nominal grain/event duration. Auto Slice switches from fixed duration to salience/event-bound driven slicing. Slice Rand adds per-grain duration variation when fixed slicing is in play.

### Support
Support is the salience budget that decides how assertively the texture can step forward. Support Rand adds per-grain variation without rewriting the detector itself.

### Envelope / gate release
Attack, Decay, Sustain, Release, Attack Curve, Decay Curve, Release Curve, and Gate Release Mode shape the playback envelope. Free, Tight, and Clamp change how the tail behaves after gate-off.

### File handling
Output Mode, Auto-Reload, Reload Texture, and Max Load control how the slot-0 material is loaded and whether you hear Dry+Wet or Wet Only.

### Global shaping
Global Pitch, Material Profile, Env Mix, Env Speed, and Debug Overlay sit on top of the core engine. Material Profile changes the behavior priors of the current source rather than swapping engines outright.

### Triggering / MIDI
Trigger Source chooses whether the engine follows Input, Sidechain, or MIDI. MIDI Note Mode selects Mono or Poly note behavior.

---

## Routing / notes
Typical routing:

- **1/2** = main input
- **3/4** = sidechain trigger source when selected
- MIDI = note/gate source when selected

The file slot is central. Texture does not make sense without a loaded material.

---

## Notes
### Important current-source ideas

- **Auto Slice** uses local salience peaks and event bounds.
- **Gran Synth** lets MIDI notes own pitch and gate like a synth while still allowing input/SC support behavior when useful.
- **Material Profile** changes priors and behavior bias, not the fundamental engine architecture.

### Practical advice

Start simple. A loaded texture plus Mix, Sensitivity, Flow, Focus, Grounding, Slice, and Support will tell you very quickly whether you are in the right neighborhood.

---

## In one sentence
Texture is the repo’s full performance texture engine: file-driven, detector-aware, and capable of live, sidechain, or MIDI-controlled behavior.
