# Sample

## What it is
Sample is a **load-and-go psycho multisampler**.

The whole point is immediate usefulness:

- load files into **slot 0**
- play notes
- get variation right away
- shape the feel with a compact set of macros

The current source goes well beyond the older README. It now includes hybrid playback, transient handling, onset modes, dynamics control, and optional FluxBridge-aware motion behavior.

---

## Why use it
Use it when you want a folder of samples to become an instrument **fast**.

It is especially good for:

- quick sketching
- one-shot banks
- found-sound instruments
- exploratory sound design
- randomized playback
- texture instruments
- “turn this pile of audio into something playable right now” workflows

---

## Quick start
1. Insert the plugin.
2. On **slot 0**, choose **Open Multiple...** and select the files you want.
3. Play MIDI notes.
4. Pick **Random** or **Sequence** mode depending on whether you want wandering variation or ordered stepping.
5. Shape the result with the envelope, macro, and playback controls.

---

## Main controls
### Bank handling
Force rescan slot 0 tells the plugin to refresh the load-and-go bank immediately. In normal use the current source also auto-polls the slot and queues bank reloads when the selection changes.

### Envelope
Attack, Decay, Sustain, Release, and Hold shape how each triggered sample behaves over time.

### Tone / feel
Color, Focus, Width, Liveliness, Truth, and Output are the fast musical macros. Color warms or brightens, Focus softens or clarifies, Width changes spread, Liveliness adds motion and micro-variation, Truth moves between matched/cohesive behavior and raw individuality, and Output trims the result.

### Pitch / choice
Pitch transposes the instrument globally. Sequence Mode switches between Random and Sequence sample choice.

### Playback body
Stretch, Weight, Mono Source, and Dynamics shape how the bank feels physically. Dynamics in the current source is not just gain; it adds controlled loudness and density with internal protection.

### Transient handling
Transient Mode chooses how strongly attacks are protected: Auto, Punch, or Off.

### Playback engine
Playback Mode selects Raw, Tape, or Hybrid. Raw stays direct. Tape is the slower, heavier path. Hybrid keeps the sample head more direct, uses body-phase/grain behavior for sustain, and can borrow donor tails after note-off.

### Hybrid details
Granular Body and Tail Borrow decide how much of the body and release are steered by the more synthetic / donor-assisted side of the engine.

### Onset alignment
Onset Align Mode switches between Classic and Gaussian behavior. Gaussian mode anchors starts more strongly to detected onsets and makes the current source feel more consistent across mixed material.

---

## Notes
### FluxBridge-aware behavior

The current source can listen for a **FluxBridge Receiver CC burst** on a fixed lane block. When that exact block is present, the sampler adds subtle motion-driven tone, focus, width, pan, and body contour behavior. It is designed to stay musical instead of turning every hit into obvious wobble.

### Good first recipe

- Random mode
- low Attack
- moderate Release
- a little Liveliness
- set Pitch to a useful register
- use Color and Focus first
- finish with Width and Output

### Mental model

Do not think of Sample as a full painstaking zone editor. Think of it as:

> “I have a bank of sounds. Make it playable and alive with almost no setup.”

---

## When it is the wrong tool
Use a traditional multisampler instead when you need exact key mapping, velocity-layer authoring, or realistic note-by-note instrument recreation.

---

## In one sentence
Sample turns a loose bank of audio files into a playable, variable instrument with almost no setup.
