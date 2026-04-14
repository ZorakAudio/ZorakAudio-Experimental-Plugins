## What it is

Load-and-Go MultiSampler is a **zero-setup multisampler**.

You point it at a group of samples and play.

No manual zones. No bank editor. No key mapping session. No tedious setup before it becomes useful.

It is built for the fast path:

- load a bunch of files
- press a key
- get variation immediately
- shape the feel with a few meaningful controls

---

## Why use it

Most multisamplers make you do admin work before you can make sound.

This one does the opposite.

It is meant for:

- quick sketching
- texture instruments
- found-sound banks
- percussion and one-shots
- randomized sample performance
- exploratory sound design

It is especially good when you want a folder of samples to become an instrument **fast**.

---

## What makes it different

### 1. No manual mapping
Traditional multisamplers often expect you to build zones, assign roots, and organize layers.

This one is **load samples and go**.

### 2. The whole bank is playable from any key
A single MIDI note can trigger **any sample in the bank**.

You are not locked into “this key only plays this file.”

### 3. It is designed for variation
Instead of repeating the same hit over and over, it can move through the bank in a more alive way.

### 4. The controls are about feel
The main controls are not there to bury you in technical setup. They are there to let you quickly shape:

- envelope
- pitch
- stereo width
- liveliness
- tonal color
- clarity/focus
- output level

---

## Quick start

1. Insert the plugin.
2. On **slot 0**, choose **Open Multiple...**
3. Select the samples you want in the instrument.
4. Play MIDI notes.
5. Adjust the controls to taste.

That is it.

The plugin auto-rescans the bank when the file selection changes, so you can swap the sample set and keep going.

---

## How to think about it

Do **not** think of it like a piano sampler where every note must be mapped carefully.

Think of it more like this:

> “I have a set of sounds. Turn them into a playable instrument with useful variation.”

That is the correct mental model.

---

## Main controls

## ADSR
These shape the overall envelope of each triggered sample.

### Attack
How quickly the sound comes in.

- low = immediate
- high = softer fade-in

### Decay
How long it takes to fall from the initial peak down toward the sustain level.

### Sustain
The held level while the note is down.

### Release
How long it takes to fade out after note-off.

---

## Pitch
Global pitch control.

Use it to shift the whole instrument up or down.

Range: **two octaves down to two octaves up**.

Good for:

- turning one bank into multiple instruments
- making impacts heavier
- making textures smaller/brighter
- creative repurposing of the same sample set

---

## Liveliness
Controls how much variation the instrument injects from trigger to trigger.

Low liveliness:

- steadier
- more consistent
- more repeatable

Higher liveliness:

- more movement
- more small differences between hits
- less stale repetition

Use this when the bank feels too static or too machine-like.

---

## Random / Sequence
Controls how samples are chosen.

### Random
Each trigger chooses from the bank with variation.

Best for:

- organic playback
- avoiding obvious repetition
- exploratory sound design

### Sequence
Walks through the bank in order.

Best for:

- consistent auditioning
- predictable stepping through samples
- controlled rhythmic use

---

## Color
Broad tonal tilt.

- lower = warmer / darker
- higher = brighter / lighter

Use it to move the bank toward a darker body or brighter edge without digging into EQ.

---

## Focus
Shifts the sense of softness versus clarity.

- lower = softer / smoother
- higher = clearer / more present

Useful when you want the bank to either sit back or speak more clearly.

---

## Width
Controls stereo spread.

- lower = tighter / narrower
- higher = wider / more open

Useful for turning the same bank into either a centered instrument or a wider texture.

---

## Truth
Balances raw sample character versus a more controlled presentation.

Lower values lean toward a more managed, even behavior.
Higher values preserve more of the original sample differences.

Use it like this:

- lower = more cohesive bank behavior
- higher = more raw individuality

---

## Output
Final output level.

Use it to level-match the instrument after shaping it.

---

## UI controls

### Scroll wheel
Hover a control and scroll to adjust it.

### Right-click or double-click reset
Right-click or double-click a control to return it to its default value.

---

## Best use cases

This instrument shines when:

- you have many similar one-shots
- you want one key to explore an entire bank
- you want fast results instead of setup work
- you want variation without building a huge sampler program by hand
- you are sketching ideas and do not want workflow friction

---

## When it is the wrong tool

Use a traditional multisampler instead when you need:

- exact note-by-note mapping
- carefully authored velocity layers
- realistic chromatic instrument recreation
- precise bank programming

This instrument is about **speed, variation, and immediacy**, not obsessive manual zone design.

---

## Design philosophy

The goal is simple:

**remove setup friction and make a bank of samples instantly playable.**

It is designed around:

- fast loading
- immediate playability
- variation without tedious programming
- controls that shape the result in obvious ways

In other words:

> less administration, more sound.

---

## Beginner recipe

If you do not know where to start:

1. Load a bank of samples into slot 0.
2. Set **Random** mode.
3. Set **Attack** low.
4. Set **Release** somewhere moderate.
5. Add a little **Liveliness**.
6. Use **Pitch** to find the most useful register.
7. Adjust **Color** and **Focus** until it sits right.
8. Use **Width** and **Output** to finish.

That will get you most of the way there.

---

## In one sentence

Load-and-Go MultiSampler turns a pile of samples into a playable, variable instrument with almost no setup.