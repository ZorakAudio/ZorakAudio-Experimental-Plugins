# TextureXY

## What it is
TextureXY is a **stripped-down XY-driven texture instrument** built from the Texture family.

The core idea in the current source is extremely direct:

- draw a path
- release to play it
- let X steer **Stable → Wild**
- let Y steer **Soft → Active**

The motion itself becomes the performance.

---

## Why use it
Use TextureXY when you want the expressive feel of the Texture ecosystem, but you want it driven by a hand-drawn gesture path instead of a conventional detector pipeline.

---

## Quick start
1. Load the source texture file.
2. Draw a motion path in the pad.
3. Release to play it in **OneShot**, **Loop**, or **PingPong** mode.
4. Set **Pitch**, **Grain**, **Travel**, and **Level** to get the basic feel right.
5. Choose **Naive** or **Smart** picking depending on whether you want straightforward scan following or coherence-aware texture-style source choice.

---

## Main controls
### Pitch
Global pitch anchor for the instrument.

### Grain
Nominal grain size.

### Travel
Source scrub speed. In the current UI, 1.0x corresponds to real-time source motion.

### Level
Output level.

### Playback
OneShot, Loop, or PingPong path playback.

### Picking
Naive follows the scan path more directly. Smart uses the texture-style descriptor system, phase pools, and coherence-aware candidate choice.

---

## Notes
This plugin is intentionally more performance-forward than menu-heavy.

The pad labels matter:

- horizontal motion = more stable to more wild
- vertical motion = softer to more active

That means the gesture shape itself is part of the sound design.

---

## In one sentence
TextureXY turns a drawn gesture into a playable texture-performance path.
