# Contour

## What it is
Contour is a **file-driven contour texturizer**.

Load a texture into **slot 0**, then let the plugin apply that material against the live input envelope. It sits somewhere between texture injection, granular re-contouring, and envelope-following resynthesis.

---

## Why use it
Use it when you want the shape of the live source to stay meaningful, but you want a loaded material to take over the actual texture.

---

## Quick start
1. Load the texture file into **slot 0**.
2. Set **Mix** and **Texture Gain** first.
3. Use **Precision**, **Grain Size**, **Granular Feedback**, **Jitter**, and **Pitch** to decide how tightly the texture follows the source and how granular it feels.
4. Choose **Dry+Wet** or **Wet Only** output mode.
5. Shape the contour response with the built-in envelope controls and **Input Threshold**.

---

## Main controls
### Texture / blend
Mix, Texture Gain, and Output Mode decide how much of the loaded material replaces or supports the original source.

### Granular behavior
Precision, Grain Size, Granular Feedback, Jitter, and Pitch determine how tightly or loosely the applied texture behaves.

### File handling
Auto-Reload, Reload Texture, and Max Load control how the slot-0 source is refreshed and how much of it is loaded.

### Envelope shape
Env Attack, Hold, Decay, Sustain, and Release shape how the live contour drives the texture.

### Input Threshold
Stops very low-level input from constantly triggering the engine.

---

## Routing / notes
This plugin depends on the **Texture file slot**. Without a loaded texture, there is nothing to contour with.

---

## In one sentence
Contour lets a loaded texture inherit the dynamics and phrase shape of a live input.
