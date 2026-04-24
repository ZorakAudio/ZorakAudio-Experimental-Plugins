# GesturePad

## What it is
GesturePad is a **draw-and-play MIDI control surface**.

You draw a gesture in the pad and the plugin turns that motion into MIDI output. It can transmit:

- X and Y CC motion
- a “primary motion” CC derived from gesture analysis
- note output
- CC + note output together

The current source is much broader than a simple XY pad. It includes playback modes, note generation, motion-lane routing, and an advanced bank of extra CC outputs for speed, velocity, acceleration, jerk, and error signals.

---

## Why use it
Use GesturePad when you want **performed motion** instead of a static LFO.

It works well for:

- live macro movement
- XY synth morphing
- expressive automation recording
- motion-driven note generation
- turning drawing gestures into reusable MIDI phrases

---

## Quick start
1. Insert GesturePad before the instrument or effect you want to control.
2. Set the **MIDI Channel** and choose the CC numbers for **X**, **Y**, and **Primary Motion**.
3. Pick a playback mode such as **Direct**, **Loop**, **OneShot**, or **PingPong**.
4. Draw a gesture in the pad. Enable **Auto Play After Draw** if you want it to fire immediately.
5. Choose **CC Only**, **Note Only**, or **CC + Note** depending on whether you want pure modulation, notes, or both.

---

## Main controls
### Playback
Mode, Playback Speed, Smoothing, and Min Point Distance control how the drawn motion is captured and replayed. Use these first to decide whether the gesture feels tight, smooth, fast, slow, or repeatable.

### Basic CC output
CC X and CC Y are the direct pad outputs. CC Primary Motion is a third lane that follows whichever movement feature you assign as the dominant gesture descriptor.

### Primary motion analysis
Primary Motion Lane Source chooses what that main analysis lane represents. The current source can derive it from Speed, Velocity X/Y, Acceleration X/Y, Jerk X/Y, or Error X/Y.

### Draw/play behavior
Emit While Drawing sends data live as you draw. Auto Play After Draw starts playback when the gesture is finished. CC Deadband and Speed Sensitivity help tame noisy or over-busy output.

### Note generation
Output Mode, Note Pitch Source, Note Velocity Source, Note Base, Note Span, and Fixed Note Velocity turn the pad into a note performer instead of only a CC surface.

### Advanced CC lanes
The current source exposes additional per-feature CC slots: Speed, Velocity X, Velocity Y, Acceleration X, Acceleration Y, Jerk X, Jerk Y, Error X, and Error Y. Use them when you want multiple gesture-derived lanes at once.

---

## Routing / notes
GesturePad is a MIDI generator. Route its MIDI output to whatever should receive the CCs or notes.

A simple first setup is:

- X CC -> filter cutoff
- Y CC -> morph or mix
- Primary Motion -> resonance, FM amount, or another “energy” control

---

## Notes
This is not a static XY controller anymore. The current source behaves more like a **gesture analyzer + playback engine** with optional note synthesis on top.

If you only use X and Y, you will miss a large part of what the plugin can do now.

---

## In one sentence
GesturePad turns drawn motion into playable MIDI control, note, and gesture-analysis data.
