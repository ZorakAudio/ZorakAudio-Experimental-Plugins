# Sample

## What it is
Sample is a **load-and-go psycho multisampler**.

The point is immediate usefulness: load files into **slot 0**, play MIDI notes, get variation right away, and shape the bank with a compact set of macros.

The current build includes hybrid playback, transient handling, onset modes, Clean, Room Tame, Walk selection, Push salience control, Hybrid Release→Attack handoff, continuous envelope slopes, dynamics, and optional FluxBridge-aware motion behavior.

---

## Quick start
1. Insert the plugin.
2. On **slot 0**, choose **Open Multiple...** and select files.
3. Play MIDI notes.
4. Choose **Random**, **Sequence**, or **Walk**.
5. Shape the result with Envelope, Playback, Clean, Room Tame, Push, and Output.

---

## Selection modes
**Random** chooses from the whole bank with anti-repeat weighting.

**Sequence** walks through the loaded files in order.

**Walk** bridges from the previous sample using brightness, focus, transient richness, length, pitch confidence, body span, and tail span. In Hybrid, donor grain choices can use the same coherence bias.

---

## Envelope
Attack, Hold, Decay, Sustain, and Release shape each triggered sample.

**Shift + Wheel** on Attack, Decay, or Release changes slope from **-100% to +100%**. **0% is linear**. The defaults preserve the original envelope behavior:

- Attack slope: **+100%**
- Decay slope: **-100%**
- Release slope: **-100%**

Hybrid Release keeps the granular body alive under the Release envelope. When a new Hybrid note arrives during Release, the new Attack can inherit the current Release level and matching body phase for a smoother handoff.

---

## Playback engines
**Raw** plays the source directly.

**Tape** uses the slower, heavier rate path.

**Hybrid** keeps the sample head direct, sustains from body grains, can borrow matched donor tails, and supports smoother Release→Attack transitions.

---

## Cleanup and depth
**Clean** reduces low-level hiss, room bed, and sample dirt while protecting attacks.

**Room Tame** restrains obvious late-room smear without acting like a heavy de-reverb.

**Push** is a salience-budget control. Higher values push the instrument farther back in the mix by reducing foreground-grabbing presence, air, click, and wide high-band splash. **100% is intentionally extreme background placement.**

---

## Tone and feel
**Color** warms or brightens.

**Focus** softens or clarifies.

**Width** changes spread.

**Liveliness** adds variation.

**Truth** moves from matched/cohesive behavior toward raw sample individuality.

**Weight** adds short body bloom.

**Dynamics** adds controlled density with internal protection.

**Output** trims the final result.

---

## FluxBridge
When the fixed FluxBridge CC lane block is present, Sample adds subtle motion-driven tone, focus, width, pan, and body contour while protecting transient-rich hits.

---

## Good uses
Sample is strongest for quick one-shot banks, found-sound instruments, texture instruments, sketching, randomized playback, and exploratory sound design.

---

## Wrong tool
Use a traditional multisampler when you need exact key zones, velocity-layer authoring, or realistic note-by-note instrument recreation.

---

## In one sentence
Sample turns a loose bank of audio files into a playable, variable instrument with almost no setup.
