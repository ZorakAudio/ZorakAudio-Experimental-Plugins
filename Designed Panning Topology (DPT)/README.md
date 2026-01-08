# Designed Panning Topology (DPT)

**SAFE Headphone Psychoacoustic Panner**

DPT is a **minimal, mix-safe headphone panner** that improves left/right localization using only the binaural cues humans actually rely on—without HRTFs, rooms, or spatial gimmicks.

It is designed to be **immediately usable**: drop it on a track, turn one knob, move on.

---

## What DPT Is

DPT is a **headphone-only psychoacoustic panner** that combines:

* Equal-power panning (stable loudness)
* Interaural Time Difference (ITD) on the *far ear only*
* Head shadow (high-frequency loss) on the *far ear only*
* A small, conservative crossfeed from near → far ear

All cues are:

* **Angle-aware**
* **Amount-controlled**
* **Human-scale**
* **Hard-bounded for safety**

There are no modes, no debug features, and no ways to misconfigure it badly.

---

## What DPT Is Not

* ❌ Not a spatializer
* ❌ Not a binaural renderer
* ❌ Not an HRTF plugin
* ❌ Not a room or distance simulator
* ❌ Not a width enhancer
* ❌ Not intended for speakers

If you want elevation, distance, or “sound around your head,” this is the wrong tool.

DPT is about **clean, believable horizontal placement** on headphones.

---

## Controls (SAFE Set)

### **Azimuth**

Left/right position.

* `-100` = hard left
* `+100` = hard right

### **Amount**

Overall strength of psychoacoustic cues.

* `0%` → behaves close to a centered, gentle pan
* `100%` → full pan with maximum (but still bounded) ITD, shadow, and crossfeed

This is the main control you’ll touch.

### **Output (dB)**

Final trim only.

* No internal gain staging depends on this.
* Included for workflow convenience.

That’s it. No hidden modes.

---

## Quick Start (30 seconds)

1. **Insert DPT on a track**

   * Mono or stereo is fine (DPT pans the mono mid internally for stability).

2. **Wear headphones**

   * DPT is intentionally headphone-only.

3. **Set Azimuth**

   * Place the sound where you want it horizontally.

4. **Adjust Amount**

   * Start around **50–70%** for natural placement.
   * Push higher only if you want stronger separation.

5. **Trim Output if needed**

You are done.

---

## What’s Happening Under the Hood (Brief)

DPT follows a **strict, minimal topology**:

1. **Mid extraction**

   * Uses mono mid as the panning source for predictable behavior.

2. **Equal-power pan law**

   * Prevents level drop across the stereo field.

3. **Far-ear ITD**

   * Up to ~0.63 ms delay, clamped to human anatomy.

4. **Far-ear head shadow**

   * Gentle low-pass filter; never fully muffled.

5. **Conservative crossfeed**

   * Fixed, low-level, angle-aware.
   * Reduces headphone fatigue without collapsing the image.

6. **Hard safety**

   * Bounded delays
   * No feedback paths
   * No boosts
   * Output hard-clipped to ±8
   * No NaNs, no idle noise

Every choice favors **mix translation and stability** over impressiveness.

---

## Why Only Headphones?

Because:

* ITD and crossfeed are **physically wrong on speakers**
* Speaker playback already has acoustic crosstalk
* Applying headphone cues on speakers causes comb filtering and image smear

Rather than expose a mode switch, DPT removes the possibility entirely.

This avoids misuse and keeps the plugin honest.

---

## Recommended Uses

* Dialogue placement
* Foley and SFX positioning
* Animation and film mixing
* Game audio (non-HRTF pipelines)
* Headphone-first music production
* Any situation where stereo placement must remain mix-safe

---

## When Not to Use DPT

* Mixing for speakers only
* Needing distance or depth cues
* Wanting elevation or “3D space”
* Already using a full binaural renderer

DPT is deliberately boring in isolation.
That’s why it works in a mix.

---

## Design Philosophy

DPT is built on three rules:

1. **Only implement cues humans actually use**
2. **Only apply them where they are physically valid**
3. **Never let the user break the mix**

Minimal controls are not a limitation.
They are the product.

---

## Summary

DPT is a **SAFE psychoacoustic panner**:

* Drop-in
* Headphone-correct
* Mix-reliable
* Impossible to “overdo” accidentally

It does less than most spatial tools.
That restraint is the feature.
