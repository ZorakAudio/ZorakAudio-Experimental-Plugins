# Temporal Structural EQ (TSEQ)
### Time-domain “detail EQ” — sculpt structure by timescale, not by frequency

**TSEQ** is an EQ that doesn’t target bands of Hz.  
It targets **bands of time**.

It decomposes audio into 5 “detail buckets” by subtracting progressively smoother versions of the signal (Savitzky–Golay smoothing differences). Each bucket represents structure at a specific **temporal scale**:

- Bucket 1: **0.25 ms** (ultra-fine edges)
- Bucket 2: **0.75 ms** (snap/attack skin)
- Bucket 3: **2.0 ms** (texture detail)
- Bucket 4: **5.0 ms** (motion/articulation)
- Bucket 5: **10 ms** (macro contour)

Then you boost/cut each bucket with a single signed % control.

This makes it ideal for:
- Removing “grit” without dulling brightness
- Adding transient edge without EQ harshness
- De-texturing noisy detail while keeping body
- Shaping articulation/motion in foley and SFX
- Extracting control signals (CV) from micro vs macro activity

---

## What It’s Doing (Conceptually)

1) A delay line buffers the signal (PDC aligned).  
2) Multiple Savitzky–Golay smoothers create progressively smoother trajectories of the same audio.  
3) Adjacent smoothers are subtracted to form **detail bands in time**:

- `b0 = dry - smooth(0.25ms)`
- `b1 = smooth(0.25ms) - smooth(0.75ms)`
- `b2 = smooth(0.75ms) - smooth(2ms)`
- `b3 = smooth(2ms) - smooth(5ms)`
- `b4 = smooth(5ms) - smooth(10ms)`

4) Each bucket is **gated** by an RMS threshold to avoid lifting noise/floor.  
5) Bucket gains are smoothed to avoid zippering.  
6) Output = dry + (weighted sum of gated buckets), then DC-blocked and peak-limited on the detail path.

Net effect: you’re boosting/cutting “structure” at different time resolutions.

---

## Controls

### Threshold (dB)
Activity gate for the buckets.
- Lower threshold → more constant detail extraction (can raise noise/roomtone texture).
- Higher threshold → only strong events trigger buckets (cleaner, more selective).

If you hear “hash” on quiet material, raise Threshold.

### Output Gain (dB)
Final trim after processing.  
Use it to level-match. This plugin can easily trick your ear by loudness.

### Bucket 1–5 (%)
Signed bucket gains.
- Positive = **add** that time-scale detail.
- Negative = **remove** that time-scale detail.

Bucket time meanings (as labeled in the UI):
- **B1 (0.25 ms):** micro-edge / grit / tiny clicks
- **B2 (0.75 ms):** snap / transient skin
- **B3 (2.0 ms):** texture detail
- **B4 (5.0 ms):** articulation / movement
- **B5 (10 ms):** macro contour / swell structure

---

## Outputs (Pins)

### Output L/R
Processed audio.

### CV Fine (pin 3)
Envelope follower of the *fine* activity:
- Derived from Buckets 1–2 magnitude
- Smoothed with attack/release
Use this for: transient-driven modulation, micro-shimmer, micro-ducking, grit detection.

### CV Coarse (pin 4)
Envelope follower of the *coarse* activity:
- Derived from Buckets 3–5 magnitude
Use this for: macro motion control, swell detection, “gesture” modulation, movement-following reverb/width.

---

## Quick Start Recipes

### “De-grit without dulling”
- Threshold: −36 to −24 dB (keep it selective)
- Bucket 1: −20 to −60
- Bucket 2: −10 to −40
- Others: 0
- Output: level-match

### “Add bite / definition”
- Threshold: −54 to −42 dB
- Bucket 2: +15 to +50
- Bucket 1: +5 to +25 (sparingly)
- Bucket 3: +0 to +20 (optional)

### “Reduce chatter / harsh texture”
- Threshold: −42 to −30 dB
- Bucket 3: −10 to −50
- Bucket 2: −0 to −25 (optional)

### “Add motion / articulation (foley, cloth, body)”
- Threshold: −48 to −36 dB
- Bucket 4: +10 to +40
- Bucket 5: +5 to +25

### “Flatten uneven swells”
- Threshold: −42 to −30 dB
- Bucket 5: −15 to −60

---

## UI / Workflow Notes

- **Drag** on a bucket to set its signed gain quickly.
- **Shift+click** auditions a bucket (solo mode) to learn what each time scale contains.
- **Right-click** resets the hovered control to default.
- **Mouse wheel** edits the hovered control (CTRL modifies step size).

---

## Safety / Behavior Notes

- Detail path is **DC-blocked** and has a **ceiling limiter** to prevent runaway spikes when boosting buckets.
- The gate is **per bucket** (RMS-based), so a bucket can be quiet/disabled while others remain active.
- Gains are smoothed to avoid zipper noise when automating.

---

## Mental Model

Think of TSEQ as:
- A **time-domain multiband enhancer/dehancer**
- A way to sculpt “texture vs motion vs contour” independently
- A generator of modulation signals that track **fine** vs **coarse** structure

If a normal EQ asks “*which frequencies are too much?*”  
TSEQ asks “*which time scales are too much?*”