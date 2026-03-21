# Roomalizer Mini (Safe ER Placement) — README

Roomalizer Mini is a minimal, SAFE early-reflection (ER) placer for quick “put it in a space” positioning without a full reverb tail. It’s designed for speed: one preset selector + a handful of macro knobs, while the engine auto-maps to stable predelay, ER timing, diffusion, distance EQ, stereo decorrelation, and output limiting.

The goal is not realism by brute force. The goal is **credible spatial cues** that survive mono and don’t explode your gain staging.

---

## What it does (in plain terms)

- **Creates an early-reflection field** using a fixed set of taps (ER “events”) placed inside a safe timing window.
- **Adds subtle diffusion** (allpass) so ERs feel like a room and not like discrete echoes.
- **Shapes distance** using predelay + ER timing + a distance-style EQ (HPF rises, LPF falls with depth).
- **Adds width** via small decorrelation delays and controlled crossfeed.
- **Protects output** with a safety limiter (instant attack, smooth release).

This is best thought of as a “spatializer / placer,” not a long reverb.

---

## Controls

### Placement (Preset)
Chooses a macro profile that changes internal caps and defaults for common source categories:
- **Neutral**: balanced room cues
- **Voice**: more predelay, tighter width, gentler ER density
- **Foley**: controlled early cues, moderate diffusion
- **SFX**: wider caps, punch-preserving transient behavior
- **BG**: larger, denser, more “environmental” ER field

### Depth
Perceived distance. Increasing Depth:
- increases predelay and pushes ERs later
- increases ER prominence (within safe bounds)
- applies distance EQ (more HPF, less LPF)
- slightly reduces wet presence (auto de-presence)

### Size
Room scale. Increasing Size:
- spreads ER taps farther apart (larger time window)
- increases diffusion density (less “tap tap tap,” more “space”)
- subtly changes the ER topology to read larger

### Width
Stereo spread of the ER field. Increasing Width:
- increases decorrelation delay between L/R ER streams
- increases crossfeed contribution to widen the image
- stays below the “this is chorus” zone by design

### Mix
Constant-power wet/dry blend. This keeps perceived loudness stable as you sweep Mix.

### Tone (Wet Tilt @ ~4 kHz)
Tilts the wet field brightness:
- positive = brighter, more present reflections
- negative = darker, softer reflections
Depth also nudges the wet field slightly darker so “farther” feels farther.

### Mono-safe (Low-end centered)
Reduces **only low-frequency side energy**, keeping bass centered while preserving width in the highs.

---

## Recommended use

- **Voice**: start with *Voice* preset, Depth 20–50, Size 20–40, Width 10–30, Mix 10–25.
- **Foley**: *Foley* preset, Depth 10–35, Size 20–50, Width 15–40, Mix 10–30.
- **SFX**: *SFX* preset, Depth 20–70, Size 30–70, Width 25–70, Mix 10–45.
- **Backgrounds**: *BG* preset, Depth 30–90, Size 50–100, Width 20–60, Mix 20–60.

---

## Why it stays “SAFE”

- ER timing is clamped to avoid obvious echo timing.
- Mix/Width have preset-specific caps to prevent extreme settings from breaking the image.
- Filters are clamped to safe ranges (no “Nyquist death”).
- Output limiter prevents spikes during parameter changes and dense ER moments.
- Mono-safe reduces only low side energy instead of collapsing everything to mono.

---

## Monitoring / Debugging

Use REAPER’s wet/dry or delta monitoring if you want to hear what the ER field is adding. The ER content should sound like “space cues” rather than a reverb tail.

---

## Notes / Limitations

- This is not a late-reverb / hall simulator. If you need tails, put a proper reverb after it.
- ER taps are deterministic (stable and predictable), not randomized.
- The psychoacoustic target is *placement*, not “room realism at all costs.”

---
