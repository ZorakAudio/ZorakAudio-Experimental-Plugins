# Click-Be-Gone (SG)
### Psychoacoustic Temporal De-Click & Micro-Smoothing

**Click-Be-Gone (SG)** removes **high-frequency needle clicks, micro-splats, and temporally implausible artifacts** without dulling texture or tone.

Instead of working in the frequency domain, it operates in **time**. Each sample is evaluated for *temporal plausibility* using a Savitzky–Golay (SG) predictor. When a sample behaves like an out-of-place spike—brief, broadband, and inconsistent with its neighbors—it is softly rewritten. Everything else passes untouched.

The result is **less irritation and listening fatigue**, not less detail.

---

## What it’s good at
- Removing needle-like clicks in wet granular sounds  
- Taming tiny digital burrs from edits or resampling  
- Cleaning up micro lip clicks and tongue ticks in vocals  
- Reducing “nervous” high-frequency tension without de-essing  
- Smoothing abrasive detail while preserving articulation  

At **low Sensitivity**, it can act as a **gentle vocal or dialogue smoother**, removing only the most implausible micro-events.

---

## What it’s not
- Not a spectral repair tool  
- Not a de-esser  
- Not a transient shaper  
- Not meant for long pops, hard clipping, or buffer crackle  

It edits **only when something doesn’t belong**.

---

## How it works (briefly)
- A high-frequency *edge view* detects sudden micro-events  
- A relative detector compares each event to its local context  
- A Savitzky–Golay predictor tests temporal plausibility  
- Implausible samples are softly replaced, with a capped mix  
- A short hold ensures short click trains are fully covered  

All processing is level-independent and designed to fail gracefully.

---

## Controls
- **Amount** — How strongly detected artifacts are rewritten  
- **Sensitivity** — How strict the plausibility test is  
- **HPF Focus** — Which high-frequency region is examined  
- **Mode (Fast / Medium / Slow)** — Optimized presets for different click widths [0/1/2 in VST3]
- **Monitor (Output / Delta)** — Hear the result or only what’s being removed  [0/1 in VST3]

---

## Latency
- Fixed, very low latency (required for temporal prediction consistency)

---

## Design philosophy
Click-Be-Gone (SG) prioritizes **psychoacoustic plausibility over aggressive correction**.

If it isn’t confident a sample is wrong, it leaves it alone.

That restraint is why it sounds natural.
