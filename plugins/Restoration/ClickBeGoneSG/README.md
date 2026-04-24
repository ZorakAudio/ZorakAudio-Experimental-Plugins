# Click-Be-Gone (SG)

## What it is
Click-Be-Gone (SG) is a **high-frequency click and splat remover** aimed at wet, granular, or otherwise delicate material where ordinary dulling would do more harm than good.

The current Faust source uses Savitzky–Golay style prediction and replacement logic rather than treating the whole signal like a generic noise problem.

---

## Why use it
Use it when the problem is:

- HF needle-clicks
- little wet splats
- tiny granular spikes
- brief brittle defects

and you do **not** want to sand off the whole texture just to make the defect disappear.

---

## Quick start
1. Insert it on the material that contains the clicks or splats.
2. Raise **Amount** until the defects start disappearing.
3. Use **Sensitivity** to decide how aggressively events are detected.
4. Set **HPF** so the detector listens to the brittle problem area instead of the whole signal.
5. Use **Monitor = Delta** when you want to hear what is being removed.

---

## Main controls
### Amount
How much corrective replacement is allowed.

### Sensitivity
How easily the detector decides a small HF event is suspicious.

### HPF
Moves the detector upward so the plugin focuses on the clicky region.

### Mode
Behavior preset for the current source. Fast, Medium, and Slow change the predictor ladder, gating feel, hold behavior, and maximum replacement attitude.

### Monitor
Output or Delta. Delta is the easiest way to check whether you are removing the right thing.

---

## Notes
This is a targeted defect remover, not a broadband de-noiser. If the whole source is just harsh, use a different tool.

---

## In one sentence
Click-Be-Gone removes small HF defects while trying to leave the living texture of the source intact.
