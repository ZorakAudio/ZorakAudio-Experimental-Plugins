# Vocal Air Recovery (VAR)

## What it is
VAR is a **stereo air-restoration tool** built around bounded HF expansion plus a synthetic high-air halo.

Important naming note: the current repo metadata presents this plugin as **Vocal Air Recovery (VAR)**, while the current Faust source declaration string says **Vocal Air Restore (VAR)**. This README keeps the repo-facing name but documents the source behavior as it exists now.

The current source describes the process very clearly:

- a level-invariant, LoG-like curvature detector looks for the right kind of missing air behavior
- a bounded high-frequency expansion stage adds real HF lift
- a band-limited noise layer adds a controlled “air halo” rather than just raw fizz

---

## Why use it
Use VAR when a vocal or airy source feels **closed-in, dull, or de-aired**, but a normal exciter sounds too fake or too aggressive.

---

## Quick start
1. Start with **Air Amount** low to moderate.
2. Raise **Sensitivity** until the air behavior starts to wake up naturally.
3. Set **Detector Floor** high enough that silence and low-level junk do not keep the detector open.
4. Level-match by ear after insertion.

---

## Main controls
### Air Amount
Overall amount of air restoration. In the current source this also scales the bounded HF expansion ceiling and the added halo contribution.

### Sensitivity
How easily the curvature detector decides the source should open up.

### Detector Floor
Prevents the detector from reacting to very low-level content or empty space.

---

## Notes
VAR is not meant to be a broad “make everything brighter” knob. The best settings are usually the ones that make you miss it when bypassed, not the ones that scream “exciter.”

---

## In one sentence
VAR restores upper-air openness by adding real HF lift and a controlled synthetic halo without turning the source into brittle hype.
