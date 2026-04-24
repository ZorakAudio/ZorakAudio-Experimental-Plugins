# Reverb Expanding Downwards (RED)

## What it is
RED is currently best thought of as a **compact wet-tail tamer** rather than a full general-purpose dynamics processor.

Important naming note: the repo metadata currently calls this **Reverb Expanding Downwards (RED)**, but the current Faust source declaration still presents it as **Reverb Tail Tamer (Wet 1/2, Ref In 5/6)**. This README follows the repo-facing RED name while documenting what the source actually does today.

The current source is a small reference-keyed reverb return controller:

- wet return on 1/2
- reference input on 5/6
- smooth wet-only gain control
- pass-through of the extra channels

---

## Why use it
Use RED when you want a **small, practical wet-return governor** and do not need the fuller role-aware behavior of RTT.

---

## Quick start
1. Feed the wet return to channels 1/2.
2. Feed the reference or driving source to channels 5/6.
3. Set **Amount** for the maximum duck depth.
4. Use **Sensitivity** to decide how easily the return is judged too loud relative to the reference.
5. Set **Release** for how fast the tail recovers.

---

## Main controls
### Amount
Maximum duck depth in dB.

### Sensitivity
How easily the wet return is considered too loud relative to the reference.

### Release
Recovery time for the wet gain reduction.

---

## Routing / notes
The current Faust source uses a **6-in / 6-out** layout.

- **1/2** = wet return to be controlled
- **5/6** = reference input
- **3/4** = passed through unchanged

Use RED when you want the return to react to a reference without rebuilding the more elaborate RTT routing.

---

## In one sentence
RED is the lean wet-tail controller in the catalog: simple controls, reference-aware ducking, and minimal routing drama.
