# TSEQ

## What it is
TSEQ is a **Temporal Structural EQ**.

Instead of splitting audio by frequency bands, the current source splits it into **five time-scale detail buckets** using Savitzky–Golay style smoothing differences. That means you can boost or reduce detail by how fast or slow it behaves, not only by where it lives spectrally.

---

## Why use it
Use TSEQ when you want to change the amount of:

- fast detail
- slower contour
- transient bustle
- temporal clutter
- shape movement

without reaching first for a normal EQ.

---

## Quick start
1. Start by adjusting the five **Bucket Gain** controls.
2. Set each bucket’s **Threshold** so it only engages when that detail layer is actually active.
3. Use the per-bucket **Knee** controls to make activation softer or firmer.
4. Trim the result with **Output Gain**.

---

## Main controls
### Bucket gains
Buckets 1–5 are signed percentages. Positive adds that time-scale detail. Negative removes it.

### Thresholds
Each bucket has its own activity gate so silence and room tone do not get textured unnecessarily.

### Knees
Each bucket has its own activation softness.

### Output Gain
Final trim after the temporal reshaping.

---

## Routing / notes
Outputs:

- **1/2** = audio out
- **7/8** = CV Fine / CV Coarse

The current source exposes CV Fine for the faster activity region and CV Coarse for the slower activity region, so TSEQ can also be part of a modulation setup.

---

## In one sentence
TSEQ lets you EQ time-scale detail instead of only EQing frequency content.
