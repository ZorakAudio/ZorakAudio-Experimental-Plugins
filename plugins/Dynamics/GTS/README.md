# Gaussian Transient Shaper (GTS)

## What it is
GTS splits the signal into **attack** and **sustain** by using a true Gaussian FIR blur.

In the current Faust source:

- the blurred path acts like the sustain component
- the aligned dry-minus-blur path acts like the attack component
- you rebalance them with dedicated gain controls

That makes it a very direct transient shaper with a more explicitly defined split than the usual “attack/sustain” black box.

---

## Why use it
Use GTS when you want to:

- sharpen attacks
- soften attacks
- pull sustain down
- make sustain bloom more
- rebalance punch versus body in a very controlled way

---

## Quick start
1. Start with **Attack Gain** and **Sustain Gain** at 0 dB.
2. Set **Gaussian Sigma** to decide how broad the sustain blur should be.
3. Raise or lower **Attack Gain** to emphasize or soften the transient portion.
4. Raise or lower **Sustain Gain** to thicken or dry out the body.
5. Use **Mix** and **Output Gain** to blend and level-match.

---

## Main controls
### Gaussian Sigma
The width of the Gaussian smoothing window, in milliseconds. Lower values behave more like very fine transient separation; higher values create a broader sustain estimate.

### Attack Gain
Gain applied to the extracted attack component.

### Sustain Gain
Gain applied to the Gaussian-smoothed sustain component.

### Mix
Wet/dry blend between the shaped result and the aligned dry path.

### Output Gain
Final gain trim.

---

## Notes
The current source declares **128 samples of latency** because the Gaussian FIR is 257 taps wide and the dry path is delay-aligned to match it.

That latency is part of the design, not a bug.

---

## In one sentence
GTS uses Gaussian blur to split attack from sustain and lets you rebalance them directly.
