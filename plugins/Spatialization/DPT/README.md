# Designed Panning Topology (DPT)

## What it is
DPT is a **minimal psychoacoustic panner** with separate speaker and headphone behavior.

In Speakers mode it behaves like a clean equal-power pan. In Headphones mode it adds far-ear realism through a small time difference, head-shadow style darkening, and a controlled diffuse fill so hard pans feel less synthetic.

---

## Why use it
Use DPT when you want a panner that stays simple but feels more natural than a bare balance control.

---

## Quick start
1. Set **Mode** to Speakers or Headphones.
2. Move **Position** left or right.
3. Raise **Natural** if you want more realism and less synthetic hard-panning feel.
4. Trim with **Output** if needed.

---

## Main controls
### Position
Left/right placement.

### Natural
Macro amount for the added psychoacoustic realism in the current source.

### Mode
Speakers or Headphones behavior.

### Output
Final trim.

---

## Notes
Automation is smoothed in the current source so sweeps across center stay clean.

---

## In one sentence
DPT is the clean, simple natural-feel panner in the catalog for either speaker or headphone use.


## Cooperative idle

The native host may sleep after 8192 consecutive silent mono samples, settled pan/naturalness controls, and exact zero-input fixed points of the active headphone filters. Both modes are covered. The host also requires exact silent output and no pending activity. New audio, controls, or host events wake processing. Offline rendering always processes. Stock REAPER continues processing normally; the readiness variable is a native-host hint.
