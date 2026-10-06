# ERB Tilt

## What it is
ERB Tilt is a **perceptual tilt EQ** built in ERB space instead of ordinary straight-line frequency thinking.

The current source does four things:

- applies a bright↔dark tilt across an ERB-spaced filterbank
- lets you choose the pivot frequency
- compensates loudness so tilt changes do not fool you as easily
- adds a roughness guard to reduce harsh high-band modulation artifacts

---

## Why use it
Use it when a normal tilt EQ feels too coarse, too loudness-biased, or too easy to overdo.

---

## Quick start
1. Set **Tilt** for brighter or darker voicing.
2. Move **Pivot** until the hinge point feels right.
3. Keep **Comp** high if you want louder/quieter perception to stay controlled while auditioning.
4. Raise **Roughness Guard** if brightening starts to turn into fizzy harshness.

---

## Main controls
### Tilt
Perceptual tilt amount. Positive is brighter, negative is darker.

### Pivot
The frequency that stays most anchored while the tilt rotates around it.

### Comp
Loudness compensation based on the current source’s A-weighted ERB energy matching.

### Roughness Guard
Upper-band anti-harshness protection.

---

## In one sentence
ERB Tilt gives you a brighter/darker macro that stays more perceptual, more level-aware, and less nasty than a crude tilt filter.

## Native FAUST filter-bank implementation

The compiled plugin now uses `@faust` for its sample filter bank, while EEL keeps control calculations and the existing display. FAUST with its LLVM backend is needed at build time; it is not a runtime dependency.

Stock REAPER JSFX does not support `@faust`. Its original source remains available as `src/ERB Tilt.jsfx`; use that file directly in REAPER. The native build entry is `src/ERB Tilt Faust.jsfx`.

Detector state still evolves during silence, so this plugin does not grant cooperative sleep.
