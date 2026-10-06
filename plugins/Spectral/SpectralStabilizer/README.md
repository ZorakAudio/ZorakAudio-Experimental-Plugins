# Spectral Stabilizer

## What it is
Spectral Stabilizer is a **SAFE excess-only tonal stabilizer**.

The current source does not boost missing bands. It only attenuates **excess** energy relative to a smoothed baseline. That makes it more about calming spectral swing than about “EQing a sound into shape.”

---

## Why use it
Use it when a source keeps swinging between too bright, too spiky, too muddy, or too uneven from moment to moment—even though the average tone is broadly okay.

---

## Quick start
1. Start with the default settings.
2. Raise or lower **Sigma** to decide how smooth the baseline should be.
3. Increase **Depth** until the tonal swing calms down.
4. Use **MotionBias** to keep transient-rich material from being over-grabbed.

---

## Main controls
### Sigma
How smooth the expected spectral baseline is. Higher = steadier but slower and potentially duller. Lower = livelier and more reactive.

### Depth
How strongly excess energy is pushed down.

### MotionBias
How much fast spectral movement is allowed to pass before stabilization grabs it.

---

## Notes
If the result starts feeling “held,” smeared, or pumped, back Depth down before assuming Sigma is wrong.

---

## In one sentence
Spectral Stabilizer calms tonal swing by shaving off excess energy instead of boosting missing bands.

## Native FAUST filter-bank implementation

The compiled plugin now uses `@faust` for its sample filter bank, while EEL keeps control calculations and the existing display. FAUST with its LLVM backend is needed at build time; it is not a runtime dependency.

Stock REAPER JSFX does not support `@faust`. Its original source remains available as `src/Spectral Stabilizer.jsfx`; use that file directly in REAPER. The native build entry is `src/Spectral Stabilizer Faust.jsfx`.

Detector state still evolves during silence, so this plugin does not grant cooperative sleep.
