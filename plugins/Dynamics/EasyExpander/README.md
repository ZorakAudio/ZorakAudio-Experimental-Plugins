# EasyExpander

## What it is
EasyExpander is a **minimal broadband downward expander** driven by a perceptually weighted ERB detector.

The current source keeps the audio path simple. The plugin is mainly about detector behavior, thresholding, and smooth gain reduction rather than a long page of extra features.

---

## Why use it
Use EasyExpander when you want to:

- push down low-level wash or spill
- clean up tails between phrases
- keep pauses quieter
- gate gently without hard chatter

---

## Quick start
1. Set **Threshold** where the source should stop being considered “active.”
2. Raise **Depth** until the cleanup is doing useful work.
3. Adjust **Contour** to decide whether the action should feel softer or more assertive.
4. Use **Detector HPF** and **Detector LPF** to decide what should trigger the expander without changing the tone of the output.

---

## Main controls
### Threshold
The activation point for the expander.

### Depth
Maximum amount of downward expansion.

### Contour
The character of the action: gentler and softer at lower settings, harder and more gate-like at higher settings.

### Detector HPF
Keeps low-frequency content from triggering the detector when that is not helpful.

### Detector LPF
Keeps upper-frequency content from driving the detector when you want a more body-focused response.

---

## Notes
The point of this plugin is not feature count. The point is a **clean detector workflow**.

If a source feels like it is gating for the wrong reasons, fix the detector filters first instead of immediately chasing the threshold.

---

## In one sentence
EasyExpander is a clean, ERB-aware downward expander for tidying material without overcomplicating the job.
