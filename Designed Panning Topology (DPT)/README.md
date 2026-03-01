# DPT Lite — Natural Psychoacoustic Panner (Headphones/Speakers)

DPT Lite is a minimal panner designed to feel “normal” in both speakers and headphones.

- **Speakers mode**: clean, predictable **equal-power panning**.
- **Headphones mode**: adds a small set of psychoacoustic cues so hard pans don’t feel like a sound is stapled to one ear.

This plugin is intentionally small: one Position knob, one “Natural” macro, one Mode switch, and Output trim.

---

## What it does

### Speakers Mode (clean pan)
- Uses an equal-power law to maintain stable perceived loudness while panning.
- No timing or coloration tricks.

### Headphones Mode (binaural-ish pan)
Adds three cues that commonly help headphone panning feel less artificial:

1. **Far-ear floor (balance fix)**
   - Prevents the far ear from going unnaturally silent on hard pans.
   - Increases with **Natural** and with pan amount.

2. **Interaural Time Delay (ITD)**
   - Delays the far ear slightly to mimic arrival time differences around the head.
   - Max delay ≈ **0.66 ms** at hard pan, scaled by **Natural**.

3. **Head shadow (spectral darkening)**
   - Low-passes the far ear more as pan increases, imitating high-frequency shadowing by the head.
   - The cutoff drops as pan increases and as **Natural** increases.

4. **Micro-diffuse far-ear fill**
   - Adds a tiny “spread” component to the far ear using short taps (~1–3 ms) that are also low-passed.
   - This helps reduce razor-sharp ear isolation without turning into reverb.

All headphone processing is designed to stay **stable** and **monophonic-predictable** (it derives from a mono sum internally), so it behaves consistently across sources.

---

## Controls

### Position (L/R)
Moves the source left/right.
- -100 = hard left
- +100 = hard right

In Headphones mode, this also drives the strength of the psychoacoustic cues (more pan = more cues, depending on Natural).

### Natural
Macro amount for “real-world” headphone cues.
- 0%: mostly a clean pan with minimal coloration
- Higher: more ITD, more head shadow, more far-ear fill, more diffuse support

Rule of thumb:
- **Low** for surgical placement and minimal tonal shift
- **High** when hard pans feel too synthetic or fatiguing on headphones

### Mode
- **Speakers**: equal-power pan only
- **Headphones**: adds ITD + shadow + diffuse far-ear fill

### Output (dB)
Final level trim after processing (±12 dB).

---

## Practical starting points

- **General headphone panning**: Natural 40–70%
- **Super-clean / technical placement**: Natural 0–25%
- **Wide stereo illusion without “one-ear stapling”**: Natural 70–100%, then adjust Output to match loudness

---

## Notes / Design intent

- **Automation smoothing (~20 ms)** is used for Position and Natural to prevent zipper noise and clicks.
- Output is safety-clamped to prevent runaway peaks (this is a panner, not a loudness generator).
- This is not a full HRTF binaural renderer—just the small set of cues that reliably improves headphone pans in a minimal CPU footprint.

---

## When not to use it

- If you need **true externalization** (front/back, elevation, room interaction), you want a full binaural/HRTF solution.
- If your mix already has strong spatial cues from early reflections/room mics, keep Natural lower to avoid over-coloring.

---

## Version / License

Include your preferred version tag and license statement here.