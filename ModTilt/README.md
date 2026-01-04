# ModTilt (SAFE)
### Envelope Tilt Shaper (not EQ) — Minimal + SAFE

**ModTilt reshapes *gain motion* over time.**  
It “tilts” the signal’s **envelope modulation** so **fast movement** (micro-transients / chatter) can be emphasized or suppressed relative to **slow movement** (body / swells), without applying any frequency filtering.

Internally it:
1) Detects a linked-stereo envelope (RMS-ish)  
2) Splits envelope motion into **slow** and **fast** components around a **Pivot (Hz)**  
3) Applies a dB “tilt” to those motion bands  
4) Converts the result into a **bounded gain ratio**, smooths it, and applies it to the audio  
5) Auto-trims average level so loudness stays stable

This is a **time-domain “feel” shaper**: less jitter vs more snap, without changing the tone via EQ.

---

## What it’s good at
- **Wet/granular textures:** reduce “nervous sparkle chatter” without dulling the sound
- **Vocals/dialogue:** tame micro-tension and spitty edge without classic de-esser artifacts
- **Ambience/rooms:** keep loudness and presence while reducing agitation
- **SFX layers:** make movement feel more “solid” or more “electric” without tonal EQ

---

## Controls
### 1) Tilt (dB)
Tilts envelope motion around the pivot:
- **Negative**: favors **slow** motion → smoother / thicker / calmer
- **Positive**: favors **fast** motion → snappier / more animated / more “alive”

### 2) Pivot (Hz)
The crossover between slow vs fast envelope motion.  
Typical range: **2–5 Hz**.
- Lower = more motion treated as “fast”
- Higher = only very quick motion treated as “fast”

### Mix
Blends processed vs dry.  
(Useful for subtlety; processed path is auto-trimmed.)

---

## Safety / behavior notes
- Gain ratio is hard-clamped to a safe range (won’t explode levels)
- Linked-stereo detection avoids image wobble
- Auto-trim keeps long-term loudness stable

---

## Quick Start (recommended)

ModTilt is designed to be hard to misuse. Start here:

- **Mix:** 1.0  
- **Pivot:** ~3 Hz  
- **Tilt:** **−6 dB**  

For most material, **−6 dB Tilt** immediately feels more natural.  
This setting suppresses fast envelope chatter while reinforcing slower body movement, reducing irritation and fatigue without dulling tone.

From there:
- If the sound still feels **nervous / jittery**, try **−3 to −6 dB**.
- If it feels **flat / sleepy**, move Tilt toward **0 dB** or slightly positive.
- If you can clearly “hear the plugin,” reduce Tilt magnitude or lower Mix.

A good rule of thumb:
- **0 dB** = raw / literal motion  
- **−6 dB** = natural / comfortable  
- **+ dB** = excited / animated

### Creative tip: tension & release without EQ or compression

Automating **Tilt** from negative to positive can create a strong sense of
**tension and release** without changing overall loudness, dynamics, or spectral balance.

- **Negative Tilt** → calmer, grounded, held-back motion
- **Positive Tilt** → nervous, urgent, energized motion

Because ModTilt reshapes *envelope motion* rather than frequency or level,
this kind of automation increases perceived intensity without obvious processing.

Try slow ramps (e.g. −6 dB → 0 dB or +1 dB) leading into a drop, impact, or climax.

