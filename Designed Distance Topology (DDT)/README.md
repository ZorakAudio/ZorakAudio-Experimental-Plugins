# Designed Distance Topology (DDT) v2
### Distance + room perception shaper — topology-driven, not a “reverb knob soup”

**DDT moves a sound forward/back in space** by manipulating the cues your brain actually uses to judge distance:
- **Direct vs diffuse balance** (how much “source” vs “room”)
- **Early reflection timing/density** (how soon and how many reflections arrive)
- **Air absorption** (high-frequency loss grows with distance and time)
- **Stereo coherence** (far sources feel less “wide-source,” while rooms can stay wide)

It’s not trying to be a physically perfect room simulator.
It’s a **perceptual distance tool** designed to be fast to use and hard to break.

---

## What it does (in plain steps)

Internally DDT:
1. Builds a deterministic “reflection cloud” (multi-tap delays) from your settings  
2. Collapses the **source** toward mono as Distance increases (stable far imaging)  
3. Splits into **Direct** and **Diffuse** energy (distance shifts that ratio)  
4. Applies **air absorption** with time-dependence (late energy is darkest)  
5. Adds **stereo decorrelation** to the diffuse field via small ITDs  
6. Mixes wet/dry with an energy-style crossfade and applies output trim  
7. Hard-clamps final output as a last-resort safety guard

---

## What it’s good at

- **Pushing sounds back** without obvious EQ moves  
  (dialogue, vocals, guitars, synth leads)
- **Making “closer vs farther” layers** in dense productions  
  (foreground vs background staging)
- **Game/film perspective shifts**  
  (footsteps down a hall, distant machinery, off-screen cues)
- **Making a stereo source feel “far”** without a weird wide wobble  
  (Distance collapses the source while Width widens the room field)

---

## Controls

### 1) Distance
The main perspective knob.
- Low = more direct, earlier reflections, tighter space impression
- High = more diffuse dominance, later/denser reflections, mild attenuation, more mono source

### 2) Spread
Controls how “scattered” the reflection timing feels.
- Low = reflections cluster more (tighter, more focused)
- High = reflections distribute more evenly and densely (more diffuse)

### 3) Air Absorb
High-frequency damping that increases with distance and time.
- Low = brighter, more “near”
- High = darker, more “far / through air,” especially in late energy

### 4) Width
Stereo width of the *diffuse field* (not the dry source).
- Low = more centered/mono room return
- High = wider, more decorrelated space impression

### 5) Room Size
Scales the time geometry of the reflections.
- Low = small/tight environment
- High = longer reflection window and later early→late transition

### 6) Quality (Eco → Extreme)
Density vs CPU.
Higher quality allows more taps and a longer reflection window.

### 7) Amount (Wet %)
Wet blend.
- 0% = dry only
- 100% = pure DDT output (use Monitor to audition components)

### 8) Output (dB)
Final trim after mixing.

### 9) Monitor (Normal / Direct / Diffuse / Bypass)
Debug and learning tool:
- **Direct**: only the direct component (after distance/air filtering)
- **Diffuse**: only reflections (early+late)
- **Bypass**: audition input as the monitor signal (most meaningful at Amount=100%)
- **Normal**: the full result

---

## Safety / behavior notes

- **Tap count and delay times are bounded** (Quality cap + buffer clamp) so it stays stable in realtime.
- **Topology is deterministic**: the “room pattern” doesn’t randomly reshuffle every block.
- **Final output is hard-clamped** as a last-resort guard against runaway gain.

---

## Quick Start (recommended)

Start here for most material:

- **Amount:** 100%  
- **Quality:** Moderate  
- **Distance:** 30–50  
- **Room Size:** ~50  
- **Spread:** ~50  
- **Air Absorb:** ~40  
- **Width:** ~55  
- **Output:** 0 dB

Then:
- Want it **farther**? Increase **Distance**.
- Want a **bigger environment** (not just farther)? Increase **Room Size**.
- Want less “sparkly agitation” in the space? Increase **Air Absorb**.
- Want the room to feel **less clustered**? Increase **Spread**.
- Want a **wider room** without widening the dry source? Increase **Width**.

---

## Example recipes

### “Background dialogue” (behind the camera)
- Distance: 65  
- Room Size: 55  
- Spread: 60  
- Air Absorb: 55  
- Width: 40  
- Amount: 100%  
- Output: -1 to -3 dB (match level)

### “Close but in a room” (subtle depth)
- Distance: 15–25  
- Room Size: 40  
- Spread: 35–50  
- Air Absorb: 15–30  
- Width: 50–65  
- Amount: 40–70%

### “Big space wash (but still readable)”
- Distance: 45  
- Room Size: 85  
- Spread: 70  
- Air Absorb: 40–60  
- Width: 75  
- Amount: 100%

---

## Why this implementation is different

Most “distance” tools either:
- act like a simple reverb send,
- do a static EQ + volume drop,
- or require a bunch of technical parameters (predelay, diffusion, ER level, RT60…)

DDT is built around **topology**: a controlled reflection geometry that responds to a few perceptual knobs:
- **Distance** changes *multiple* cues at once (ratio, timing, density, coherence).
- **Room Size** scales time geometry without forcing you to tune a reverb model.
- **Air Absorb** is time-aware (late energy is damped more than early/direct).
- **Width** lives primarily in the diffuse field, so the source can stay stable.

In short: you steer “where it feels like it is,” not “how many milliseconds the 9th reflection is.”

---

## Psychoacoustics (why this works)

Humans judge distance using a mash-up of cues:
- **Direct-to-reverberant ratio**: farther sources have relatively more room energy.
- **Early reflection timing/density**: bigger/farther spaces produce later, denser arrivals.
- **High-frequency loss**: air and surfaces reduce HF over time and distance.
- **Interaural coherence**: wide, decorrelated diffuse energy reads as “space,” while far sources often read as more coherent/centered.

DDT targets those cues directly, so you get strong distance perception without needing aggressive EQ or obvious compression.

---
