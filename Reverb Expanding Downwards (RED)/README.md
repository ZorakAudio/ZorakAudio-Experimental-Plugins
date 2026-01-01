# RED — Reverb Expanding Downwards

**RED** is a wet-only reverb controller that keeps reverb tails *proportional* to their source, even when monitoring conditions make tails hard to judge.

It does **not** duck reverb based on absolute level.  
It controls reverb based on **relative balance**.

---

## What RED Does

RED continuously compares:

- **Wet return level** (reverb output)
- **Reference level** (the signal that fed the reverb)

When the reverb tail becomes *too loud relative to its source*, RED smoothly reduces **only the wet signal**.  
When balance is restored, RED releases the reduction naturally.

The result is reverb that:
- Feels spacious, not washed
- Preserves early reflections
- Doesn’t “float up” after note-off
- Remains stable across gain staging changes

---

## Core Idea: Ratio-of-Return

Traditional duckers ask:  
> “Is the sidechain loud?”

RED asks:  
> “Is the reverb louder than it *should* be, relative to what created it?”

This makes RED:
- Level-independent
- Resistant to pumping
- More transparent than gates or classic duckers

---

## Signal Routing

**Inputs**
- Channels **1/2**: Wet return (post-reverb)
- Channels **5/6**: Reference signal (typically the dry source)

**Outputs**
- Channels **1/2**: Processed wet signal
- All other channels: Passed through untouched

> RED is designed for hosts that support multichannel routing (e.g. REAPER).

---

## Controls

### Amount (dB)
Maximum reduction RED is allowed to apply to the wet signal.  
This is a *ceiling*, not a constant reduction.

### Sensitivity (%)
How aggressively RED reacts when the reverb dominates its source.
- Lower values: more forgiving, longer tails
- Higher values: tighter control, faster restraint

### Release (ms)
How quickly reverb recovers after reduction.  
Also influences how gently RED engages after the reference signal ends.
- Longer: cinematic, floating tails
- Shorter: controlled, mix-ready space

---

## How It Behaves

### Reference Active
- Reverb blooms naturally
- Early reflections pass largely untouched
- Reduction engages only if the tail overwhelms the source

### Reference Stops
- Brief grace period before control engages
- Tail is gently restrained
- Decay remains natural (no chopping or muting)

### Reference Returns
- Reduction releases faster
- Clarity is restored immediately

---

## What RED Is Not

- Not a gate  
- Not a traditional expander  
- Not a sidechain ducker  
- Not a reverb replacement  

RED is a **relative balance stabilizer** for reverb returns.

---

## Design Philosophy

- **Perceptual space > raw level**
- **Smooth motion > hard thresholds**
- **Early reflections are sacred**
- **Control should feel invisible**

Designed for real-world mixing:
- Low-volume monitoring
- Noisy environments
- Long cinematic tails
- Dense arrangements

---

## Technical Notes

- RMS-based energy detection
- Gain-stage invariant
- Soft-knee response
- Wet-only processing
- Stable behavior even when the reference signal is muted

Dynamics are intentionally smooth rather than sample-deterministic, favoring musical motion over rigid timing.

---

## Known Limitations

- Requires multichannel routing (not all hosts support channels 5/6)
- Not a sample-accurate clone of the original JSFX reference
- Intended for reverb **returns**, not ambient generators

---

## In One Sentence

**RED keeps reverb big without letting it get stupid.**
