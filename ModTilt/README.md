# ModTilt (SAFE)

**Envelope Tilt Shaper**

Reshapes modulation energy above and below a pivot frequency while preserving loudness, dynamics, and gain safety.  
Stereo-linked, self-trimming, and bounded by design.

> Use it to reshape **movement**, not add it.

---

## What It Does

ModTilt operates on the **amplitude envelope**, not the spectrum.

1. Detects a linked stereo RMS envelope  
2. Separates slow baseline from modulation  
3. Splits modulation around a pivot frequency  
4. Tilts fast vs slow motion in opposite directions  
5. Recombines and applies as a gain ratio  
6. Smooths, clamps, and auto-trims the result  

No pumping. No clipping. No gain drift.

---

## Controls

### Tilt (dB)
Redistributes modulation energy.

- Positive values emphasize **fast detail**
- Negative values emphasize **slow movement**
- Internally split ±½ tilt above / below the pivot

---

### Pivot (Hz)
Defines the modulation crossover frequency.

- Lower = more motion treated as *fast*
- Higher = more motion treated as *slow*

---

### Mix
Dry / wet blend after processing.

Auto-trim remains active at all mix values.

---

## Safety Guarantees

- Hard gain clamp (~−3.5 dB to +3.5 dB instantaneous)
- Relative envelope floor prevents collapse
- Ratio smoothing prevents zipper noise
- Continuous auto-trim preserves perceived loudness
- Stereo-linked detection prevents image shift

You cannot make this plugin blow up a mix.

---

## What It’s Good For

- Adding detail motion without harshness  
- Reducing pumping or breathy envelopes  
- Making modulation feel intentional and controlled  
- Vocals, pads, ambiences, foley beds  
- Pre-conditioning modulation before downstream FX  

---

## What It’s Not

- Not a compressor  
- Not an EQ  
- Not a transient designer  
- Not a loudness maximizer  

Think **motion sculptor**, not dynamics hammer.

---

## Design Philosophy

ModTilt treats modulation as a first-class signal.

By shaping envelope curvature with bounded ratios and automatic compensation, it stays musical, predictable, and fatigue-free.

Set it once. Forget it’s there. Notice the mix feels better.
