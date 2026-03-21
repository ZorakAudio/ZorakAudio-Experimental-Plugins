# ERB Tilt EQ (Perceptual)
### ERB-spaced tilt + perceptual loudness compensation + roughness guard

**ERB Tilt EQ (Perceptual)** is not a simple shelving EQ.  
It tilts the spectrum in **ERB space** (Equivalent Rectangular Bandwidth), which more closely follows how human hearing groups frequencies.

The result:  
- Bright ↔ Dark moves feel smoother and more “natural”  
- Midrange doesn’t collapse when boosting highs  
- Tonal shifts remain coherent across complex material  

This is a perceptual tool, not a surgical EQ.

---

# What Makes This Different

### 1) ERB-Spaced Tilt
Instead of linear Hz spacing, gain is distributed across **16 ERB bands**.  
This produces a perceptually even slope from lows to highs.

You don’t get:
- Hollow mids
- Strange shelf resonance
- Artificial “EQ curve” artifacts

You get a tonal rebalancing that feels integrated.

---

### 2) Loudness Compensation (A-Weighted)
Tilting a spectrum changes perceived loudness — especially when boosting highs.

This plugin:
- Measures pre-tilt vs post-tilt energy
- Uses A-weighting (human loudness curve)
- Applies smoothed global gain correction

Result:  
You can evaluate tonal change without being fooled by volume.

---

### 3) Roughness Guard (High-Band Modulation Control)

Harshness is rarely just “too much high end.”  
It’s often **amplitude modulation in upper bands** (fast envelope divergence).

The Roughness Guard:
- Detects fast vs slow envelope difference in high ERB bands (~2 kHz+)
- Reduces harsh AM salience via dynamic soft saturation
- Only activates when roughness is present

It preserves brightness while reducing grit.

This is not a static high-cut.  
It is modulation-aware.

---

# Controls

## Tilt (dB)
Perceptual spectral slope.

Positive → Brighter  
Negative → Darker  

Internally clamped for safety (prevents runaway curves).

Typical range:
±1 to ±4 dB for mastering  
±6 to ±10 dB for creative reshaping  

---

## Pivot (Hz)
The “hinge” frequency.

Bands below move opposite bands above.  
Because this operates in ERB space, the pivot behaves perceptually consistent across the spectrum.

Common uses:
- 800–1200 Hz → Vocal-centric tilt
- 1500–2500 Hz → Presence/clarity pivot
- 300–600 Hz → Weight reshaping

---

## Comp (Loudness) %
Blends in A-weighted loudness matching.

0% → No compensation  
100% → Full perceptual match  

Recommended:
10–40% for subtle control  
50–100% for mastering comparisons  

Smoothed to avoid pumping.

---

## Roughness Guard %
Modulation-driven harshness control (high bands only).

0% → Off  
100% → Maximum modulation suppression  

Best for:
- Over-bright dialogue
- Distorted material
- Saturated synths
- Cymbal splash harshness

Does not dull tone unless roughness is present.

---

# When To Use It

• Master bus tone shaping  
• Dialogue brightness control  
• Foley integration  
• Sound design contrast control  
• Pre-compression tonal balancing  
• Quick “make it darker/brighter” without breaking the mix  

---

# Gain Staging & Behavior Notes

- Extreme tilt changes spectral balance significantly — level match when judging.
- Loudness comp is smoothed and bounded (0.25x–4x) for stability.
- Roughness guard only acts above ~2 kHz.
- Designed to be stable on any material (SAFE architecture).

---

# Philosophy

This is a **perceptual rebalancer**, not a traditional EQ.

It manipulates:
- Spectral slope (ERB domain)
- Perceived loudness (A-weighted)
- High-band roughness (modulation-aware)

Use it to shift emotional tone quickly:
- Dark = intimate / grounded
- Bright = open / exposed
- Guard = refined instead of abrasive

It is intentionally minimal.
Four controls.
No unnecessary complexity.