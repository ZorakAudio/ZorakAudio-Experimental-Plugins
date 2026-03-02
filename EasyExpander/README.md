# EasyExpander (ERB Detector)

**Minimal Downward Expander with Perceptual Detector**

---

## What It Is

EasyExpander is a **broadband downward expander** driven by an **ERB-spaced multi-band detector**.

Important distinction:

* Audio path = **clean, untouched**, except for one smooth gain multiplier.
* Detection path = **ERB-style filterbank** → weighted loudness estimate → soft-knee expander.

You get perceptually aware noise reduction without the twitchy, metallic chatter typical of simple RMS gates.

This is not a gate that slams shut. It’s a controlled downward expander designed to reduce the noise floor while preserving tone and dynamics.

---

## Why ERB Detection?

Most expanders measure raw broadband RMS. That overreacts to low rumble and underreacts to mid-range content.

This plugin:

* Splits the detector into **8 ERB-spaced bands (≈80 Hz – 8 kHz)**
* Applies perceptual weighting (mid-band emphasis)
* Smooths power per band
* Recombines into a single loudness estimate

Result:
The detector behaves more like hearing, not a voltmeter.

The gain reduction remains broadband for phase integrity and tonal safety.

---

## Core Controls

### Threshold (dB)

Level where expansion begins.

* Higher threshold → more material reduced
* Lower threshold → more transparent, subtle cleanup

Measured against the ERB detector loudness.

---

### Depth (dB)

Maximum attenuation below threshold.

* 6–12 dB → gentle floor control
* 18–30 dB → clear noise tightening
* 40+ dB → gate-like behavior

Depth is hard-limited for safety.

---

### Contour (0–100)

Controls envelope behavior and effective ratio.

Internally maps to approx **1.5:1 → 4:1 expansion ratio**.

* 0 = gentle, smooth, glue-like
* 50 = controlled tightening
* 100 = firm, punchy floor clamp

Higher contour = faster and more assertive reduction feel.

---

## Detector Shaping (Detection Only)

These filters **do not touch the audio path**.

They only change what the detector “listens” to.

### Detector HPF (Hz)

High-pass for detection.

Use when:

* Low rumble is falsely opening the expander
* You want speech or mids to drive expansion

Example: 80–150 Hz for dialogue.

---

### Detector LPF (Hz)

Low-pass for detection.

Use when:

* Hiss is falsely opening the expander
* You want body over air to drive the gate

Example: 6–10 kHz for vocal cleanup.

---

## Internal Behavior (Why It Feels Stable)

Several stability layers prevent chatter:

* Soft knee (6 dB ramp into ratio)
* 2 dB hysteresis (prevents flutter at threshold)
* Separate open/close time constants
* Adaptive close speed (faster when transients end)
* Multi-band smoothed power detection

This is why it closes smoothly instead of snapping.

---

## Gain Computer Summary

Below threshold:

```
GR = (Threshold - DetectorLevel) × (Ratio - 1)
```

Clamped by Depth.

Above threshold:

```
GR = 0
```

Then smoothed.

No upward expansion. No compression. No tone shaping.

Only clean downward gain control.

---

## Typical Use Cases

Vocal cleanup
Dialogue floor control
Foley tightening
Room tone shaping
Synth tail cleanup
Reverb return cleanup

For transparent vocal work:

* HPF detector around 100 Hz
* LPF detector around 8–12 kHz
* Threshold around -45 dB
* Depth 12–20 dB
* Contour 40–60

---

## What This Is Not

Not multiband processing
Not spectral gating
Not dynamic EQ
Not upward expansion
Not noise reduction via subtraction

It is perceptually informed broadband downward expansion.

---

## Metering

The GFX displays:

* Detector level (dB)
* Threshold and hysteresis markers
* Gain reduction amount
* State indicator (OPEN / EXPANDING)

If the lamp is lit, you are below threshold.

---

## Philosophy

Most expanders react to amplitude.

This one reacts to structured, smoothed, perceptually weighted energy.

Cleaner floor control.
Less tonal damage.
No transient murder.

Minimal surface. Controlled behavior. Stable results.
