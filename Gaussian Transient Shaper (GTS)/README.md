# Gaussian Transient Shaper

Gaussian Transient Shaper is a linear-phase transient processor that separates **attack** and **sustain** using time-scale decomposition rather than envelope detection.

It uses a true Gaussian FIR filter to split the signal into:
- **Sustain** (slow, low-frequency temporal energy)
- **Attack** (the residual: fast transient detail)

These components are recombined with independent gain control and latency-aligned mixing.

---

## Why Gaussian?

Most transient shapers rely on envelope followers and thresholds, making them:
- level-dependent,
- nonlinear,
- program-dependent.

This plugin instead uses a **Gaussian time-domain filter**, which provides:
- smooth, ripple-free behavior,
- exact linear phase,
- predictable results independent of input level.

“Time” here means *time scale*, not detector speed.

---

## Controls

### Control Group
- **Gaussian Sigma (σ) [ms]**  
  Sets the time scale used to separate attack and sustain.
  - Small values (≈0.3–1 ms): only very sharp transients are treated as attack.
  - Medium values (≈2–4 ms): drum hits, plucks, consonants.
  - Larger values (≈6–8 ms): broader impacts and macro punch.

- **Attack Gain [dB]**  
  Boosts or attenuates the transient (residual) component.

- **Sustain Gain [dB]**  
  Boosts or attenuates the sustained body of the sound.

### Gain Group
- **Mix**  
  Blends between the aligned dry signal and the shaped signal.

- **Output Gain [dB]**  
  Final level trim for gain matching.

---

## Technical Notes

- **Linear-phase FIR processing**  
  The plugin introduces a fixed latency equal to half the FIR length.
- **Level-independent behavior**  
  Input level does not affect transient detection.
- **No internal saturation**  
  The processor is linear by design; use output gain to manage level.

---

## Use Cases

- Tightening or softening drum transients
- Adding punch without envelope pumping
- Sculpting vocal consonants
- Sound-design and SFX transient control
- Parallel transient shaping via Mix

---

## License

MIT License.
