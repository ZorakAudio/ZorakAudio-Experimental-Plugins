## Vocal Air Recovery

**Vocal Air Recovery** is a perceptually driven high-frequency restoration tool designed to recover clarity, presence, and “air” in vocals **without harshness, hiss, or exaggerated EQ**.

Instead of boosting treble blindly, it analyzes a vocal’s **predictability and spectral density** to distinguish between genuinely missing high-frequency detail and content where noise or sibilance would be exaggerated. Only perceptually relevant components are enhanced, while transients, consonants, and already-present detail remain intact.

This makes it especially effective on:

- dull or over-de-essed vocals  
- heavily processed or denoised dialogue  
- intimate close-mic recordings lacking openness  
- stacked vocals that need separation without edge  

The result is **clarity without brittleness**, **openness without fatigue**, and air that feels *recovered*, not added.

Designed to be **safe, minimal, and gain-invariant**, Vocal Air Recovery works just as well for subtle dialogue repair as for expressive vocal production.


## Parameter intuition

Vocal Air Recovery intentionally exposes only two controls. The DSP does the rest: detection, safety limiting, and gain normalization are handled internally so the plugin stays predictable under automation and gain staging.

### Air Amount (%)
Controls the **maximum amount of high-frequency “air” the plugin is allowed to recover**.

- Low values: subtle openness and clarity; mostly “just a little more life.”
- Medium values: restores presence on dull/over-de-essed vocals without turning into EQ hype.
- High values: stronger recovery, but still bounded internally to avoid harshness.

Think of this as the **ceiling** on how much air can be added — not a treble boost.

### Sensitivity (%)
Controls **how readily the detector decides HF detail is missing** (vs. already present or likely to be noise/sibilance).

- Lower sensitivity: only recovers air when the loss is obvious; safest for bright or sibilant vocals.
- Higher sensitivity: recovers air more often and from subtler cues; useful for denoised/flattened vocals.

Think of this as the detector’s **picky vs. eager** setting.

### Practical starting points
- Dull vocal: **Air 30–50**, **Sensitivity 40–60**
- Over-de-essed / denoised dialogue: **Air 40–70**, **Sensitivity 55–80**
- Already bright / sibilant: **Air 10–35**, **Sensitivity 20–45**

### What to listen for (fast diagnosis)
- If it starts to sound “fizzy” or emphasizes ess: lower **Sensitivity** first, then **Air Amount**.
- If it’s doing nothing: raise **Sensitivity** first, then **Air Amount**.
