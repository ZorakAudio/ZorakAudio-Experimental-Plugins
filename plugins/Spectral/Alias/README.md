# Alias

Alias is a JSFX that **extracts a body layer from intentionally downsampled / aliased side paths** and blends that layer back under the original signal.

It was built around a specific observation:

- a **clean / high-rate** path preserves top-end detail and transient clarity
- a **lower-rate / aliased** path can create useful low-mid density and perceived weight
- that weight often feels more *connected* to the source than synthetic sub generators, generic noise layers, or envelope-followed fillers, because the added layer is still derived from the input material itself

Alias turns that into a controllable effect.

---

## What it does

Alias creates up to three parallel body paths:

- **48k path** — mild, upper-body reinforcement
- **24k path** — stronger, usually the main body lane
- **12k path** — deepest and dirtiest, easy to overdo

Each lane is driven, rate-reduced, band-shaped, slightly decorrelated, and then summed into a single wet/body layer. That wet layer can then be:

- **added to the dry signal**, or
- **auditioned on its own** in **Wet Only** mode

A dedicated wet-only HP/LP stage lets you trim mud, rumble, or excess edge **without touching the dry path**.

---

## Why it works

Most “body” processors either:

- synthesize unrelated low-frequency content,
- generate generic saturation across the whole signal, or
- add noise or harmonics that are not tightly tied to the source.

Alias does something different.

It first removes some of the direct low end with **Seed HP**, then uses downsampled / held side paths to create folded, correlated byproducts from the remaining material. Those byproducts are filtered into a usable reinjection band and mixed back in as a secondary body layer.

The result is often:

- denser,
- more anchored,
- less hollow after pitch-shifting,
- and more “part of the source” than a fake bass layer.

This is especially useful after pitch or granular processing, where the source already contains a lot of high-frequency or ultrasonic detail that can be folded into a musically useful body layer.

---

## Philosophy

Alias is **not** trying to be clean or technically neutral.

It is deliberately using artifacts as material.

The design philosophy is:

1. **Harvest useful artifacts, not broadband damage**
2. **Keep the added layer correlated with the source**
3. **Band-limit the reinjection so the effect reads as weight, not fuzz**
4. **Preserve the dry path** and let the user sculpt the wet path independently
5. **Make the result obvious in Wet Only, but subtle in full context**

In practice, that means the best result is usually **felt more than heard**.

---

## Host sample-rate behavior

Alias adapts to host sample rate.

- At **96 kHz**, all three lanes are available: **48k / 24k / 12k**
- At **48 kHz**, the **48k lane is disabled**, because there is no meaningful “down to 48k” stage from a 48k host
- At lower rates, lanes disable themselves when they no longer make sense

This is expected behavior, not a bug.

---

## Quick start

### 1) For pitch-shifted or granular material

Start here:

- **Body Mix**: 20–35%
- **24k Path**: 40–65%
- **12k Path**: 5–25%
- **48k Path**: 10–35% (only if host rate allows it)
- **Seed HP**: 800 Hz – 1.8 kHz
- **Focus**: 120–220 Hz
- **Drive**: 6–14 dB
- **Smart**: 40–70%
- **Decohere**: 0.3–1.2 ms
- **Density**: 20–45%
- **Inject**: +3 to +10 dB
- **Wet HP**: 25–60 Hz
- **Wet LP**: 250 Hz – 1.5 kHz

Then watch the **Focus-band add** meter. A good working range is usually around **15–35% of dry**.

### 2) For plain source thickening

Start more conservatively:

- lower **12k Path**
- lower **Drive**
- lower **Inject**
- keep **Wet LP** lower so the reinjection stays supportive rather than obvious

### 3) To clean mud fast

Use the spectrum panel:

- drag the **blue HP handle** right to remove boom / rumble
- drag the **amber LP handle** left to darken or de-fizz the added layer

Remember: these filters affect the **wet path only**.

---

## Parameter guide

## Body Mix

Global amount of body reinjection.

This controls how much of the harvested body layer is allowed forward before the later wet gain stages. It is a broad “how much effect” control.

Use this first for overall depth, then fine-tune with **Inject**.

---

## 48k / 24k / 12k Path

These are **color controls** as much as they are amount controls.

### 48k Path
The cleanest and most restrained body lane. Useful for subtle upper-body support.

### 24k Path
Usually the core lane. Good balance of weight and control.

### 12k Path
The roughest lane. Best for depth, aggression, and obvious reinforcement. Easy to overdo.

A good rule:

- make **24k** do most of the work,
- season with **48k**,
- use **12k** as a spice, not the base.

---

## Seed HP

High-pass filter **before** alias-body generation.

This determines what part of the source is allowed to fold down into the body layer.

- **Lower Seed HP** = more direct, obvious reinforcement
- **Higher Seed HP** = more “detail-derived” body from upper content

If the body feels too disconnected or too synthetic, lower this.
If the result feels boomy or too literal, raise it.

---

## Focus

Center of the reinjection behavior.

This influences the per-lane band targets and the meter match band.

- **Lower Focus** = deeper weight
- **Higher Focus** = more punch / chest / upper body

For many use cases, **120–220 Hz** is the sweet zone.

---

## Drive

Nonlinear excitation inside the body-generation lanes.

More Drive means more fold material, more density, and more obvious effect. It is one of the fastest ways to make the wet layer feel stronger.

Too much Drive can make the added layer feel rough, buzzy, or smeared.

---

## Smart

Adaptive allowance based on the source balance.

Bright or detail-heavy input tends to get more allowance. Already-heavy or dark input gets less.

This helps keep Alias from piling too much into material that already has enough body.

- lower Smart = more manual / fixed behavior
- higher Smart = more adaptive restraint

---

## Decohere

Micro-delay used to keep the wet layer from comb-filtering against the dry path.

This is not meant to sound like an audible echo.

Useful range is typically **0.2–1.5 ms**.

Too little can feel phasey.
Too much can feel detached or double-tracked.

---

## Density

Controls wet-path compression / smoothing.

The harvested body can be spiky and bursty. Density makes it more even and sustained.

- lower Density = more reactive, raw, flickery
- higher Density = steadier, thicker, more glued

If the effect disappears too easily, raise Density.
If it starts to feel flattened or too constant, lower it.

---

## Inject

Post-wet gain for the added layer.

This is the main “make it felt” control once the body color is right.

If the effect is correct in Wet Only but too subtle in context, raise **Inject** before overdriving everything else.

---

## Output

Final output trim after sum.

Use this for level matching.

Always level match before deciding whether the effect is genuinely better.

---

## Audit

Switches between:

- **Add to Dry**
- **Wet Only**

Use **Wet Only** to understand what Alias is adding.
Use **Add to Dry** to judge whether that addition actually improves the source.

---

## Wet HP / Wet LP

These are **wet-only** 12 dB/oct filters shown in the spectrum panel.

They do not touch the dry signal.

### Wet HP
Use this to remove rumble, boom, and excessive bloom from the added layer.

### Wet LP
Use this to keep the added layer supportive rather than fuzzy or distracting.

A lot of the time, the wet path sounds better when it is narrower and duller than you expect in solo.

---

## Reading the display

## Path Weights
Shows each lane’s contribution range and lane meter.

## Spectrum panel
Shows the **post-filter wet layer**. This is what the HP/LP filters are shaping.

## Waveform panel
Top = dry reference. Bottom = added body. The wet view is auto-zoomed so the added layer remains visible even when quiet.

## Input level / Body level
General signal presence meters.

## Smart gate
Shows how much the adaptive Smart behavior is allowing through.

## Focus-band add
Shows how much wet energy is being added in the match band relative to the dry signal.

This is one of the most useful “sanity” meters in the plugin.

## Density gain reduction
Shows how much the Density stage is compressing the wet layer.

---

## Practical tips

- **Wet Only should be obvious.** The full mix result usually should not be.
- If the effect is too weak, try **Inject** before over-cranking everything else.
- If the result gets muddy, raise **Wet HP** first.
- If it gets fuzzy or distractingly audible, lower **Wet LP**.
- If it feels phasey, increase **Decohere** slightly.
- If it feels too fake, lower **Seed HP** or back off **12k Path**.
- If the body collapses on some material, lower **Smart** or raise **Density** a little.
- Use **Wet Only** to shape the character, then switch back to **Add to Dry** and trim until it feels like the dry signal simply became more anchored.

---

## A simple workflow

1. Set the lane balance so Wet Only has the right character
2. Use **Wet HP / LP** to remove useless mud and fuzz
3. Use **Decohere** to keep the wet layer from fighting the dry path
4. Use **Inject** to bring the effect into the “felt” range
5. Use **Output** to level match
6. Judge in context, not in solo

---

## What Alias is not

Alias is not:

- a transparent mastering EQ
- a clean subharmonic generator
- a neutral saturation plugin
- a pitch shifter

It is a **signal-derived artifact harvester** designed to turn controlled alias-style byproducts into a usable body layer.

That is the point.
