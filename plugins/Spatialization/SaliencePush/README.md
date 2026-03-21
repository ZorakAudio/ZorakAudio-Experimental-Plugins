# Salience Push (Minimal + SAFE)

**Salience Push** is a psychoacoustic utility designed to push sounds back in a mix **without obvious EQing, compression artifacts, or pumping**.
Instead of reducing volume directly, it regulates **perceptual attention** by controlling the elements that make a sound jump forward:

* upper-band brightness
* transient spikes
* stereo motion
* spectral contrast vs surrounding material

The result is a sound that feels **farther away and less intrusive while remaining natural and intact**.

This processor is **attenuation-only**, slow-moving, and capped to avoid destructive processing. It is meant to behave predictably across many types of sources.

---

# Philosophy

Most mix tools reduce level or apply heavy EQ when something feels too forward.
However, the ear locates foreground elements mainly through:

• high-frequency clarity
• transient sharpness
• stereo movement
• contrast with surrounding material

Salience Push targets those mechanisms instead of simple gain reduction.

This allows sources to **sit back naturally** while preserving identity.

---

# Controls

## Profile

Selects a tuning profile optimized for different source types.

### Neutral

Balanced behavior suitable for general-purpose use.

### Vocals

Protects intelligibility and formants while gently reducing forwardness.

Best for:

* backing vocals
* distant narration
* vocal layers
* breathy elements

### SFX

More assertive transient and brightness control.

Best for:

* impacts
* drips
* bright FX
* stylized elements

### Foley

Protects realism cues and body while controlling pokey details.

Best for:

* cloth
* footsteps
* object handling
* sticky textures

### BG

Designed for ambience and beds that should stay behind the foreground.

Best for:

* ambience
* reverb returns
* environmental beds
* tonal washes

---

## Push

Controls how strongly the processor pushes the sound backward.

Increasing Push will gradually:

• reduce brightness emphasis
• reduce stereo attention
• reduce salience spikes
• subtly soften transient prominence

Typical ranges:

| Source      | Push  |
| ----------- | ----- |
| Vocals      | 35–50 |
| Drips / SFX | 40–60 |
| Ambience    | 45–65 |

---

## Tame

Controls suppression of **short transient spikes**.

These spikes often cause:

* clickiness
* harsh drips
* pokey consonants
* splashy FX

Low values preserve detail.
High values soften sharp micro-transients.

Typical ranges:

| Source      | Tame  |
| ----------- | ----- |
| Vocals      | 20–35 |
| Drips / SFX | 45–70 |
| Ambience    | 10–30 |

---

## Preserve

Protects character, gloss, and stereo life.

Low values allow stronger suppression of brightness and width.

High values preserve:

* air and shine
* stereo movement
* vocal clarity
* wet detail

Typical ranges:

| Source      | Preserve |
| ----------- | -------- |
| Vocals      | 70–90    |
| Drips / SFX | 60–85    |
| Ambience    | 35–60    |

---

## Output

Simple output trim after processing.

Use this only for level matching during comparison.

---

# Routing

Input channels:

```
1/2 = main signal
3/4 = optional reference / dialogue
```

If a reference signal is present on channels 3/4, the processor will bias its suppression relative to that signal's presence band.

If no reference is provided, the plugin automatically switches to a **self-salience detector**.

---

# SAFE Design Principles

The processor follows several design rules to avoid destructive processing:

• **attenuation-only** (never boosts)
• **hard capped attenuation**
• **slow smoothing** to prevent pumping
• **no broadband auto-gain tricks**
• **formant protection for vocals**
• **width restraint only in upper bands**

These rules allow it to be inserted freely without catastrophic results.

---

# Example Use Cases

## Bright Wet Drips / Clicky FX

Profile: **SFX**

```
Push     40–55
Tame     45–70
Preserve 60–85
```

Softens the clicky spike while keeping the slick wet character.

---

## Reverb Return

Profile: **BG**

```
Push     45–65
Tame     10–25
Preserve 40–60
```

Keeps the reverb behind the foreground without dulling it completely.

---

## Backing Vocals

Profile: **Vocals**

```
Push     35–50
Tame     20–35
Preserve 70–90
```

Moves vocals deeper in the mix while maintaining natural tone.

---

## Foley Details

Profile: **Foley**

```
Push     35–50
Tame     25–45
Preserve 60–80
```

Reduces distracting pokiness while keeping tactile realism.

---

# Technical Overview

Internally the processor analyzes three perceptual regions:

* **Form / body**
* **Edge / presence**
* **Air**

From these it derives a **salience demand signal** based on:

• spectral contrast
• transient overshoot (fast vs slow energy)
• stereo dominance

The demand signal drives a bounded attenuation stage which distributes suppression across presence and air bands with optional side-channel restraint.

All actions are smoothed with long time constants to maintain natural behavior.

---

# Intended Role

Salience Push is not meant to replace EQ or compression.

Instead it is best used as:

* a **foreground management tool**
* a **depth control**
* a **harshness stabilizer**
* a **mix decluttering processor**

Think of it as **an attention regulator**.

---

# Author Notes

Designed for psychoacoustic mix management and experimental sound design workflows.

Minimal interface, bounded behavior, and profile-driven tuning allow the processor to behave predictably across a wide range of sources.

---

