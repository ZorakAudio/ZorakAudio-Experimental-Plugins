# Hyperreal 3D Panner (JSFX)

A headphone-focused spatial panner that places a source in a believable stereo soundfield using **interaural timing, head shadow, distance modeling, occlusion, and early reflections**.

Designed for **headphones**, sound design, and psychoacoustic spatial placement where the goal is:

> “Whoa… that sounds like it’s coming from *there*.”

This is not a full HRTF renderer or Dolby Atmos replacement.
Instead, it focuses on the cues that produce strong spatial perception while remaining **lightweight, responsive, and automation-friendly**.

---

# Core Idea

Humans localize sound using several cues:

• **ITD (Interaural Time Difference)** – micro-delays between ears
• **ILD (Interaural Level Difference)** – head shadow changes loudness by frequency
• **Spectral cues** – pinna-style filtering hints direction
• **Distance cues** – air absorption + direct/reverb balance
• **Early reflections** – geometry of nearby surfaces

This plugin simulates those cues in a simplified, controllable way suitable for realtime mixing.

---

# Features

### 3D Spatial Placement

Place sources around the listener using:

* azimuth (left / right / front / back)
* distance
* width (point vs spread)

### Distance Modeling

Distance affects:

* volume falloff
* high-frequency air absorption
* direct vs reflected energy

### Occlusion Simulation

Simulates sound being partially blocked by surfaces:

* muffled highs
* preserved lows
* stronger reflection presence

### Early Reflection Engine

A sparse early reflection model creates the **externalization effect** that pushes sources outside the head.

### Externalization Enhancer

A subtle **pinna-style spectral cue** improves front/back separation and perceived realism.

### Micro Motion Stabilizer

Tiny slow movement can prevent headphone images from collapsing inward.

Optional and subtle by design.

### Visual Spatial Field

The GFX interface displays a top-down listening field.

You can:

* **drag the orange source dot**
* visually place the sound
* see distance and azimuth update instantly

---

# Controls

## Azimuth

Controls horizontal direction around the listener.

Range:
`-180° (far left)` → `+180° (far right)`

Front/back differences affect the character of the spatial cue.

---

## Distance

Controls perceived distance.

Higher values introduce:

* level falloff
* air absorption
* stronger room presence

---

## Room

Controls early reflection strength.

Higher values create stronger externalization but may reduce precision.

---

## Occlusion

Simulates the source being blocked by an object or wall.

Effects include:

* stronger high-frequency damping
* preserved low frequencies
* increased reflection dominance

---

## Width

Controls apparent source size.

Low values create a **tight point source**.
Higher values create a **diffuse or wide emitter**.

---

## Output Trim

Final gain after spatial processing.

Useful for level matching.

---

## Externalize

Controls strength of spectral cues that push the source **outside the head**.

Higher values increase spatial character.

---

## Micro Motion

Adds extremely small slow movement to the spatial position.

Helps stabilize headphone perception.

Use subtly.

---

# Visual Placement

The main display represents a **top-down listening field**.

* center = listener
* orange dot = sound source

You can **click and drag the dot** to control:

* azimuth
* distance

This is the fastest workflow for spatial placement.

---

# Typical Workflow

1. Insert plugin on a **mono source**
2. Drag the source in the visual field
3. Adjust **Distance**
4. Add **Room** for externalization
5. Dial **Externalize** until the source feels outside the head
6. Use **Width** for source size

---

# Example Settings

### Close Dialogue

```
Distance: 0.15
Room: 0.20
Width: 0.05
Externalize: 0.30
```

### Foley in a Room

```
Distance: 0.35
Room: 0.40
Width: 0.08
Externalize: 0.45
```

### Distant Source

```
Distance: 0.70
Room: 0.55
Occlusion: 0.10
Externalize: 0.50
```

---

# Recommended Use

Best used on:

* foley
* environmental sounds
* ambience elements
* creature / body sounds
* spatial design layers

Works especially well in **headphone listening environments**.

---

# Limitations

This plugin is not:

* a full HRTF renderer
* a Dolby Atmos system
* a personalized binaural engine

Instead it provides **fast, controllable psychoacoustic spatialization**.

---

# Design Philosophy

The goal is **maximum perceptual effect per CPU cycle**.

Rather than simulate every aspect of acoustics, the plugin focuses on the cues the brain uses most strongly for spatial perception.

Small cues, applied correctly, produce convincing results.

---

