# Reverb Tail Tamer V3  
**Role-Aware, Permission-Based Wet Compression**

---

## What This Is

Reverb Tail Tamer V3 is not a traditional sidechain compressor.

It does **not** duck the wet signal simply because something gets loud.

Instead, it evaluates:

> Is this amount of reverb justified by the current vocal or SFX energy?

Only when the wet return exceeds what the input has effectively “earned” does gain reduction occur.

The result is preserved emotional bloom with controlled space and improved intelligibility.

---

## Routing Contract

- **Wet (post-reverb return):** Channels 1/2  
- **Vocals key input:** Channels 5/6  
- **Other / SFX key input:** Channels 7/8  

Channels 5/6 and 7/8 are used for analysis only.  
Only the wet return on 1/2 is processed.

---

## Core Model

### 1. Wet Energy (Ey)

Measured from the reverb return.

### 2. Justified Energy (Ex)

Built from role-aware contributions:

- Vocal reference × Vocal permission  
- Other reference × Other permission × Other authority  
- Combined using a power-sum (not linear addition)

### 3. Return Ratio

The system evaluates:

```

rdB = Wet level − Justified level

```

If this difference exceeds a sensitivity-dependent threshold, wet gain reduction is applied.

Implications:

- Loud reverb is allowed if source energy supports it.
- Quiet input cannot sustain excessive tail energy.

---

## Role Logic

### Vocal Permission

Triggered by:
- Transient excitation (fast vs slow envelope comparison)
- Sustained speech floor detection

Vocal memory decays more slowly to preserve speech continuity.

---

### Other Permission

Triggered by:
- Level-based presence
- Mild excitation assist for sharp SFX

Other authority is reduced while vocals are active or recently active.  
This prevents small incidental sounds from “re-opening” old vocal tails (anti-resurrection).

---

## Tail Behavior

### Grace Window

When references drop out, ducking ramps in gradually.  
This preserves natural tail bloom immediately after dialogue or SFX ends.

### Aging

Very long unattended tails are progressively reduced over time.  
Prevents ambient buildup in dense mixes or across edits.

---

## Controls

### Amount (max duck dB)

Sets the maximum possible wet attenuation.

---

### Sensitivity

Controls how easily excess return is detected.  
Higher values lower the effective threshold and increase compression ratio.

---

### Release

Controls how quickly wet gain recovers.  
Also influences grace timing and long-tail aging duration.

---

### Attack

Controls how quickly ducking engages.  
Slower values preserve early tail bloom and transients.

---

### Mix

Parallel blend of processed wet and original wet.

---

### Other Authority

Determines how strongly non-vocal material is allowed to justify reverb.  
Automatically reduced during vocal dominance.

---

## Why Use This Instead of Traditional Sidechain Ducking?

Traditional ducking:
- Reduces wet whenever input increases
- Often sounds mechanical or overly dry

Reverb Tail Tamer V3:
- Reduces wet only when it exceeds justified return
- Preserves density when earned
- Removes excess when unsupported

This keeps:
- Dialogue intelligible  
- SFX impactful  
- Space controlled  
- Emotional tails intact  

---

## Intended Use Cases

- Dialogue-heavy animation mixes  
- Dense SFX environments  
- Hybrid vocal/SFX scenes  
- Cinematic reverb control  
- Preventing ambient accumulation across edits  

---

## Design Philosophy

This is a return governance system, not a ducking gimmick.

It models:
- Authority  
- Permission  
- Memory  
- Suppression  
- Time-aware decay  

The result is controlled space that still feels natural and earned.
