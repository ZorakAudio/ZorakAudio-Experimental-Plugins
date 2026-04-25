# SOMA

## What it is
SOMA is a **GUI-only psychoacoustic / somatic limiter**.

Push the input, let the limiter catch the peaks, then use the psychoacoustic and body controls to decide where the gain reduction damage goes and what gets reinforced while the mix is being squeezed.

It sits somewhere between loudness control, perceptual gain allocation, transient preservation, and GR-keyed body injection.

---

## Why use it
Use it when you want the mix to get louder without every part of the signal being crushed equally.

Standard limiting turns peaks down. SOMA tries to make the limiter act more like a loudness allocator:

- preserve the material that defines the mix
- sacrifice less important energy first
- add controlled body when gain reduction happens
- keep the result dense without making it feel dead

---

## Quick start
1. Set **Drive** first. Start around **+4 to +6 dB**.
2. Set **Ceiling** next. Use **-0.5 to -1.0 dB** if this is going near final output.
3. Choose a **Style**:
   - **Punch** for impact
   - **Dense** for smooth level
   - **Body** for pressure
   - **Clean** for safer limiting
   - **Wild** for destructive loudness
4. Tune **Preserve** until the foreground stops collapsing.
5. Raise **Somatic Body** until the limiter feels physical, then back it off before it sounds fake.
6. Set **Release** by ear. Faster = more aggressive. Slower = smoother.
7. Use **Lookahead** to trade punch for cleaner peak control.
8. Hover any control for help. Right-click or double-click a control to reset it.

---

## Main controls

### Loudness / ceiling
**Drive**, **Ceiling**, **Lookahead**, and **Release** decide how hard the limiter is hit, where the output stops, how early it reacts, and how quickly it recovers.

### Psychoacoustic behavior
**Preserve** protects important midrange, presence, and transient information from being flattened too aggressively.

Higher Preserve means more of the apparent foreground survives heavy limiting. Lower Preserve behaves closer to a normal limiter.

### Somatic behavior
**Somatic Body** adds low and low-mid reinforcement keyed from the gain-reduction envelope.

No limiting means little to no body addback. More gain reduction means more controlled pressure.

### Motion
**Motion** adds tiny sub-JND movement so the limited signal feels less static.

Use low values for mastering-style work. Push it higher for sound design and unstable density.

### Stereo / safety / display
**Stereo Link** controls how much the left and right channels share gain reduction.

**Guard** keeps the limiter from getting too reckless. Leave it on unless you are deliberately abusing the plugin.

**Display Range** changes the visible meter and scope depth only. It does not change the sound.

---

## Styles

### Clean
Smoother, safer limiting. Longer recovery, less body push, reduced hype.

### Punch
Fast recovery with stronger transient and presence preservation. Best first choice for impact.

### Dense
Slower, heavier leveling. Good when you want the mix glued and consistently loud.

### Body
More low and low-mid reinforcement. Best for pressure, weight, and tactile material.

### Wild
Fast, hyped, and unstable on purpose. Use for destruction, sound design, or loudness abuse.

---

## GUI behavior
All native JSFX sliders are hidden. The plugin is meant to be controlled from the custom interface.

- Drag tiles horizontally to change values.
- Drag the **Drive** and **Ceiling** badges directly on the graph.
- Hover any control for a tooltip.
- Right-click a control to reset it.
- Double-click a control to reset it.

The reset values are editable near the top of `@init`.

---

## Hand tuning
The top of `@init` contains the practical tweak zone.

Use it to change:

- default reset values
- tooltip size and spacing
- double-click timing
- badge drag sensitivity
- style multipliers
- presence addback
- air addback
- body addback
- motion depth

After editing constants, save the file and reload or rescan the JSFX in REAPER.

---

## Routing / notes
SOMA is a stereo in / stereo out limiter.

It does not require sidechains, file slots, or routing tricks.

This is **not a certified true-peak limiter**. For final delivery, put a true-peak meter after it. If reconstruction overs are showing up, lower the ceiling to **-0.5 dB** or **-1.0 dB**.

---

## In one sentence
SOMA turns limiter gain reduction into perceptual loudness, protected impact, and body pressure instead of simple flat crushing.
