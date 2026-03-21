# Designed Occlusion Topology (DOT)
### Occlusion / behind-the-wall shaper — topology-driven, not a static “muffle EQ”

**DOT makes a sound feel occluded** — like it’s behind a door, around a corner, or inside another space.

Instead of a simple low-pass, it aims for a more *perceptual* result:
- Controlled high-frequency loss
- Minimum-phase style filtering (so it feels like a physical obstruction)
- A “leak floor” so it never turns into total blanket-muffle unless you want it

---

## What it does (in plain steps)

Internally DOT:
1. Chooses a material-style magnitude curve (Topology)
2. Scales and shapes it (Amount / Size / Stretch)
3. Converts it into a minimum-phase-ish filter shape
4. Applies it with a controllable leak floor
5. Applies output trim

---

## What it’s good at

- “Behind the wall / off-screen” perspective shifts (film/game)
- Occluding background sources so foreground reads clearly
- Making a source feel inside a vehicle/room/boxy space (with the right settings)

---

## Controls

### 1) Topology
Material-style occlusion curve preset.

### 2) Amount
How strong the occlusion is.

### 3) Brightness (dB)
Overall brightness compensation.

### 4) Color
Tone bias (warm ↔ sharp).

### 5) Size
Moves the main roll-off region (small ↔ large occluder).

### 6) Stretch
Frequency scaling of the curve.

### 7) Output (dB)
Final trim.

### 8) Leak Floor (dB)
Minimum brightness that always leaks through.

---

## Quick Start

- Topology: Default  
- Amount: 40–70  
- Size: ~50  
- Leak Floor: -36 to -24 dB  
- Brightness: adjust to taste  
- Output: level-match

---

## Safety / behavior notes

- DOT is a filter: it can change perceived loudness. Always level-match.
- Extreme Amount + very low Leak Floor can sound intentionally “blocked.”
