# Ambience Discipline System (ADS)
### Background-bed controller — keeps ambience present but not distracting

**ADS is for ambience and room “beds”**: the stuff that should feel *there*… but shouldn’t fight dialogue, vocals, or foreground detail.

It does this by shaping the cues your brain uses to decide what’s *important* vs *background*:
- **Level + motion** (busy beds feel “louder” than meters suggest)
- **Spectral brightness** (too much top feels “closer”)
- **Stereo spread** (wide, unstable beds grab attention)
- **Depth / density** (beds feel better when they sit *behind* foreground)

It’s not a magic “mix it for me” box.
It’s a **stability tool** designed to keep beds behaving like beds.

---

## What it does (in plain steps)

Internally ADS:
1. Measures bed energy and motion (how “active” the background feels)
2. Applies **depth discipline** (bed stays behind the foreground)
3. Applies **width discipline** (reduces splashy, unstable stereo when needed)
4. Shapes **tone** (tilt + HPF) to avoid low build-up and harsh “air”
5. Optionally applies **ducking** (keyed / dialogue-friendly behavior)
6. Trims output to match level

---

## What it’s good at

- Room tone and ambience beds in **film/game**
- “Constant” background layers in dense music productions
- Keeping wide ambience from feeling like it’s glued to the listener’s ears
- Preventing low rumble / buildup from stacking across scenes

---

## Controls

### 1) Fit
Overall intensity / strength of the system.

### 2) Tone (dB)
Brightness tilt for the bed.
- Negative = darker / farther
- Positive = brighter / closer

### 3) Width Discipline
How strongly ADS reins in unstable or attention-grabbing stereo width.

### 4) Depth
Depth discipline (expander-like behavior that helps push the bed back).

### 5) HPF (Hz)
High-pass filter to remove rumble and low build-up.

### 6) Dialog Duck (dB)
Extra ducking amount when a dialogue/key signal is present.

### 7) Output Trim (dB)
Final level trim after processing.

### 8) Salience Budget
“How much attention is the bed allowed to take?”
Lower = more background, higher = more present.

---

## Quick Start (recommended)

- Fit: 40–60  
- Depth: ~40  
- Width Discipline: ~50  
- HPF: 70–120 Hz  
- Tone: 0 to -3 dB  
- Dialog Duck: 0–3 dB (if needed)  
- Output Trim: level-match

---

## Safety / behavior notes

- ADS is designed to be **hard to break**, but it is **not a limiter**.
- Extreme settings can still change perceived loudness — level-match with Output Trim.
