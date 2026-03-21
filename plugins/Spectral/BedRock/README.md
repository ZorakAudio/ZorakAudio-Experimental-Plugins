# Bedrock  
### Adaptive Subharmonic Synthesizer & Translation Engine

Bedrock does not boost low end.

It synthesizes a controlled sub foundation that locks to pitch when possible, falls back to division when needed, and manages energy so the mix does not collapse under sustained low-frequency load.

This is a **foundation generator**, not a resonant bass enhancer.

---

## Core Architecture

Bedrock contains two sub-generation engines:

### 1. Frequency Divider Engine
- Tracks waveform zero-crossings  
- Generates f/2 and f/4 square-derived subharmonics  
- Extremely stable for percussive or noisy material  
- Works even when pitch is unstable  

### 2. Pitch-Locked Sine Engine
- Estimates period from rising-edge timing  
- Produces phase-coherent f/2 and f/4 sine oscillators  
- Engages when pitch confidence is high  
- Cleaner and more musical for tonal bass  

A real-time confidence system measures:
- Period jitter  
- Edge stability  
- Timeout behavior  
- Source level  

The plugin automatically crossfades between engines.

---

## Band-Limited Sub Generation

The synthesized sub passes through:

- 4th-order high-pass at the floor frequency  
- 4th-order low-pass at the ceiling frequency  

This prevents:
- DC creep  
- Infrasonic build-up  
- Mid-bass smearing  

---

## Psychoacoustic Harmonic Translation

Pure sub does not translate on small speakers.

Bedrock:
- Soft-saturates the generated sub  
- Bandpasses it into the ~90–300 Hz region  
- Mixes this controlled harmonic layer back in  

Result:
- Sub weight on full-range systems  
- Audible bass perception on limited playback systems  

---

## Transient Gating & Sustain Logic

The sub layer is driven by:

- A transient detector (fast vs slow envelope comparison)  
- Optional sustain injection (Cinematic mode)  
- Sidechain-triggerable gating (channels 3/4)  

This allows:
- Tight impact reinforcement  
- Cinematic bass bloom  
- Controlled sustain without mud  

---

## Energy Management

Sustained low frequencies fatigue listeners and consume headroom.

Bedrock includes:

### Short-Term RMS Energy Clamp
~1.8 second integration window.  
Prevents long-term low-frequency overload.

### Low-Frequency Guard (<40 Hz)
Reduces steady infrasonic energy when no transient is active.

### Wet Path Peak Limiter
Prevents runaway sub amplitude before summing with dry signal.

### Output Safety Limiter
Final ceiling to avoid clipping.

---

## Controls

### Style
Defines internal bias and behavior:

- **Anchor** – Tight reinforcement, conservative harmonic translation  
- **Cinematic** – Sustained bloom with ducking control  
- **Impact** – Aggressive transient reinforcement, lower pitch-confidence bias  

---

### Amount
Scales:
- Sub gain  
- Harmonic translation  
- Limiter depth  
- Energy target  

Higher values increase perceived weight, not just amplitude.

---

### Depth
Controls octave emphasis:

- 0% = reinforces base octave  
- Mid = emphasizes -1 octave  
- High = emphasizes -2 octave (f/4 dominance)  

---

### Tightness
Controls:
- Gate release time  
- Confidence threshold for sine engine  
- Energy clamp aggressiveness  
- AM modulation depth  

Higher values = cleaner, shorter, more controlled.  
Lower values = bloom and sustain.

---

### Tone
Shifts band limits of:
- Fundamental sub  
- Harmonic translation layer  

Lower values = deeper, darker.  
Higher values = more audible bass presence.

---

## Routing

### Standard Insert
Place on bass, kick, or impact track.

### Sidechain Trigger
Feed kick or transient source into channels 3/4.  
Pitch remains derived from the main input.  
Gate responds to sidechain signal.

---

## What It Is Not

- Not a simple EQ boost  
- Not a static sub oscillator  
- Not a one-band enhancer  

It is a pitch-aware sub synthesizer with energy governance.

---

## Intended Use Cases

- Reinforcing weak kick fundamentals  
- Adding floor to thin bass  
- Impact reinforcement in cinematic sound design  
- Creating weight without uncontrolled boom  
- Maintaining translation across playback systems  

---

## Operating Advice

Low frequencies accumulate energy faster than mids.

Start conservative:
- Amount: 20–40%  
- Depth: moderate  
- Tightness: mid-to-high  

Increase gradually while watching headroom and RMS behavior.

If you hear obvious sub distortion, you are likely pushing beyond the intended energy envelope.