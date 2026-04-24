# Ambience Discipline System (ADS)

## What it is
ADS is a **slow, attenuation-only ambience governor** for room beds, background layers, and environmental support.

It is designed to keep a bed behind the foreground instead of turning the bed into a second lead element. The current source is built around four ideas:

- trim body, presence, and air when the bed gets too forward
- rein in excess side energy so the bed stays wide without becoming splashy
- optionally duck against a dialogue or key signal on channels 3/4
- cap “attention capture” with the salience-budget stage instead of relying on brute-force broadband ducking

---

## Why use it
Use ADS when the background is technically quiet but still **feels** too present.

It is especially useful on:

- ambience beds
- room layers
- environmental support stems
- wide noisy backgrounds
- supporting reverbs that should stay supportive instead of flashy

---

## Quick start
1. Insert ADS on the ambience or room bed you want to keep under control.
2. If you want foreground-aware behavior, feed the dialogue or key source to channels 3/4.
3. Set **Fit** for overall discipline, then use **Width Discipline** and **Salience Budget** to stop the bed from grabbing attention.
4. Add **Dialog Duck** only as needed. In many cases Salience Budget is the smarter first move.
5. Use **Output Trim** last to level-match.

---

## Main controls
### Fit
The main overall discipline amount. Higher settings push the bed farther back by trimming the parts of the ambience that feel too forward.

### Tone
Shifts the voicing from more pink/open toward more brown/darker support. Use it to keep a bed soft and supportive instead of bright and splashy.

### Width Discipline
Reins in side energy so the bed can stay wide without spraying attention into the stereo field.

### Depth
A gentle broadband downward-expander amount. This is the part that helps the bed sit behind the foreground without obvious pumping.

### HPF
Keeps low-end build-up from making the bed feel bulky or cloudy.

### Dialog Duck
Optional sidechain-based ducking from channels 3/4. Use it when a foreground voice or key source needs a little extra space.

### Output Trim
Final level trim after the discipline stages.

### Salience Budget
The smartest “stay in your lane” control here. It limits attention capture using presence level, transient behavior, and high-band side motion instead of acting like a plain volume duck.

---

## Routing / notes
**Input 1/2** is the ambience bed.

**Input 3/4** is optional and acts as the dialogue / key reference for ducking and salience-aware behavior.

A good starting point for beds is usually:

- Fit: moderate
- Tone: a little darker than flat
- Width Discipline: enough to stop splash
- Depth: gentle
- Salience Budget: moderate to fairly high

---

## When it is the wrong tool
ADS is not the right tool when you want obvious pumping, rhythmic keying, or big audible dynamic effects. It is built to behave like a governor, not a showy compressor.

---

## In one sentence
ADS keeps ambience supportive, wide, and under control without letting it steal the spotlight.
