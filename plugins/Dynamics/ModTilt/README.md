# ModTilt

## What it is
ModTilt is a **SAFE envelope-tilt shaper**.

This is not a normal tonal tilt EQ. The current Faust source works on the **movement** of the signal envelope, splitting the modulation into slower and faster components around a pivot and then tilting the balance between them.

In practice that means you can make a signal feel more:

- punchy
- snappy
- calm
- weighty
- dynamically “fast” or “slow”

without using a conventional compressor shape.

---

## Why use it
Use ModTilt when the problem is not basic level control, but the **character of the movement**.

It is useful when you want more front-edge urgency or more slow-body weight without rewriting the whole dynamics chain.

---

## Quick start
1. Start with **Tilt** near zero and **Mix** at 100%.
2. Push Tilt positive for a faster, more forward motion balance.
3. Pull Tilt negative for slower, heavier, more grounded motion.
4. Adjust **Pivot** to decide where the split between slow and fast behavior sits.
5. Back off **Mix** if you want a gentler blend.

---

## Main controls
### Tilt
The main fast-vs-slow envelope tilt amount. Positive values favor the faster portion of the movement. Negative values favor the slower portion.

### Pivot
The modulation pivot that decides where “slow” hands off to “fast.” This is about envelope behavior, not a traditional audio EQ pivot.

### Mix
Wet/dry blend.

---

## Notes
The current Faust source includes internal SAFE clamping and auto-trim behavior. It is built to be hard to break and easy to audition quickly.

---

## In one sentence
ModTilt changes the balance of slow versus fast envelope motion, not the static tone of the source.
