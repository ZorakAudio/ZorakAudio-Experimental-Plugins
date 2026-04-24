# Reverb Tail Tamer (RTT)

## What it is
RTT is a **permission-based wet compressor** for reverb returns.

The core idea in the current source is not “duck whenever something happens.” It is:

> let the tail exist when the foreground justifies it, and pull it back when it no longer earns its space.

The current version is role-aware. It treats vocals and other/SFX inputs as separate authorities and uses that to decide how much reverb is allowed to stay alive.

---

## Why use it
Use RTT when ordinary sidechain ducking feels too blunt for reverb returns.

It is especially useful when:

- vocal reverb needs to stay musical but not swamp the mix
- other foreground sounds should still be able to justify some wet energy
- old tails should not re-open just because a tiny unrelated event happened later

---

## Quick start
1. Put the wet reverb return on **1/2**.
2. Feed the vocal authority signal to **5/6**.
3. Feed other or SFX authority to **7/8**.
4. Set **Amount** and **Sensitivity** first.
5. Shape recovery with **Release** and **Attack**, then trim how much non-vocal material can justify wet with **Other Authority**.

---

## Main controls
### Amount
Maximum wet gain reduction ceiling.

### Sensitivity
How easily the wet return is judged too hot relative to justified input.

### Release
Recovery time. It also influences the tail-grace behavior of the current source.

### Attack
How quickly ducking is applied once excess wet is detected.

### Mix
Parallel blend for the wet control path.

### Other Authority
How much the non-vocal authority input is allowed to justify reverb compared with the vocal authority lane.

---

## Routing / notes
Current routing contract:

- **1/2** = wet return
- **5/6** = vocals sidechain key
- **7/8** = other / SFX sidechain key

This is the larger, more role-aware tail controller in the repo. If you just need a lighter reference-keyed wet controller, RED is the smaller sibling.

---

## Notes
The current JSFX includes anti-resurrection logic and tail-aging behavior specifically to stop stale wet energy from becoming “alive again” for the wrong reasons.

---

## In one sentence
RTT is the smart reverb-return governor: it ducks wet when the tail stops being justified, not merely when a trigger fires.
