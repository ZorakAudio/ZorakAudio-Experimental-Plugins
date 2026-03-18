# Gesture MIDI Pad XL — patched build

This patch keeps the original instrument intact and adds **hidden extra motion-to-CC routing** so you can map multiple movement derivatives at once.

## What changed

The original file already had one “third lane” that could be switched between:

- Speed
- Vel X / Vel Y
- Acc X / Acc Y
- Jerk X / Jerk Y
- Err X / Err Y

But it could only feed **one CC at a time**.

This patch keeps that original lane and adds a hidden **EXTRA CCs** panel so you can optionally assign **additional motion sources to their own CC numbers** at the same time.

### Example

You can now do things like:

- Primary motion lane -> CC 11
- Vel X -> CC 12
- Acc X -> CC 13
- Jerk Y -> CC 16

Or leave the primary lane off and only use the extra hidden routes.

## Patch goals

This was done as a **patch**, not a rewrite.

The existing structure, drawing logic, playback logic, note logic, transport section, note range UI, and most of the custom UI remain unchanged. The added work is focused on motion routing only.

## New behavior

### 1. The old “Speed CC” is now the **Primary Motion CC**

The visible motion lane in the main UI still works the same way, but the labels were clarified:

- `CC Speed` -> `CC Primary Motion`
- `Emit Speed CC` -> `Emit Primary Motion CC`
- `Motion Lane Source` -> `Primary Motion Source`

That visible lane is still the fast, obvious way to use one motion-derived CC.

### 2. Hidden **EXTRA CCs** menu

A new button appears in the **ADVANCED & STATUS** card:

- `EXTRA CCs`

That opens a hidden routing panel with one optional CC slot for each motion source:

- Speed
- Vel X
- Vel Y
- Acc X
- Acc Y
- Jerk X
- Jerk Y
- Err X
- Err Y

Each row can be:

- assigned to any CC number
- turned **OFF**

The menu is hidden by default, so the main UI stays uncluttered.

### 3. Extra routes are independent from the primary lane

The primary motion lane still uses:

- `Emit Primary Motion CC`
- `CC Primary Motion`
- `Primary Motion Source`

The new hidden extra routes do **not** depend on that toggle. They are separate optional outputs.

That means you can:

- keep the primary lane on and add more routes
- turn the primary lane off and only use extra routes
- use both at the same time

### 4. CC picker now supports **OFF** for extra routes

When assigning one of the hidden extra routes, the CC picker now includes a **TURN OFF** button.

That disables the route cleanly without deleting anything else.

## Motion sources

### Unipolar source

- **Speed**  
  Encoded as normal 0–127 CC.

### Bipolar sources

These use **center-zero MIDI encoding**:

- 64 = center / zero
- below 64 = negative
- above 64 = positive

Sources:

- **Vel X**
- **Vel Y**
- **Acc X**
- **Acc Y**
- **Jerk X**
- **Jerk Y**
- **Err X**
- **Err Y**

### What “Err” means

`Err X` / `Err Y` are the current prediction error against a simple one-step motion estimate:

- predict next X/Y from previous position + previous velocity
- compare the real motion to that prediction
- emit the difference

Musically, this tends to emphasize sudden direction changes, surprises, and non-linear hand motion.

## Shared controls

The extra CC routes reuse the existing engine. They share the same motion preprocessing as the original file:

- **Smoothing**
- **CC Deadband**
- **Speed Sensitivity** (for Speed)
- **Motion Lane Sensitivity** (for signed motion features)

That means:

- raising **Motion Sense** makes Vel/Acc/Jerk/Err outputs more dramatic
- raising **Speed Sense** makes Speed stronger
- raising **Deadband** reduces MIDI chatter on all CC routes
- raising **Smoothing** softens everything before modulation is emitted

## Main UI workflow

### Simple use

1. Draw in the pad.
2. Set **Output Type** to `CC Only` or `CC + Note`.
3. Turn on **Primary Motion CC** if you want the visible motion lane.
4. Pick a **Primary Motion Source**.
5. Assign its CC number.

### Advanced use

1. Open **EXTRA CCs**.
2. Assign any additional motion sources to their own CCs.
3. Leave unused rows set to **OFF**.
4. Record-enable the REAPER track and capture the generated MIDI.

## Notes about MIDI output

### When CCs are emitted

Motion CCs only emit when CC output is active:

- `Output Type = CC Only`
- or `Output Type = CC + Note`

If `Output Type = Note Only`, the extra CC routes remain configured but do not transmit.

### Duplicate CC numbers

The patch allows multiple sources to target the same CC number.

That is legal, but it is usually a bad idea unless you want intentional interaction. If two sources feed the same CC, the later send wins block-by-block.

Use different CC numbers for clean modulation routing.

## Backward compatibility

This patch is intentionally conservative:

- existing X/Y CC behavior is unchanged
- existing note output behavior is unchanged
- existing gesture drawing/playback behavior is unchanged
- existing primary motion lane is still there
- new advanced routing is opt-in and hidden by default

If you never open **EXTRA CCs**, the script behaves almost exactly like before.

## Files included

- `Gesture MIDI Pad XL.jsfx` — patched JSFX file
- `Gesture MIDI Pad XL.patch` — unified diff against the original file
- `README.md` — this file

## Quick summary

If you only need one motion lane, use the visible controls exactly like before.

If you want multiple derivative modulations at once, open **EXTRA CCs** and assign:

- Speed -> one CC
- Velocity -> another CC
- Acceleration -> another CC
- Jerk / error -> more CCs as needed

That gives you layered movement-driven modulation without bloating the default UI.
