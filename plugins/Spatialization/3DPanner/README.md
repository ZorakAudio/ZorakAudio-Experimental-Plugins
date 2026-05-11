# 3DPanner

## What it is
3DPanner is a **headphone-focused, mouse-first perceptual 3D panner**.

The current build is **Hyperreal 3D Panner V6.1 — Perceptual Position + Visual Cues**. It is built around direct object placement instead of raw degree dialing.

The source combines several perceptual cues:

- softened lateral / front-rear placement
- Spatial Throw for directional cue intensity
- microdelay / interaural timing
- head-shadow style darkening
- pinna-style outside-head cues
- distance loss and air absorption
- occlusion filtering
- room and early-bounce cues
- micro motion for static-source stability
- Automation Safe smoothing
- a canvas-based UI for direct placement and visual feedback

It is not just a left/right pan with a width knob. It separates **where the object appears visually** from **how hard the spatial cue stack fires**.

---

## Why use it
Use it when a sound should feel like it sits in a headphone space instead of merely moving across the stereo field.

3DPanner is meant for fast placement, animation, and automation-friendly movement. You grab the source, place it, then decide how strongly it should assert that position.

---

## Quick start
1. Insert 3DPanner on the sound you want to place.
2. Drag the orange source in the stage canvas.
   - X = **Lateral**
   - Y = **Depth Front/Rear**
3. Use the mouse wheel over the canvas to adjust **Distance**.
4. Adjust **Spatial Throw** if the position feels too subtle or too extreme.
5. Use **Size**, **Push Out**, **Room**, **Occlusion**, and **Micro Motion** to shape the object.
6. Leave **Automation Safe** on for drawn automation and fast movement.
7. Trim the result with **Output Trim**.

---

## Main controls

### Lateral
Left/right object placement.

Negative values place the object left. Positive values place it right.

### Depth Front/Rear
Front/back object placement.

Positive values place the object toward the front wall. Negative values place it toward the rear wall.

### Distance
How close or far the source feels.

V6.1 uses a tighter, higher-resolution distance range than the older raw-distance model, so small movements are easier to dial in.

### Spatial Throw
Spatial cue intensity.

This controls how strongly the placement asserts itself through ITD, head shadow, pinna cues, and lateral weighting. It does **not** directly change output gain.

Use this when the source is visually in the right place but sounds too narrow or too exaggerated.

### Size
Apparent source size.

Low values make a tight point source. High values make the source wider and more diffuse.

### Push Out
Outside-head realism.

This is the renamed and more intuitive version of Externalize. It strengthens cues that help the source feel less trapped inside the head.

### Room
Amount of early reflection / room cue contribution.

Higher values make the space more audible and externalized, but less surgically dry.

### Occlusion
How blocked, hidden, or screened the source feels.

Higher values darken and smear the direct path while letting room/reflection energy become more dominant.

### Micro Motion
Tiny movement added to reduce headphone collapse on static sources.

Use low values for subtle life. Use higher values when the source should feel more animated.

### Output Trim
Final gain trim after spatial processing.

---

## Automation / advanced controls

### Automation Safe
Caps aggressive spatial behavior and uses safer smoothing for automation curves.

Leave this enabled for most automation. Disable it only when you deliberately want sharper, more extreme movement.

### Cue Curve
Controls how much moderate visual angles are perceptually softened.

Higher values make mid-angle placements less extreme while still allowing strong side placement when needed.

### Motion Smooth
Movement smoothing time in milliseconds.

Higher values make automation smoother and less jumpy. Lower values make movement more immediate.

### Room Size
Scale of the room-cue geometry.

Higher values make the virtual room larger and increase early reflection timing.

---

## Visual cues

The UI is part of the workflow. The canvas is not decorative; it shows what each spatial control is doing.

### Orange source
The solid orange orb is the visual object position.

### Ghost source
The ghost orange orb is the effective cue position after Spatial Throw and Cue Curve compression.

This makes the plugin honest: the object can be visually placed at 45° left while the actual cue intensity is softened for a more natural result.

### Size cue
Size is shown with an orange halo / cloud around the source.

Bigger halo means a larger or more diffuse source. It does not mean farther away.

### Distance cue
Distance is shown with range rings and distance dashes from the listener.

Farther positions show more range separation without turning the source into a width effect.

### Push Out cue
Push Out is shown with a listener-centered shell.

A stronger shell means the sound is being pushed farther outside the head. It does not mean the object itself moved farther away.

### Room cue
Room is shown with cyan wall bounces, wall-hit markers, and room glow.

The cyan wall markers represent early bounce cues that help the sound externalize.

### Motion cue
Motion is shown with a green dotted trail around the source.

More motion produces a stronger trail.

---

## Mouse controls

### Canvas
- Drag source = set Lateral + Depth Front/Rear
- Mouse wheel = Distance
- Shift-drag = fine placement
- Ctrl/Cmd-drag = lateral-only movement
- Alt/Option-drag = distance-only movement
- Ctrl/Cmd + wheel = Spatial Throw
- Alt/Option + wheel = Push Out
- Right-click canvas = reset position

### Control lanes
- Drag lane = set value
- Shift-drag lane = fine edit
- Mouse wheel over lane = quick edit
- Right-click lane = reset that control

### Snap chips
Use the snap chips for fast common placements:

- Center
- 30L
- 45L
- 60L
- 30R
- 45R
- 60R
- Rear

These set the visual object placement. Spatial Throw still controls how aggressively that placement is rendered.

---

## Notes
The current UI is designed around object placement, not degree management.

Older versions exposed raw **Azimuth** and **Externalize** as primary controls. V6.1 replaces that workflow with **Lateral**, **Depth Front/Rear**, **Distance**, **Spatial Throw**, and **Push Out**.

This is an intentional compatibility break. Presets from older versions should be rebuilt by ear instead of copied blindly.

---

## In one sentence
3DPanner places a headphone source as an object in a perceptual room, then lets you control how strongly that placement, distance, size, room, and outside-head cues are expressed.
