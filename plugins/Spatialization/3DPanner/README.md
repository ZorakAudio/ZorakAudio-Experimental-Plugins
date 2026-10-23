# Hyperreal 3D Panner

A headphone-focused, mouse-first 3D panner with two original render models:
**Artistic** for softened perceptual placement, and **Physical** for per-ear
geometry, fitted KEMAR spectra, propagation and room paths. The maintained
V7.3.0 implementation incorporates faithful Fast EEL optimizations; it does
not substitute a simplified FAUST renderer.

## Start here

Place it on a stereo track and listen on headphones. Drag the orange source
in the stage canvas and use the mouse wheel for distance. The solid orange
object is the visual position; its ghost shows the effective cue position after
Spatial Throw and Cue Curve compression. The POV canvas range limits ordinary
drag placement near center; quick controls and snap chips still allow the full
range. Hover the canvas controls for guidance.

Choose Artistic or Physical with Render Model. **FT / m** in the header
switches Imperial feet and Metric metres; feet is the default for both models.
The preference is saved per instance. Right-click it to return to feet.
This only converts the visual readouts: 10 m displays as 32.8 ft, and every
existing source position, distance curve, room/ear height, automation value,
control gesture and audio calculation keeps its original meaning.
Current control is entirely local. The retired manager
and string slots are inert and retained for old state/automation identities;
this revision does not subscribe to 3DPannerManager.

The unchanged Distance mapping covers about 0.66–32.81 ft (0.20–10 m).
Its default is about 2.24 ft (0.683875 m). Reducing it brings the source closer
through the original model. The added far-distance presence/level restraint is
inactive once settled below 3.28 ft (1 m); near-ear contrast can increase for
off-centre point sources. There is no added close-range bass boost. Changes
remain smoothed; any prior far-distance RMS restraint releases over about
700 ms when moving closer.

## Artistic proximity and distance anti-salience

Open **MODEL** with **Artistic** selected. Try **Proximity 75%** and
**Anti-Salience 65%**, then move Distance while listening on headphones.
For a dense, peak-normalized wet texture, try both controls at 100% and
Distance at its maximum. New native VST3/CLAP instances default to
Proximity 75% and Anti-Salience 65%. Older saved states missing these controls
explicitly restore Proximity 0% and Anti-Salience 65%, including when loaded
over an active instance. Existing saved values, including an explicit 0%,
remain unchanged. At 0% the original Artistic audio path is retained exactly.
Right-clicking Proximity selects this legacy 0% sound. Standalone REAPER JSFX
keeps its original 0% default; native saved-state rules do not run in REAPER.

Proximity uses the existing 0.20–10 m distance mapping. Below 1 m it adds
bounded, attenuation-only near-ear contrast to point emitters. Beyond 1 m it
progressively lowers direct level and close presence relative to the existing
room cues, reaching full distance-cue strength at 10 m. Preserved stereo side
content and the Ambience Bed feed also follow distance; a broad bed can no
longer bypass the added distance treatment. Beds retain their stereo field.

This is a small distance-specific processor, separate from Anti-Salience MX.
It uses no MX foreground reference, persistent anchor tax, modulation-debinding,
scatter network or core level lock.

Anti-Salience adds stereo-linked density/onset presence restraint and slow
RMS attenuation for loud distant sources. The linked level stage is capped at
12 dB before strength weighting. It does not raise quiet material, restore
lost level, add reverb, or use separate left/right gain detectors. Set it to
0% for the proximity cues alone. It needs Proximity enabled and acts in the
far region; source character and desired prominence remain mix decisions.
Output Trim still acts last and can intentionally override the level cue.

These additions affect Artistic only. The Physical renderer and separate
Hyperreal variants are unchanged. Near-ear contrast is a creative geometric
approximation, not a new measured near-field HRTF. Digital RMS references are
mix references, not calibrated acoustic SPL. Perceived externalization still
depends on source material, headphones, listener and the rest of the mix.

## Source facing and internal head/body POV

Artistic **MODEL** also offers four optional controls. Their defaults preserve
the previous external sound, including when older projects are loaded.

- **Facing** rotates emission without moving the source: 0 degrees faces the
  listener, +/-90 is side-on, and +/-180 faces away. It is relative to the
  listener, so moving the source preserves the chosen facing relationship.
- **Directivity** enables frequency-dependent beaming. Try 100% and sweep
  Facing from 0 to 180. Low frequencies radiate more broadly; off-axis presence
  and highs recede. Each wall/corner reflection uses its own departure angle.
  The source arrow shows facing. At 0% the original emission is retained.
- **Internal POV** blends toward sound inside the head/body. At 100% it is
  head-locked, independent of external Distance, Facing, Proximity, Occlusion
  and room/late-field settings. Mono is centered; Stereo/Bed/Dual retain side
  information. Intermediate amounts blend external placement and internal sound.
- **Body Tone**, active with Internal POV, ranges from clear internal sound
  at 0% to darker, less wide body-transmitted colour at 100%. Default 50%.
  Output Trim remains final. Right-click resets any new control.

Directivity and Internal POV both default to 0%. They do not alter Physical.
The directivity curves are a generic authored radiator, not measured human-voice
or instrument data. Body Tone is an auditory approximation of transmission,
not anatomical modelling or physical bone/tissue vibration. Own voice combines
air and body conduction; no single bass boost reproduces it. Already recorded
room or binaural cues in the input remain part of the recording.

## Controls and views

- Lateral and Depth Front/Rear place the object; Distance controls near/far.
- Spatial Throw controls cue strength; Cue Curve compresses moderate angles.
- Size widens the apparent source; Push Out and Room emphasize externalization
  and early room cues. Cyan markers show early reflection hits.
- Occlusion colors blocked sources; Micro Motion adds small movement.
- Elevation uses Artistic height cues or Physical source height. Middle-drag
  or Shift-wheel adjusts it. Physical preserves range while changing elevation.
- Mono, Stereo, Bed and Dual define input interpretation. Input Width Preserve
  and Bed Anchor retain stereo/ambience structure rather than forcing every
  source into a point emitter.
- Automation Safe caps aggressive throw and enforces gentler smoothing. Motion
  Smooth sets the movement response; Output Trim controls final level.

Physical settings include frequency-scaled Pinna Profiles, Distance Gain,
Travel Time, Air Loss, room dimensions and listener ear height. The profiles are
variants of a KEMAR fit, not individual listener measurements. The fitted data
covers −40 to +90 degrees; lower directions retain the −40-degree spectrum.
Room cues fade outside room boundaries. In FOLLOW view, drag to change listener
yaw/pitch, Shift for fine control, and Alt-drag/Alt-wheel for roll.

The Scene drawer controls Late Field, Space Size, Protect and source presets.
Late Field at zero fades to bypass and clears the original tail; raising it
activates the local SceneVerb field. This differs from alternative FAUST
products whose disabled returns may keep private histories warm.

## Idle behavior and automation

The default native build waits for one second of output below −140 dBFS before
automatic idle suspension. Any nonzero input, parameter change or other host
wake event resumes processing. The canvas may continue drawing while audio
sleeps. Automatic suspension is a CPU-saving heuristic; use Never Sleep when
continuous internal evolution is required. Offline processing always advances
DSP. The native CLAP host bridge reports actual changed canvas parameters,
including button controls, through global live parameter events.

## Builds and alternatives

Build the default with:

```text
python scripts/build.py --only "Hyperreal 3D Panner" --config Release
```

Native builds use Legacy graphics and retain the established plugin/parameter
identity. HyperrealFast keeps a separate compatibility identity. HyperrealFaust
and HyperrealHybrid are separate designed alternatives; their Physical model
differs substantially and they do not replace this product transparently.
See [implementation, benchmarks and limits](../../../docs/Hyperreal-Panner.md).

Performance and numerical tests are scoped evidence, not a guarantee of
localization for every listener, preset or host. Audition placement and motion
with your material. KEMAR attribution: Bill Gardner and Keith Martin,
MIT Media Laboratory, 1994; the source includes the dataset reference.
