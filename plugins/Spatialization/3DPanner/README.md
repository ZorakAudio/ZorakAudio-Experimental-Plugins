# Hyperreal 3D Panner

A headphone-focused, mouse-first 3D panner with two original render models:
**Artistic** for softened perceptual placement, and **Physical** for per-ear
geometry, fitted KEMAR spectra, propagation and room paths. The maintained
V7.1.2 implementation incorporates faithful Fast EEL optimizations; it does
not substitute a simplified FAUST renderer.

## Start here

Place it on a stereo track and listen on headphones. Drag the orange source
in the stage canvas and use the mouse wheel for distance. The solid orange
object is the visual position; its ghost shows the effective cue position after
Spatial Throw and Cue Curve compression. The POV canvas range limits ordinary
drag placement near center; quick controls and snap chips still allow the full
range. Hover the canvas controls for guidance.

Choose Artistic or Physical with Render Model. Artistic retains its original
distance mapping in metres; Physical displays feet while stored room/ear-height
values remain metres. Current control is entirely local. The retired manager
and string slots are inert and retained for old state/automation identities;
this revision does not subscribe to 3DPannerManager.

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
