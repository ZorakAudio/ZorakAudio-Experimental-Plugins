# Hyperreal Panner Fast

A compatibility identity for the optimized Hyperreal V7.1.2 renderer, now also the default implementation of 3DPanner. Its Artistic and fitted-KEMAR Physical models, original filters, delay interpolation, geometric update cadence, smoothing and room/tail behavior remain.

The change specializes the fixed 28-ear-path rendering loop, each six-filter cascade and each five-coefficient update with constant offsets. The order of additions and state updates is preserved. This version remains EEL; it is not advertised as a FAUST conversion. The separate [Hyperreal FAUST Panner](../HyperrealFaust/README.md) is the new FAUST design.

The canvas and controls work as in the current 3DPanner source. That source is local-only; historical manager-link behaviour does not apply to V7.1.2. 3DPanner now contains these same optimizations under its existing identity; this separate identity is retained for sessions that already use HyperrealFast.

Build with `python scripts/build.py --only HyperrealFast`. Regenerate from the maintained 3DPanner implementation using `python tests/faust/panner_candidates.py`, then rerun qualification. The checked generator is the maintained patch; avoid hand-editing the generated copy.

See [the audit](../../../docs/Hyperreal-Panner.md) for complete-plugin timing, null comparisons and limitations. Fixed-offset specialization reduces overhead; it does not remove the original Physical renderer's substantial workload.

## Quick start

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
