# Hyperreal FAUST Panner

A new headphone panner built with an embedded `@faust block` renderer. It reuses Hyperreal's current standalone canvas and automation controls, but deliberately has its own sound. It is **not the original measured KEMAR renderer**.

Drag the source on the canvas, use the distance wheel, and adjust Throw, Size, Push Out, Room, Occlusion and Micro Motion. The original Mono, Stereo, Bed and Dual mode controls remain. Open the Physical settings drawer for listener yaw/pitch/roll, distance gain, travel time, air loss, coloration profile and room dimensions. The Scene drawer controls the late field, space size, protection and role presets.

The render models are two voicings of the new design:

- **Artistic:** shaped pan cues and softened distance gain.
- **Physical:** approximate inverse-distance gain, bounded propagation time and room-boundary weighting. This is a designed cue model, not a measured HRTF simulation.

FAUST owns smoothing, fractional delays, head shadow, comb-based pinna coloration, six early reflection taps per ear and an eight-comb stereo late field. EEL handles the interaction and block-rate targets. FAUST computes once per host block, without EEL audio-stage handoffs, JIT compilation or worker threads during playback.

The late field uses low-frequency protection, damping, ducking and width controls. Its histories remain warm while the return is off; enabling it can reveal the existing tail. Moving propagation delays can bend pitch. Startup controls fade in from zero rather than reproducing the original initialization. Travel time is bounded to 65534 samples, early delays to 16382, and pinna delays to 126. These bounds make the design rate-dependent at extremes.

The existing V7.1.2 source has no manager IPC implementation. This variant consequently retains local operation; an older 3DPanner README describes a historical manager link that is not present here.

This is a separate native VST3/CLAP identity; installing it does not replace 3DPanner. The embedded source requires this repository's compiler and FAUST LLVM at build time, and is not stock REAPER JSFX. Use `python scripts/build.py --only HyperrealFaust` to build it.

Maintained DSP: `tests/faust/panner_renderer.dsp`. Recreate the interaction-shell integration with `python tests/faust/panner_candidates.py`, then requalify it. The generator checks integration boundaries. See [the audit](../../../docs/Hyperreal-Panner-Variants.md) for measurements and tested limits. Numerical checks do not establish subjective localization quality; audition it on headphones.
