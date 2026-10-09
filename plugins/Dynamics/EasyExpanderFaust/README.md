# EasyExpander Faust

A minimal downward expander with an ERB-weighted detector. It preserves EasyExpander's controls and custom meters, with the detector/expander audio implemented in an embedded FAUST section. This has a separate plugin identity from EasyExpander.

## Quick start

1. Insert it on a stereo track and play the material to clean up.
2. Set **Threshold** near the level below which spill or tails should be reduced.
3. Raise **Depth** to set maximum attenuation; adjust **Contour** from gentle to gate-like action.
4. Adjust **Detector HPF/LPF** if low rumble or high hiss triggers it incorrectly.
5. Compare quiet passages and attacks, rather than only loud continuous audio.

## Controls and routing

- **Threshold (dB)**: activation point, initially -40 dB.
- **Depth (dB)**: maximum reduction, initially 24 dB.
- **Contour**: opening/closing character, initially 50; lower is gentler.
- **Detector HPF**: low-frequency rejection in the detector; 0 disables it.
- **Detector LPF**: high-frequency detector limit; 20 kHz is its open setting.

The detector filters do not EQ the output signal. Use stereo input/output 1/2. The gain and detector histories are stateful; quiet input still needs to advance them. Offline renders always advance DSP. Native idle behaviour also depends on the selected sleep mode; use continuous processing when comparing output with an always-active reference.

## Implementation and comparison

EEL owns initialization, sliders and graphics. The compiler infers slider aliases, signal inputs and scalar bindings for FAUST; exported final-sample values feed the meters. FAUST has its own DSP state.

The recorded matched active-processing comparison was 15.821 s versus 3.476 s (4.55x), with 58,558,936 bit-identical float samples. That is a specific supplied-recording experiment, not a universal ratio. Earlier threshold-based Auto Sleep results did not null against continuously active processing. See the [sleep audit](../../../docs/validation/EasyExpander-Sleep-Audit.md) and [FAUST integration contract](../../../docs/JSFX-Faust-Sections.md) for scope and limitations.

## Building and help

Build with `python scripts/build.py --only EasyExpanderFaust --config Release`. The compiler needs FAUST's LLVM backend at build time; the plugin does not ship a FAUST compiler. The embedded section requires this repository's compiler rather than stock REAPER JSFX.

The `?` button shows this README embedded in both native formats. Rebuild to update installed help text.
