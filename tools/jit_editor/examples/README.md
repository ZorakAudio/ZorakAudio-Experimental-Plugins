# JIT Editor examples

Choose an example from **Load example...** in Edit, then press **Run**. The dropdown embeds these exact source files, so it works without an example directory beside the installed plugin. The archive also includes the files for inspection, copying and **Open source...**. Each source is self-contained; Faust uses the bundled `stdfaust.lib`.

| Example | What to learn | First experiment |
| --- | --- | --- |
| [JSFX stereo utility](StereoUtility.jsfx) | Ordinary sliders, `@slider` setup, sample DSP and GFX meters | Set Width to zero; both outputs become mono. |
| [JSFX envelope gate](EnvelopeGate.jsfx) | Linked detection and attack/release gain smoothing entirely in EEL2 | Raise Threshold, then lengthen Close time. |
| [FAUST smooth filter](SmoothFilter.dsp) | Pure Faust controls, smoothing, a resonant filter and wet/dry routing | Sweep Cutoff; increase Q cautiously because resonance adds gain. |
| [FAUST stereo compressor](StereoCompressor.dsp) | Shared stereo gain, unit conversion, continuous controls and a binary bypass | Lower Threshold; listen with and without Bypass. |
| [Hybrid gain](HybridGain.jsfx) | The smallest useful EEL setup -> Faust block -> custom EEL UI example | Drag the blue gain bar; watch the input meter. |
| [Hybrid studio channel](StudioChannel.jsfx) | A complete channel-strip effect with Faust DSP, exported meters and an EEL interface | Boost Bass, add Drive, lower Threshold, then blend Wet or bypass. |

## The studio channel at a glance

```text
EEL @block: convert dB/ms controls to gain/seconds
       |
FAUST @faust block: stereo EQ -> saturation -> linked compression
                   -> wet/dry -> output gain
       |                         |
       | audio                   | final meter values
       v                         v
host stereo output          EEL @block: meter dB conversion
                                 |
                            EEL @gfx: draw controls/meters
                                 |
                            slider_automate -> host parameters
```

The source has four labelled stages. EEL owns twelve automatable slider declarations and the interface. Its first `@block` converts drive/makeup/output dB to linear gains and attack/release milliseconds to seconds. Faust imports those ordinary scalars and slider aliases automatically. No manual Faust sliders or binding declarations are needed for those shared values.

The Faust section performs the expensive per-sample work: first-order bass/treble shelves, a constant-Q mid bell, normalized tanh saturation, a shared stereo compressor detector/gain, a wet/dry blend and output gain. Parameter smoothing lives inside Faust so controls can be updated once per callback without turning its DSP into one-sample calls. The saturation is a waveshaper, not a physical triode/diode model or an alias-free algorithm. Try the editor's 2x/4x oversampling when adding drive; changing oversampling resets DSP history.

`input_peak`, `output_peak` and `gain_reduction` are declared by EEL in `@init` and defined by Faust. Matching top-level Faust definitions automatically export signal values into those EEL globals. They are internal meter outputs and do not create additional host audio pins. The envelope detectors are sample-rate DSP; EEL sees their final values after the buffer. Gain reduction is the compressor's wet-path gain reduction, even while bypassed. The input/output meters show peaks, not LUFS or RMS.

The second `@block` follows Faust in source order and converts the peak exports to dB for the UI. Its explicit block boundary keeps these display calculations outside a sample-feedback dependency. There is no EEL `@sample` stage in this example. The compiler metadata records one non-fused, explicit block Faust stage; it computes over the host buffer, rather than being called once per sample.

The `@gfx` helper `control(...)` draws a label, value and bar, tracks an active drag, and returns its new value. The caller assigns that value to the corresponding slider and calls `slider_automate` when it changes. All declarations begin their labels with `-`, hiding duplicate generated widgets while retaining host parameters. The bypass button changes the wet blend and keeps the processing state warm; it does not skip DSP. Output gain still applies during bypass. A settled bypass at 0 dB output is dry audio; the short smoothing transition is intentional.

This is an original educational effect. It is not a replacement or bit-exact conversion of a catalog plugin, and it does not claim that Faust is faster than an equivalent EEL implementation. Its purpose is to show a clear division of DSP and interface responsibilities using the actual shared runtime.

## How to extend it

Faust-local/library definitions take precedence over inferred EEL inputs. For example, `ratio` and `mix` can resolve to library functions. The first block uses distinct `channel_ratio` and `channel_wet` captures instead; Faust cannot mistake them for those library names. Use descriptive, distinct capture names when adding controls. This keeps legal Faust name resolution rather than overriding its library semantics.

1. Add an EEL slider declaration with a stable slot/alias. Use a normal label for a generated control, or a `-` label when drawing it yourself.
2. Convert units in the first `@block` when useful. Refer to the alias or converted scalar inside Faust; the compiler infers the input binding.
3. Keep full-buffer DSP in `@faust block`. If a later EEL audio stage needs a continuously changing Faust scalar on every sample, redesign around streams/block boundaries or use the documented sample mode deliberately.
4. For a display output, declare an EEL global in `@init`, define its signal in Faust, and read it after the completed block or in GFX. Do not confuse a final scalar value with an entire signal buffer.
5. Keep mouse/string/drawing work in GFX. Notify host parameter changes with `slider_automate` so automation and last-touched behavior can follow the control.

## Save, Run and DAW presets

**Run** compiles the draft without saving it to disk. **Ctrl+S**, while the source editor is focused, saves the full UTF-8 source and then compiles/runs it. A source opened from disk has a save target; a newly written/pasted/example program gets a Save dialog. Choosing a resource **Source folder...** only sets import lookup and does not make its virtual `editor.jsfx` an overwrite target. Cancelling Save does not compile. A failed save leaves the running program intact; a successful save followed by a compile error retains the running program but the edited source remains saved.

DAW presets/project state contain the full applied source, separate draft, selected language, numeric/string controls, source/import origin, explicit save target, selected file paths, host settings and explicit `@serialize` state. A preset is not a source-file save. Preset loading recompiles the stored program and restores its saved controls; imports, images and sample files remain external. These examples need only the bundled Faust standard library. Arbitrary DSP history is not migrated unless explicitly supported through serialization. Successful Run/Ctrl+S uses fresh DSP history and declaration defaults; editing alone does not alter the active program.

## What is tested

The executable example check runs every embedded source at 44.1 and 48 kHz with callbacks of 1, 17, 256, 1024 and 4096 frames. It checks finite audible output and stereo pins, JSFX mono width, linked compressor stereo ratio, hybrid full-block stage metadata, actual GFX drags/host parameter changes, exported channel-strip meters and settled bypass audio. The same examples are tested through both frontend paths. This is functional validation, not a listening evaluation or a performance comparison.
