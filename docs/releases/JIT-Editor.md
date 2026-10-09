# ZorakAudio JIT Editor — build your own audio tools inside your DAW

**Write an effect. Press Run. Hear it in your session.**

Turn audio ideas into working effects and instruments inside a **VST3 or CLAP**. Open an existing source, start from a documented example, or write your own. The compiler comes with the download—no separate development setup required.

Use **JSFX/EEL2**, the language behind REAPER's scripted effects, or **FAUST**, a language for describing audio processing. Combine them to pair FAUST audio processing with a custom JSFX interface.

![JIT Editor: running a hybrid FAUST and JSFX program with a custom interactive interface](https://raw.githubusercontent.com/ZorakAudio/ZorakAudio-Experimental-Plugins/main/docs/media/jit-editor.gif)

*From source to an interactive effect: a 30-second demonstration.*

## What you can do

- **Experiment while you listen.** Editing leaves the applied program running. Press Run to apply a successful build; compilation errors retain the previous program.
- **Create the interface with the sound.** Custom graphics, automatic controls, hidden sliders and source-defined audio routing follow your program. Hide the editor to use the full interface.
- **Save an effect as a preset.** DAW presets store the program and its controls. Ctrl+S saves source, compiles and runs.
- **Build beyond a simple filter.** MIDI, sample banks, imports, images and background analysis support instruments and larger effects.

Unicode editing, familiar copy/paste/undo shortcuts, resizing and control resets are included.

## Start with the examples

Six examples take you from a stereo utility, gate, filter and compressor to a **complete studio channel strip** with EQ, saturation, linked compression, blending and meters. Its source shows how audio processing and an interactive interface fit together.

Choose **Load example...**, press Run, and start changing it. [Annotated examples and first experiments](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/tools/jit_editor/examples/README.md).

## Downloads and installation

The Editor has **its own release**, with separate CLAP and VST3 packages for **Windows x64** and **Linux x64** (Ubuntu 24.04 baseline). **macOS Editor support is pending.**

Keep **JITEditor.runtime** beside the CLAP plugin, or copy the complete VST3 bundle. Close the DAW before replacing an older build, then rescan. The installed name still includes **PoC**.

## A few things to know

A successful **Run or Ctrl+S starts with the program's declared defaults** and resets its processing history. Loading a preset restores saved controls. Program changes can click.

Presets store source; **imports, images and samples remain external**. Keep those files with your project and use Open source or Source folder to locate dependencies.

This is an experimental development tool. Use trusted code: faulty programs can hang or crash the DAW. Editing is available in this standalone Editor; the catalog plugins remain separate products.

[Full Editor guide](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/tools/jit_editor/README.md) · [Validation scope](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/tools/jit_editor/VALIDATION.md) · [Report an issue](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/issues).
