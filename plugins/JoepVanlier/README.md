# Joep Vanlier / Saike native plugin catalog

The 50 vendored JSFX entries live in `plugins/JoepVanlier/<PluginKey>/`. Each has its own source tree, metadata, license and user README. These READMEs are embedded in VST3 and CLAP builds and displayed by the `?` button; documentation changes require a rebuild to reach an installed binary.

Each help page describes the effect, a quick workflow, routing and source-declared controls. Canvas parameters are identified as hidden automation controls. Dummy/deprecated slots are omitted. Native packaging and compiler/runtime optimization preserve the DSP algorithms.

## Performance

Each leaf has a WDL/EEL2 native x64 JIT versus LLVM DSP measurement or an explicit reason its workload is not comparable. The [catalog comparison](../../docs/Joep-Performance.md) explains the method. DSP timings exclude the JUCE host callback and graphics thread; they cannot establish full REAPER or open-editor CPU savings, especially for analyzers and GFX-driven effects.

## Licensing and upstream material

The original upstream README remains in `_UPSTREAM_README.md`. The conversion manifest records original source paths; its `JoepValier` category/entry spelling is historical.

The upstream root license is MIT, but spectral analyzer files explicitly declare LGPL. Those declarations are retained. Super Spreader credits original work by lkjb and has no per-file license field. Consult each leaf's `LICENSE.upstream` and source headers for applicable credits and terms.
