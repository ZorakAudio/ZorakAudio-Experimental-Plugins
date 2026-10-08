# Standalone JIT Editor — shared AOT/JIT runtime, Windows x64

This standalone VST3/CLAP bundles the existing Python/llvmlite JSFX compiler and Faust LLVM compiler. Users do not need Python, LLVM, Faust, Visual Studio, or an internet connection installed separately.

The standalone editor now uses production runtime components shared with the AOT plugins. It no longer maintains separate PoC implementations of Faust processing, deferred tasks, or sample pools. The existing plugin filename and host identity still include “PoC” so this refactor does not introduce another plugin identity. This does not enable source editing in ordinary AOT catalog plugins.

The editor's **Run** button uses the standard production compiler. There is no frontend selector in the user interface. The experimental C++ resolver/parser remains available to developer qualification tools and for compatibility with older saved projects. It feeds the same production lowering/LLVM emitter and does not remove the bundled Python dependency.

The [documented examples](examples/README.md) are embedded in **Load example...**: two JSFX effects, two pure Faust effects, a small hybrid and a complete hybrid studio channel. The channel strip uses Faust for full-buffer EQ/saturation/linked compression and EEL for automatable custom controls and exported meters. Source files are also included in the archive.

## Install and run

Close the DAW before replacing the previous build. For CLAP, extract the archive and keep `JITEditor.runtime` beside `ZorakAudio JIT Editor PoC.clap` in a CLAP search directory. For VST3, copy the entire `.vst3` bundle. Rescan in the DAW. The compiler payload is required; the plugin binary alone is insufficient.

The normal view displays the plugin graphics with an **Edit** button in the corner. Press **Edit** to reveal source and compilation controls. Use **Open source...** for a local JSFX/DSP file, or write source/load an example, then press **Run**. A successful Run returns to the plugin view; **Show plugin** also closes the source pane. Editing alone does not change audio. The previous program continues during compilation and after a compilation failure. **Default** restores stereo passthrough and removes program controls. Each successful Run initializes a fresh DSP state and uses declared slider defaults; saved projects restore numeric/string controls, selected file paths, and values explicitly saved by `@serialize` when recompiling their saved source. Other DSP histories are not automatically migrated. Changes can click; no crossfade is guaranteed.

For pasted code, use **Source folder...** to select the original source directory containing its dependency folders. The selected folder is shown in Edit and saved with the project. The editor uses a virtual `editor.jsfx` path there; it does not write your code to that file. `provides:` does not download dependencies: the files must exist. Image lookup supports direct paths, `Resources` inside the selected folder, and a sibling `Resources` folder using the shared production decoder. Imported files and images remain external and must stay available when reopening a project.

The source editor supports Ctrl+A/C/X/V/Z/Y and Ctrl+Shift+Z. The window is resizable. Generated controls scroll when they do not fit. Programs without executable `@gfx` code open in a compact controls view with no empty graphics pane; empty/fully hidden controls also occupy no space. A program with neither graphics nor controls displays just the toolbar. **Edit** expands the source view and **Show plugin** returns to the compact view. Routine factory/running messages have no footer; the Run button shows compilation in progress, and failed compilation shows its diagnostic. The plugin view gives GFX the full available area. In editing mode, GFX remains beside the source when present. After compilation and host reconfiguration, GFX can run before the first audio callback, including while the host is paused. Audio adoption still occurs at a processing-block boundary. Host-rate preparation invalidates the old processing configuration and recompiles the applied program; audio can be silent until that recompile completes. Background Run retention does not promise uninterrupted playback through a host-rate reconfiguration.

Source, comments, string literals, control labels/choices, file paths and project state support UTF-8. The source editor uses a monospace grid, including fallback Unicode glyphs, so mouse selection, caret placement and deletion agree with the displayed text. Wide glyphs fit one codepoint cell; this is a code editor, not a grapheme-aware typesetting editor. EEL identifiers retain the production language's existing syntax. JSFX string operations retain their byte-oriented semantics; Unicode support does not change `strlen` to count characters. Script GFX can use `gfx_setfont` with a system font for Unicode text; the legacy bitmap font does not contain every Unicode glyph.

## JSFX, controls, and GFX

**Ctrl+S**, while the source editor has focus, saves its UTF-8 source and then compiles/runs. A file opened with **Open source...** is the save target; a new/pasted/example draft opens a Save dialog. A resource **Source folder...** is not a disk-save target. Cancelling Save does not compile; failed saves retain the running program and show an error. A saved draft can still fail compilation, retaining the old program while leaving the edit saved. Ordinary **Run** does not write source to disk. Source files use a temporary file before replacement; no writes occur on the audio callback.

DAW presets/project state already store complete programs: applied source, separate draft, language, controls, paths, host settings and explicit serialization. They do not need the original top-level source file to recompile the stored text, but imported source/assets/sample files remain external. The explicit source save target is also stored; older presets without it use a Save dialog on Ctrl+S. Loading an example or restoring Default clears the previous disk-save association. Preset restore keeps saved control values; successful Run/Ctrl+S starts from declaration defaults and fresh DSP history.

```text
desc:Gain and native GFX
slider1:gain=0.5<0,1,0.01>Gain
@sample
spl0 *= gain;
spl1 *= gain;
@gfx 400 300
gfx_set(0.1,0.2,0.3,1);
gfx_rect(0,0,gfx_w,gfx_h);
gfx_set(0,1,0,1);
gfx_rect(20,20,(gfx_w-40)*gain,30);
```

Slider declarations use the shared production parser: numeric sliders 1–256, named aliases, labels, defaults, step sizes, choice menus, reversed ranges, and log/square curves. The editor generates controls from these declarations. It no longer presents 16 artificial normalized controls. Declared hidden sliders start hidden, and explicit `slider_show` calls can hide/reveal widgets. String-input sliders have text widgets and remain outside the numeric host-automation list.

Right-click a generated slider, choice control, or its label and choose **Reset to default**. The menu shows the declared default value or choice. Numeric sliders also reset on double-click. String controls include reset in their usual cut/copy/paste menu. Numeric/choice resets notify the host inside a parameter gesture, so host automation can record them. A reset changes only that control: it does not recompile source or clear DSP history. Hidden controls stay hidden; custom JSFX GFX controls keep their source-defined mouse behavior. **Default** in the source toolbar still restores the factory passthrough program, rather than resetting a single control.

`@gfx` is compiled by the same native legacy/GFX compiler path as AOT plugins. The drawing bridge, strings, image bank, file primitives, rasterizer, menu bridge, and menu overlay are shared with production. GFX execution uses a dedicated worker; audio and source editing continue while a modal GFX menu waits for selection. Mouse buttons/modifiers, wheel movement, keyboard input, file-drop paths, and the common cursor shapes are forwarded. Per-instance state and strings are retained across frames. The dynamic editor gives GFX a private scalar view so DSP scratch variables cannot interrupt drawing loops. Audio publishes scalar snapshots without waiting for the GFX worker; permitted GFX writes, slider changes and visibility changes return at audio callback boundaries. RAM and runtime services retain their production sharing rules; this is not an atomic snapshot of the entire heap or string graph.

This integration is exercised by pixel, text, slider-feedback, keyboard, wheel, and modal-menu tests; it is **not a claim that every JSFX host service or every production UI behavior is already implemented**. Opening source sets its directory as the resource/import root. Pasted source has no original project directory; use **Open source...** when it needs relative imports or assets. Imported dependencies and assets remain external files at their saved paths. Runtime `slider_show` controls widget visibility, and GFX slider automation is forwarded through parameter gestures.

## Inferred input/output pins

The editor uses the production compiler's `io_channels` metadata. `splN` reads/writes and explicit `in_pin:` / `out_pin:` declarations determine the required channel counts, up to 64. Example:

```text
@sample
spl0 *= 0.5; spl1 *= 0.5;
spl2 *= 0.5; spl3 *= 0.5;
```

This declares a four-channel program. The existing AOT rules conservatively mirror an unspecified side and fall back to stereo for code with no audio usage. Use `in_pin:none` to explicitly declare a JSFX generator with no input pins. Faust pipelines use their actual port counts, so a pure mono Faust process exposes one input/output and a Faust generator can expose no inputs.

A changed source can change the host parameter list and pins. CLAP requests a host restart, commits the new schema while deactivated, and requests parameter/audio-port rescans. VST3 requests I/O reconfiguration and refreshes its parameter/controller caches while deactivated. The new program is adopted at an audio-block boundary after this handshake. A host must honor reconfiguration requests; the editor cannot force the DAW's track routing to supply additional channels. Automation for removed parameters is not preserved as a portable contract. Parameter changes are applied at callback boundaries.

Both public-ABI host harnesses check parameter changes and stereo/four-channel transitions; the CLAP harness additionally checks an explicitly declared zero-input generator. These are automated host tests, not confirmation of every DAW's UI behavior. A live REAPER reconfiguration test remains useful.

## Mixed JSFX and full-buffer Faust

```text
slider1:gain=0.5<0,1,0.01>Gain
@block
level = gain;
@faust block
process = spl0 * level, spl1 * level;
```

The compiler infers `level`, `spl0`, and `spl1`. Legacy bare `@faust` retains the production compiler's sample semantics. Explicit `@faust sample` interleaves with sample stages and can require one-sample calls; choose `@faust block` for full-buffer processing. Captured scalar streams, fused sample islands, and generated tables use the production execution plan. Faust executes over the callback's full frame count. Unfused EEL sample stages use the existing native bulk loop. Pure **FAUST** mode wraps source in `@faust block`:

```text
process = _,_ : *(hslider("Gain",0.5,0,1,0.01)),
                 *(hslider("Gain",0.5,0,1,0.01));
```

Unbound Faust sliders, numeric entries, buttons, and checkboxes become generated host controls. Faust `button` controls are momentary: hold to send 1, release to return to 0. Checkboxes and 0–1 sliders/entries with step 1 use toggle buttons and boolean host parameters. Continuous 0–1 controls, such as gain/mix with step 0.01, remain sliders with their full range. Generated binary controls support the same right-click default menu. Closing the editor releases a held momentary button. JSFX-inferred Faust zones retain their existing bindings. Faust bargraphs do not yet become output meter widgets. Combined JSFX/Faust programs have at most 256 editable controls.

## Compilation and runtime

1. A worker starts a private, bundled Python helper with an isolated import path.
2. The standard production Python frontend resolves/preprocesses source and parses EEL, using the existing lowering, symbol/slider binding and LLVM emitter. Developer tools can select the experimental C++ resolver/parser to feed that same pipeline. The bundled Faust LLVM compiler supplies Faust stages. LLVM O2 is applied and metadata is returned.
3. The plugin links native entrypoints with the bundled LLVM ORC runtime, binds supported runtime services, and initializes state away from the audio callback.
4. Host reconfiguration precedes adoption of the initialized program. Retired native code is released on the compiler worker after audio and GFX stop using it.

New Run/Default requests invalidate older jobs and terminate their helper processes, including their Faust child processes. Closing the plugin also closes its compiler job. There is no arbitrary 120-second compiler cutoff: large programs can take several minutes. No Python executes on the audio thread. Each instance owns its DSP state, strings, RAM, and JIT code. Heap indexing uses the production memory services; JIT preallocates the declared JSFX memory bound (8M cells by default), capped at 32M cells. `options:maxmem=...` changes that bound. Scalar variables use a separate allocation sized from compiler metadata; there is no 4096/32768-cell JIT limit. Deferred-task snapshots use that same actual variable extent.

Older saved projects retain their applied frontend when reopened, including on host sample-rate changes. Pressing the user-facing **Run** button selects the standard frontend for the replacement. Developer checks can still select C++ explicitly; a native frontend error does not silently fall back to Python. Pure FAUST source uses Faust's compiler directly. Native parsing has a 512 parser/AST-depth limit in addition to the editor's other existing limits. Mixed plans currently make a second native parse of the planner-generated EEL markers. Do not assume that migration compiles faster; JSON transfer and Python syntax-tree reconstruction add overhead.

The editor uses the production source resolver/preprocessor, EEL compiler, native GFX, MIDI builtins, FFT/memory builtins, communication/gmem runtime, deferred-task runtime, sample pools, and mixed Faust execution plan. It supports `@serialize` with `file_var`, `file_mem`, and `file_string` on handle zero; source text and selected paths are also stored in the host state. Actual host transport/track information is forwarded when supplied. `pdc_delay` is reported to the host asynchronously. Enhanced `#FILE:` slots expose file-selection buttons; decoding and pool publication happen on worker threads. Existing `file_open`/`file_riff`/`file_mem` calls use the native file backend; disk operations belong in initialization/GFX, not audio callbacks.

These are implemented capabilities, not a blanket promise of compatibility with every plugin. [VALIDATION.md](VALIDATION.md) records the test coverage and its limits. Parameters are applied at callback boundaries; `slider_next_chg` returns the current value and no remaining intra-block point, as in this repository's AOT adapter. Faust bargraphs do not yet become output meter widgets. Foreign Faust functions/variables/constants and `soundfile` still require additional integration. This editor is Windows x64 only and retains explicit resource bounds: 4 MiB source (also checked after import expansion), 256 combined controls, declared heap limits, and existing task capture/count limits. A program exceeding a bound fails with a diagnostic.

The plugin view exposes the production **Oversampling** and **Sleep** settings. Oversampling supports Off/2x/4x/8x and resets DSP history when changed. Sleep defaults to Auto and also offers Input, Events, Free run, Never, and Explicit, using the same inference, readiness, task, MIDI, and wake rules as AOT. Offline rendering stays awake. Sleep is saved as a plugin setting; oversampling is also a host parameter. Default restores passthrough while retaining these host settings.

Both execution modes use the same compiled-section dispatcher, Faust execution engine, numeric/FFT/atomic/string/slider services, file and sample-pool machinery, task heap hooks, MIDI lifecycle, audio routing/oversampling, host environment and idle policy. JIT supplies a program-specific entrypoint table and metadata rather than statically linked generated symbols. Its dynamic editor and host-reconfiguration adapter remain specific to the standalone plugin. Script-authored slider values retain production precision while host parameter notifications are acknowledged.

Each editor program owns a separate ORC module. Generated Faust table bindings are module-private, with prepare before publication and serial audio execution; JIT machine code is not shared between instances. Native LLVM remains loaded for the host process lifetime, so replacing a runtime DLL still requires closing the host. Do not copy a new payload over a running host's loaded DLL. A future versioned runtime cache can make distribution folder replacement easier.

The C++ migration design is in `CPP-MIGRATION.md`. The [native frontend](../native_compiler/README.md) passes source/token/unlowered-AST comparison checks and remains bundled for developer qualification and saved-project compatibility. It does not emit LLVM itself. Production Python lowering/emission remains in use until the complete native implementation passes differential equivalence and host tests.

Native DSP and GFX execute inside the DAW. The compilation process boundary does not sandbox running code. Infinite loops or invalid native operations can still hang or crash it; use trusted local code. A linter, debugger, execution watchdog, and macOS/Linux support remain future work.

## Build and validation

Configure `tools/jit_editor` with CMake/Ninja/Clang on Windows, specifying `Python3_EXECUTABLE` for the installed Python with llvmlite. Build `JITEditor_VST3`, `JITEditor_CLAP`, `jit_editor_check`, `jit_editor_clap_check`, and `jit_editor_vst3_check`. The build generates the bridge header through the production compiler. Checked-in patches are automatically checked at configure and build time by both AOT and JIT builds; conflicts are reported without discarding local edits. Dynamic host-reconfiguration hooks are gated to the editor targets; the shared CLAP event-notification corrections also apply to catalog builds. See [building and CI](../../docs/Build-and-CI.md) for publishing without dependency forks and the platform matrix. `package_runtime.py` and `package_bundle.py` stage explicitly supplied local Python/Faust installations; they do not download or install tools. `scripts/ci_jit_editor.py` builds and qualifies usable Windows CLAP/VST3 archives with the matching compiler payload.

The shared-runtime tests compare the frozen pre-refactor AOT implementation against current DSP and complete processor builds, and execute JIT programs through the actual processor and packaged CLAP/VST3 interfaces. Catalog coverage includes all 83 JSFX entries; loaded-bank tests cover Sample tape/granular playback and Corpus analysis/playback using synthetic WAV fixtures. See [VALIDATION.md](VALIDATION.md) for exact scope, reproducible checks, and remaining limitations. Host-delivered Ctrl shortcuts still require a live REAPER check. Distribution archives retain bundled dependency notices; broader public redistribution still needs the existing licensing review.
