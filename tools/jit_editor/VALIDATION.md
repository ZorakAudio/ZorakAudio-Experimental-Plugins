# Shared AOT/JIT runtime qualification

## Linux qualification — 8 October 2026

The standalone editor now builds and runs on x86-64 Ubuntu 24.04 under WSL2.
This port changes OS integration and packaging; DSP/GFX still use the production
AOT/JIT runtime. The tested payload uses Python 3.11.17, llvmlite 0.46.0 and
Faust 2.81.2 built with LLVM 18.1.3. macOS editor support remains separate work.

`scripts/ci_jit_editor.py` passed against the real Linux CLAP and VST3 archives,
extracted into a different directory containing spaces and Unicode. Native
runtime/public-interface tests run with PATH empty and developer compiler/Python
environment settings removed. The bundled Python import paths stay inside the
payload. No system Python/Faust/LLVM is needed by the running editor.

| Gate | Linux result |
| --- | --- |
| Native frontend contract | Passed, including numeric parsing under a non-default locale |
| Packaged compiler | All 24 positive/negative cases passed for each frontend |
| Shared runtime | Standard and optional C++ frontend suites passed |
| Interface | Controls/defaults, binary Faust controls, hidden/empty panes, Unicode glyph/caret/clipboard, imports/images, saved state and Ctrl+S checks passed |
| Examples | All six documented JSFX/Faust/hybrid programs passed |
| Resources and host settings | Nested imports, image roots, preset resource restoration and 1/2/4/8x oversampling/rate/MIDI checks passed |
| Sample | Three generated WAVs loaded; tape and granular MIDI playback produced finite nonzero audio and GFX/state save passed |
| Corpus | Three generated WAVs completed actual analysis; MIDI playback produced finite nonzero audio and GFX/state save passed |
| Public CLAP/VST3 | Extracted plugins passed processing, dynamic controls, inferred pins and host restart/rescan checks |
| Compiler lifetime | Both supervisor tests passed, including parent exit terminating the Python worker and its descendant |
| Graphics concurrency | All 64 columns survived all 60 synthetic frames; actual Sample retained all 61 bank bars and its unchanged non-flat EQ over 200 frames and 445 MIDI retriggers |
| Shared Faust compiler | 12 compiler/import contracts and 19 mixed execution fixtures passed on Windows and Linux |

The Linux save test exposed a POSIX directory-replacement behavior; source saving
now rejects a directory target explicitly and the test checks that it survives.
Archive isolation also caught a distro `sitecustomize.py` shadowing the private
startup module. The packager excludes the distro file. Windows Faust include
roots with Unicode are mirrored to relative private paths because its executable
uses narrow file arguments; the standard shared compiler handles this too.

These are WSL build and public-interface results, not a live Linux DAW or realtime
latency certification. The earlier frozen AOT comparison and full JIT catalog
results below are Windows historical evidence, not reruns of those matrices on
Linux. Full Linux AOT catalog compilation is tracked separately from this editor
qualification. Native execution remains in-process with the existing safety limits.

## Windows package requalification — 8 October 2026

The Windows CLAP and VST3 archives were rebuilt and passed the same expanded
`scripts/ci_jit_editor.py` qualification, including isolated extraction into a
Unicode path, all 24 compiler cases per frontend, native frontend contracts,
shared runtime, controls, Unicode/save/examples, imports/images, oversampling
and loaded Sample/Corpus banks. The public test loaders now decode Windows
command-line arguments and module paths as Unicode; both extracted plugin
interfaces passed. The Linux-specific supervisor remains a Linux-only gate.
This run does not replace the historical frozen AOT comparison below or claim
that the entire Windows AOT catalog was rebuilt on 8 October.

## Windows interface and runtime evidence — 7 October 2026

The interface/Unicode follow-up removes the user-facing frontend selector, uses the standard compiler for the editor's Run button, and displays a footer only for failed compilation. Empty GFX/controls collapse. The source editor uses a monospace primary font and draws fallback Unicode glyphs in the same codepoint grid used by caret/hit-testing. Its focused check exercises glyph pixels, caret/backspace, undo, clipboard, tabs and horizontal scrolling, UTF-8 source/import/image paths, labels/choices/string defaults, GFX text/image pixels, saved Unicode source/draft/path and numeric/string controls, failed-Run retention and fresh successful-Run defaults. Pure Faust Unicode labels and controls-only layout are checked too. Run with `jit_editor_check.exe --interface-unicode` (optional screenshot directory). The generated-control checks and public packaged interface/compiler checks are rerun for this delivery. The full catalog, frozen AOT comparisons and Sample stress results below are earlier qualification evidence, not reruns for this UI change. No shared AOT DSP/runtime source changes are part of this follow-up.

The initial runtime-consolidation qualification below is retained as historical evidence. The controls/GFX follow-up has separate logs and a new delivery report: its full 83-plugin catalog and six complete AOT-processor matrix were not rerun. Current follow-up checks include the 44-case frozen AOT DSP/state comparison, both JIT runtime suites, both generated-control suites, 24 independent WDL semantics cases (including multiline `while`), packaged interfaces/compiler, oversampling/resources, and the focused graphics tests described here.

The follow-up fixes native dynamic GFX scalar ownership: DSP and GFX no longer execute on the same scratch-variable cells. The editor adapter freezes a scalar view, retains private GFX locals/host drawing state, and publishes permitted writes and slider/visibility changes at callback boundaries. The audio publisher uses preallocated nonblocking snapshot slots. Runtime services, task/heap ownership, drawing, strings, files and sample pools retain their production implementations. This does not provide a transactional snapshot of RAM or strings, and it does not change ordinary AOT GFX execution modes.

The original shared-cell fault reproduced on 60/60 synthetic frames (zero of 64 expected columns at minimum); the corrected adapter produced all 64 columns on 60/60 frames. Actual Sample loaded 61 generated WAVs and retained all bank bars and the unchanged non-flat EQ response across 200 frames with repeated MIDI notes. Separate checks extract Sample's actual display equations and compare native GFX with WDL EEL at 387 frequency/settings points per frontend, including flat EQ, peaks, resonant cuts and slopes. They verify the display equation, not equality between its cutoff visual approximation and every nonlinear audio transfer function. See the follow-up report for exact counts, errors and binary hashes.

This refactor replaces the editor's separate PoC runtime services with production components used by AOT plugins. The compiler remains the existing Python/llvmlite pipeline, with an optional C++ resolver/parser feeding its production lowering and LLVM emitter. AOT publication remains statically linked; JIT publishes an instance-specific native entrypoint table. Neither runtime consolidation nor the hybrid frontend establishes a DSP performance improvement.

## What is shared

The subsequent save/examples update adds focused-editor Ctrl+S disk save followed by Run, with an explicit save target kept separate from a virtual import/source-folder origin. The interface check exercises actual UTF-8 file replacement, preset restoration of the save target, save-failure retention and compile/run after save. Six documented sources are embedded in the dropdown and included in the archives. `jit_editor_check.exe --examples` checks them at 44.1/48 kHz with variable buffers, stereo routing and finite audio, hybrid non-fused block metadata, real GFX parameter changes/meters, settled bypass and complex program/control preset restoration. Repeat with `--examples --cpp-frontend` for developer backend coverage. This is functional validation, not a DSP speed comparison.

| Responsibility | Production implementation used by both |
| --- | --- |
| Compiled sections and mixed execution | `JsfxCompiledProgram.h`, `JsfxFaustPlan.h`, `JsfxFaustEngine.h`; emitted bulk `jsfx_process_block` is used directly |
| Native service binding | `JsfxRuntimeExports.inc` is the authoritative service manifest for the helper, binder and package |
| Numeric, FFT, atomics, strings, sliders | Production builtins and native strings; shared declaration/value mapping, parameter retention and GFX gesture rules |
| Files and sample pools | `JsfxFileRuntime.h`, `JsfxSamplePoolOperations.h`, builtins: real decoders, cache/loader, slots, generation/publication, adoption and read policy |
| Deferred tasks and heap lifetime | Production task runtime, `JsfxTaskRuntimeHooks.h`, `JsfxHeapMemory.h`; real clone/preserve/adopt/rebind operations |
| Explicit serialization | `JsfxSerialization.h`, with real handle-zero `file_var`, `file_mem`, `file_string` transactions |
| Host processing | `JsfxAudioHost.h`, `JsfxHostEnvironment.h`, `JsfxMidiHost.h`, `JsfxIdleRuntime.h`: routing, oversampling, transport, note cleanup, sleep/wake rules |
| GFX and resources | Production native GFX bridge, renderer, strings, menus, image decoding and source-relative resource lookup |
| Safety and reset | `JsfxStateVariables.h`, `JsfxStateReset.h`, `JsfxProcessingSafety.h`: ownership, reset and audio/MIDI silencing on detected memory faults |

The copied JIT Faust, task-hook and sample-pool implementations were removed. JIT still needs an adapter for dynamic source editing, compiler process management, ORC module ownership, per-program metadata and host parameter/port reconfiguration. That adapter binds the common runtime rather than interpreting a smaller language or substituting successful no-op services.

Native shared state uses ABI 17: the fixed typed state owns a pointer/count for scalar variables allocated to the actual compiler-reported extent. The ordinary AOT publication ABI is retained. Task-private snapshots and arenas use the actual variable extent. The previous 4096/32768-cell JIT restrictions were removed; Protosynth executes through the actual JIT processor. Rebuilds must pair their generated headers/metadata with the matching native runtime.

## Publication and host behavior

- Compilation, LLVM optimization and initial construction occur off the audio thread. The private program receives restored controls/strings/files before `@init`; slider aliases are available during initialization and reapplied before `@slider`.
- The host installs the matching controls and port layout before audio may adopt the candidate. Adoption happens at a callback boundary. Failed Run keeps the last working program and its source/resource origin.
- Programs remain pinned while GFX uses them; retirement ends outstanding MIDI notes and unloads code on the worker after readers finish. At handover/reset, old note-offs precede new note-ons at the same timestamp. Compiler jobs own their child processes and terminate on replacement, cancellation or plugin destruction.
- Oversampling uses production 1/2/4/8x routing/resampling and reset policy. Host-rate preparation recompiles the applied source with its selected frontend while retaining host controls, strings and selected files. Audio can be silent until that recompile completes; this is separate from failed-Run retention under an unchanged host configuration.
- Idle behavior uses production inference/readiness/override rules. Pending tasks, pool adoption, events and explicit wake requests prevent unsafe sleep. Offline rendering stays awake.
- DSP/GFX slider notifications retain the script's original double value while host float notifications are acknowledged. GFX notifications are published before host conversion and request the next `@slider` pass. Unrequested writes are not automatically advertised as host automation.

## Recorded validation

The raw pre-refactor snapshot under `build/runtime-unification/baseline` is hash-checked by the comparison tools. It is not regenerated from the refactored implementation. An existing callback signature build error in the frozen non-native GFX processor is adapted in a separate probe copy using a one-argument forwarding lambda; the raw snapshot remains unchanged. This is the only probe adapter applied to that frozen processor.

| Gate | Scope and evidence |
| --- | --- |
| Frozen AOT DSP/state | 44 bit-exact before/after cases across publication and native layouts; numeric/FFT/atomics/strings/control fixtures and checked state |
| Frozen complete AOT processors | EasyExpander, CMD, HyperrealHybrid, Amaranth, Sample, Corpus: two sample rates, four oversampling modes, variable callback sizes, audio/MIDI hashes, parameters/defaults, latency and saved-state size |
| AOT loaded banks | Sample tape/granular and Corpus analysis/playback using three synthetic WAVs and actual editors; both baseline and refactor must finish and produce finite nonzero output; this gate does not compare audio bits |
| Production build | CMD CLAP Release rebuild after extraction |
| JIT runtime suites | Standard and C++ frontend paths: DSP, native GFX, aliases, hidden controls, pins, restore, serialization, files/pools, tasks, MIDI including same-note handover/reset ordering, memory faults, failed Run, Default and direct editor shortcuts |
| JIT catalog | All 83 JSFX entry points, including imported Joep plugins and Protosynth, run through the actual processor; varied inputs/MIDI and callback sizes, GFX and state save; final rerun results are recorded separately |
| JIT loaded banks | Sample tape/granular and Corpus complete actual analysis and emit finite nonzero MIDI-triggered playback; generated synthetic WAVs only |
| State extent/lifecycle | More than 33,000 variables with a real deferred private snapshot; actual catalog Protosynth; source-relative nested imports/images and saved/failed-draft origin checks |
| Host settings | Off/2x/4x/8x, varying blocks, rate reprepare, numeric/string restoration, latency, MIDI and Faust sample-rate tables; Auto/Never/Explicit sleep behavior and wake/offline rules |
| Native semantic oracle | 23 checks against the independent WDL EEL implementation |
| Task contracts | 11 checks in ordinary and native layouts with the actual linked task runtime |
| Packaged interfaces | Public CLAP/VST3 tests load matching packaged binaries: Run, failed Run, Default/state, dynamic controls, aliases, port changes, host restart/rescan; CLAP also checks zero-input processing |
| Packaged compiler | 21 positive/negative cases for each frontend with empty PATH and isolated bundled Python imports |

Repository logs and result matrices are under `build/runtime-unification`. The delivered qualification report names the final runs and their outcomes. Compilation times collected during parallel validation are elapsed test times, not controlled compiler or DSP benchmarks.

## Reproduce

Use the configured local toolchain and the Python installation used for packaging:

```text
cmake --build build/jit-editor --config Release --parallel 4 --target jit_editor_bank_check JITEditor_CLAP JITEditor_VST3
build/jit-editor/jit_editor_bank_check.exe
build/jit-editor/jit_editor_bank_check.exe --cpp-frontend
build/jit-editor/jit_editor_bank_check.exe --oversampling
build/jit-editor/jit_editor_bank_check.exe --source-resources
build/jit-editor/jit_editor_bank_check.exe --large-state
python tests/runtime/jit_catalog.py --exe build/jit-editor/jit_editor_bank_check.exe
python tests/runtime/compare_frozen_aot.py
python tests/runtime/compare_frozen_host.py
python tests/runtime/compare_frozen_host.py --banks --plugins Sample Corpus
```

Loaded-bank mode accepts `--loaded-bank` followed by the actual Sample or Corpus source path. Public wrapper checks accept a staged plugin with its matching compiler payload. `CompilerCheck.py` accepts the staged `JITEditor.runtime` path and optionally `--frontend cpp-frontend`. `jit_editor_bank_check` is the same test source and production static library as `jit_editor_check`, allowing qualification without replacing an executable being used by another check.

## Limits and practical qualification boundary

The catalog is a default-state processing/UI/save smoke gate. It is not an original-versus-JIT audio null test for 83 complete plugins, nor a guarantee for every preset, sample rate, file format, host or long-running stress scenario. AOT exact comparison coverage is the listed fixtures/processors; the loaded-bank comparison adds functional coverage without a null claim. Source/expanded-source is bounded at 4 MiB, combined controls at 256, and existing heap/task count/capture limits remain. Test harnesses have deadlines; the editor no longer imposes the old 120-second compilation cutoff.

The same AOT limitations carry into JIT: parameters apply at callback boundaries; `slider_next_chg` reports the current value and no remaining sub-block change point. Legacy `file_read`/`file_write` semantics are not a new generic writable-file API. Disk calls belong in init/GFX; background sample pools provide audio reads. The production slider acknowledgment policy can retain an exact value a host float cannot represent; this does not make host automation double-precision.

Host-delivered Ctrl shortcuts still require a live REAPER check. Direct key-handler tests and the focused-editor Windows hook do not certify REAPER's accelerator path. Public wrapper tests do not certify every host's routing/UI behavior. Faust bargraph meter widgets, foreign functions/variables/constants and soundfile integration remain unavailable. A linter/debugger, native execution watchdog, crossfade/state-history migration and macOS support remain future work. Native DSP is not sandboxed; an infinite guest loop or an invalid native operation can still hang/crash the host.

The loaded LLVM DLL/shared library is retained for the host process lifetime. Close the DAW before replacing `JITEditor.runtime`. Compiler process cleanup does not unlock a DLL loaded by the DAW; this build has no versioned runtime cache. Distribution archives retain dependency notices; public redistribution needs the existing licensing review. Python-to-C++ lowering/emission migration remains separate work described in [CPP-MIGRATION.md](CPP-MIGRATION.md).
