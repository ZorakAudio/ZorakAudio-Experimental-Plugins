# JSFX compatibility and source imports

This reference replaces the early single-Abyss showcase and pre-migration concurrency audit. The maintained source resolver is `scripts/jsfx_source.py`; broad native coverage is documented in [JoepVanlier compatibility](JoepVanlier-Native-Compatibility.md) and the [catalog checkpoint](Plugin-Catalog-Regression.md).

## One source for DSP and graphics

The build expands imports into `JSFXExpanded.jsfx`, embeds that same text in `JSFXSource.h`, and records dependency hashes/section ownership in `JSFXImports.json`. It searches the importing directory, package root and dependency subdirectories within explicitly allowed roots. Missing, ambiguous, escaping or cyclic imports fail; unique case-insensitive fallback is supported. Comments and string literals do not become import directives. Libraries must have sections rather than an unsupported sectionless body.

Imported `@init` sections execute once in dependency postorder, then the main initialization. For other ordinary sections a main definition wins even when empty; otherwise the first postorder imported definition supplies the fallback. Imported headers do not inject sliders/options into the main effect. Mixed scripts support repeated timeline sections as specified in [the FAUST contract](JSFX-Faust-Sections.md).

Sources containing `<? ... ?>` preprocessing blocks are expanded through the Cockos/WDL preprocessor before import/metadata parsing. Root `config:` declarations supply compile-time defaults; they are not runtime automation controls. This is build preprocessing, not an embedded playback-time compiler. Build dependencies and supported constructs still constrain what can be compiled.

## Native semantics and opt-ins

Nested instance receivers, persistent locals, namespaced calls, overloaded/redefined functions, packed character constants, multiline strings and EEL precedence have dedicated compatibility work. This is broader than the initial Abyss milestone, but not a promise of every EEL lvalue or REAPER host service.

`jsfxCompatibility.eel2Stores` injects `options:za_eel2_stores=1` into generated input without editing vendored sources. Checked assignments clear NaN, infinity and subnormal results to positive zero. This conservative checked-store option is not a proof of all WDL optimizer/numerical behavior. Preserve the option when a qualified package depends on it.

EEL graphics memory configuration can declare explicit shared ranges so graphics-owned animation RAM is not overwritten by DSP snapshots. Native publication mode instead requires an ownership contract; Native Legacy uses live shared atomic cells. Read [publication graphics](Native-GFX-Publication.md) and [Legacy graphics](Native-GFX-Legacy.md) before migrating heap-heavy canvases. Atomic per-cell access prevents undefined data races but does not make a whole table/frame transactional.

Custom `@serialize` bodies and `gfx_idle/gfx_idle_only` are not supplied by native Legacy. Host parameter/path persistence does not replace a script's own sample/table serialization. REAPER scheduling, random sequences, host metadata and every file/string service are not universally equivalent.

## Validation and vendored code

`--correctness-check` runs an eligible ordinary native script against the vendored WDL/EEL shadow engine. Task-enabled, FAUST-mixed and Native Legacy builds reject it because its assumptions cannot evaluate those extensions/concurrent state. Use their dedicated runtime and complete-processor fixtures instead.

`python scripts/verify_jsfx_snapshot.py <plugin-directory>` checks a vendored package against its recorded local hashes. Optional upstream verification requires network access and does not update the sources. Preserve upstream notices, `upstream.json` and package licenses. A local integrity hash is not an upstream identity proof.

Current coverage reports are versioned checkpoints: their source/runtime fingerprints and fixture constraints matter. Zero error in a short audio/MIDI sequence does not prove all presets, custom serialization, platform behavior or REAPER equivalence. Production wrapper builds and actual DAW testing are separate from processor/editor fixtures.
