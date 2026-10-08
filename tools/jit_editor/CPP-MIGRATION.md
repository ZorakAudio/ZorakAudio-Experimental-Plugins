# Native C++ compiler migration assessment

This is a migration design, not a completed C++ compiler replacement. Phase 1 has a separate [native frontend library/helper](../native_compiler/README.md): source resolution, WDL preprocessing, tokens and unlowered syntax trees, with differential checks against the production compiler. The editor bundles that helper for developer checks and older saved projects; its user-facing Run button uses the standard compiler without a frontend checkbox. The native frontend resolves/parses JSFX and passes its syntax tree into the existing production Python lowering and LLVM emitter. This is not a Python-free compiler. Audio, native GFX, task workers, LLVM ORC execution, host integration, file decoding, and sample-pool runtime are C++ already.

## Recommendation

A full C++ compiler is feasible and coherent. Implement it as a reusable library plus a small standalone compiler helper, initially selected by an experimental backend flag. Keep Python as the differential reference until the native backend passes explicit equivalence gates. Do not remove the existing compiler merely because a handful of effects compile with the replacement.

Retain the process boundary during the first native implementation: malformed source or a compiler crash should not take down REAPER. The editor can use the C++ library through that helper, and the same library can later be linked into JUCE when the debugging and crash-recovery tradeoffs are acceptable. DSP execution remains in the host. Compiling C++ instead of Python does not sandbox DSP or guarantee better audio performance.

## Actual scope

The current Python core is approximately 7,450 lines; its LLVM emitter accounts for approximately 2,740 lines. The lexer is approximately 191 lines, the parser approximately 510. Shared source resolution is 409 lines, preprocessing integration 133, Faust integration 372, and deferred-task integration 161. This is approximately 8,500 lines of semantic implementation to assess, not just a parser rewrite. Counts include comments and supporting code and are a sizing aid rather than an effort estimate.

| Component | Native replacement | Equivalence obligation |
| --- | --- | --- |
| Source/import resolver | C++ filesystem resolver with provenance and package roots | Library postorder, diamond deduplication, cycles, missing dependencies, preprocessor/config defaults, section fallback and override order |
| Preprocessor | Reuse the existing WDL/EEL2 C++ helper | Identical expanded source, include roots, config values, diagnostics |
| Lexer and parser | C++ token/span types and AST | Case folding, literals, comments, operators, precedence, reference outputs, loop syntax and function declarations |
| Semantic analysis and lowering | Explicit symbol/receiver maps and lowering passes | Named sliders, persistent locals, dotted receivers, function redefinitions, argument evaluation, section rules |
| Optimization plans | Port existing analyses before changing optimization policy | Mutation and alias barriers, loop and section invariants, observable store behavior |
| LLVM emitter | LLVM C++ API, current native target and ORC-compatible layout | Every state offset and capacity, calling convention, float precision, stores, memory bounds, external symbol signatures |
| Task lowering | C++ AST capture/lifetime analysis and callback emission | Private state, captures, reductions, dependencies, cancellation, arenas and runtime safety contracts |
| Faust integration | Preserve current inference/planning; later choose CLI or libfaust | Legal Faust, diagnostics-based imports, UI metadata, tables, captured streams, fused versus block stages, export feedback |
| Metadata/header generation | Versioned compiler result schema and deterministic header generator | Pins, aliases, defaults, visibility, source origins, capabilities, state ABI and native imports |

Important EEL details include assignment filtering of subnormals/NaN/infinity, approximate comparisons, memory-index rounding, integer conversions and shifts, signed zero, evaluation order, function-local persistence and receiver specialization. Passing an ordinary arithmetic test does not establish these semantics.

Faust's backend and the JSFX backend still need a compatible LLVM version and target layout. The current integration also repairs a specific Windows Faust bitcode CRLF issue and initializes generated-table sampling rates for the bundled Faust version. Those behaviors belong in the migration inventory; a new native API does not automatically solve them.

## Proposed library boundary

`compile(request, cancellation) -> result` owns the entire frontend and returns native-target LLVM bitcode plus versioned metadata, dependency/provenance information, and structured diagnostics. The request includes source text, source path/package roots, mode, preprocessing/config values, sample rate, target settings, and resource limits. Diagnostics carry filename, span, phase, severity and message. Compilation never runs from the audio callback. Loading/publication remains separate so a failed compile preserves the old program.

The backend selector must be explicit (`python-reference` or `cpp-experimental`). For supported syntax, native failure should be visible rather than silently changing semantics or substituting zero-return runtime functions. A developer comparison mode can invoke both backends, compare results, and execute them through the same runtime.

## Equivalence gates

1. Generate a checked reference corpus from the current compiler before replacing stages. Include every repository entry point, imports, generated/preprocessed sources, positive fixtures, and negative diagnostics.
2. Compare tokens and normalized AST/lowering results. Ignore ephemeral IDs and temporary paths, but compare symbol binding, persistent storage, source spans and section order.
3. Compare metadata exactly: input/output pins, slider aliases/ranges/defaults/hidden flags, GFX sizes/resources, runtime imports, task callbacks, Faust stages/bindings/tables and state offsets.
4. Execute both backends with the same runtime, deterministic input and initial state. For the same algorithm and target, compare output sample bits and observable scalar/RAM/string state. Exercise silence, impulses, random signals, parameter changes and blocks of 1, 17, 256, 1024 and 4096 samples. Investigate discrepancies rather than relaxing thresholds globally.
5. For concurrent tasks, compare documented results, ordering/dependency invariants, bounded resources and cancellation outcomes. Wall-clock completion order is not a bitwise equivalence requirement. Use deterministic reductions and stress program retirement while workers are active.
6. Compare GFX draw commands and resulting images with the same fonts/assets/input traces; test mouse, modifiers, keys, menus, drops and slider visibility/automation. Actual host-delivered shortcuts require an integration test, not just direct calls to a key handler.
7. Compare public CLAP/VST3 behavior: dynamic schemas, mono/multichannel/generator ports, restart/rescan, state restore, failed Run retention, destruction and multiple instances. Add a real REAPER qualification pass before making the replacement the default.
8. Measure source resolution, parsing/lowering, IR construction, LLVM optimization, machine-code linking, peak memory and package size separately. Include Amaranth and the larger Sample/Corpus programs. Do not infer audio speedups from compiler language.

Byte-for-byte LLVM IR is not a useful gate: names, ordering and optimizer versions can differ without changing behavior. Conversely, a matching audible render alone can miss metadata, UI, MIDI and state regressions.

## Packaging and performance

The current installed compiler payload is approximately 174 MB. The single `llvmlite.dll` is 106,601,984 bytes (approximately 101.7 MiB); it includes LLVM and its binding interface. The Python subtree is approximately 133 MB and already includes that DLL. Therefore, replacing Python does not save the whole Python subtree: most of it is LLVM that a native compiler still needs. The current Faust payload is approximately 40 MB.

A selective native LLVM build could reduce the LLVM footprint by including only required targets, ORC/JIT and optimization components. That saving is unmeasured and must be established with a real build. LLVM static linkage into each plugin can also duplicate code and enlarge the plugin; a shared compiler/runtime is a separate packaging choice.

C++ is likely to reduce Python object overhead, startup and frontend IR construction costs, but LLVM optimization and Faust compilation are already native and remain part of compile time. The same generated DSP will not become faster merely because its frontend is rewritten. The editor now reports phase timings. In a local Amaranth run, the initial helper spent 33.34 seconds total: source resolution 0.13, frontend 0.93, IR emission/Faust 3.17, IR preparation 3.02, LLVM optimization 8.41, and import validation 17.67 seconds. Scanning symbols once alone did not fix the delay: per-function IR printing dominated validation. Printing the module once reduced validation to 0.47 seconds and total measured compilation to 16.46 seconds, with byte-identical emitted IR for this source. These are individual local runs, not a statistical benchmark or a host render speedup.

After that correction, approximately 4.1 seconds remain in resolution/frontend/IR emission, 3.0 in IR preparation, and 8.8 in LLVM optimization for this example. Porting Python may reduce frontend/emission overhead and let preparation avoid text round trips, but the optimizer cost remains. Use phase and peak-memory measurements on several programs before promising a native compiler speedup.

## Execution order

Phase 1 is implemented in `tools/native_compiler`: a versioned frontend protocol, native source resolver/lexer/parser, production Python reference adapter, generated/negative fixtures, catalog discovery and frozen reference replay with dependency provenance. The initial 230-case differential run matched all 83 repository entry points; the final executable also passed the expanded 241-case frozen corpus (83 entry points plus 158 fixtures). This proves frontend equivalence for the checked corpus; it does not establish lowering, LLVM, DSP or host equivalence. The frontend remains selectable in developer tools and feeds existing production lowering and symbol analysis. Porting those passes is the next migration target; native LLVM emission remains later work.

First freeze the request/result ABI and differential fixtures. Port source resolution plus tokens/AST. Port lowering and symbol analysis, then LLVM emission and runtime import validation. Port task and mixed-Faust planning. Run the complete equivalence/host suite under an explicit native-backend flag. Only after that should packaging drop Python and the native backend become the default.

This is a substantial compiler engineering project with multiple reviewable milestones. An automatic Python-to-C++ translation or an alternate minimal EEL interpreter would not satisfy the requirement to preserve the existing compiler's behavior. The current editor fixes should not depend on completing this conversion.
