# Native compiler migration: phase 1

Implemented: a reusable C++ frontend library and a standalone Windows x64 helper. The native executable resolves JSFX imports, applies compile-time config defaults through the existing WDL/EEL2 preprocessor, extracts sections, lexes EEL and builds normalized unlowered ASTs. It does not start Python, invoke the Python compiler, or depend on a Python installation. JUCE core supplies JSON and strings; vendored WDL supplies preprocessing.

The JIT Editor bundles this helper for developer qualification and older saved projects. Its adapter passes native syntax trees into the existing production lowering and LLVM emitter; Python remains bundled. The user-facing Run button uses the standard compiler, with no frontend checkbox. This helper alone does not produce LLVM output or full runtime metadata and cannot replace `compiler_worker.py`.

## Build and run

Use CMake/Ninja/Clang in a Windows development environment with the repository's JUCE submodule present:

```text
cmake -S tools/native_compiler -B build/native-compiler -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-compiler --parallel 4
ctest --test-dir build/native-compiler --output-on-failure
```

Explicit compiler/Ninja paths may be needed when they are absent from PATH. The targets are `jsfx_native_frontend`, `jsfx_frontend` and `jsfx_frontend_check`. Windows helper targets reserve 8 MiB stack. Native library users must arrange the same compiler-thread stack budget. No build step downloads dependencies, installs plugins or rewrites production compiler files.

```text
jsfx_frontend absolute-request.json absolute-result.json
python tools/native_compiler/Check.py build/native-compiler/jsfx_frontend.exe --all-plugins --freeze-reference build/native-compiler/reference-v1.zip --output build/native-compiler/equivalence.json
python tools/native_compiler/Replay.py build/native-compiler/jsfx_frontend.exe build/native-compiler/reference-v1.zip --output build/native-compiler/replay.json
```

The differential check requires the existing development Python/llvmlite environment for its reference compiler. Replay uses only standard Python libraries as a test runner; it does not import the compiler or LLVM. Every native helper invocation runs with an empty PATH. The frozen archive contains relocation-normalized requests, expected results, fixture dependency files, compiler hashes and hashes of repository source dependencies. Replay fails when reference provenance changes.

See [the protocol](PROTOCOL.md) for encoding, diagnostics, resource limits, cancellation, import semantics and equivalence fields. The direct native contract test covers deterministic output, cancellation, protocol rejection, source size, recursive parser/left-associated AST limits and concurrent parsing.

## Qualification boundary

The initial full differential run passed all 83 catalog entry points plus 147 fixed/generated fixtures, including Sample, Corpus, Amaranth and Protosynth. The final executable passed all **241 frozen comparisons: 83 catalog sources and 158 fixtures**. Additional numeric configuration fixtures cover wide hexadecimal defaults, invalid fractional hexadecimal defaults and non-finite preprocessing definitions; protocol fixtures reject malformed flags/paths and unknown settings. Windows CRLF behavior, diamond imports, fallback/override order, repeated init sections, mixed section order, hidden metadata wildcard paths, string escapes/NULs/UTF-8, EEL precedence, function qualifiers, loops, task syntax and negative diagnostics are explicitly covered.

Matching tokens and unlowered ASTs does not establish lowered symbol bindings, variable/state layout, complete compile success, DSP performance or audio correctness. The subsequent shared-runtime editor refactor removed the former 32768-scalar state limit separately; parsing alone did not establish that capability. Native Faust parsing/compilation is not part of this phase; bodies remain opaque and intact.

Observed single-run frontend timings include normalized token/AST construction on both sides, not LLVM emission or DSP rendering. The Python reference timing excludes process startup; native process timing includes startup and result JSON serialization. Amaranth: 1.75 s reference / 1.20 s native process; Corpus: 10.15 / 2.38; Sample: 5.13 / 5.39. These timings came from a correctness run, not a controlled statistical benchmark. They demonstrate why the language change alone should not be advertised as a universal speedup. The approximately 2 MB helper contains no LLVM emitter/runtime or Faust compiler; it is not an estimate of the finished compiler distribution size.

## Next phase

Port function extraction/redefinition handling, receiver specialization, persistent locals and symbol/slider binding from the production pipeline. Extend the reference protocol with normalized **lowered** ASTs and symbol tables, preserving storage and evaluation order. Gate all 83 sources plus dedicated binding/alias/evaluation-order fixtures before beginning LLVM emission. Keep the JIT Editor on Python until the complete metadata/bitwise execution/host gates in [the migration design](../jit_editor/CPP-MIGRATION.md) pass.
