# Native frontend phase 1 qualification — 2026-10-07

Phase 1 is implemented locally in `D:\Dev\ZorakAudio-Experimental-Plugins\tools\native_compiler`. The JIT Editor and AOT compiler still use the production Python backend. This milestone adds a separate native C++ library and helper; it does not change any plugin's audio output, runtime, controls or installed binary.

## Results

| Check | Result | What it establishes |
| --- | --- | --- |
| Production Python versus C++ | 230/230 initially | Exact resolved text/provenance, tokens/spans and unlowered AST fields for 83 catalog sources plus 147 fixtures |
| Expanded fixture differential checks | 158/158 | Additional malformed-protocol and wide/invalid/non-finite config cases |
| Final executable versus frozen reference | **241/241** | All 83 catalog sources and all 158 fixtures, with production/dependency provenance verified |
| Native library contract | **8/8** | Determinism, cancellation, protocol rejection, source limit, parser nesting, left-associated AST depth, decimal locale independence, concurrent parsing |
| Imported-file diagnostic | Passed | An imported sectionless file is named in the structured error, rather than only the root source |
| Native helper environment | Empty PATH throughout | Native resolution/preprocessing/parsing needs no Python, Faust CLI or developer tool lookup |
| CMake build / CTest | Passed | Clang 21 Release Windows x64 build and registered native contract test |
| Whitespace validation | Passed | Repository diff check with the existing CRLF convention respected |

The exact final helper SHA-256 is `79cfbee04af1490f92041fe36064bee596d1526f628edc21a5ae29fdb11dbf94`. Its size is 2,043,392 bytes. The frozen reference carries hashes of the production Python frontend/resolver/tasks implementation, preprocessing bridge, vendored WDL sources and 233 repository source dependencies. Test data and measurements are in `build/native-compiler`; the frozen archive is `reference-qualified.zip` and the final replay report is `replay-qualified.json`.

The full catalog includes Sample, SampleFaust, Corpus, Amaranth, Hyperreal variants and Protosynth. All were accepted and matched at this frontend stage. Protosynth's success does not lift the editor's separate 32768-scalar limit; there is no scalar layout/LLVM emission here yet.

## Implementation and fixes found by the gate

The native resolver preserves package-relative import search, postorder initialization, diamond deduplication, cycles/missing/ambiguous import errors, empty root overrides, imported fallback sections, repeated initialization bodies, mixed section order and coarse section ownership. WDL preprocessing runs directly in C++; root config defaults reach imported units. The differential check caught Windows text-mode CRLF expansion and the native path resolver's handling of not-yet-saved source paths. Both were corrected rather than normalizing away the differences.

The parser preserves the production compiler's EEL precedence, assignment target rules, approximate/exact operator spellings, line continuations, sequence blocks, both while forms, loop bodies, named/dotted functions, qualifiers, string assignment rewriting and all five deferred forms. Unicode string bytes, packed character literals, hexadecimal/mask literals, overflow/underflow and exact binary64 AST values are represented explicitly. Native numeric conversion uses a fixed C decimal locale so a host's comma locale cannot reinterpret constants.

The helper rejects unknown protocol fields, wrong option types and oversized/deep source with diagnostics. AST depth is bounded independently of parser recursion: a very long left-associated expression cannot evade the recursion guard and overflow the JSON writer's stack. Cancellation is cooperative; WDL preprocessing remains synchronous and needs an external helper deadline for pathological programs. It is not a code sandbox.

## Performance observations

These are single observations from a correctness run, not controlled benchmarks. Python timing includes resolution plus token/normalized-AST construction without Python startup. Native process timing also includes process startup and result serialization. Neither includes symbol lowering, LLVM, Faust compilation, linking or audio processing.

| Source | Python reference frontend | Native helper process |
| --- | ---: | ---: |
| Amaranth | 1.75 s | 1.20 s |
| Protosynth | 5.62 s | 4.36 s |
| Corpus | 10.15 s | 2.38 s |
| Sample | 5.13 s | 5.39 s |
| SampleFaust | 5.30 s | 5.47 s |

The results are mixed. They do not justify a universal frontend speedup claim, a DSP speedup claim or an estimate of the complete compiler size. The approximately 2 MB helper omits LLVM and Faust, which will dominate much of the eventual bundled payload.

## Remaining scope

Faust source is preserved as opaque section text. There is no native Faust parser/LLVM integration in this milestone. Builtin/service validation, named-slider binding, pin/control metadata, receiver specialization, persistent local storage, lowered symbols/AST, optimization and LLVM emission remain on the Python side. Exact imported-file source maps also remain future work; EEL spans refer to the expanded stream, with coarse section ownership available.

The next phase is symbol analysis and function lowering. It should expose normalized lowered ASTs and stable symbol/storage maps through an explicit protocol extension, compare all 83 sources plus binding/alias/redefinition/evaluation-order fixtures, and keep the editor on Python. Switching the editor comes only after LLVM/state ABI, bitwise audio/state, tasks/GFX and host equivalence gates pass.
