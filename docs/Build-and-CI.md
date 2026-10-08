# Building and publishing

## Dependency patches

Keep JUCE and clap-juce-extensions at the commits recorded by this repository's
submodules. Publish the wrapper patches with this repository; separate dependency
forks are not required. Do not commit a new submodule pointer solely to capture
uncommitted wrapper edits.

After a fresh checkout:

```sh
git submodule update --init --recursive
python scripts/build.py --config Release --tag dev --out dist
```

The normal builder, direct AOT CMake configuration and JIT CMake configuration
invoke `tools/jit_editor/apply_wrapper_patches.py`. CMake also checks at build
time, before compiling the wrappers. A reused build directory therefore repairs
a wrapper reverted since configuration. Already-applied patches are left alone.
Direct VST3-only AOT configuration only requires the JUCE patch/dependency.

Both dependencies are preflighted before either patch is applied. Unrelated edits
are preserved; conflicting edits stop the build with a diagnostic. A cross-platform
file lock serializes simultaneous patch checks. Nothing resets, fetches or commits
the dependency repositories. An unexpected later apply failure rolls back only
patches applied by that invocation.

To verify or prepare explicitly:

```sh
python tools/jit_editor/apply_wrapper_patches.py --check
python tools/jit_editor/apply_wrapper_patches.py
python tests/build/test_wrapper_patches.py
```

The small `m` on each submodule in Git status is expected after patching. Git stores
the patches, applier and CMake integration in the main repository; it does not store
the dirty submodule files themselves. After updating either pinned dependency,
review/regenerate the corresponding patch and run these checks before publishing.

## CI coverage

The existing **Build & Release** workflow now also runs for branch pushes and pull
requests. It has these build jobs:

| Job | Formats | Architectures |
| --- | --- | --- |
| Windows catalog | VST3, CLAP | x86-64 |
| macOS catalog | VST3, CLAP | universal2: arm64 and x86-64 |
| Linux catalog | VST3, CLAP | x86-64, Ubuntu 24.04 |
| Windows JIT Editor | VST3, CLAP with bundled compiler | x86-64 |

Branch/PR catalog builds use `--smoke`: DDT, ERBTilt, HyperrealFast, ModTilt and
Saike BandJoiner. This covers ordinary JSFX graphics, mixed JSFX/Faust, native
Legacy graphics, pure Faust and the automatic Joep Legacy path. Tags matching
`v*` or `R*` build the entire catalog. **Run workflow** defaults to the entire
catalog; uncheck **full_catalog** for the representative set. Windows JIT builds
and qualification run in every case.

Every catalog and JIT job uses the shared setup action to apply and verify the
JUCE and CLAP wrapper patches before installing the native toolchain or compiling.
A patch conflict fails the job immediately; each job also verifies the patches
after its build, before uploading plugin archives.

The setup action initializes Python 3.11 and pins llvmlite 0.46.0. It verifies
actual C++ and LLVM Faust output, LLVM bitcode/layout parsing and native object
emission before compiling plugins. macOS preflight also emits both architecture
objects. Compiler versions and paths are recorded in `build/ci/environment.json`.

Faust is pinned to 2.81.2 with upstream release-asset SHA-256 checks. Windows uses
the official installer. Linux/macOS build the complete source release natively
with LLVM 18 and explicitly enabled C++/LLVM backends, avoiding differences in
rolling distro/Homebrew Faust packages. Only the AOT plugins are universal2; the
Faust build tool runs natively on the runner. Platform compiler/SDK and LLVM 18
patch versions still follow their runner/package manager and are recorded by logs.

macOS explicitly passes both CMake architecture flags and DSP object targets,
stages complete CLAP/VST3 bundles, signs them ad hoc and verifies both slices and
signatures before upload. These are development signatures; Developer ID signing
and notarization are not configured. Linux's supported runtime baseline is the
Ubuntu 24.04 build environment; compatibility with older glibc is not promised.

Windows JIT CI builds the production preprocessor, optional native frontend and
both plugin formats. It packages Python/Faust/LLVM, checks both compiler frontends,
executes shared-runtime and controls/Unicode/save/example checks, and loads the
actual packaged CLAP and VST3 through their public interfaces. The test executable
uses a copy of this build's matching payload. It is never paired with an older
installed payload. The archives retain compiler/dependency notices and examples.

Catalog archives and usable Windows JIT Editor archives are separate Actions
artifacts. Diagnostic artifacts include compiler versions and JIT logs/screenshots.
Tagged releases publish three catalog archives plus two Windows JIT archives only
after all build/qualification jobs pass. `R*` tags are regular releases; `v*` tags
retain the existing prerelease policy. Branch, PR and manual builds do not publish
GitHub releases. No additional repository secrets are needed for this unsigned CI.

## Qualification limits

This setup does **not** port the standalone JIT Editor to macOS/Linux. Its LLVM
loader, helper process containment, keyboard hook and compiler packaging currently
use Windows APIs/layouts; its CMake project continues to reject other platforms.
Cross-platform AOT catalog builds and a cross-platform editor are separate claims.

Local checks cover the Windows build/payload and LF/CRLF dependency fixtures,
concurrency/conflicts, repeat builds and CMake repair of a reverted wrapper. macOS
and Linux binaries require successful hosted builds before those outputs are
qualified. CI does not certify every DAW, live host keyboard handling, runtime
sandboxing or public redistribution terms. Full catalog compilation is a build
gate, not an audio null-test matrix.

Upstream references: [recursive checkout](https://github.com/actions/checkout),
[Faust 2.81.2 release](https://github.com/grame-cncm/faust/releases/tag/2.81.2),
[llvmlite binary installation](https://llvmlite.readthedocs.io/en/v0.46.0/admin-guide/install.html).
