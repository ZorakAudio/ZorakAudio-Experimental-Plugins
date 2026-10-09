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

Windows catalog CI puts build intermediates under `RUNNER_TEMP/za-catalog/windows`
to keep MSVC's nested VST3 output paths below its traditional 260-character limit.
Local builds retain `build/<platform>` by default. For a deep checkout, pass
`--build-root` with a short directory; relative paths resolve against the repository.
`--clean` and `--clean-only` clean only the selected platform under that base.
The builder checks Windows VST3 paths before compilation and reports a shorter-root
remedy. Plugin filenames, IDs and packaged collection layouts are unaffected.

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

Compiler/runtime performance checks run in catalog CI on the first shard:

```sh
python tests/tasks/test_tasks.py
python tests/runtime/test_performance_contracts.py
python tests/tasks/test_corpus_matrix.py
```

These checks cover worker capability inference, notification wakeups, idle
parking and shutdown, cached host-variable bindings, and idle IPC with direct
messages, cross-process delivery, contention retries and peer liveness.
They also check shared-memory sizing, publication locks, moved ownership and
production `gmem` namespace isolation.
macOS uses names within Darwin's 31-byte limit and a private regular-file setup
lock because Darwin rejects `flock` on POSIX shared-memory descriptors. That
lock runs only during attachment/initialization. Linux additionally exercises
the Darwin name/locking strategy under a syscall contract shim; this is not a
substitute for the native macOS CI check. Windows/Linux retain their existing
shared-memory names and normal locking strategy.

The oversized-attachment fixture compares a request against the existing
attachment's actual backing capacity, rather than assuming a 4 KB allocation.
Darwin arm64 can round shared-memory backing to 16 KB, so a 16 KB request against
an originally requested 4 KB segment is not necessarily oversized. The Linux
Darwin shim models that 16 KB rounding, and the test still requires requests
beyond the backing capacity to fail without resizing or erasing existing data.
CI prints the checkout SHA and individual test names to distinguish failures
from older runs. This fixture correction does not change runtime DSP.

Windows/macOS AOT emission carries the selected optimization level through to
Clang's machine-code backend. Its IR optimizer is disabled at that final step
because the tuned LLVM pipeline has already optimized the module. Linux's
native object emission continues to use the configured LLVM target machine.
This preserves the existing floating-point rules and bounded inliner.

On Windows with cached release libraries,
`python tests/runtime/profile_host_changes.py --plugin joep_amaranth` compares the
working runtime against HEAD, using identical optimized input IR and a full
production callback. It also checks audio/MIDI bits, parameters, latency and
saved-state size across two rates and four oversampling settings. This is a
local before-commit probe; its numbers are specific to that plugin and setup.

The small `m` on each submodule in Git status is expected after patching. Git stores
the patches, applier and CMake integration in the main repository; it does not store
the dirty submodule files themselves. After updating either pinned dependency,
review/regenerate the corresponding patch and run these checks before publishing.

## CI coverage

**Build & Release** builds the catalog. **JIT Editor Build & Release** builds the
standalone Editor independently. Both run for branch pushes and pull requests,
with these platform jobs:

| Job | Formats | Architectures |
| --- | --- | --- |
| Windows catalog | VST3, CLAP | x86-64 |
| macOS catalog | VST3, CLAP | universal2: arm64 and x86-64 |
| Linux catalog | VST3, CLAP | x86-64, Ubuntu 24.04 |
| Windows JIT Editor | VST3, CLAP with bundled compiler | x86-64 |
| Linux JIT Editor | VST3, CLAP with bundled compiler | x86-64, Ubuntu 24.04 |

Branch/PR catalog builds use `--smoke`: AntiSalienceMX, ERBTilt, 3DPanner, ModTilt and
Saike BandJoiner. This covers ordinary JSFX graphics, mixed JSFX/Faust, native
Legacy graphics, pure Faust and the automatic Joep Legacy path. Tags matching
`v*` or `R*` build the entire catalog. **Run workflow** defaults to the entire
catalog; uncheck **full_catalog** for the representative set. The separate Editor
workflow builds and qualifies Windows and Linux on branch/PR pushes, its own
`jit-v*` tags, or its own manual **Run workflow**. Catalog tags do not trigger an
Editor build, and Editor tags do not trigger catalog builds. Full catalog builds split the
catalog into four disjoint groups per platform. Tagged releases merge those
groups only after checking that every group and all distributable catalog plugins are present,
without changing plugin bytes, bundle modes, symlinks or signing metadata.
The merger also requires a nonempty CLAP binary and VST3 binary at each plugin's
platform-specific bundle path; a manifest entry alone cannot satisfy this gate.

The release policy checksum normalizes LF/CRLF line endings, so Windows, macOS
and Linux agree even when Git uses different checkout line endings. The merger
also accepts old raw-byte checksums for this exact policy's LF and CRLF forms;
it still rejects changed policies, missing shards and missing/empty binaries.

If final assembly fails after all catalog jobs succeed, reuse the uploaded
`catalog-shard-*` artifacts from that run. Download all twelve and unwrap each
artifact ZIP once into its own folder under `build/catalog-shards`. Keep the
inner plugin ZIPs intact, especially macOS bundles: assembly preserves their
executable modes, symlinks and signed contents without extracting them on Windows.
With the same release sources/policy and the corrected merger, run:

```sh
python scripts/merge_catalog_archives.py --input build/catalog-shards --output dist/catalog --tag R2026.10.09-02 --collections
```

Use the original run's tag in place of the example. This produces the three
all-platform collection ZIPs locally without invoking a compiler. Upload those
ZIPs to the matching GitHub release after reviewing them; no full rebuild is needed.

For a packaging-only run in GitHub Actions, push the corrected packager and
`.github/workflows/recover-catalog.yml` to the default branch. Open **Actions →
Recover catalog packages → Run workflow**, select that branch, and enter the
original release run ID (for example `37991931940`). This workflow downloads only
that run's twelve saved build artifacts and uses its exact original commit for
catalog definitions and collection policy. The current corrected packager
produces an artifact named `catalog-collections-recovered-<run ID>` containing
the three ZIPs. It does not compile plugins or publish a release. Expired or
incomplete artifacts stop recovery before packaging.

The local equivalent can pass `--catalog-root` with a separate checkout of the
original build commit, while running the corrected merger from the current checkout.

Every catalog and JIT job uses the shared setup action to apply and verify the
JUCE and CLAP wrapper patches before installing the native toolchain or compiling.
A patch conflict fails the job immediately; each job also verifies the patches
after its build, before uploading plugin archives.

Catalog jobs also run `python scripts/check_plugin_readmes.py`. It checks every
buildable leaf, rejects placeholder or invalid help, checks local Markdown links,
and compiles the generated headers to verify exact UTF-8 bytes. Both native
formats share this embedded README path. Updating documentation requires a
rebuild; installed binaries do not read loose READMEs at runtime.

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

Windows/Linux JIT CI builds the production preprocessor, optional native frontend and
both plugin formats. It packages Python/Faust/LLVM, checks both compiler frontends,
executes shared-runtime and controls/Unicode/save/example checks, and loads the
actual packaged CLAP and VST3 through their public interfaces. Qualification
extracts the archives into another directory containing spaces and Unicode,
removes developer compiler/Python environment settings and empties PATH. The test
executable uses a copy of this extracted build's matching payload. It is never
paired with an older installed payload. Linux additionally checks compiler child
process cleanup and uses a virtual display for the real JUCE interface checks.
Both platforms also exercise nested imports/image resources, oversampling/rate
changes and loaded Sample/Corpus banks made from synthetic WAV fixtures.
The archives retain compiler/dependency notices and examples. Linux uses tar.gz
archives to preserve executable permissions and relative native-library paths.

Catalog archives and usable Windows/Linux JIT Editor archives are separate Actions
artifacts. Diagnostic artifacts include compiler versions and JIT logs/screenshots.
Full manual/tag builds also upload `catalog-collections`: **Essentials**, **All**
(non-Joep) and **JoepVanlier**, each containing Windows, macOS and Linux binaries.
Essentials reuses compiled All payloads; it does not rebuild them. The single
release policy excludes IPCProbeA/B and retires SaliencePush in favor of the new
AntiSalienceMX identity. See [release selection and installation](Release-Collections.md).
Raw platform shards are separate `catalog-shard-*` artifacts; smoke builds cannot
produce full collections.

## Independent release targets

| Target | Workflow | Tag | Published assets |
| --- | --- | --- | --- |
| Catalog | `.github/workflows/release.yml` | `R*` or `v*` | Three all-platform collection ZIPs: Essentials, All and JoepVanlier. |
| JIT Editor | `.github/workflows/jit-editor-release.yml` | `jit-v*` | Windows CLAP/VST3 ZIPs and Linux CLAP/VST3 tar.gz archives. |

Each publishes a **different GitHub Release** after its own build/qualification
jobs pass. A failing or slow Editor build cannot block catalog publication.
Existing catalog tags keep their policy: `R*` regular releases, `v*` prereleases.
Editor releases are prereleases and explicitly do not become the repository's
Latest release. macOS Editor assets cannot be added until its platform port and
qualification exist; the catalog's universal2 support is separate.

For example, push a new `R1` tag to release the catalog, or a new `jit-v0.1.0` tag
to release the Editor. Use distinct new tags for subsequent versions. Branch,
PR and manual builds upload Actions artifacts without publishing a GitHub
Release. A manual run of one workflow does not launch the other. Workflows
already running use their original configuration; this split applies after it
has been pushed.

New release bodies come from [catalog notes](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/docs/releases/Catalog.md) and
[Editor notes](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/docs/releases/JIT-Editor.md). Review/edit those Markdown files before
tagging. Rerunning publication keeps an existing release's body and replaces its
named assets. Missing or empty expected archives stop publication before release
creation. No additional repository secrets are needed for this unsigned CI.

## Qualification limits

The standalone JIT Editor supports Windows and Linux; macOS still requires its
platform port and its CMake project explicitly rejects macOS. Windows and Linux
use the same production compiler and DSP/GFX runtime. The platform adapter covers
library loading, locating the private compiler payload and owning compiler child
processes. Linux uses a dedicated process supervisor and relative ELF library
paths; it does not require a system Python/Faust/LLVM installation to run.

Local qualification uses Windows and Ubuntu 24.04 under WSL2. WSL builds and
public CLAP/VST3 checks provide Linux evidence; hosted CI remains an independent
fresh-checkout gate. macOS binaries still require successful hosted verification.
Dependency checks cover LF/CRLF fixtures, concurrency/conflicts, repeat builds and
CMake repair of a reverted wrapper. CI does not certify every DAW, live host keyboard handling, runtime
sandboxing or public redistribution terms. Full catalog compilation is a build
gate, not an audio null-test matrix.

Upstream references: [recursive checkout](https://github.com/actions/checkout),
[Faust 2.81.2 release](https://github.com/grame-cncm/faust/releases/tag/2.81.2),
[llvmlite binary installation](https://llvmlite.readthedocs.io/en/v0.46.0/admin-guide/install.html).
