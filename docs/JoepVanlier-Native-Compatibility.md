# JoepVanlier native compatibility

This work targets the 50 JoepVanlier packages configured in this repository.
Their entry points and imports run unchanged through native Legacy mode.
`scripts/build.py` now selects Legacy automatically for every configured
JoepVanlier plugin, and the direct `build_jsfx_aot()` helper enforces it for
sources under `plugins/JoepVanlier/`. No native flag is needed. The publication
prototype flag cannot override this policy. Other JSFX use their manifest selection or the default EEL graphics path;
pure Faust is unaffected. The matrix below is a fingerprinted checkpoint,
not a fresh sweep of every subsequent compiler or runtime change.
DSP and graphics use one fixed-capacity guest heap and the same numeric state;
native instances do not create an EEL graphics VM or synchronize an EEL heap.

The result matrix and screenshots accompanying this document distinguish native
generation, an independent WDL audio/MIDI comparison, and a production JUCE
editor lifecycle check. Passing those checks is a compatibility milestone, not
qualification of every preset, control, algorithm option, or host platform.

The [result matrix, source manifests, and screenshots](joep-native-qualification/README.md)
record the tested files.

## Compatibility changes

- The parser follows EEL operator precedence and boolean tolerance, handles
  packed character constants, multiline strings, overloaded and redefined
  functions, and accepts the declaration forms used by the catalog.
- Function lowering preserves receiver namespaces and persistent locals.
  Native loops use WDL's iteration budget and observable return values.
- Mutable strings and pattern captures use a shared native store. Native file
  services support sample decoding and write directly into the DSP heap.
- The JUCE renderer implements offscreen images, resource loading, blits,
  gradients, pixels, fonts, text metrics, keyboard input, cursors and file drops.
- Initial host values include a 120 BPM fallback and the active channel count.
  Exhausted `slider_next_chg` returns `-1`; the host currently supplies
  block-boundary automation rather than a sub-block event stream. This prevents
  Filther's automation iterator from repeatedly processing a nonexistent point.
- Hidden editor construction does not execute `@gfx` before layout. Release
  frames follow visible editor activity; this keeps simulations from freezing
  their persistent bounds to a temporary 1×1 canvas.
- Empty `@sample` sections preserve audio with separate output buffers as well
  as in-place buffers, without a per-sample guest-state loop.
- Compiler function streaming and short block labels reduce memory consumed by
  large native graphics modules. Large arithmetic leaf helpers share a body
  per section and receive pointers to their own receiver cells; persistent
  locals retain their original slots and shared accesses remain atomic.
  Small helpers and helpers with namespace members, nested calls, heap writes,
  strings, or host
  callbacks retain receiver specialization. Guest code remains compiled at
  LLVM `-O2`.

## What the checks cover

The numerical runner compiles the original `@init`, `@slider`, `@block`, and
`@sample` in the vendored portable WDL interpreter and compares them with the
full native object. It compares every configured output channel, including
BandSplitter’s ten and Drums’ 24 outputs. It uses 48 kHz float audio, variable
block sizes from 1 to 511 samples, a deterministic signal and tail, and short note/CC/pitch-bend MIDI
messages. Graphics does not execute in this numerical comparison. String and
language fixtures separately compare against WDL. Native CPU timings and the
reference loop timings include different harness work and are not a controlled
performance benchmark.

MIDI Arp's default bank is empty. Its fixture programs four identical steps in
both independent test heaps after initialization and requires generated MIDI
output. This changes test instance state, not the plugin source or object.
A separate Drums fixture enables 24-channel routing and uncoupled hats, then
triggers all twelve percussion notes and compares every output with WDL.
Those option writes resolve named slider aliases to their canonical cells.

The editor runner uses the production JUCE processor with concurrent audio,
transport, MIDI, two editor lifetimes, host parameter notifications, resizing,
and state save/restore. It drops a real stereo WAV onto native controls in
Amaranth, Bric-a-Brac, Partials, Protosynth, Yutani, and Drums, and requires
successful native decoding through bulk or scalar reads. Files and their mouse
hit-test position travel together in the input queue, so a pad does not depend
on a preceding mouse move. Some fixtures try additional visible positions until
the source accepts the drop. This does not qualify every drop zone or format. Screenshots wait for a
published canvas, not merely entry into guest graphics code. The runner requires
finite audio, audio-thread progress, and no guest heap fault.

Editors share one test host identity to reuse JUCE compilation. This verifies
production processor control flow, not 50 independently packaged VST3/CLAP
identities, DAW discovery, or a REAPER pixel comparison. The test host processor
is compiled without optimization to limit build memory; each guest object uses
`-O2`. Production host performance needs a separate optimized-host benchmark.

Two sweep runs, Nuker and Squashman, returned without their terminal result
records. Both completed normal reruns and five further diagnostic lifecycle
repetitions each. The initial cause remains undetermined; the result report
retains the incomplete and repeated records. This checkpoint does not establish
long-run shutdown reliability.

## Remaining gaps

Custom `@serialize` code is not executed. Host parameters and file paths have
state support, but guest-authored pattern banks, embedded samples, and other
custom tables are not yet restored through their JSFX serialization routines.
This matters for several packages and prevents a claim of complete feature
compatibility.

Non-image companion data currently resolves through filesystem candidates,
including the compiled source directory. Image assets are embedded, but shipping
relocatable bundles for plugins that read `.dat` or other companion files still
needs asset packaging and installed-location checks. This checkout-based
qualification must not be used as proof that those files exist on another
machine after installing a binary bundle.

Exact REAPER thread scheduling, UI interactions and PRNG interleaving are not
reproduced. Ordinary compound updates can lose writes; multi-cell structures
are not transactional. Mutable string operations and explicit atomic services
can contend on mutexes. Real-time contention, multiple-instance `gmem`, SysEx,
all MIDI buses, every parameter setting, long runs, and Windows/macOS are outside
the current suite. `gfx_idle`/`gfx_idle_only` remain unsupported.

Large requested heaps are allocated once. DuskVerb requests 220 million doubles
(1.76 GB decimal) per instance; this is a practical memory constraint even
though heap copying has been removed. See [the concurrency and allocation
contract](Native-GFX-Legacy.md) before changing heap limits or synchronization.
The existing EEL fallback and WDL FFT/LICE remain linked; removing those linked
dependencies is a separate change from removing the executing EEL graphics VM.

## Build and reproduce

```bash
git submodule update --init --recursive
export CMAKE_BUILD_PARALLEL_LEVEL=2
python scripts/build.py --only JoepVanlier --config Release
python -m unittest tests.jsfx_showcase.test_native_gfx_compiler \
  tests.jsfx_showcase.test_native_gfx_legacy tests.jsfx_showcase.test_source_resolver
python tests/jsfx_showcase/joep_full_qualification.py --out build/joep-native --jobs 1
```

Use a single compiler job for the large packages. The editor runner additionally
requires a configured JUCE plugin build with `ZA_JOEP_TEST_RUNNER=ON` and a
desktop or Xvfb display; pass that build as `--editor-build`. The numerical runner
requires the portable WDL oracle library from `tests/jsfx_showcase`, passed as
`--wdl-build`. Source manifests and compiler/runtime fingerprints accompany each
result so a cached pass cannot stand in for a different tested version.


For the Linux qualification fixtures, first generate the catalog as above, then
configure the WDL oracle without building unrelated showcase targets:

```bash
cmake -S tests/jsfx_showcase -B build/joep-wdl \
  -DSHOWCASE_GENERATED="$PWD/build/joep-native/joep_amaranth"
cmake --build build/joep-wdl --target showcase_eel showcase_numeric_runtime eel_eval
python tests/jsfx_showcase/joep_full_numerics.py --out build/joep-native \
  --wdl-build build/joep-wdl --seconds 2 --reuse
```

To configure the common editor fixture, copy the first guest into its build
folder before configuring CMake:

```bash
mkdir -p build/joep-editor
for f in JSFXDSP.h JSFXDSP.o JSFXSource.h JSFXResources.h; do
  cp "build/joep-native/joep_amaranth/$f" "build/joep-editor/$f"
done
cmake -S cmake/plugin -B build/joep-editor \
  -DZA_ROOT="$PWD" -DCMAKE_BUILD_TYPE=Release \
  -DPLUGIN_NAME="Joep native fixture" -DPLUGIN_SLUG=JoepFixture \
  -DPLUGIN_CODE=JpFx -DMANUFACTURER_NAME=ZorakAudio -DMANUFACTURER_CODE=Zora \
  -DBUNDLE_ID=com.zorakaudio.joepfixture -DPLUGIN_VERSION=0.0.1 \
  -DPLUGIN_TYPE=jsfx -DPLUGIN_JSFX_OBJ="$PWD/build/joep-editor/JSFXDSP.o" \
  -DZA_JOEP_TEST_RUNNER=ON -DZA_NATIVE_GFX_LEGACY=ON
python tests/jsfx_showcase/joep_full_qualification.py --out build/joep-native \
  --editors-only --editor-build build/joep-editor --reuse
```

The numerical/editor fixtures currently use Linux tools and a running display.
For a headless editor run, provide an Xvfb display through `DISPLAY`. The
production category build uses the normal platform build dependencies and
packages each plugin under its own identity.
