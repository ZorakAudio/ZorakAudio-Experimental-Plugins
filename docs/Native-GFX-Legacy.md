# Native GFX Legacy: shared-state contract

`--native-gfx-legacy` compiles DSP and `@gfx` into native code that accesses the
same live guest variables, slider cells, and local heap. No ownership contract,
heap mirror, publication view, or GFX-to-DSP heap copy is required. The original
Sample source at the checkout's base commit runs in this mode without a source
migration. JoepVanlier packages always select this compatibility path in the
standard build and direct AOT build helper. Other JSFX opt in explicitly.
This is not a completed replacement of every EEL language feature or GFX API.

## Build and allocation

Initialize dependencies and build an eligible plugin:

```bash
git submodule update --init --recursive
python scripts/build.py --only Sample --native-gfx-legacy --config Release
```

For non-Joep JSFX, `--native-gfx-prototype` selects bounded publications;
omitting both flags uses a manifest override when present, otherwise EEL graphics. JoepVanlier always selects Legacy,
including when the prototype flag is supplied. Faust selects neither mode.
The two explicit flags are mutually exclusive. Legacy mode
also rejects custom section/loop scalar hoisting and the shadow correctness
monitor: their assumptions do not hold for concurrently changing guest state.

Legacy mode uses C++20, lock-free aligned 64-bit cells, and generated runtime
state ABI 5. Rebuild the object and header together; a header from another mode
is not interchangeable. CMake derives the C++ requirement from that header.
The default and publication modes retain ABI 3 and C++17.

`options:maxmem=N` sets the fixed capacity in **double cells**, not bytes.
The accepted range is 1 through 268,435,456; the default is 8,388,608 cells
(64 MiB per instance). Construction allocates and touches the zeroed pages
before processing begins. Each instance owns its allocation until destruction.
Reprepare and oversampling resets retain the heap address and capacity.
`__memtop()` reports that capacity; `freembuf()` does not resize it. Invalid
accesses fault the invoking DSP or GFX context instead of reallocating or
accessing outside the allocation. A failed allocation leaves a faulted runtime.

This removes guest-heap reallocation, not every allocation in the plugin.
Sample loading, drawing, existing FFT permutation scratch, and other host
services have their own storage. Pages are not locked into physical memory.
Large capacities increase startup time and resident memory even for scripts
that touch few cells; four default instances reserve 256 MiB for guest heaps.

## Concurrency contract

Guest numeric loads/stores use LLVM `monotonic` atomics and C++ relaxed
`atomic_ref<double>` accesses. This preserves live per-cell interleaving without
introducing C++ data-race undefined behavior. Variables, heap cells, slider
aliases, `spl`, `srate`, and `samplesblock` resolve to the shared DSP state.
Named sliders resolve directly to their canonical slider cell.

Ordinary assignment does **not** provide a transaction or publication barrier.
`counter += 1` is a load, arithmetic, and store; two writers can lose updates.
Readers can see a mixture of old and new cells in a table, waveform, or UI
frame. Script flags and double buffers retain their original algorithmic races.
This mode does not retrofit a synchronization protocol into existing scripts.

`atomic_get`, `atomic_set`, `atomic_add`, `atomic_exch`, and
`atomic_setifequal` serialize through one per-instance mutex, following the
vendored `WDL/eel2/eel_atomic.h` model. Simple lvalue operands are read inside
that critical section, including `atomic_add(x,x)`; expression operands use
private temporaries. Add/set return the new value, compare/set returns the old
value with WDL's `1e-5` comparison tolerance, and exchange swaps both lvalues
and returns the new first value. These calls can block on contention. Ordinary
guest reads/writes do not acquire that mutex and can still interfere with an
explicit atomic operation. A publication protocol must consistently use the
explicit atomic APIs at its synchronization point.

Bulk helpers are also atomic-aware. `memset`, overlap-aware `memcpy`, file/MIDI
outputs, sample reads, and message buffers access guest cells individually.
WDL FFT/convolution arithmetic gathers into private scratch and commits cells
individually, preventing raw non-atomic math from touching a concurrently shared
buffer. This is not an atomic FFT transaction: an overlapping writer can be
overwritten on commit. FFT access reserves two 65,536-double thread-local banks
(1 MiB per participating thread), in addition to existing permutation scratch.
First-use scratch costs and cache contention still need real-host measurement.

The GFX execution context is separate from guest storage: drawing commands,
fonts, input queues, fault status, and PRNG bookkeeping are private. Mutable
strings share an instance-owned store, protected by a mutex. String operations
from DSP can therefore block when contending with GFX.
There is a lifecycle mutex around GFX execution and host reset, not around
ordinary audio processing. Audio uses a try-lock for oversampling reinit and
defers it if GFX is executing. The host must quiesce audio for `prepareToPlay`,
as with the existing implementation. Held input receives one release frame on
hide/close; hidden editors then stop running GFX. Script-authored sliders are
not overwritten every block by unchanged host parameters. Actual host changes
and explicit slider notifications continue through the host automation bridge.

## Supported scope

The LEGACY renderer implements the 31 drawing/input calls used by the catalog,
including offscreen images, resource loading, blits, gradients, pixels, fonts,
text metrics, keyboard input, cursors, menus and file drops. It also implements
13 worker file calls, mutable string operations and `match`/`matchi` pattern
captures. GFX file loading writes the shared heap directly. Audio file loading
continues through the existing host cache. Drawing setup calls from `@init`
use an instance-owned graphics context that survives editor closure.

Legacy mode includes heap writes, bulk memory, FFT/inverse/permutations/
convolution, `gmem`, dynamic slider/sample references and explicit atomics.
Parser compatibility covers EEL precedence, overloaded/redefined functions,
receiver namespaces, packed character constants, multiline strings and loop
budgets. Shared heap allocation remains fixed for the instance lifetime.

`gfx_idle` / `gfx_idle_only` remain unsupported. Custom `@serialize` code is
still not executed; host parameter/file-path state is saved, but a JSFX's own
sample-in-preset or custom table serialization is not supplied by that state.
Exact REAPER scheduling/PRNG behavior, every EEL lvalue construct and every
host service are not claimed. WDL FFT/LICE and the existing EEL fallback remain
linked. Native legacy instances execute neither an EEL GFX VM nor an EEL heap
mirror; removal of the linked fallback is a separate step.

See [JoepVanlier native compatibility](JoepVanlier-Native-Compatibility.md) for
the current full-source catalog matrix. The earlier results below describe the
initial Sample/concurrency implementation, before the catalog extension.

## Historical implementation qualification

The checked implementation was built on Linux x86-64 with GCC 13, llvmlite
0.50, and the repository's recursive JUCE/CLAP submodules. It passed:

- 29 compiler tests covering both native modes and unsupported feature gates.
- Actual AOT DSP/GFX concurrent execution: 200,000 explicit atomic increments
  without loss, an aliased addend test, producer/publication ordering, concurrent
  bulk writes, direct GFX-to-audio visibility, named slider aliases, persistent
  locals, stable allocation, and last-cell/one-past-end bounds.
- 13 WDL scalar comparisons and eight bulk comparisons (the initial real FFT
  plus seven additional sequences), with maximum observed FFT/bulk error zero.
  These include complex/real inverse transforms, permutations, convolution,
  overlapping copy, and maximum 32,768-point transforms.
- AddressSanitizer on C++ helpers and the WDL oracle. Generated AOT code was
  **not** ASan-instrumented; no Clang was installed. LeakSanitizer was disabled
  because this execution environment prevents its `/proc` thread inspection.
  No ThreadSanitizer claim is made.
- Default compiler comparison against the pre-legacy compiler: 58 scripts have
  identical LLVM; 19 retain their existing compiler errors and one its existing
  import-expansion error. This is not a claim that all 78 plugins run natively.
- Actual Sample VST3 and CLAP builds and the production JUCE editor: original
  source, sample bank growth/shrink, MIDI audio and logging, native controls and
  menus, balanced automation gestures, resize, focus/hide, reprepare, two editor
  lifetimes, and four concurrent instances. Heap publication cells/capacity and
  publication time are all zero in legacy mode.

The editor qualification build enables diagnostic hooks used by the runner.
The distributed VST3/CLAP are rebuilt with those hooks disabled; there is no
test-only scalar-bank capture in those artifacts.

Benchmark results and the final regression/editor logs accompany the package.
The benchmark runs the actual processor with identical original Sample source,
numeric settings, a three-file synthetic bank, 48 kHz/512-sample blocks, MIDI,
and no editor. It isolates DSP overhead; it does not measure a DAW deadline,
GFX contention, or every polyphony/oversampling setting. Keeping this mode
opt-in is appropriate until those workloads and supported target platforms are
qualified. Windows/macOS builds, DAW/pluginval, and execution inside REAPER
were not tested in this Linux checkpoint. Later Windows product checks in the
CMD, Sample Faust and Hyperreal reports do not broaden this fixture's coverage.

Five alternating 2,000-block trials per mode measured the following. Both
produced the same observed audio peak, 0.0680148. These are local measurements,
not a general performance guarantee.

| Metric (microseconds per block) | Default DSP | Shared legacy DSP |
| --- | ---: | ---: |
| Median of run means | 140.576 | 164.682 |
| Median block time across trials | 121.363 | 146.240 |
| Range of run means | 131.830–146.565 | 162.703–178.075 |
| Median p95 across trials | 220.393 | 251.269 |

The median run mean increases **17.1%**, and median block time **20.5%**, with
shared cells in this workload. An earlier five-trial run measured 20.8% mean
overhead, illustrating the noise in this shared execution environment.
There is no editor or heap publication in either benchmark run; the result
therefore measures the DSP-side cost, not the net cost with graphics open.

Relevant primary references: [REAPER JSFX sections/options](https://www.reaper.fm/sdk/js/js.php),
[REAPER atomic APIs](https://www.reaper.fm/sdk/js/advfunc.php),
[LLVM atomic guide](https://llvm.org/docs/Atomics.html), and the vendored
`src/WDL/eel2/eel_atomic.h`. The implementation supplies the contract above;
it does not claim proof of equivalence to REAPER's closed host implementation.
