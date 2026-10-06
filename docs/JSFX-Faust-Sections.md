# Mixed JSFX / Faust sections

For a condensed current interface and the other language extensions, start with
[the consolidated guide](DSP-JSFX-Guide.md). Explicit modes are `@faust block`
and `@faust block when control`; `@faust sample` is not implemented. Unqualified
`@faust` retains automatic dependency-preserving execution.

`@faust` contains Faust source, including its normal imports, definitions and
`process` expression. The installed Faust compiler's LLVM backend is invoked
at build time. Its LLVM module is linked with the EEL2 module before the existing
AOT optimization/object-emission path. Plugins do not load libfaust, compile,
JIT, or run a Faust worker at playback time. Verified locally with Faust 2.81.2
(LLVM 17), llvmlite's LLVM reader and Clang 21 on Windows x64.

```eel
slider1:0.5<0,1,0.01>Gain
@init
meter = 0;
@block
level = slider1;
@faust
import("stdfaust.lib");
meter = abs(spl0);
process = spl0 * level, spl1 * level;
@sample
spl0 *= 0.9;
spl1 *= 0.9;
```

No bridge declarations are required. `level` is a JSFX input, `spl0` / `spl1`
are audio signals, and `meter` publishes a Faust signal into the JSFX global
of that name. `meter` is initially declared in JSFX so its ownership is explicit.

## Inference and scope

Faust resolves its own definitions, libraries, function parameters and local
scopes normally. When its compiler reports an unresolved name, the bridge
imports it only if it names an existing JSFX scalar, a declared slider alias,
`slider1` through `slider256`, `spl0` through `spl63`, `srate` or `samplesblock`.
JSFX names are matched case-insensitively. Namespaced values such as
`params.gain` are imported through a generated Faust environment; matching
parameterless definitions inside a Faust environment can export those fields.
Unknown names fail with the originating section line; they never silently
become a zero-valued control. Faust-local definitions take precedence.

JSFX scalar inputs become private, hidden Faust control zones and are refreshed
before each compute call. No visible sliders are added. `splN` inputs become
signal ports fed from the corresponding audio channel. Standard Faust `_`
inputs also work, mapped in order from channel zero. Named signal ports precede
implicit ports internally; the compiler records the complete wiring in metadata.

Top-level, parameterless Faust definitions matching existing ordinary JSFX
globals become output signals appended to the internal process outputs. Their
values are assigned back to JSFX, with the normal EEL2 assignment filtering.
Slider aliases can also be output destinations; slider change notification is
queued and the alias scalar is updated. Names defined only in Faust stay private.
Audio outputs from `process` map in order to `spl0`, `spl1`, etc. Definitions of
`splN` inside Faust are local signal definitions, not imperative JSFX assignments.

This does not turn Faust into an imperative language: a self-referencing Faust
definition still needs lawful Faust feedback/delay syntax. Shared JSFX RAM,
`gmem`, strings and JSFX function calls are not implicitly imported. Faust
foreign functions still need definitions available to the final native linker.
The `__za_` prefix is reserved for the generated Faust bridge.

## Ordering and rates

For scripts containing Faust, the source order of every `@block`, `@sample`,
and `@faust` section defines the DSP timeline. Repeated sections are supported.
`@init`, `@slider`, `@gfx` and serialization keep their existing lifecycle roles.
Scripts without Faust keep the existing JSFX order and ABI.

Each `@block` stage runs once per host processing block, at its position in the
timeline. Audio stages between block stages form a processing group. Independent
stages process entire buffers in order, allowing Faust to compute a full block
once. A block after an audio group sees the preceding group's final sample state.

When an EEL sample section updates a scalar consumed by Faust, reads a scalar
produced by Faust, or another Faust section consumes those scalar outputs, the
whole group is interleaved sample by sample. Multiple EEL sample sections also
use this path because they can alias arbitrary JSFX heap state. Dynamic or
unknown EEL mutations are treated conservatively. In that path `compute(1)`
preserves the dependency rather than converting a changing control to a block
constant. Pure audio routing alone does not require this fallback.

The metadata includes ordered stages, imported/exported bindings, Faust version,
source hashes, DSP layouts and which groups are sample fused. A scalar output
exposes its final sample outside a group; within a fused group it is visible to
the next stage on the same sample. Processing adds no block of latency. Faust's
own filters/delays retain their normal latency and private state.

## Runtime guarantees and limits

Each processor owns independent Faust DSP state. The host initializes/reinitializes
it alongside JSFX, including sample-rate and oversampling changes. Generated Faust class tables are also instance-owned. The LLVM bridge redirects
table access through thread-local pointers that are rebound before each compute
call, so processor instances and migrated audio threads cannot borrow each
other's tables. Rate-dependent table banks for 1x/2x/4x/8x are generated during
host preparation; rate changes select a prepared bank rather than generating
tables or allocating generator state on the audio thread. Native Legacy graphics retains its lifecycle
lock; the existing EEL shadow correctness monitor is rejected for mixed builds
because stock EEL cannot execute the extension. Custom section/loop hoisting is
also rejected so it cannot move statements across language boundaries.

Buffers and DSP memory are allocated/prefaulted during host preparation. The
mixed processing function does not allocate, free, lock, compile or JIT. Faust
executes on the host audio thread. These bridge guarantees do not override
behavior inside user-supplied foreign functions or existing JSFX code. The host reserves scratch capacity for its
maximum 8x oversampling factor; oversampling reinitialization resets existing
Faust state without reallocating it. Buffers beyond prepared capacity fail with
a memory fault and silence rather than reallocating in processing; the host must
prepare again for a larger maximum block size. Zero-length processing runs its block
stages without advancing sample/Faust audio state.

Bounds: 64 audio channels, 128 internal signal ports, 256 inferred imports and
256 MiB of private DSP state plus the four table banks per Faust section. Faust uses double precision;
conversions to host floats happen at the outer pipeline boundary. Deadline
behavior still depends on the DSP program and host scheduling. Memory backing
for legal Faust delays is part of the instance, not the JSFX heap.

Relative Faust imports search the script directory and standard Faust libraries.
The compiler CLI accepts repeatable `--faust-include` directories; the normal
repository build supplies the original JSFX directory even after JSFX import
expansion. `JSFX_FAUST_COMPILER` can select a different Faust executable.

## Validation and example

`plugins/Dynamics/EasyExpanderFaust` is a separate motivating plugin, retaining
EasyExpander's controls/setup/graphics and moving its detector/gain kernel into
Faust. The original plugin is unchanged. `tests/faust/test_mixed.py` links real
LLVM-generated modules and exercises order, input/output inference and execution
modes. `profile_kernel.py` compares the original and mixed kernels at several
rates/buffer sizes with parameter changes. The JUCE supplied-file profiler also
checks complete processor/editor/reset behavior. Test results cover those cases,
not every Faust library, foreign function, platform or DAW.

Faust reference: [architecture/backend documentation](https://faustdoc.grame.fr/manual/architectures/)
and [language syntax](https://faustdoc.grame.fr/manual/syntax/).

The installed Faust 2.81.2 LLVM backend needs a version-specific correction to
initialize the sampling frequency in its private table generators; a rate-dependent
table fixture verifies this at 48/96 kHz and after reset. Table initialization
cannot capture live JSFX controls: such unsupported bindings fail compilation.
Table generators should use constants or Faust's sample-rate value. Unsupported
backend table constant-expression layouts also fail compilation explicitly.

Measured product results, active/offline benchmark conditions and rejected
integration boundaries are consolidated in [FAUST qualification](FAUST-Qualification.md).
This API document does not use historical test counts or Auto Sleep timings as
current performance claims.

## Cooperative sleep audit

The host supports automatic silence/event modes and explicit cooperative sleep.
Default/Auto selects permission-based sleep for scripts declaring
`za_sleep_ready`; otherwise it uses idle options/topology. Cooperative sleep
requires a fresh `za_sleep_ready = 1` grant, exact-zero output, no input/event
activity, and no outstanding task obligation. Automatic threshold skipping is
a heuristic rather than a universal null guarantee. Never sleep and the other
saved selectors are available. Offline rendering always advances DSP. See
`Cooperative-Sleep.md`. EasyExpander makes no grant because its envelopes
continue changing.

Earlier comparisons with shared automatic sleep did not establish equivalence
to continuously active DSP. See the matched active/offline comparison in the
qualification summary.

## Explicit block streams and conditional execution

`@faust block` requires block execution. Imports written by the nearest preceding
`@sample`, including its reachable EEL helpers, become private per-frame streams.
The runtime captures each value after that sample stage; FAUST receives the whole
sequence rather than the final scalar value. These streams do not add host audio
pins. Other imports remain scalar controls. An intervening `@block` does not
overwrite the already captured stream. Use separately named block controls when
that is the intended dependency.

The compiler rejects explicit block mode when the island still requires sample
interleaving. An explicit `@block` boundary can separate phases, but the author
must preserve shared heap, reset and event dependencies across that boundary.
The original unqualified `@faust` behavior remains compatible.

`@faust block when enabled` adds a scalar block control. When zero, the stage
passes audio through, leaves its private DSP history and scalar exports unchanged,
and skips `compute`. A sample stage may not write this condition. Restoring
enabled resumes its history; it does not automatically clear a tail.

For a pipeline that needs bounded state feedback, `options:za_faust_quantum=256`
opts into chunks of up to 256 frames. The first stage must be `@block` and runs
once for the original host callback. All subsequent stages, including later
`@block` hooks, repeat for each chunk. `samplesblock` remains the original host
frame count; sample clocks and MIDI ingestion must stay in the initial host
stage. Chunks add no audio latency. A final one-frame remainder is avoided by
shortening the preceding chunk; an actual one-frame host callback remains legal.
This option accepts 2–4096 and requires an entirely unfused pipeline.

Stream and audio buffers are allocated during preparation. Conditional capture
routes are selected once per sample stage, not rediscovered per frame. There is
no allocation, compiler, JIT or worker scheduling in these audio callbacks.

Sample's character integration uses this mechanism to preserve dry passthrough,
share filter histories with an EEL fallback and feed silence bookkeeping back at
chunk boundaries. It retains EEL when solo, analyzers, Contour, DeCrust, latency
or output fades prevent commuting the character residual past final output trim.

Unfused EEL sample stages in a mixed pipeline execute through a generated LLVM bulk loop. Its sample body is inlined into the loop, and named-stream routes are selected once per stage. This avoids a C++ callback into EEL for every frame. Fused scalar-feedback islands retain their per-sample execution semantics.
