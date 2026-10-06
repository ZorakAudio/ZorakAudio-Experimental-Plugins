# Structured background tasks

DSP-JSFX now compiles deferred bodies into background callbacks. No `@worker`
section is required. This is an extension to this repository's compiled language,
not stock REAPER EEL2. It is an initial bounded implementation, not a claim of
production qualification for arbitrary existing scripts.

## Interface

```eel
task = defer(
  calculate_X();
  defer(
    calculate_Y();
    defer(calculate_Z(););
  );
  7; // This task's scalar result.
);

mapped = defer_for(i, count,
  calculate_value(i); // One scalar result for each iteration.
);

energy = defer_reduce(i, count, SUM, 0,
  task_buffer_read(input, i) * task_buffer_read(input, i);
);

next = defer_after(energy, task_result(energy) * 2;);
joined = defer_all(task, mapped, next);

task_status(joined) == TASK_SUCCEEDED ? (
  total = task_result(next);
  last = task_value(mapped, count-1);
);
task_cancel(task);
task_release(task);
```

Bodies are unevaluated at submission. Ordinary scalar globals are copied into
private state; surrounding function parameters and locals are captured by value.
Each parallel iteration starts with a fresh copy and a private index variable.
Writes to these variables never update the submitting DSP/GFX state. Helpers
called by deferred code use that same private state.

Children may run as soon as they are submitted. A parent's completion includes
all descendants, including children created by parallel iterations. Siblings
may run concurrently. A parent's scalar result is its own body's last expression,
not its final child's result. A body ending in `defer(...)` therefore returns the
numeric child handle; add an explicit final scalar expression when desired.

`defer_after` captures state when submitted, not when its dependency finishes.
Read dependency results explicitly with `task_result` or `task_value`. Dependencies
are retained even if their submitting caller releases its handle. Joins retain
all dependencies. Child handles are retained until their parent terminates and
then automatically released; root handles require `task_release`.

## Status, rejection and cancellation

Statuses: `TASK_INVALID=-1`, `TASK_PENDING=1`, `TASK_RUNNING=2`,
`TASK_SUCCEEDED=3`, `TASK_CANCELLED=4`, `TASK_FAILED=5`.
`TASK_BUSY=0` is reserved. `task_finished` is true for success, failure or
cancellation; check status before using results. It is false for invalid handles.
Valid retained handles have immutable results after success.

Submission and state-changing operations return immediately on contention:

| Return | Meaning |
|---|---|
| Positive | Accepted handle, or successful mutation (`1`) |
| `-1` | Capacity exhausted |
| `-2` | Busy: retry on a later callback |
| `-3` | Invalid argument or capacity bound |
| `-4` | Invalid, stale, or unavailable dependency/buffer |

Cancellation/release return `0` for an invalid task. Do not spin waiting on the
audio thread. For example, keep a root handle at zero until a submission succeeds:

```eel
@block
job <= 0 ? job = defer_reduce(i, 100, SUM, 0, i;);
task_status(job) == TASK_SUCCEEDED ? adopted = task_result(job);
za_keep_awake = job <= 0 || !task_finished(job);
```

`task_cancel` requests cooperative cancellation of the tree. The compiler checks
at task/helper entry and each `loop`/`while` iteration. A math builtin already in
progress is not forcibly interrupted. No deadline is promised. A rejected child
submission or invalid worker result/buffer read fails its parent tree; a cancelled
dependency prevents its continuation from executing. Completed tasks stay completed.
For a new file selection, cancel and release the old root handle and replace the
active handle. Cancellation after success does not erase a completed result;
the script must adopt only the currently selected load's results.

`task_status`, `task_finished`, `task_result`, `task_value`, and sealed buffer reads
use lock-free atomic loads. Reading a result before success or an invalid index
returns an error; doing so inside a worker fails that task. Result queries do not
wait or allocate. Handle numbers are instance-local and must not be persisted in
presets or transferred across instances.

## Immutable inputs and publication

```eel
input = task_buffer_create(100);
// Fill every cell, checking task_buffer_set's return value.
task_buffer_set(input, 0, 0.25);
// ... fill the other cells ...
task_buffer_seal(input);
job = defer_for(i, 100, task_buffer_read(input, i) * gain;);
```

Buffers are writable only by the submitting thread before sealing. Creation is
constant-cost and does not clear reused storage: initialize every cell that will
be read. Check every mutation's return value and retry a busy operation later.
Sealing makes the buffer immutable. Workers cannot create, mutate, seal, or release
buffers. `task_buffer_release` invalidates caller access, while conservatively
retaining storage until all active tasks in the instance finish. It can therefore
hold memory longer than an exact per-buffer reference scheme would.

Raw `mem[]`/`gmem[]`, sliders, audio samples, graphics, strings, file operations,
random state and host communication are rejected in ordinary deferred bodies, including
reachable helper functions. Capture control values into ordinary scalars before
submitting. Scalar snapshots of Legacy live globals are individually sampled;
they are not a transactional snapshot of a concurrently edited object. Use sealed
buffers to express coherent multi-cell inputs.

Parallel bodies return scalar values, exposed by `task_value(task, index)`.
Explicit arena bodies can write their private heap; parallel scalar bodies cannot. Copy results
into a replacement model incrementally in `@block`, then switch the active model
only after it is complete. This avoids implicit heap writes and large audio-side
publication copies. Multi-field descriptors can be represented as flattened
scalar indices or separate bounded batches.

## Reductions and resource bounds

Reducers: `SUM`, `PRODUCT`, `MIN`, `MAX`, `ANY`, `ALL`. The supplied identity is
returned for empty input. Results are combined in index order on a worker after
all iterations finish, independent of worker scheduling. NaNs propagate through
MIN/MAX; EEL2's normal assignment filtering still applies when storing results.
Custom reducers, scans, filters and general mutable shared buffers are not included.

Per instance: two worker threads, 32 task slots (including joins and retained
results), 4096 iterations per parallel task, 64 captured function locals, 4096
scalar variables, eight buffers of at most 65536 doubles each. Iterations are
scheduled in chunks of 32, not one queue node per iteration. Storage is allocated
and initialized at processor construction. Submission copies at most the bounded
scalar state and never allocates. Workers poll the scheduler at approximately
1 ms when idle. There is no global host-wide thread budget or work stealing yet.

Releasing a completed root task frees its slot once dependencies release their
references. Large graphs must be staged to stay within the fixed slot budget.
Nested resource exhaustion fails the tree instead of silently dropping children.

## Host integration and boundaries

DSP sections can use tasks in the normal AOT mode. `@gfx` submission requires
native graphics (`--native-gfx-legacy` or `--native-gfx-prototype`); the compiler
rejects task syntax in an uncompiled EEL graphics section. Legacy is the practical
mode for existing graphics-heavy scripts. Native display mode retains its existing
scalar ownership restrictions.

The processor owns the scheduler independently of the editor. State reset and
sample-rate/oversampling reinitialization invalidate old handles and cancel old
work. Destruction cancels and joins workers before releasing instance resources.
The current wrapper vetoes its own sleep while tasks or unacknowledged results
remain; `za_keep_awake` can also express additional adoption obligations. This
cannot force an external DAW to keep calling a suspended plugin. Task bodies
are not serialized into presets, and task API
calls from `@serialize` are rejected. Prefer coarse submissions from `@block` or
native `@gfx`; submitting new work per audio sample wastes the bounded task budget.

The stock EEL shadow correctness monitor cannot execute this extension and is
rejected for task-enabled builds. Custom scalar hoisting is also rejected with
tasks. Synchronous offline preparation/barriers are not provided: a plugin using
tasks must define what it renders before its model is ready. Corpus's preparation pipeline now uses the private arena chain described below.

## Validation

`python tests/tasks/test_tasks.py` checks compiler restrictions and links actual
AOT callbacks to the runtime in ordinary and Legacy modes. The processor fixture
also builds with the real JUCE wrapper using `ZA_TASK_TEST_RUNNER`. These checks
do not certify every platform, DAW, workload, or scheduling deadline.


## Private analysis arenas

`defer_arena(arena, dependency, body)` executes a single writer in an explicit,
private heap. It shares mutable private scalar/heap results with later writers
in the same arena. Every writer must depend on the previous writer. Ordinary
`defer_after` retains its submission-time snapshot semantics.

```eel
arena = task_arena_create(__memtop(), sample_pool_handle, source_generation);
// Retry on later blocks until task_arena_status(arena) == 3.
// Freeze the inputs/configuration while copying a contiguous prefix:
task_arena_copy_in(arena, offset, count); // at most 16384 cells per call
// Seal only after the entire heap is copied.
task_arena_seal(arena);
features = defer_arena(arena, 0, build_features(););
structure = defer_arena(arena, features, build_structure(););
pe = defer_arena(arena, structure, build_pe(););
// Once pe succeeds, copy selected model ranges in bounded batches:
task_arena_copy_out(arena, offset, count);
// Variable names are unevaluated destinations; only listed globals are copied.
task_arena_commit(arena, model_ready, model_epoch);
task_arena_release(arena);
// Release every task handle too, retrying -2 contention responses.
```

One arena per instance is bounded to 25,165,824 doubles (192 MiB), in addition
to the existing live heap. Creation queues allocation and page initialization
on a worker; release queues worker reclamation. Arena status is 1 queued,
2 allocating, 3 accepting input, 4 sealed, or 5 failed. Status is independent
of task status. All root operations use try-lock and return -2 when contended.
Copying/releasing never allocates or frees heap memory on the audio thread.
No arena operations may run inside ordinary workers; `task_report(stage,
phase, cursor, total)` is an arena-only exception. Poll its four atomic fields
with `task_arena_progress(arena, field)` (0 through 3); fields are approximate
progress, not a consistent model snapshot.

Indexed memory, FFT/memory builtins, captured slider reads, `sample_read2`, and read-only sample metadata
(`sample_get`, `sample_len`, `sample_srate`, `sample_channels`, `sample_peak`)
are permitted only in arena bodies and reachable helpers. Strings, shared
`gmem`, slider writes, audio/GFX/MIDI/file/pool mutation and random state remain
forbidden. The host pins exactly the requested immutable sample generation
throughout the arena lifetime. A generation mismatch fails allocation. Pass
zero for pool/generation for computation without sample reads. Heap growth is
forbidden. Exceptions or memory faults fail the graph. Cancellation is checked
at helper entry and every loop iteration; individual host/FFT calls finish
before the next check.

The caller must freeze model inputs during copy-in, check its own source/config
epoch before adoption, copy every required output range, then commit scalar
readiness flags. Copy-out and commit require all arena writers to have finished
successfully. They do not provide a transaction across a whole live heap:
consumers must stay disabled until the final commit. Control/UI/delay memory
should be excluded from output ranges. Corpus enforces these rules and falls
back to its earlier incremental preparation if arena execution fails.

## Worker snapshots and heap adoption (native Legacy)

Native Legacy builds can avoid callback-paced full-heap transfers:

```eel
arena = task_arena_clone(__memtop(), sample_pool_handle, source_generation);
// Freeze model inputs from submission through adoption.
// The worker allocates, snapshots and seals; wait for status 4.
task_arena_preserve(arena, controls_start, controls_count);
index = defer_arena(arena, 0, build_index(););
features = defer_arena(arena, index, build_features(););
// After the final writer succeeds, validate your source/configuration epoch:
task_arena_adopt(arena, model_ready, model_epoch);
// Adoption consumes the arena. Release each task handle as usual.
```

`task_arena_clone(size, pool, generation)` requires the complete fixed host heap
and the native Legacy ownership hooks. Other modes return -3. Cloning runs on a
worker; atomic cell reads prevent data races but do not make a concurrently
mutating model a coherent snapshot. The caller must freeze model inputs.

`task_arena_preserve(arena, offset, count)` registers up to 16 live heap ranges,
with an aggregate maximum of 262144 cells (2 MiB). These ranges are copied from
the live heap immediately before adoption. Keep them limited to controls and
playback state that must survive; model output ranges must be excluded.

`task_arena_adopt(arena, global1, ...)` requires a cloned, sealed arena, no
remaining writers, the original live heap, valid destination globals and a
current pinned sample generation. It copies preserved ranges, swaps heap
ownership, commits only the named globals, and refreshes the native graphics
heap binding while the host lifecycle lock excludes graphics execution. The
retired heap and sample lease are reclaimed by a worker. No allocation or heap
free occurs in the audio callback. The operation returns 1 on success and
consumes the arena; -2 means retry, while validation failure leaves the live
model unchanged. State reset/destruction cancels snapshots and joins workers
before their source heap can be freed.

This is a whole-model publication boundary, not a guarantee of a universal
audio deadline: preservation still copies up to the registered bound, and the
host must call the coordinator to publish. External configuration must be
validated by the caller. Corpus uses approximately 1 MiB of preserved live
state and chains indexing with the other analysis stages in one arena.
