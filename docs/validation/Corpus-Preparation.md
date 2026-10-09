# Corpus preparation: historical long-recording checkpoint

This report records the earlier fully deferred Corpus v1.9.4 preparation test.
It is not a new benchmark of the current release. The recording's private name
and local path are omitted from the public checkpoint.

## Workload and results

One approximately ten-minute recording; Windows, a standalone production JUCE
processor, 48 kHz, 256-frame callbacks, default Balanced analysis and an 8192
grain budget. Graphics mode was native Legacy, with no editor open. Both paced
runs simulated normal audio callback cadence.

| Implementation | Total preparation wall time |
| --- | ---: |
| Original callback-paced preparation | 297.181 s |
| Intermediate background graph, with remaining callback-paced indexing/transfers | 56.5608 s |
| Fully deferred indexing/analysis and worker heap snapshot | 19.7435 s |

The final path saved 93.36% of elapsed preparation time against the original
(15.05×), and 65.09% against the intermediate graph (2.86×). These totals include
loading and publication. They are elapsed waits, not playback CPU improvements
or benchmarks against REAPER's own JSFX engine.

These are single timing runs from earlier profiling, not repeated interleaved
trials. The original and intermediate timings were collected earlier with
matching defaults/rate/buffer/graphics mode. Hardware, host cadence and the
recording affect the result. The unpaced final run took 21.2677 s; busy polling
competed with the workers rather than improving this case.

## What changed and what was checked

Indexing, features, structure, context and PE run as dependent single-writer
stages in a private analysis heap. The stages remain sequential. A worker
snapshots the heap; completion adopts the new model after validating the source
and configuration, preserving about 1 MiB of live controls/playback state.
Retired heap reclamation also happens on a worker.

Decoding and final prune publication remain separate. The host must keep
processing Corpus so its coordinator can submit, observe and publish work. This
test does not certify callback deadlines or imply preparation completes while
the host has stopped calling the plugin.

The model comparison checked **746,592 cells**, including initial indexing
tables, against the intermediate v1.9.3 graph; every tested cell matched as raw
bits. Paced and unpaced final models also matched. This is a tested-model claim,
not a proof of equivalence for every recording or a complete audio null test
against the original implementation.

Reload checks during indexing, PE and just before adoption published only the
latest source generation, with no memory faults or sample-read errors. MIDI
playback and editor creation passed after preparation. See the
[task/arena contract](../Structured-Tasks.md) and
[Corpus manual](../../plugins/Spectral/Corpus/README.md) for current behavior.

## Evidence and reproduction

[The sanitized checkpoint](Corpus-Preparation-results.json) retains the original
timings, comparison and model-check metadata. It was extracted from
`build/tasks/corpus-next/fully-deferred-results.json`; paths and the recording's
name were removed, and the data was not regenerated for this release.

The original baseline was commit
`e14e5adae6be00dc42023d0dd09cf27316644d4b`. The repository's
`tests/tasks/corpus_profile.cpp` harness supports a caller-supplied recording,
model dumps, reload-at-stage checks and playback verification. A new benchmark
should record the actual source/build fingerprints and repeat both paths with
the same recording and settings. No audio was loaded or rerun to prepare these
release notes.
