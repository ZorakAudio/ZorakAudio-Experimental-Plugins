# Corpus

Corpus loads recordings, learns their acoustic structure, and plays new paths through them from MIDI notes.

## Try the local task build

Use the VST3 or CLAP in `dist/Corpus-Tasks`, then rescan it in your host. This uses the same plugin identity as Corpus; use one version at a time. Open **Corpus sources**, select your recording and wait for preparation to finish, then send MIDI notes. **Reload Analysis** rebuilds the model.

Preparation runs Index → Features → Structure → Grammar/Map/Focus → PE as dependent
background tasks. A worker snapshots the input heap and runs the same deterministic
calculations without waiting for audio callbacks between work batches. The completed
model is adopted by swapping heaps after source/configuration validation; only the
small live control/playback regions are copied at adoption. The old heap is reclaimed
on a worker. Hosts must continue processing Corpus so it can notice completion and
publish the model.

The graph owns a private 192 MiB analysis heap and pins the selected immutable
sample generation. Replacing the bank or changing analysis settings cancels
old work. Persistent controls and live delay memory are preserved across adoption.
Worker failure falls back to incremental preparation. Decoding remains on the
sample loader, and the final prune publication remains incremental. The five
analysis stages form a serial dependency chain; two available worker threads do
not make these dependent stages run simultaneously.

Use this repository's AOT build and native Legacy graphics mode, selected by
`plugin.json`; this extension is not stock REAPER JSFX. Handles/private heaps
are transient and are not saved in presets.

## Validation and rebuilding

`python tests/tasks/test_tasks.py` covers compiler restrictions, private memory,
slider snapshots, writer ordering, worker snapshots, heap adoption, cancellation and publication in ordinary
and Legacy modes. `tests/tasks/corpus_profile.cpp` profiles the real JUCE
processor with a caller-supplied file. Its `--dump-model`, `--reload-at-index`, `--reload-at-pe`,
`--reload-at-copy` and `--check-playback` options verify model equivalence,
cancellation and usable playback without loading any additional recordings.

`python tests/tasks/build_corpus.py --package` uses a separate generated fixture
for general repository regression testing. Do not use that command when a run
is restricted to a specific input recording. The supplied-file profiling and
production build can be performed separately with that fixture disabled.
