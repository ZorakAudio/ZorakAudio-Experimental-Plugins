# Structured task tests

Run `python tests/tasks/test_tasks.py` with llvmlite and clang++ installed.
The suite verifies compiler rejection rules and links generated AOT code to the
production scheduler in both ordinary and Legacy cell modes.

`processor.jsfx` and `task_processor_check.cpp` exercise the actual JUCE processor,
audio adoption, native graphics submission, editor closure and repeated prepare.
Configure the ordinary plugin CMake project with this fixture's generated AOT
object and `-DZA_TASK_TEST_RUNNER=ON`; run the resulting `task_processor_check`.
`python tests/tasks/build_processor.py` performs this build, builds VST3/CLAP,
and runs the lifecycle fixture using the locally installed toolchain.

See `docs/Structured-Tasks.md` for the language and runtime contract.

`python tests/tasks/build_corpus_graph.py` builds the real Corpus supplied-file
profiler without loading a generated recording. Pass exactly the authorized
file and a CSV output path to `build/tasks/corpus-profile/graph.exe`.
`--dump-model` includes indexing tables and downstream models; `--reload-at-index`,
`--reload-at-pe`, and `--reload-at-copy` reload that same file during indexing,
PE analysis, and the final worker stage before adoption respectively. The last
option retains its old name for compatibility; heap adoption now replaces the
output-copy phase. `--check-playback` tests MIDI output and editor creation after
timing. `--unpaced` removes the simulated host callback clock.

`python tests/tasks/build_corpus_graph.py --package` builds the production VST3
and CLAP into `dist/Corpus-Tasks` without loading any audio input.
