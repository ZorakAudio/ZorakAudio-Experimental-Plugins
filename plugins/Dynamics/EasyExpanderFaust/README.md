# EasyExpander Faust

This motivating example keeps EasyExpander's JSFX initialization, slider controls
and graphics, and implements its audio detector/expander in an embedded `@faust`
section. The original EasyExpander remains available separately.

Build with `python tests/faust/build_example.py --package`. VST3 and CLAP are
staged in `dist/EasyExpander-Faust`. The compiler requires Faust's LLVM backend
at build time; the plugin has no Faust compiler or libfaust runtime dependency.
The example has its own plugin identity and uses native Legacy graphics.

Ordinary JSFX scalars, slider aliases and `splN` signal inputs are inferred from
Faust's symbol resolution. Top-level Faust scalar definitions matching JSFX
globals publish their final sample value automatically; the graphics meters
receive these results. DSP state lives in an independent Faust instance.

`python tests/faust/profile_kernel.py` compares original and Faust kernels,
including parameter changes and multiple rates/buffers. It creates numerical
signals in memory, without loading an audio file. `build_example.py --before`
and `build_example.py` produce supplied-file JUCE profilers; invoke the resulting
`before.exe` / `after.exe` with the authorized recording and a report path.
Timing excludes file decoding and records `processBlock` time separately from
output dumping. Editor creation, closure and rate resets are also checked.

See `docs/JSFX-Faust-Sections.md` for ordering, inference and execution guarantees.

The example now uses the exact user-pasted EasyExpander baseline. Its executable
code is identical to the earlier repository source; only help/tooltip comments
and the final newline differ. The host uses explicit plugin sleep permission,
so this stateful expander remains active through quiet input. Offline renders
also always advance DSP. See `docs/Cooperative-Sleep.md` for the new contract.

Updated comparisons use continuously active processing on both sides. The older
report compared two builds sharing the previous threshold-based Auto Sleep;
although those outputs matched each other, Auto Sleep did not null against a
continuously active reference. Consult the updated idle audit and results.

Updated matched active-processing result: 15.821 s -> 3.476 s (4.55x), with 58,558,936 bit-identical float samples. See EasyExpander-Sleep-Audit.md and the accompanying JSON report.
