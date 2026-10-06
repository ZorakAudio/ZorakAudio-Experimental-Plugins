# Sample Faust character integration

Implemented locally as an optional **Sample Faust** native plugin. The maintained cached Sample remains the default. The generated entry is `plugins/Spectral/Sample/src/SampleFaust.jsfx`; its independent manifest is `plugins/Spectral/SampleFaust/plugin.json`. Neither Corpus nor sleep policy was changed in this integration.

## Complete-plugin CPU measurements

These measurements compare complete Sample processing against the already optimized, cached EEL Sample—not the older uncached implementation. Each invocation produces 32 seconds of stereo output; file loading is excluded. Active tests use three interleaved pairs per setting, reversing execution order between trials. Values below are medians of processing CPU time.

| Sample rate / host buffer | Cached Sample | Sample Faust | CPU saved |
|---|---:|---:|---:|
| 48 kHz / 64 | 13.3798 s | 13.1476 s | 1.7% |
| 48 kHz / 256 | 5.81358 s | 5.36264 s | 7.8% |
| 96 kHz / 1024 | 7.84723 s | 7.20648 s | 8.2% |

The isolated character kernel's earlier roughly 4.5–5.5× improvement does not translate into that improvement for the full sampler. Voice generation, harmonic enrichment, filters, modulation and other processing still contribute to total CPU.

With character and harmonics disabled, single paired runs measured **3.5%, 2.8%, and 0.7% more CPU**, respectively. FAUST performed zero compute calls in those runs; the remaining cost is staged execution overhead. Single pairs are sensitive to timing noise, but this is sufficient reason to keep the variant optional.

Additional Contour/DeCrust handoff tests measured 4.5% and 16.2% less CPU at 48 kHz/256 and 96 kHz/1024. Those are single pairs intended primarily to validate correctness; the 16.2% result is not a repeatable general speedup claim.

## Correctness evidence and limits

All **58,368,000 compared floating-point output values were bit-identical**: 36,864,000 in nine active pairs, 9,216,000 in two routing/handoff pairs, and 12,288,000 in three character-disabled pairs. Existing observer CSVs were byte-identical, including voice, smoothing, silence-counter and output observations. This does not independently validate every private filter history or every possible preset/input.

Active tests alternate raw/tape scenarios and exercise transistor/diode modes, 12/24 dB drives, harmonic changes, HP/LP/EQ changes, MIDI offsets, overlap and retriggering. Handoff tests additionally switch Contour and DeCrust and use generated low-level dry-input sine probes. Audio remained finite; memory-fault checks, editor creation and release/reprepare at a different sample rate passed.

Only the user-authorized FLAC was loaded as the sample bank. It is approximately 610 seconds long, but these are **32-second output workloads, not complete ten-minute renders**. No other recording was loaded. This is a native test-host comparison, not a REAPER render or listening evaluation. The Legacy editor build was exercised; interactive UI actions, all presets and other editor backends have not been exhaustively qualified.

The compiler contract suite passed **12 tests** and mixed-stage runtime suite passed **19 fixtures**, including stream capture through helpers, overwrites after capture, conditional suspension/resumption, independent instances, quantum boundaries and irregular block lengths.

## DSP and execution design

The original ordering is preserved: harmonic enrichment precedes the processing strip. The original HP/LP/EQ and five parallel character bands retain their coefficients, wet mixes, denormal handling and high-air guards. The nonlinear formulas are unchanged:

- Transistor: `y = x + 0.18*x*x - 0.10*x*x*x`, then `y/(1 + 0.62*abs(y))`.
- Diode: `x/sqrt(1 + 1.45*x*x) + 0.055*(x*x/(1 + 1.8*abs(x)) - 0.18*x)`.

There is no diode lookup approximation. Metering remains before the high-air guard; dry summation, output trim and silence accounting retain the original behavior.

The character section uses `@faust block when za_character_block`, with a processing quantum of at most 256 samples. A 1024-sample host callback therefore processes several chunks. Named per-frame streams capture existing EEL variables at the producing sample stage; block-wide controls are copied once per host callback. Unfused EEL sample stages run through an inlined LLVM bulk loop, preserving the optimization lost when calling EEL from C++ once per sample. Conditional routing is decided once per stage, and inactive FAUST computation is skipped.

Processing runs synchronously on the host's processing thread. This is block DSP, not worker-thread computation. No JIT, compilation or buffer allocation occurs during processing, and this adds no audio latency.

## Conservative fallback and ownership

The original EEL character implementation owns processing when analyzers, EQ solo, Contour, DeCrust, delay/input-latency paths, output fades or warm-up make separation unsafe. Guards also protect the silence-reset boundary. Entering FAUST seeds its histories from EEL; leaving copies histories back. A reset epoch ensures a deliberate buffer reset wins over stale state. A completed FAUST pass synchronizes histories at the host-block boundary. Zero-length callbacks do not publish nonexistent results.

This preserves behavior across the tested transitions without running both character paths on the same audio. More configurations can be optimized later only after comparable handoff qualification.

## Source and regeneration

The optional source is generated from the maintained cached `Sample.jsfx` by `tests/faust/sample_character_candidate.py`. That generator checks integration boundaries and rejects drift rather than silently introducing duplicate character processing. The maintained Sample source was unchanged by this integration.

Qualified generated-source SHA-256: `7f30152c1c4c8d9e51c5ba71cf71d17171318549a6b6c33fa52b85021636adc5`.

Regenerate with:

```text
python tests/faust/sample_character_candidate.py --output plugins/Spectral/Sample/src/SampleFaust.jsfx
```

Build with `python scripts/build.py --only SampleFaust`; this requires FAUST's LLVM backend at build time. On this machine the Visual Studio generator could not locate its compiler, so the packaging build uses the installed Clang/Ninja toolchain via `ZA_CMAKE_GENERATOR=Ninja`. The embedded source requires this repository's native compiler; it is not a stock REAPER JSFX effect.

See `docs/JSFX-Faust-Sections.md` for the explicit block, named-stream, condition and quantum API. Raw results and test logs are retained under `docs/validation/sample-faust-character`. Test-host fixture identity is separate from the actual Sample Faust product identity.

## Local package result

The Release **Sample Faust VST3 and CLAP** package built successfully with Clang/Ninja. ZIP integrity and both plugin entries were checked. The package was not installed. Numerical qualification used the native test-host build described above; this is not a claim that the packaged formats were rendered in REAPER.
