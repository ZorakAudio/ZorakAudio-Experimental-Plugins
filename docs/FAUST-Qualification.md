# FAUST integration: current decisions and measured evidence

This document consolidates the catalog, Sample/Corpus, export-pruning and perceptual experiments. It describes current decisions, while preserving the useful conclusions of older tests. Read [the guide](DSP-JSFX-Guide.md) for the interface and [the FAUST contract](JSFX-Faust-Sections.md) for execution guarantees. None of these native comparisons is a benchmark against REAPER's own EEL execution engine.

## Current implementations

| Product | Current decision | Evidence |
| --- | --- | --- |
| CMD | Original-design block FAUST under the existing native identity | 1.21–1.45× whole-processor speedup; worst tested difference below 3e-14; [qualification](CMD-Original-FAUST-Integration.md) |
| EasyExpander Faust | Separate native detector/gain example | Matched continuously active full-recording comparison: 15.821 s original / 3.476 s FAUST, 4.55×; 58,558,936 float values byte-identical |
| ERB Tilt | Native manifest selects a fixed 16-band FAUST source; original EEL retained | One complete recording run: 42.4718 / 15.7052 s, 2.70×; maximum error 5.55e-17 |
| Spectral Stabilizer | Native manifest selects a fixed 12-band FAUST source; original EEL retained | One complete recording run: 66.4130 / 17.7605 s, 3.74×; maximum error 1.86e-9 |
| Sample | Cached EEL remains the default; Sample Faust is optional | Optional variant saves 1.7–8.2% in repeated active whole-plugin tests; [detailed report](Sample-Faust-Character-Integration.md) |
| Corpus | Maintained EEL audio and deferred analysis; FAUST audio candidates remain experimental | No qualified production full-block audio migration |
| Hyperreal default | Original model optimized in EEL; separate FAUST designs exist | [Current variants and matched-model evidence](Hyperreal-Panner.md) |

The spectral-bank comparisons each cover 58,558,936 output samples. These and EasyExpander use the single authorized approximately 610-second recording, matching defaults at 48 kHz/256 and excluding decoding/dumping. They are single full-recording pairs, supported by kernel tests, not statistical full-host medians. Their editor/rate-reset and finite-output checks passed. They do not establish universal bit-exact parity, subjective equivalence or live deadlines. [EasyExpander's sleep audit](validation/EasyExpander-Sleep-Audit.md) explains why active/offline settings matter.

## Sample: exact optimizations and rejected shortcuts

The maintained EEL source caches control-only powers, reciprocal/wetness values and harmonic weights. Cache keys are checked on every use, so live changes invalidate immediately. Audio histories, transistor/diode formulas, guards and arithmetic behavior remain intact.

| Historical whole-Sample cache pair | Before | Cached | CPU saved |
| --- | ---: | ---: | ---: |
| 48 kHz / 64 | 14.0052 s | 13.5866 s | 2.99% |
| 48 kHz / 256 | 6.32017 s | 5.89758 s | 6.69% |
| 96 kHz / 1024 | 8.24378 s | 7.87699 s | 4.45% |

Each row is a single pair producing 32 seconds of stereo output from the authorized bank, with decoding excluded. All 12,288,000 compared values and sampled traces matched. Fifteen separate generated-input kernel cases compared another 15.36 million values exactly; active kernel cost fell approximately 15–23%. The later optional FAUST report uses the already-cached EEL source as its reference, so these savings must not be added arithmetically to its percentages.

Important experiments:

- **External-state EQ bridge:** moving previous/next filter histories through scalar ports each sample regressed whole Sample by 10.6–43.2% in the early fixtures. The imported/exported-state boundary dominated a small kernel; this was not proof that FAUST could not implement the math efficiently.
- **Private three-band block EQ:** twelve extracted cases compared 6,144,000 float values byte-identically. Original private-state histories, disabled-band freezing and quiet clears were represented in the graph. Three-trial medians measured 2.67–4.36×; a subsequent exclusion rerun measured 2.41–6.19× with noisy individual reference times. These are isolated three-band EQ results, excluding HP/LP, voices, analyzers and production denormal streams.
- **Thirteen-filter graph:** monolithic coupled recurrences exceeded the 120-second compilation limit. Independent standard filters compiled in 5.54 seconds, demonstrating a graph-construction problem rather than an intrinsic size limit. However bypass/reset differences reached approximately 0.66 during transitions. That candidate was rejected as a faithful replacement despite close steady output.
- **Diode lookup:** an 8,193-entry asymmetric-table approximation had maximum error around 5.36e-6 but inconsistent/slower performance. It was rejected. The accepted character variant retains the original nonlinear formulas.
- **Character private-state recurrences:** freezing disabled histories corrected early transition errors. Character-only kernels measured 4.49–5.54×; including unchanged harmonics diluted this to 1.28–1.33×. Full integration then preserved original stage ordering, metering and state handoffs, yielding the smaller whole-plugin improvement in the current optional-variant report.

The optional implementation uses explicit block streams, a bounded quantum and conservative EEL fallback for configurations where moving the residual would change ordering/reset behavior. Disabled-character single pairs measured small regressions despite zero FAUST compute calls: staged dispatch still has a cost. Keep it optional.

## Corpus: audio boundaries versus background analysis

The early RAM-owned diffusion bridge was expensive; an isolated private-delay block graph measured up to 1.80×. The complete audio candidate remained sample-fused because voice coverage, continuity, pre-limit peaks and meter bookkeeping require values from the same frame. Simply forcing independent buffer loops would change behavior.

Export pruning reduced a later complete candidate to nine scalar exports plus two audio outputs. Single matched 32-second runs measured:

| Rate / buffer | Original | Pruned candidate | Change in CPU time |
| --- | ---: | ---: | ---: |
| 48 kHz / 64 | 6.83785 s | 6.84080 s | +0.0% |
| 48 kHz / 256 | 2.58341 s | 2.37856 s | −7.9% |
| 96 kHz / 1024 | 2.63112 s | 2.40903 s | −8.4% |

All 12,288,000 normal values and another 3,072,000 clipping values matched as bytes. Both clipping runs counted 3,673 limited samples, pre-limit peak 2.35429 and output ceiling 0.98. Sampled text traces are coarse observations, not all-private-state proofs. The small single-pair gains do not qualify a full-block production migration.

Decimating diffusion targets every eight samples produced errors around 0.226–0.344 and inconsistent gains; it was rejected. The maintained indexing/features/structure/PE arena graph is a separate background-work facility. FAUST audio experiments do not measure its preparation speed.

## Smaller candidates and exports

ADS, SaliencePush and DPT already used full buffers in their early candidate tests; their regressions were not caused by `compute(1)`. Pruning unnecessary history exports reduced ADS outputs from 44 to 17 and SaliencePush from 30 to 9. FAUST materializes signal exports throughout the buffer even when JSFX observes only the final value.

The revised ADS kernel measured 0.94–1.03× and SaliencePush 1.10–1.19×. Numerical differences remained up to 3.94e-6 and 1.38e-5 respectively, and the retry driver did not enforce an acceptance threshold. Neither retry qualifies production equivalence. DPT's earlier headphone candidate was byte-null and about 1.7–1.8× faster, but regressed speaker mode; no mode-dependent FAUST regression was installed.

## Conservative sleep grants

These four scripts have explicit certificates; most plugins do not. Withholding a certificate does not mean automatic modes are unavailable—see [current sleep policy](Cooperative-Sleep.md).

| Script | Required settled state | Measured host savings in silence-heavy fixtures |
| --- | --- | ---: |
| ADS | 36 detector/gain states unchanged across a completely silent source/key block; slider changes invalidate | 62.0–70.9% |
| SaliencePush | 20 detector/gain states unchanged across a completely silent source/key block; slider changes invalidate | 76.9–89.1% |
| DPT | Entire 8192-sample mono history empty; pan/naturalness and active headphone filters at floating-point fixed points | 13.3–62.5% |
| DDT | Entire 16384-sample stereo history empty; six recursive states at exact zero-input fixed points | 41.1–62.1% |

Tests compared continuously active and actually sleeping instances byte-null, including tiny input and parameter wake. ADS can require approximately 26 seconds of default quiet settling. Active-DSP costs ranged from small gains to about 8% overhead for SaliencePush; these idle savings are not active-music/render speedups. No tails were intentionally reset to obtain permission.

## Reproduction and scope

Use `tests/faust/prepare_catalog.py` then `profile_catalog.py` for spectral banks/sleep candidates; `catalog_block_retry.py` for export-pruning experiments; `sampler_corpus_candidates.py`, `revise_sampler_corpus.py` and their profiling drivers for historical boundaries; `perceptual_candidates.py`, `perceptual_alternatives.py`, `perceptual_faust_character.py` and `profile_perceptual.py` for Sample caches/character experiments. Current complete Sample reproduction is in its dedicated report. Test-only EasyExpander identities must not be distributed as real catalog plugins.

Raw logs, hashes and comparisons remain under `docs/validation/` where present, with generated/build artifacts in `build/`. Historical tests use pinned references; rerunning against a modified production source is not the same experiment. Compiler call counts, audio nulls, state observations, editor/reset smoke checks and complete callback timings answer different questions. None replaces listening, exhaustive preset coverage or DAW-host testing.
