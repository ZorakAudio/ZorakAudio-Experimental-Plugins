# Benchmark rerun after the user-added Malwarebytes exclusion

Historical runner checkpoint: later Sample caches and an optional FAUST character
variant supersede the production-status statements below. Current decisions are
in [FAUST qualification](../../FAUST-Qualification.md).

All eight rebuilt Sample/Corpus benchmark executables remain on disk after successful execution. Their current sizes and SHA-256 hashes, plus complete comparison results, are recorded in Benchmark-Exclusion-Recheck.json. Protection settings and executable names were not changed during this rerun.

The sequential rerun completed 42 before/after case pairs: nine Corpus external-state bridge cases, twelve Sample bridge cases, nine Corpus private-delay block cases and twelve Sample private-state EQ cases. Each side uses three timing trials and a separate validation trial. The bridge/block tests passed their finite-audio, audio-error and sampled-double-state thresholds. The private EQ comparison again produced bit-identical finite, nonzero audio across 6,144,000 samples, with zero scalar calls and one FAUST call per processing buffer.

The private EQ timing range this time is 2.41–6.19x. Individual timings vary, including an elevated reference time in one setting; this range should not be treated as a precise end-to-end plugin gain. The consistent result is that the private, block-based EQ remains faster than its extracted EEL reference. Full Sample integration remains unqualified.

No new full-plugin host measurements were performed: those existing host runs had completed successfully and were not the disappearing kernel.exe failures. The thirteen-filter FAUST graph's compiler timeout is a separate build-complexity failure, not a runner disappearance; it was not retried here. Production Corpus and Sample sources still match their pinned hashes. No recording was opened in this rerun; all inputs were generated in RAM.

This confirms that the currently rebuilt runners execute under the user's exclusion. It does not independently determine why Malwarebytes originally classified the files, or establish that every earlier failed attempt completed. Previously failed launches remain failed; the fresh results supersede those attempts.
