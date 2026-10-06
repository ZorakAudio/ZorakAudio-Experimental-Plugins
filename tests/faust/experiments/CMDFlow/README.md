# CMD Flow

CMD Flow is a separate redesign of Cross-Mix Declutter. It retains cooperative
roles, shared masking decisions, TurnPulse, subtle width motion and bounded
somatic coloration. The audio engine uses twelve complementary broad bands and
runs once per full host buffer in FAUST. It does not emulate the original ERB
filterbank, and sounds different when processing is active.

Place an instance on each related track. Use the same Bus Name to connect them;
choose each track's Role, then raise Manifold for decluttering and Somatic for
peer-driven coloration. Safety Governor restrains changes. The twelve-band
layout is fixed; the old adjustable ERB-band control is hidden.

Flow automatically prefixes its shared-memory namespace, so original CMD and
Flow do not exchange incompatible data even with matching Bus Names. Instances
within a Flow bus use the existing block-level coordination protocol; decisions
are best-effort and depend on processing order, not sample-synchronous promises.

At neutral Manifold and Somatic settings, the complementary band sum reconstructs
the dry signal, with floating-point rounding. Band gain and width/coloration
transitions are smoothed. Audio is emergency-bounded to +/-8, not loudness
normalized. Start with conservative settings and normal output levels.

The UI band gain meters are block-smoothed target estimates; FAUST filter and
gain histories remain private. Energy summaries are 10-ms followers sampled at
block boundaries instead of exact original block sums. Breathing modulation is
updated per block. These changes are deliberate redesign tradeoffs.

Native builds require this project's FAUST section compiler; stock REAPER JSFX
does not support @faust. The original CMD remains available.

Flow keeps processing through silence because peer publication, expiry and
TurnPulse are ongoing obligations. No silence-only sleep certificate is claimed.
