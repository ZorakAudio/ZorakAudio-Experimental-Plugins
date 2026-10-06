# Hyperreal Panner Fast

A compatibility identity for the optimized Hyperreal V7.1.2 renderer, now also the default implementation of 3DPanner. Its Artistic and fitted-KEMAR Physical models, original filters, delay interpolation, geometric update cadence, smoothing and room/tail behavior remain.

The change specializes the fixed 28-ear-path rendering loop, each six-filter cascade and each five-coefficient update with constant offsets. The order of additions and state updates is preserved. This version remains EEL; it is not advertised as a FAUST conversion. The separate [Hyperreal FAUST Panner](../HyperrealFaust/README.md) is the new FAUST design.

The canvas and controls work as in the current 3DPanner source. That source is local-only; the older 3DPanner README's manager-link description does not apply to V7.1.2. 3DPanner now contains these same optimizations under its existing identity; this separate identity is retained for sessions that already use HyperrealFast.

Build with `python scripts/build.py --only HyperrealFast`. Regenerate from the maintained 3DPanner implementation using `python tests/faust/panner_candidates.py`, then rerun qualification. The checked generator is the maintained patch; avoid hand-editing the generated copy.

See [the audit](../../../docs/Hyperreal-Panner-Variants.md) for complete-plugin timing, null comparisons and limitations. Fixed-offset specialization reduces overhead; it does not remove the original Physical renderer's substantial workload.
