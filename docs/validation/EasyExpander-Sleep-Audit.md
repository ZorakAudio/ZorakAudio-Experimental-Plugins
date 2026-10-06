# EasyExpander source comparison and sleep audit

The supplied source and the earlier repository baseline have identical generated
LLVM. Differences are help/tooltip comments and the final newline, with no change
to executable DSP. The Faust example now keeps the supplied EEL text.

Previous threshold-based Auto Sleep was not equivalent to active processing:
454 blocks reported sleeping; 1,099,813
output samples differed; peak error 0.000888076
(about -61.0 dBFS).
890,337 differences were in blocks reported
awake, consistent with detector/gain state frozen during sleep. The first
difference was at frame 1384448 (28.843s).
Offline processing had the same failure before the fix.

At this historical checkpoint the host used cooperative-only sleep. The current
host restored automatic modes alongside cooperative permission; saved selectors
work again. Default/Auto uses cooperative permission when `za_sleep_ready` is
declared, and automatic option/topology policy otherwise. Offline processing
always advances DSP. The previous threshold recovery discrepancy below remains
useful evidence; it does not prove current automatic realtime sleep byte-null.
See [current sleep policy](../Cooperative-Sleep.md).

All 25 final null comparisons passed on the full recording and
the quiet/recovery fixture derived from it: zero changed samples, including all
legacy selector values in realtime and offline modes. Cooperative fixture checks
also passed fresh/stale/no-grant, tiny input, parameter wake, keep-awake, retained
task result, serialized-state migration and offline processing.

## Matched continuously active comparison

| Processor | Processing time |
|---|---:|
| Supplied EasyExpander, AOT | 15.821 s |
| Matching EasyExpander Faust | 3.476 s |

4.55x faster, 78.0% less processing time. All 58,558,936
float output samples bit-identical. Default sliders, 48 kHz stereo, 256-frame
blocks, offline and zero slept blocks on both sides. Editor/rate-reset checks
passed. This is one paired full-recording run; kernel results use three-trial
medians across 48/96 kHz and 64/256/1024 frames.

These measure built processors, not a timed REAPER render or a comparison to
REAPER's native EEL JIT. Decoding and output dumping are excluded from processing
time. Full render speed includes those and other host costs. Earlier bit-identical
results compared two builds using the same Auto Sleep policy, so they did not
establish equivalence to continuously active DSP.

Rebuild VST3/CLAP to receive the current policy. Other plugins must be
rebuilt; already-loaded/older binaries retain their previous sleep behavior.
The authorized recording was only read; no other audio input file was loaded.
