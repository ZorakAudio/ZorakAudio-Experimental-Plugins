# Hyperreal Panner: current products and qualification

The default **Hyperreal 3D Panner** uses the original Artistic and fitted-KEMAR Physical algorithms with qualified EEL fixed-offset optimizations. It is not a FAUST conversion. V7.1.2 is local-only: retired manager parameters remain inert for saved-state compatibility. Use the maintained plugin README for controls; this document covers implementation and measured evidence.

| Product | Audio design | Identity/status |
| --- | --- | --- |
| 3DPanner | Original model, optimized EEL | Maintained default, original IDs |
| HyperrealFast | Original-model Fast EEL | Separate compatibility identity |
| HyperrealFaust | Designed FAUST Artistic and Physical voicings | Separate alternative, changes both models |
| HyperrealHybrid | Original Artistic plus designed FAUST Physical | Separate alternative; Physical changes substantially |

## V7.3 source facing and internal head/body POV

Artistic appends Source Facing (slider49, -180..180 degrees), Source
Directivity (slider50, 0..1), Internal POV (slider51, 0..1) and Body Tone
(slider52, 0..1). Defaults and missing native-state values are 0, 0, 0 and .5.
The existing saved-state reseed also seeds their targets, so old states do not
inherit active perspective controls. Previous parameter IDs are unchanged.

Facing is relative to the source-to-listener line: 0 toward the listener,
+/-90 side-on and +/-180 away. This rotates radiation without moving position.
Generic authored three-band lobes have gains `10^(-L * (1-cos(theta))/2 * D/20)`
with L = 1.5/9/18 dB for low/mid/high and Directivity D. Complementary one-pole
bands at 450/2200 Hz colour direct left/right, preserved side and the bed.
Each of the eight image-source wall/corner paths uses its own source departure
vector; vertical cue taps use the horizontal radiator's broadside response.
Radiation gains are smoothed rather than wrapped angles. Original ER histories
remain untouched; a parallel difference carries optional reflected radiation.
There is no added convolution, delay allocation, measured source dataset,
power normalization or automatic makeup gain. These are generic artistic
patterns, not validated human/instrument radiation profiles.

Internal POV linearly crossfades after the external late field and before final
trim to a separate head-locked input path. At 100% all external geometric,
distance, source-facing and room processing is absent. Mono is centered;
Stereo/Bed/Dual retain controlled side. Body Tone reduces side by up to 65%
and blends in a 25 Hz high-pass followed by a positive sum of 700 Hz/2400 Hz
low-pass paths (weights .65/.30) plus .05 of the high-passed signal. Histories
run during its fade; unused paths skip their filters and clear histories at Off.
No resonant boost, anatomy simulation or physical bone vibration
is claimed. Already embedded room/binaural cues cannot be removed by this path.
Physical is unaffected; no heavyweight renderer is added.

Run `python tests/panner/test_source_perspective.py` for synthetic comparisons
against pinned V7.2.2, beaming spectra, direct/reflected departure responses,
internal-path independence, stereo preservation, automation, rates, tails and
trim. Native `tests/panner/test_native.py` also exercises saved-state fallback
and the actual controls. No private audio is used by these checks.

V7.3 qualification passed 71 checks / 125 synthetic renders, including exact
V7.2.2 output at neutral controls across all four source modes, three distances
and Proximity Off/75%; beaming spectra; separate reflected departure responses;
internal independence; stereo/antiphase preservation; rate/partition stability;
automation, trim, silence/tails and unchanged Physical output. Actual native
editor click/wheel/reset, saved-state cold/active restoration and explicit-value
preservation, units, resize and reprepare checks also passed.

A five-trial interleaved offline kernel probe at 48 kHz / 256 processing four
seconds of stereo wet-bed audio measured medians of .202992 s (V7.2.2) and
.215349 s (V7.3 with the new controls Off), a 6.1% increase for that workload.
Unused radiation/body filters skip work. This is a complete DSP-kernel timing,
not a DAW callback, a live deadline guarantee or a general CPU-meter prediction.


Design references: [directional sources and reflected propagation](https://www.microsoft.com/en-us/research/publication/directional-sources-and-listeners-in-interactive-sound-propagation/),
[measured voice directivity](https://pmc.ncbi.nlm.nih.gov/articles/PMC8329840/),
and [own-voice air/bone transmission](https://pubmed.ncbi.nlm.nih.gov/20707445/).
The proposed curves are authored approximations, not fitted results from those
studies. A measured-source profile would be needed for source-specific fidelity.

## V7.2.2 display units

Display Units (slider48, 0 = feet, 1 = metres) is appended as a UI-only
preference. FT / M in the header switches it; right-click selects feet. Both
new instances and older native states missing the preference default to feet.
The selected unit is saved per instance. The conversion is exactly
`display_feet = internal_metres / 0.3048`; selecting metres displays the original
value. No converted length is written into slider7, slider44 or slider45.
Distance mapping, room geometry, timing, automation, drag/wheel response and
all audio processing are unchanged. Range/height/FOLLOW readouts and related
tooltips use the same selection in both renderers.

Native qualification confirmed feet/metres button rendering, click/right-click,
saved metric preference, feet fallback for older states, and preservation of
every prior saved parameter. Audio remained bit-identical while switching units
live across all four Artistic source modes and near/far distances. The existing
legacy-state, editor, resize and reprepare checks also passed. The @slider,
@block and @sample code is identical to V7.2.1.


## V7.2 Artistic proximity additions

V7.2.1 starts new native instances at Artistic Proximity 75% (slider46)
and Distance Anti-Salience 65% (slider47). Explicit saved values always win.
Older states missing slider46 restore it to 0%; missing slider47 restores 65%.
The native host applies this before APVTS replacement and requests an audio-thread
reseed before the first block, so an old state loaded over an already active
instance does not briefly retain the new processing. Resaving persists those
explicit legacy values. The source opts into this via #NATIVE_DEFAULT metadata;
other plugins without the metadata retain their defaults and state behavior.
Standalone JSFX retains its 0% declaration because REAPER does not execute the
native host migration. Existing parameter IDs and distance values are unchanged.

Native qualification passed factory/default queries, old-state restoration before
prepare and over an actively processed distant stereo bed, exact legacy audio
from the first block, resaved legacy Off over an active instance, explicit Off
and custom value preservation, resave/reopen, plus editor interaction/resize/
reprepare. The DSP implementation is unchanged from the V7.2.0 qualification.
The panner's anti-salience processing is separate from Anti-Salience MX; it uses
no foreground reference, persistent eight-band anchor memory, modulation
de-binding, scatter network or level-lock restoration. MODEL exposes
the two new controls when Artistic is selected; Physical keeps its previous
controls and audio design. The separate generated variants were not changed.

Proximity has bounded geometric ear contrast below 1 m and a smooth log-range
far weight from 1 to 10 m. At full strength and maximum range the extra direct
attenuation is 6 dB. Existing reflections remain; preserved side content takes
the same extra direct attenuation, while the bed additionally follows the
existing direct distance law instead of retaining its previous loud bypass.
Complementary broad bands reduce proximity body, presence and air. There is
no generated room, feedback or gain restoration in the added stage.

Source energy is measured from both input channels before summing, so stereo
cancellation cannot blind density/onset detection. A shared detector uses the
louder Artistic output ear for RMS restraint. The 8 ms/250 ms source followers,
35 ms gain attack and 700 ms release keep their time constants across sample rates. The RMS mix
reference falls from 0.18 to 0.045 linear amplitude as the source recedes, with
a maximum 12 dB attenuation before the two controls weight its application.
The smoothed enable also weights the gain after its release so disabling
cannot leave a residual gain step at exact bypass. Output Trim remains last.

Run `python tests/panner/test_proximity.py` for synthetic complete-Artistic
DSP qualification. It compiles both the retained V7.1.2 reference and the
maintained source using the same compiler; no private recordings are loaded.
The suite covers exact legacy bypass, all four source modes, monotonic distance,
dense low-crest textures, low/mid/high tones, impulses, modulated material,
wet stereo, antiphase and one-sided sources, silence, low levels and overload,
rate changes, block partitioning, enable/distance automation, tail drain,
final trim and near-ear contrast. Numerical checks are not listening validation
or a guarantee that every distant sound will externalize for every listener.
The original measurements below describe the unmodified original-model path.

After a native Windows build, `python tests/panner/test_native.py` checks the
actual editor controls, wheel/reset gestures, parameter persistence, resizing
and sample-rate reprepare. It restores the temporary CMake probe configuration.
Synthetic qualification for this revision passed 57 checks / 92 renders;
the actual native editor/state/resize checks also passed. The full-scale
dense texture at maximum range was 11.2–12.2 dB lower than the
original in Mono/Stereo/Dual and 20.2 dB lower in Bed, with full Proximity and
65% Anti-Salience. These are measured level changes, not measured subjective
distance. See `build/panner-proximity/results/report.json` after a run.

Design context: [spectrum and distance experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC5479918/)
and [nearby-source spectrum experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC3188969/).
These motivate combining cues; the particular control curves are artistic choices.

## Original-model optimization

Fixed five-coefficient updates, six-filter cascades and the 28-ear-path loop use constant offsets with preserved arithmetic/state-update order. All 168 fitted filters, geometry cadence, four-point interpolation, propagation, smoothing and original late field remain. Native builds use Legacy graphics. The pre-promotion source is preserved in `tests/faust/reference/3DPanner-before-fast.jsfx`.

## Fresh promotion results

All **19 paired workloads** passed a direct comparison of the complete PCM bytes: **16,896,000 float output values were bit-for-bit identical**, including signed-zero representation. Coverage includes Artistic and Physical, stationary/moving scenes, Stereo/Bed/Dual, elevation/occlusion/travel with late field, renderer/source/late transitions, and directional probes at 48 kHz / 256; Artistic/stationary/moving cases also ran at 48 kHz / 64 and 96 kHz / 1024. Every tested output was finite. Editor, state restore and reprepare received lifecycle smoke checks outside timing. There were zero FAUST calls in either implementation.

The following fresh timings are medians of three interleaved paired trials, measuring CPU seconds for eight seconds of stereo output at 48 kHz / 256. No compilation ran concurrently with timing.

| Physical scene | Pre-promotion seconds | Promoted seconds | CPU saved |
|---|---:|---:|---:|
| Stationary | 4.365540 | 4.109970 | 5.9% |
| Moving controls | 7.659060 | 6.264420 | 18.2% |

These results verify the tested workloads; they are not an exhaustive proof for every possible input or setting. The source change preserves the arithmetic sequence and all filter/path processing, making it an actual execution optimization rather than a model substitution.

Build only the default product with `python scripts/build.py --only "Hyperreal 3D Panner"`. The shorter `3DPanner` filter also matches the historical manager product.


## Realtime idle and host parameters

The default source uses `idle=input idle_hold_ms=1000 idle_out_db=-140`; permanent Free-running/keep-awake overrides were removed. The separate Hybrid source still requests free-running histories. Do not assume every variant inherits default idle behavior.

## Why last touched was unreliable

The CLAP wrapper initialized note, port, channel and key fields to zero for UI
parameter-value events. Global parameter changes require -1 in those fields.
UI gestures also lacked CLAP's live flag. Meanwhile a legacy JSFX
`sliderchange(-1)` could produce gestures for every slider, including unchanged
ones, obscuring which control the user actually touched.

The wrapper now targets global parameters correctly and marks main-thread UI
value/begin/end events live. Audio-thread notifications are not marked live.
Native canvas change masks are filtered against actual before/after slider
values, using the processor's existing parameter comparison rules. Explicit
slider automation masks retain their behavior.

The real CLAP wrapper test exercises a legacy all-slider button request for
Render Model through the actual processor notification path. It receives exactly
one ordered live begin/value/end sequence for Render Model, with global targeting
and the correct value. Repeating the unchanged request produces no events.
This verifies host-facing events, not REAPER's actual Last Touched UI; that
integration still requires a manual REAPER retest.

## Validation and measured idle cost

Tests use generated in-memory audio; no recording is opened for idle or CLAP
notification checks. At 48 kHz / 256, both Artistic and Physical passed:

- Paused silence sleeps and stops advancing the DSP sample counter.
- An empty track while transport runs sleeps too.
- Tiny input and parameter changes wake processing.
- An active late-field tail is not cut immediately; the tested tails reach sleep
  after approximately 2.64 seconds (Artistic) and 2.50 seconds (Physical).
- An unchanged open canvas permits sleep and does not repeatedly wake DSP.
- Never Sleep, saved override restoration, and resetting Auto work.
- Offline processing continues through silence.

A clean timing with no compilation running measured 1,000 silent audio callbacks:

| Renderer | Never Sleep | Automatic idle | Callback time saved |
|---|---:|---:|---:|
| Artistic | 0.360354 s | 0.088533 s | 75.43% |
| Physical | 2.941510 s | 0.085077 s | 97.11% |

These are callback timings, not REAPER's CPU meter or total application CPU.
Host bookkeeping and canvas rendering still consume time; zero CPU is not
promised. The test checks complete callbacks rather than timing only the kernel.

The separate cooperative regression passed permission expiration, tiny-input and
parameter wake, buffer-size changes, keep-awake veto, pending/unacknowledged task
veto, result release, restored mode/state behavior, and offline activity.

Five offline output scenarios at 48 kHz / 256 compare against saved original
qualification dumps: Artistic, stationary Physical, moving Physical, late-field
and cue combinations, and renderer/source/late transitions. All 3,840,000 float
values matched as raw bytes, including signed zero. These are eight-second
workloads using excerpts from the single authorized recording, not a complete
ten-minute render. No other recording was opened.


## Alternative Physical model

These alternatives trade algorithms, not just execution backends. Original Physical uses six fitted KEMAR bands per ear path, geometry-derived early paths and its original late network. Designed FAUST Physical uses head-shadow/air/occlusion poles, comb pinna cues, six early taps per ear and eight damped feedback combs. Reflection timing, timbre, elevation/front-back cues, movement, late tails and startup differ. There is no measured-HRTF or listening-equivalence promise.

Hybrid retains original Artistic and captures raw audio/model-blend streams for one full-buffer FAUST pass. Both engines stay warm, including disabled late returns. That adds Artistic overhead and can reveal recent excitation when a return is enabled. Moving delays may bend pitch. Delay bounds are 65534 propagation samples, 16382 early samples and 126 pinna samples, making extremes rate-dependent. All-FAUST changes Artistic as well.

## Whole-plugin performance

CPU seconds for eight seconds of stereo output, median of three interleaved original/hybrid trials per row. Only the complete JUCE processBlock call is timed; decoding, parameter setters, file dumps, setup and editor checks are excluded. Offline execution, same AOT optimization and Legacy editor backend. No concurrent build or profiling ran during the main timing trials. Excerpts come only from the explicitly authorized ten-minute recording.

| Rate / buffer | Scene | Original seconds | Hybrid seconds | Result |
|---|---|---:|---:|---|
| 48 kHz / 64 | Artistic stationary | 0.913841 | 0.932551 | 2.0% more CPU |
| 48 kHz / 64 | Physical stationary | 4.637360 | 0.929726 | 4.99x faster; 80.0% CPU saved |
| 48 kHz / 64 | Physical moving | 7.980930 | 0.974642 | 8.19x faster; 87.8% CPU saved |
| 48 kHz / 256 | Artistic stationary | 0.530695 | 0.585895 | 10.4% more CPU |
| 48 kHz / 256 | Physical stationary | 4.264110 | 0.606245 | 7.03x faster; 85.8% CPU saved |
| 48 kHz / 256 | Physical moving | 7.553110 | 0.627188 | 12.04x faster; 91.7% CPU saved |
| 96 kHz / 1024 | Artistic stationary | 0.874777 | 0.985695 | 12.7% more CPU |
| 96 kHz / 1024 | Physical stationary | 8.352520 | 0.989699 | 8.44x faster; 88.2% CPU saved |
| 96 kHz / 1024 | Physical moving | 14.684400 | 1.060880 | 13.84x faster; 92.8% CPU saved |

The hybrid is slower than the all-FAUST instrument because it keeps the original Artistic engine running. It gains 5.0–13.8x over the original Physical workloads tested here, at an Artistic overhead of 2.0–12.7%. This overhead is the price of independent, continuously current histories and a simple renderer crossfade. These offline timings do not guarantee live callback deadlines or multi-instance capacity.


The historical hybrid qualification compared 12,288,000 values in 13 Artistic pairs with maximum difference zero; finite output, transitions, directional probes, editor/state/reset smoke checks and 36 control-response checks passed. This does not establish subjective Physical quality or exhaustive UI/preset coverage.

## Same-algorithm EEL versus FAUST

The designed Physical graph was also translated into optimized ordinary EEL and compiled through the native pipeline, with the same original Artistic engine on both sides. Fixed two-value histories use EEL scalars; large delay buffers remain arrays. This avoids penalizing EEL with an unnecessarily literal array translation. Both use double DSP and matched control sequences.

## Complete-plugin CPU time

CPU seconds to process eight seconds of stereo output. Each row is the median of three interleaved trials, reversing execution order between trials. Timings surround the complete JUCE processBlock call. Decode, setup, parameter setters, file dumping, and editor/lifecycle checks are excluded. No build or competing profile ran concurrently. Both builds use the same AOT optimization level and Legacy editor backend. The sole authorized recording supplied six-second excerpts followed by two seconds of silence; generated directional probes were additional tests.

| Sample rate / buffer | Physical scene | Optimized EEL2 seconds | FAUST seconds | FAUST execution improvement |
|---|---|---:|---:|---:|
| 48 kHz / 64 | Stationary | 1.550940 | 0.907210 | 1.71x; 41.5% CPU saved |
| 48 kHz / 64 | Moving controls | 1.577340 | 0.940405 | 1.68x; 40.4% CPU saved |
| 48 kHz / 256 | Stationary | 1.139750 | 0.566258 | 2.01x; 50.3% CPU saved |
| 48 kHz / 256 | Moving controls | 1.175760 | 0.610315 | 1.93x; 48.1% CPU saved |
| 96 kHz / 1024 | Stationary | 2.044950 | 0.974025 | 2.10x; 52.4% CPU saved |
| 96 kHz / 1024 | Moving controls | 2.137360 | 1.051520 | 2.03x; 50.8% CPU saved |

The measured whole-plugin improvement is **1.68–2.10x for the same algorithm**, rather than the larger different-model comparison. Keeping the unchanged Artistic engine in both measurements dilutes the Physical-only gain. No subtraction-based estimate is presented as a measured isolated-renderer result. Offline CPU totals do not guarantee live callback deadlines or multi-instance capacity.

## Correctness evidence

- **25 paired workloads**, covering stationary/moving Physical at three rate/buffer combinations, original Artistic selection, Stereo/Bed/Dual, wet late field, renderer/source/late transitions, and left/right generated probes.
- **23,808,000 stereo float output values compared**; largest absolute difference **1.8189894035458565e-12**, largest workload RMS difference **2.0766414267292749e-15**. The acceptance thresholds were established before running: maximum error below 2e-6 and RMS below 2e-7. Actual differences are vastly smaller.
- FAUST processed exactly one full host buffer per callback, all frames, with zero scalar calls. The EEL reference made zero FAUST calls.
- Every run checked finite/nonzero output and processing memory faults, then editor/state/reset lifecycle outside timing. These are smoke checks rather than exhaustive UI or live-host qualification.
- The scalar-history optimization and literal-array version both passed the same 25-pair numerical checks. The first reference comparison also produced a maximum difference of 1.82e-12.


Typed private histories and generated buffer loops can avoid repeated guarded EEL memory/store operations, but this benchmark does not attribute a percentage of savings to each mechanism. It measures native execution pipelines, not interpreted REAPER JSFX against FAUST. Further EEL optimization can narrow the difference.

## Build, reproduction and limits

Build the default with `python scripts/build.py --only "Hyperreal 3D Panner"`; the shorter `3DPanner` filter also matches its historical manager. Separate product filters are `HyperrealFast`, `HyperrealFaust` and `HyperrealHybrid`. Native FAUST variants need the FAUST LLVM compiler to build, not to play back.

Default promotion tests: `tests/faust/profile_panner_promotion.py`. Alternative generator: `panner_candidates.py` with `panner_renderer.dsp`; hybrid timing: `profile_panner_hybrid.py`; same-model EEL translation and comparison: `panner_equivalent.py` and `profile_panner_equivalent.py`. Requalify regenerated sources against their pinned references.

Windows release CLAP/VST3 packages were built under the appropriate identities. Package integrity, qualified-source identity and actual CLAP load/parameter/finite-audio shutdown smoke checks passed. VST3 was not independently loaded in a VST3 host. No timing is a REAPER render, a universal deadline, or a listening test. No package was installed or published by these qualifications. Older comparisons use pre-promotion references; the default has since received the idle/notification fix.
