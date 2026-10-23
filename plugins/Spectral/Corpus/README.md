# Corpus

Corpus loads recordings, learns their acoustic structure, and plays new paths through them from MIDI notes.

## Quick start

1. Insert the current Corpus VST3/CLAP and open **Corpus sources**.
2. Select your recordings and wait for preparation to finish; the host must keep processing the instance.
3. Send MIDI notes and start with the default **Balanced** analysis and **8192** grain budget.
4. Change Force target, Variation and Coherence to steer selection; use the map and browser to inspect the learned material.
5. Use **Reload Analysis** to rebuild. Changes marked **Apply** need a new analysis pass.

## Controls and playback

The custom canvas owns the hidden slider parameters and their saved/automation values. **Force target**, **Variation**, **Coherence** and **Detail density** guide traversal; **Source scope**, **Identity anchor** and role/descriptor filters restrict eligible material. Map descriptor axes, focus and drawn paths steer where it travels.

**MIDI lifecycle** selects One-shot, Gated or Natural. Attack/Decay/Sustain/Release shape playback; Pitch changes generated tuning, Velocity influence scales note response, and Output is the final trim. Analysis quality and grain budget trade preparation detail for work and memory. Pool RAM budget controls residency. Changing source/configuration cancels obsolete analysis rather than publishing it as the current model.

## Child cloud

The cloud is a single responsive section in Explore and Controls: a companion
count, visible mode cards with explanations, a plan preview, and Level, Grain
Density, Stereo Width, Similarity and Timing Spread. Gold marks the primary;
green marks independent companions. The preview shows settings, not live grain
positions. Zero companions leaves the primary alone. Children never recurse.

Count (76), Mode (77), Density (78) and Timing Spread (79) are now real host
parameters. Level (37), Similarity (38) and Width (53) keep their existing IDs.
Presets, copy/paste, project recall and automation use ordinary parameter state;
they no longer depend on private cells/custom serialization. New notes capture
the complete cloud plan, including width and similarity. Existing clouds retain
their captured settings. Level is compensated by 1/sqrt(child count), with a
small mode-dependent trim; it is not a loudness guarantee for correlated grains.

| Mode | Actual behavior |
| --- | --- |
| Follow | Independent grains following the same structural movement; softer attacks and a body bias. |
| Resonance | Favors LF weight, body and tails; adds no resonator or oscillator. |
| Debris | Favors motion, crest and detail at a lower energy target; does not guarantee short grains. |
| Contrast | Alternates brightness, weight and motion offsets across companions. |
| Shadow | Favors calmer, lower-energy body/tail material with fewer attacks; adds no low-pass or pitch shift. |
| Body bed | Formerly Fill gaps: denser body/detail overlap, not a literal gap detector. |
| Mixed | Cycles Follow, Resonance, Body bed, Debris, Shadow, Contrast; one child is Follow. |

All modes are soft selection biases. Available recordings, scope, identity
locks and hierarchy still determine what can be realized. Siblings discourage
exact/near duplicate grain choices but can fall back in a tiny corpus.

Fresh instances start with zero children. Older native states missing Count
restore an internal -1 sentinel, retaining legacy Morphology's one/two-child
plan until an explicit count is chosen; explicit zero always disables it.
Missing Mode/Density/Timing restore Follow/60%/20%. The source opts into the
existing #NATIVE_DEFAULT host rules; no shared runtime behavior is changed.
Private child choices from an older native state were never stored and cannot
be recovered if already lost.

## Transient smear

**Smear character: Grain** uses eight overlapping, Hann-windowed micro-grains
with independently chosen causal offsets. Their read speeds stay at unity;
phase scattering comes from changing excerpt offsets at silent window edges,
not pitch modulation. Windows grow from about 7 to 40 ms and offset spread from
0.3 to 95.3 ms. Grain timing, weights and linked gain are identical for L/R:
true mono remains mono, and the renderer adds no stereo decorrelation.

The amount retains parameter45. The lower range is deliberately gentle;
above the middle, transient softening grows into a continuous granular haze.
An independent deterministic random stream leaves source choices untouched.
Short-term RMS compensation is bounded from 0.5 to 2; a linked soft guard catches
cloud peaks without the old dry-peak clamp erasing delayed grains. The scene's
active voice end times cap the cloud with a short final fade. No feedback or
unbounded tail is used. Zero is an exact bypass.

**Classic** retains the former three short stereo-linked all-pass sections and
crest guard. Dry/all-pass phase interference can sound comb-like even in mono;
the classic algorithm itself uses identical channel coefficients. A mono
source can still produce stereo earlier in the engine when child Width is on.
Smear character saves as parameter80. Fresh instances use Grain; old native
projects restore Classic until changed. Both options are visible in Controls
and Explore, alongside the amount.

This is entirely Corpus DSP: fixed buffers, bounded eight-head reads, no new
native builtin/library, FFT or audio-thread allocation. The delay/window arena
ends at cell24076624, below the25165824-cell heap. Seven adoption-preservation
ranges total207904cells, below the262144-cell runtime limit.

## Duration CC envelope timing

**Duration CC ADSR: Relative** (the default) makes Sampler MIDI Hub's default
Instrument envelope follow the requested event length. Attack, Decay and
Release keep their proportions of Corpus's **One-shot** window. For example,
the default 800 ms window with 5/100/180 ms stages becomes 25/500/900 ms for a
4-second request. Sustain and the three curve shapes remain unchanged.
Ordinary MIDI notes and Generate retain their existing timing.

Choose **Absolute** to keep inherited stages in milliseconds. The choice is
available with the envelope in Explore and Controls and saves as host
parameter 75. Hub's explicit Attack/Release percentages override either mode;
Decay fits the remaining body without shortening a requested Release. If
Attack and Release together exceed a Fit window, they shrink proportionally.
Fit ends at the declared deadline; Free and Tail start Release there and may
finish afterward. On-time End packets do not cut off that tail. Continue
retains its family's timing choice and instrument release reference, resizes
relative Release for its new window, and preserves already-scheduled samples.
Custom amplitude curves already traverse the event duration independently.

## Per-grain taming

**Grain Tame** in Grain Texture works before summation. At each new voice,
cached RMS and peak measurements across the chosen excerpt produce a fixed,
attenuation-only gain. RMS uses a soft 6 dB knee and 4:1 ratio above 0.16;
peak evidence independently trims toward 0.9. The estimate visits at most
64 cached tiles and reads no PCM. It never boosts quieter recordings.

A gentle stereo-linked soft peak guard catches spikes missed by the probes.
It is linear below 0.7 and approaches 1.0 at full strength, before the grain's
window and event envelope. It has no pumping attack/release envelope and keeps
the stereo balance. Zero bypasses both stages exactly; 100% is the new default.
The amount is frozen per voice and saved as host parameter 74. REPEAT retains
the already-reduced gain and original guard amount without applying gain twice.
Auditions are unchanged. The final peak catch still handles summed overlaps.

## Pitch behavior

**Tape / Coherent** appears beside the performance controls in Controls and at
the top of Explore. Tape preserves the original speed-linked pitching: −24 st
plays each excerpt four times longer; +24 st plays it four times faster.
Coherent keeps each selected excerpt's nominal duration and movement while
transposing its internal audio, so the scheduler, child timing and source joins
continue to follow the recomposition. Both modes use the same pitch slider.
Tape is the compatibility default; no reanalysis is required to switch.

Coherent now uses a prepared native stereo spectral engine per voice. Rather
than multiplying every frequency by the full pitch ratio, it transposes the
low/mid body and smoothly reduces the shift toward the upper noisy spectrum.
Cached tonal/periodic confidence widens the conventional pitch range for tonal
grains; rough/noisy material and attacks retain more upper texture. This is a
perceptual pitch effect: upper frequencies do not all move by the full interval.

The engines preserve phase relationships across time, frequency and stereo.
Short windows handle noisy grains; longer windows stabilize confident tones.
Both are allocated, configured and warmed during processor construction.
Playback does bounded native spectral work with no allocation or PCM search.
Loaded-source lookahead compensates spectral latency while staying strictly
inside the selected excerpt; no MIDI output delay is added. Zero pitch remains
an exact bypass. Spans shorter than 12 ms retain Tape rendering.

Pitch/character are latched per voice. REPEAT restores them and the natural
duration even after live controls change. Source auditions stay original.
Host parameter 73 remains Tape / Coherent; no extra tuning controls are needed.
The native reader uses pinned MIT-licensed Signalsmith Stretch 1.3.2 and
Signalsmith Linear 0.6.4 (see the license files supplied with the build).

## Curved ADSR

Explore and Controls include an envelope preview and independent **Attack curve**,
**Decay curve** and **Release curve** controls. Drag from **Fast** through **Linear**
to **Slow**; right-click restores Linear. Sustain remains a level. Default curves
preserve the former linear ADSR. Each new event captures its curves, including
causal early release and Continue snapshots, and REPEAT restores the recorded
settings. These are host parameters 70–72; changing them needs no reanalysis.
ADSR → Curve includes the selected curvature when baking the freehand shape.

## Source channels and preparation recovery

The **Source channels** switch at the top (also in Controls) selects **Original**
or **Mono** for every loaded recording. Mono averages left/right before grain
panning; the original files and host sample pool are unchanged. Playback glides
to the setting, and analysis rebuilds automatically after existing events finish.
Child-cloud spread can still create stereo output from mono sources. The mode
is saved/automated as host parameter 69 and defaults to Original.

Preparation detects failed, cancelled and invalid task handles, retires the
obsolete graph, and automatically rebuilds through the incremental fallback.
Transient clone contention retries before indexing starts; it cannot enter half
of the incremental pipeline and strand Features behind the graph gate.
A failed writer can no longer leave the progress display waiting for a final
dependent task that will never succeed. Recovery counters remain available in
the bounded analysis snapshot for diagnosis.

## Grain texture

In **Explore** or **Controls**, choose **Rounded** for a softer grain texture or
**Soft rise** for a slower entrance. **Softness** broadens the individual grain
fades, independently of the event ADSR/MSEG. Softer windows automatically use
closer grain spacing; Flow/Overlap can add density. **Grain Balance**, under
Controls → Advanced → Map / Renderer, optionally moves the peak earlier/later.

**Overlap Level** offers **Energy** for noise-like material, **Amplitude** for
correlated repeats, and **Off** for the original summation. Compensation is
bounded and applied within each walker, preserving separate notes, child-cloud
levels, solo-grain fades, and event release. It does not replace output headroom
or the safety peak catch. Start with Rounded, 65% Softness and Energy.

The four controls use new host parameters 65–68, supporting presets, copy/paste
and automation without custom serialization or reanalysis. Changes affect future
grains; REPEAT captures the grain windows and overlap mode it recorded. Choose
Original and Off to restore the previous grain renderer. Source auditions retain their
original short edge fades. Transient Diffusion remains a separate control for
transients inside the selected audio.

## Background preparation

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

`python tests/tasks/test_corpus_cloud.py` compiles the actual cloud/smear functions
in Native Legacy mode. It checks true mono, linked scaled stereo, exact zero,
five sample rates, gentle low amounts, strong high amounts, diffuse impulse
energy, end gating, per-event cloud snapshots, Mixed ordering and legacy Off.
The supplied-file profiler's `--check-cloud` checks project state/migration,
all seven actual modes, true-mono render isolation, selection-plan identity,
and writes a four-amount listening comparison. `--cloud-screenshots DIRECTORY`
captures the native cloud and smear controls.

`python tests/tasks/test_corpus_grains.py` compiles the actual grain functions
and mixer in Native Legacy mode and checks window bounds, overlap continuity,
note isolation, event/stop release, original rendering, source auditions,
recorded replay, mono fold/glide, and exact parameter notifications above slider64.
It also verifies Tape/Coherent duration and transposition, stereo linkage, source
bounds, exact zero-pitch bypass, cached excerpt gain, attenuation-only behavior,
soft-guard continuity/bounds, independent quiet voices, and replay snapshots.
It loads no recordings.

`python tests/tasks/test_corpus_preparation.py` exercises successful, failed and
cancelled arena writers against the actual coordinator and task runtime, including
retirement, busy-clone retries, empty banks and invalid-handle detection. `python tests/tasks/test_corpus_triggers.py`
checks MIDI/Generate seeding and velocity targets, plus curved ADSR, early release,
per-event curve snapshots and Continue history.
`python tests/tasks/test_corpus_duration.py` compiles the actual CC decoder and
ADSR timing functions. It checks default Instrument and explicit percentages,
Relative/Absolute, four sample rates, short/long windows, all three boundaries,
Continue snapshots, on-time End, checksum rejection and ordinary-note timing.
The supplied-file profiler's `--check-duration` checks actual audio deadlines,
audible releases, Continue and waveform-exact REPEAT after changing controls.

`python tests/tasks/test_tasks.py` covers compiler restrictions, private memory,
slider snapshots, writer ordering, worker snapshots, heap adoption, cancellation and publication in ordinary
and Legacy modes. `tests/tasks/corpus_profile.cpp` profiles the real JUCE
processor with a caller-supplied file. Its `--dump-model`, `--reload-at-index`, `--reload-at-pe`,
`--reload-at-copy` and `--check-playback` options verify model equivalence,
cancellation and usable playback without loading any additional recordings.
`--check-mono` verifies both source modes rebuild and Mono metadata/width are consistent.
`--compare-triggers` writes repeated-MIDI/Generate comparison data beside the CSV;
`--screenshots OUTPUT_DIRECTORY` captures the actual native ADSR UI at two widths.
`--check-pitch-tame` compares both pitch modes and taming on/off across −24 to
+24 semitones, stresses multiple notes, and checks exact waveform replay after
changing live controls. `--pitch-screenshots OUTPUT_DIRECTORY` captures the
actual native pitch and Grain Tame controls.

`python tests/tasks/build_corpus_graph.py` builds that supplied-file profiler;
add `--package` to build production VST3/CLAP instead. Coherent rendering uses
native spectral processing and costs more than Tape, particularly with dense polyphony.
The eight-note stress test can exceed the audio deadline at256frames/48kHz;
normal tested single-note Coherent averaged about0.55ms per256-frame block.
See `docs/Spectral-Sample-Pitch.md` for the native API and ownership rules.

`python tests/tasks/build_corpus.py --package` uses a separate generated fixture
for general repository regression testing. Do not use that command when a run
is restricted to a specific input recording. The supplied-file profiling and
production build can be performed separately with that fixture disabled.
