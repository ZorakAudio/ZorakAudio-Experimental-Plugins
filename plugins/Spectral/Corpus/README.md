# Corpus

Corpus loads recordings, learns their acoustic structure, and plays new paths through them from MIDI notes.

## Grain detection styles

**Grain Detection** is always in the main header, beside Source channels at
wide sizes and on its own row at compact sizes. All three styles are saved,
automatable, and rebuild analysis after current events finish. Existing projects
and fresh instances default to **Auto**, preserving the former behavior.

| Style | Detection and playback |
| --- | --- |
| Aggressive | Shorter duration targets and more sensitive local change cuts. Playback uses the detected unit itself. |
| Auto | Existing variable Micro inference, adaptive region expansion and playback ceiling. |
| Coherent | Sustained pauses and persistent acoustic changes close longer passages. Brief dips do not force cuts, and there is no fixed duration limit by default. Playback preserves the detected region. |

Coherent exposes **Boundary Sensitivity** and **Limit / ms (0 = None)** directly
under the header. Higher sensitivity allows more cuts. The optional limit uses
a logarithmic millisecond control, with fine Shift-drag and Alt-wheel adjustment.
Its cuts follow the internal sensor grid. For old preset and automation compatibility,
the host parameter retains its original seconds domain; the editor converts to ms. A zero limit allows a steady
source to remain one passage, even beyond several seconds. The source end is
always a boundary; different files never merge. A limit splits an overlong region
into successive grains; it does not discard the remaining audio.

These are acoustic styles, not sentence recognition. Long passages can contain
changing timbre or multiple words, and their aggregate descriptors cannot track
rapid trajectory turns inside the sounding passage. Increase sensitivity for
more responsiveness. Aggressive and Coherent require **Micro = Auto**; a positive
manual Micro setting takes precedence. Coherent shows an Enable Auto button in
that case.

Coherent bypasses Auto's 610/680 ms inference bounds, adaptive excerpt ceiling,
and event-fraction/path excerpt caps. It uses short edge fades and near-end
crossfades, rather than overlapping whole long phrases. Event duration, Note Off,
stops, pruning, source bounds, and intentional Macro REST still apply. An 800 ms
one-shot cannot finish a five-second passage: choose a longer one-shot or hold a
Gated note to hear longer passages. Tape pitch changes playback duration;
Coherent pitch retains it as before. REPEAT stores the actual source windows,
fades, pitch settings, and scheduled decisions.

Preparation scans and aggregates a bounded number of sensors per step, including
the incremental fallback. It does not add audio-thread PCM analysis or change the
shared compiler/runtime. Host slots 242-244 append Detection, Coherent Sensitivity
and Coherent Limit; slots 1-241 retain their identities.

## Quiet grains and Length

The main header's **Quiet grains: Avoid gaps** is enabled by default. Cached
source-relative level and quiet-bin coverage reduce selection of long, nearly
silent grains when audible eligible material exists. This is an attenuation-free
selection policy, not an audio gate. Short soft details and recordings whose
whole level is low are retained. **Allow** restores the unrestricted selection
policy; a deliberately low Energy Target or Ceiling also permits quiet material.
An all-quiet eligible scope falls back to that scope instead of producing no plan.
No detection boundaries or learned ECs are changed by this policy.

**Length** is a descriptor under the **SIZE** group, available to trajectories,
map axes, the browser, identity locks and pruning. It describes the detected
source span before pitch, using a logarithmic shortest-to-longest range within
the loaded corpus. Equal lengths use the midpoint. The trajectory ruler and
request/actual readout show milliseconds. Long sounding passages still finish
before the next selection; a Length request cannot trim a passage midway.

New instances start in **Coherent pitch**. Existing saved Tape choices and
older states missing the pitch selector retain Tape. Grain-detection Auto
remains the default. These are separate choices.

## Quick start

1. Insert the current Corpus VST3/CLAP and open **Corpus sources**.
2. Select your recordings and wait for preparation to finish; the host must keep processing the instance.
3. Send MIDI notes and start with the default **Balanced** analysis and **8192** grain budget.
4. Change Force target, Variation and Coherence to steer selection; use the map and browser to inspect the learned material.
5. Use **Reload Analysis** to rebuild. Changes marked **Apply** need a new analysis pass.

## Controls and playback

The custom canvas owns the hidden slider parameters and their saved/automation values. **Force target**, **Variation**, **Coherence** and **Detail density** guide traversal; **Source scope**, **Identity anchor** and role/descriptor filters restrict eligible material. Map descriptor axes, focus and drawn paths steer where it travels.

**MIDI lifecycle** selects One-shot, Gated or Natural. Attack/Decay/Sustain/Release shape playback; Pitch changes generated tuning, Velocity influence scales note response, and Output is the final trim. Analysis quality and grain budget trade preparation detail for work and memory. Pool RAM budget controls residency. Changing source/configuration cancels obsolete analysis rather than publishing it as the current model.

## Trajectories

The editor's Y axis automatically fits the observed descriptor range in the
loaded corpus, with a small margin. The ruler, curve, actual-choice trace,
support bars, drawing, point edits and shape buttons share this scale. Descriptor
scores and saved curve targets keep their original units: loading a different
corpus changes the view without rewriting the curve. Previously saved targets
outside the view remain saved and are drawn at its edges. The range describes
corpus evidence before source scope, locks and pruning; their restrictions can
further narrow what can be selected. Empty evidence uses the full range and a
uniform descriptor gets a finite centered range.

**Event** fits a known one-shot or duration-CC event. For an unknown-duration
Gated note it repeats over **Seconds** while held, so trajectories keep moving
beyond the first pass. Explicit **Once** still reaches its end and holds.
Repeat and Bounce retain their existing behavior. The clock guides future
grain selections; it does not force cuts inside a long Coherent passage.

The **Trajectories** workspace replaces the buried Explore editor. Add up to
twelve descriptor lanes, or choose a starting point such as **Bright to dark**
or **Smooth ending**. Each lane offers its own descriptor, Strength, mode,
interpolation, mute, Solo and Duplicate. Muting retains the curve; Remove frees
the slot. **All lanes** exits Solo without unmuting anything.

**Points** exposes eight fixed time handles. Drag a handle vertically to reshape
its neighboring segments. **Draw** paints finer detail across all 32 samples.
Flat, Rise, Fall, Arch, Dip, Pulse and Stairs replace the shape; Reverse and
Invert transform it. **Undo** swaps the current and preceding shape for that
lane during the current editing session. Smooth, Linear and Step interpolation
remain available. Step now includes the final endpoint correctly.

| Mode | Request |
| --- | --- |
| Target | Favor descriptor values near the curve. |
| Ceiling (Maximum) | Penalize values above it; values below remain compliant. |
| Floor (Minimum) | Penalize values below it; values above remain compliant. |

Strength is an absolute weight. Previously it canceled out of the weighted
average when only one lane was enabled. Lanes now contribute independent costs,
so fading one to zero does not amplify the others. **Influence** scales their
combined pull from 0 to 200%; zero bypasses trajectory scoring and candidate
injection. Bypass/Resume retains the previous UI influence amount. Source scope,
eligibility, explicit identity locks and structural choices still apply.

| Clock | Behavior |
| --- | --- |
| Event | Fit a known one-shot/duration-CC window, including its envelope ending. Held/Natural events use Seconds for one pass, then hold. |
| Once | Traverse one timed pass using Seconds, then hold the end. |
| Repeat | Restart after every timed pass. |
| Bounce | Traverse forward, then return; the complete cycle takes two passes. |

Each event has its own clock. **Start offset** changes its normalized starting
position. Cached successor lookahead follows the same clock through Repeat and
Bounce boundaries. This clock is separate from the older two-axis map gesture
and amplitude ADSR/MSEG timing.

Teal shows the requested curve. Gold shows the newest event's actual primary
grain choices, with a moving clock cursor while active. Gray edge bars show
cached descriptor-bin support across the corpus, before source scope and locks.
Last error is raw normalized descriptor distance (or one-sided violation);
evidence reports descriptor confidence separately. A low-confidence tonal/PE
measurement is not proof of a good match. Bypassed or muted lanes show no false
achievement trace. Long grains cannot change character halfway through their
existing excerpt, and recordings cannot provide descriptor values they lack.

The four strongest active lanes reserve up to 24 cached candidates before the
general acoustic graph fills the search budget. Overall capacity remains 64;
PE retrieval leaves space for acoustic alternatives. There is no corpus scan,
PCM analysis, FFT, allocation or new pitch renderer in this selection path.
A fixed 1600-cell cache stores the three trajectory scoring instants and up to
256 tagged scores per instant for the current selection. Repeated successor
checks reuse identical results; collisions recompute rather than approximate.

Influence (81), Clock (82), Seconds (83), Start offset (84), Solo (85), and all
twelve Strength parameters are saved/automatable host parameters. Strength IDs
are 87, 100, 113, 126, 139, 152, 165, 178, 191, 204, 217 and 230. Lane metadata and
curves also save through host parameter state, including painted strokes and
presets. Internal curve-storage parameters are implementation data; edit curves
in the workspace. The 32 samples use 0..126 quantization (at most 0.4% rounding
error); three samples fit exactly in a 21-bit integer even through float host
state. Each stroke commits through the existing acknowledged staging protocol.
Analysis reload preserves the committed lanes. Recorded REPEAT retains its
original grain decisions when live trajectories change. Undo history is not
saved. Older native states did not store private trajectory tables, so already
lost curves cannot be recovered; missing new parameters restore empty lanes.

The shared wrapper declaration parser now accepts three-digit slider IDs while
retaining its existing 1..256 bound. It previously recognized only 1..99 despite
the compiler/runtime supporting 256. This changes recognition for plugins that
declare those higher slots; slots below 100 and audio rendering are unchanged.
No parameter was renumbered. The additional preserved trajectory state/cache uses 1756
cells; the seven adoption ranges now total 209660 cells, below the 262144 limit.


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
ranges total209660cells, below the262144-cell runtime limit.

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

## v2.3.0 render continuity

Transient Smear now uses sample-rate-aware 1.5 ms attack / 6 ms release smoothing
for its dry/wet blend in both Classic and Grain modes. This reduces abrupt blend
steps around sharp transients without changing the cloud windows, source offsets,
grain decisions, shared stereo processing or zero-amount bypass. It adds no
per-sample coefficient calculation. The grain detector, EC learning and adaptive
inference of a Coherent limit are unchanged in this release.
