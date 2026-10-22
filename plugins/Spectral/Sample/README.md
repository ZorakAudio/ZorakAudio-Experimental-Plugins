# Sample

## Source level matching

**Legacy remains the default.** The original Truth slider, its parameter ID,
automation, gain calculation and grain matching remain intact in Legacy mode.
Older saved states without the new mode explicitly restore Legacy, including
when loaded over an instance previously set to another mode. This restore rule
is independent of any future fresh-instance default.

**Ctrl+click Truth** selects **Legacy / None / RMS / Loudness**, following the
Anti-Salience profile selector. **Ctrl+right-click Truth** restores Legacy.
None bypasses automatic source gain matching; Truth still controls its existing
creative grain behavior. RMS and Loudness blend their correction in dB from
Truth 0 (fully matched) to Truth 1 (original source gain).

In RMS/Loudness, **Shift+drag Truth** adjusts the selected target. New host
parameters expose matching mode (slider65), loudness target (slider66, -23),
RMS target (slider67, -18.4 dBFS), maximum boost (slider68, +6 dB), and source
peak ceiling (slider69, -3 dBTP). All original parameter slots are retained.
The targets/limits affect only the new modes. Settings are latched when a note
starts, including donor-grain level corrections, so editing a mode/target does
not reinterpret an existing voice's base gain.

Whole-file stereo/mono measurements are cached by the Sample Pool's existing
background loader. Matching performs no full-file scan or allocation during
playback. Loudness uses K-weighting, a consistent short-event estimate below
750 ms, an energy blend through 1250 ms, and standard integrated gating for
longer recordings. Short-event values are estimates, not integrated LUFS.
Silence and unusable measurements receive no automatic boost. The source peak
constraint applies before velocity, Liveliness, granular gain and processing;
it is not a limiter for the final polyphonic output.

The fixed source-target matching does not compensate Dynamics, Weight, Output,
or other sound-design processing. No existing project is migrated automatically.

## What it is
Sample is a **load-and-go psycho multisampler**.

The point is immediate usefulness: load files into **slot 0**, play MIDI notes, get variation right away, and shape the bank with a compact set of macros.

The current build includes hybrid playback, transient handling, onset modes, Clean, Room Tame, Walk selection, Push salience control, Hybrid Releaseâ†’Attack handoff, continuous envelope slopes, dynamics, and optional FluxBridge-aware motion behavior.

---

## Quick start
1. Insert the plugin.
2. On **slot 0**, choose **Open Multiple...** and select files.
3. Play MIDI notes.
4. Choose **Random**, **Sequence**, or **Walk**.
5. Shape the result with Envelope, Playback, Clean, Room Tame, Push, and Output.

---

## Selection modes
**Random** chooses from the whole bank with anti-repeat weighting.

**Sequence** walks through the loaded files in order.

**Walk** bridges from the previous sample using brightness, focus, transient richness, length, pitch confidence, body span, and tail span. In Hybrid, donor grain choices can use the same coherence bias.

---

## Envelope
Attack, Hold, Decay, Sustain, and Release shape each triggered sample.

**Shift + Wheel** on Attack, Decay, or Release changes slope from **-100% to +100%**. **0% is linear**. The defaults preserve the original envelope behavior:

- Attack slope: **+100%**
- Decay slope: **-100%**
- Release slope: **-100%**

Hybrid Release keeps the granular body alive under the Release envelope. When a new Hybrid note arrives during Release, the new Attack can inherit the current Release level and matching body phase for a smoother handoff.

---

## Playback engines
**Raw** plays the source directly.

**Tape** uses the slower, heavier rate path.

**Hybrid** keeps the sample head direct, sustains from body grains, can borrow matched donor tails, and supports smoother Releaseâ†’Attack transitions.

---

## Cleanup and depth
**Clean** reduces low-level hiss, room bed, and sample dirt while protecting attacks.

**Room Tame** restrains obvious late-room smear without acting like a heavy de-reverb.

**Push** is a salience-budget control. Higher values push the instrument farther back in the mix by reducing foreground-grabbing presence, air, click, and wide high-band splash. **100% is intentionally extreme background placement.**

---

## Tone and feel
**Color** warms or brightens.

**Focus** softens or clarifies.

**Width** changes spread.

**Liveliness** adds variation.

**Truth** moves from matched/cohesive behavior toward raw sample individuality.

**Weight** adds short body bloom.

**Dynamics** adds controlled density with internal protection.

**Output** trims the final result.

---

## FluxBridge
When the fixed FluxBridge CC lane block is present, Sample adds subtle motion-driven tone, focus, width, pan, and body contour while protecting transient-rich hits.

---

## Good uses
Sample is strongest for quick one-shot banks, found-sound instruments, texture instruments, sketching, randomized playback, and exploratory sound design.

---

## Wrong tool
Use a traditional multisampler when you need exact key zones, velocity-layer authoring, or realistic note-by-note instrument recreation.

---

## In one sentence
Sample turns a loose bank of audio files into a playable, variable instrument with almost no setup.

## Release selection

The original Sample is included in Essentials and All. [Sample Faust](../SampleFaust/README.md) is withdrawn following reports of display/EQ faults and clicks that trigger host auto-mute. Its source is retained for investigation. The two builds have separate plugin identities; replace a Sample Faust instance with Sample rather than renaming its binary.


## Layers and Coherent pitch

The Layers page offers Single, Double, Triple and Loose starting points. They
edit the current instance only. Tap Voices 1-4 to set the count. Support, pitch
spread and timing spread use direct sliders: click, drag or wheel; hold Shift
for finer mouse adjustment; right-click restores the control's default. Level,
pitch and delay are visible for every included layer. Normal windows use level
faders; compact windows use two-column cards. Tap an empty card to add its layer.
Layer levels follow source matching, so their intentional differences remain.
There is no personal-start button or change to defaults in other plugins.

All layer pan defaults to centre. Open Pan to opt into spread or per-layer pan.
Different selects distinct eligible bank sources where possible; Same reuses the
first enabled layer's source. Exclusion and the active bank filter apply to all
layer choices. Preview plays the group; clicking a bank sample still auditions
that sample alone. Supporting layers fade through the attack when Shared attack
is enabled. Layers share the triggering key/velocity and release together; a
pending delayed layer is cancelled if its note is released before it begins.

Original retains speed-linked pitching. Coherent uses the prepared native stereo
spectral reader to separate intentional pitch from sample duration and retain
more upper texture. Tape Stretch remains speed-based. Pitch and layer setup are
latched on new triggers. Exact zero-pitch single-layer playback bypasses spectral
processing. Very short spans, authored Hub loops and excess concurrent grains
retain Original rendering; the Layers view reports grain fallback. Coherent
costs more CPU with dense polyphony; the instrument retains its 16-note-voice
budget and steals an old layer group together when space is needed.

New instances use Coherent with one centred layer. Old projects missing the new
controls restore Original with one layer; saved matching modes, targets,
parameters and bank filters remain intact.


## Native mixer layout (v76)

The wide Layers view occupies the right sound panel and keeps Playback and
Envelope visible. Its EQ tab restores the equalizer in that panel. Medium
windows keep Playback beside Layers/Envelope/EQ. Compact windows use the same
section tabs with a two-column layer grid; one and two layers occupy one row.
The default three-layer setup fits without scrolling at 680×637 and 640×600.
Optional pan and chord controls can extend the panel; short windows can scroll.
The full-bank plot retains all loaded samples, data-scaled axes, outlier marks,
filter thresholds, exclusion and audition controls in a smaller header area.
Opening or idly repainting the mixer does not write or round host parameters.
This revision changes graphics, layout and test-only geometry diagnostics; it
does not change audio sections, parameter declarations, defaults or state migration.
