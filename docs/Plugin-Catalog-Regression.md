# Non-Joep catalog regression checkpoint

This is the recorded 33-plugin snapshot, not a live catalogue count or a fresh
qualification of later FAUST/task/native changes. Current decisions are in
[FAUST qualification](FAUST-Qualification.md).

This checkpoint covers all 33 configured plugins outside JoepVanlier: 28 JSFX
in their normal EEL/publication mode and 5 Faust plugins. It checks the edited
production compiler, processor and editor code. The per-plugin matrix, terminal
worker records, source manifests and actual JUCE screenshots are in
[catalog-regression](catalog-regression/README.md).

| Completed check | Result |
|---|---|
| Production code generation and compilation | 33/33 |
| Editor/audio lifecycle, resize/reopen and parameter state | 33/33 |
| Actual EEL GFX frame progress | 28/28 JSFX |
| Standard-call WDL audio/MIDI comparisons | 17/17 exact matches |
| Plugins requiring custom host services | 11; production-host checks |
| Loaded convolution/texture service workers | 5/5 |
| Automatic build-mode policy tests | 6/6 |
| Standard BandJoiner and GTS VST3/CLAP builds | Both passed |

The [loaded service screenshots](catalog-regression/contact-loaded.png) and
[TextureXY before/after](catalog-regression/contact-TextureXY-fix.png) show the
visual fixes. All 33 default editors were inspected in
[sheet 1](catalog-regression/contact-editors-1.png) and
[sheet 2](catalog-regression/contact-editors-2.png).

## Build policy

Every configured JoepVanlier JSFX now selects native Legacy automatically in
`scripts/build.py`. The direct `build_jsfx_aot()` helper enforces the same policy
for entries under `plugins/JoepVanlier/`. A request for the publication prototype
does not override it. The compiler flag and CMake host flag use the same
effective selection. Faust always selects neither JSFX native mode; other JSFX
preserve their existing default or explicit selection.

The shadow EEL correctness monitor cannot observe concurrently shared Legacy
state coherently. A selection containing Joep plugins therefore rejects
`--correctness-check` before build/staging mutations. Six policy tests cover all
50 Joep entries, all 28 other JSFX and all 5 Faust entries, direct helper calls,
main compiler/CMake routing and rejection timing.

The normal BandJoiner build is additionally exercised without a native flag,
and the normal GTS Faust build with the global Legacy flag. These are separate
production VST3/CLAP wrapper checks, not an assertion that every individual
wrapper has been qualified. See the release logs and artifact hashes in the
result directory.

## Runtime coverage

Each guest uses the actual production processor and editor in a reusable Linux
JUCE host. JSFX guests use their production O2 generated DSP objects; Faust
guests use real Faust 2.70.3 generated C++ and the production Faust processor.
The shared host is compiled at O0 to reduce rebuild cost. It uses one Sample
build identity, so this is a guest/runtime regression test rather than a
per-plugin format/identity validation or performance benchmark.

The lifecycle fixture checks:

- Variable-size audio blocks while the editor is open and being resized.
- Two editor lifetimes, including destroy/reopen, with completed worker records.
- Actual EEL GFX frame progress for all 28 JSFX, not merely a visible empty canvas.
- Finite processing at 48 and 96 kHz and host parameter-state restoration.
- Screenshots of opening, resizing and reopening every editor.
- Sample and Corpus loading three stereo WAVs, Corpus analysis becoming ready,
  and audible MIDI-triggered output from otherwise silent input.
- IPCProbeA and IPCProbeB each running as a real sender/receiver pair, with
  received messages and nonzero receiver debug audio.

Separate loaded-service workers cover PsychoConvolver, Contour, Texture,
TexturePM and TextureXY. File paths go through the real host loading service.
Older slot-zero plugins may consume only the first file, despite being offered
three paths. Readiness checks use published GFX fields, not private DSP fields
that are intentionally absent from publication snapshots. Contour/Texture
select Wet Only. TextureXY receives real JUCE mouse down/drag/up events and must
render the expected connected path without stray segments, and produce audio
after release with silent input. PsychoConvolver uses a
3072-frame stereo broadband impulse/tail fixture spanning three 1024-frame
partitions, with finite output and a bounded peak.

## WDL comparison

Seventeen non-Joep JSFX that use standard numeric/MIDI host calls compare their
unchanged generated DSP objects against the project's actual WDL/EEL2 audio
sections. The fixture supplies identical deterministic audio, transport and
short MIDI events over two seconds at 48 kHz, using block lengths
1/17/64/511/128/33. It checks every configured output, finite values, native heap
bounds and short-message MIDI differences. The production slider callbacks and
numeric helpers are extracted verbatim; slider aliases are mirrored exactly as
the normal default host does. Retired, unused string sliders in 3DPanner are
excluded from numeric default configuration, without editing its source/object.

This is an audio comparison, not an independent GFX renderer comparison. Eleven
plugins use custom sample, file or IPC callbacks and are explicitly recorded as
NOT_COMPARABLE in that fixture. They use the production-host checks described
above. A dummy WDL callback returning zero would not establish compatibility.

## Fixes found

PsychoConvolver's existing source used negative-exponent scientific literals
that failed the embedded EEL GFX `@init` compiler. Its drawing helpers then
failed to resolve and the frame worker never ran. Equivalent decimal constants
fix this without changing the AOT DSP object: both versions produced SHA256
`63966a2c7253d6a20dc5f62d6c700af3d5053556f983185c6261901d50bbc339`.
Its default GFX height increases from 390 to 520 so both knob rows fit below
the IR preview. The fresh editor screenshot was visually inspected.

TextureXY's default dual-heap path also needed an ownership declaration. Its
gesture arrays were overwritten by DSP publications during drawing; the
baseline screenshot shows a fan of stale segments rather than the drawn path.
The UI now owns cells 0–3071 (three 1024-point arrays), with continuous FROM_GFX
publication, while DSP supplies cells 7168–9567 (the 1200-bin min/max waveform).
An explicit memory policy prevents reverse copies over the gesture arrays.
The corrected fixture checks the actual rendered path because `point_count`
is UI-owned and absent from DSP publication snapshots. It additionally requires
audio from silent input after mouse release. This source-only ownership change
leaves its AOT DSP object byte-for-byte unchanged, SHA256
`0b1f7dd4fdea04f02cdcd2332bed7c2d26f27bf9d2b6f4f70ebee488678fdb7b`.

Contour and Texture displayed flat waveform summaries after loading even
though their DSP produced audio. File-service scratch grew the logical heap,
moving the automatic high mirror away from their metadata addresses. Contour
now declares DSP-to-GFX waveform/heat, voice and scope ranges; Texture declares
its reserved metadata region bidirectionally, retaining UI CC/browser writes.
These declarations supplement the automatic mirror, independent of where its
suffix moves. The loaded fixture additionally requires waveform pixels across
more than 20 rows. Both objects remain byte-for-byte unchanged:
Contour `ce135a57079c8890faf4930982b23ef423cb443e21d27841cae0aeaeefcc69a8`;
Texture `7c24866f1ebb2c3c0a51a132ea9fda6093de182ee5be00f9fa1a30d243481490`.

The initial test setup also had two defects: it checked the wrong IPC receiver,
and the Faust link omitted required cached JUCE font modules. Those were fixed
in the fixtures without a production runtime change. Initial failed records
are preserved separately; the final matrix requires successful reruns.
The first GTS format build also contained an empty cached JUCE GUI object.
Deleting that object and rebuilding produced both production wrappers. The
cause of the empty object is undetermined; its failed build log is retained.

## Scope and recovery

These checks do not certify every parameter setting, preset, GUI control,
custom `@serialize` routine, long-duration behavior, installed companion assets,
DAW routing or Windows/macOS build. Generic host parameter state is distinct
from executing guest-authored JSFX serialization. Non-Joep native opt-in modes
are not covered by this default-mode matrix. The earlier Joep checkpoint and
its limitations remain documented in
[JoepVanlier native compatibility](JoepVanlier-Native-Compatibility.md).

The workspace was cleared during qualification. Sources were restored from the
previous archive after verifying its checksum and all 6043 source-file hashes;
the real base checkout and all four recursive submodule revisions were restored.
The 33-plugin code generation, editor checks, WDL comparisons and loaded-service
workers were rerun after recovery. Missing pre-recovery evidence was not counted
as a completed result. Source/evidence checkpoints were saved during recovery.
The compiler and aggregate runtime fingerprint still match the earlier
50-plugin Joep checkpoint; that entire Joep sweep was not rerun for this policy
change. BandJoiner's new normal-build check validates automatic Legacy selection.

Reproduction commands and fixture constraints are in
[tests/plugin_catalog](../tests/plugin_catalog/README.md). The source bundle
includes recursive submodule contents, a cumulative patch against the recorded
base commit and a SHA256 file manifest.
