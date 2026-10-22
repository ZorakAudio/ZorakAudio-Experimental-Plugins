# Native spectral sample pitch

`sample_pitch_read2(pool, sample, start, frames, semitones, age,
voice, character, outL, outR)` is a stateful AOT sample reader (standard and Native Legacy builds).
The compiler emits `DSPJSFX_HAS_SPECTRAL_PITCH` only for programs using it;
the AOT wrapper then prepares and warms its 64 native voice slots during
processor construction. Other plugins do not allocate these engines.

`start`/`frames` describe an immutable allowed source excerpt. `age` is an
integer output-frame offset at the current engine sample rate. `voice` is
0..63 and must uniquely own the slot while active. `character` is a latched
0..1 tonal confidence. Source generation, sample, excerpt, pitch, character,
native rate or a rewind reset that slot. Identical repeat inputs are deterministic.

The reader uses source lookahead and zero padding outside the permitted
excerpt to account for spectral latency without delaying MIDI playback.
No reads cross the excerpt boundary. A prepared 2048-point engine handles
noisy grains; a 4096-point engine handles confident tonal grains, both at
4x overlap. Output is cached in fixed256-frame chunks. A call processes at
most8 chunks; large age jumps fail rather than doing unbounded catch-up work.
No allocation, configuration, file I/O or PCM correlation search occurs in
playback. Spectral processing is synchronous native work and costs more CPU
than ordinary sample reads. Eight-note/40-voice stress at256frames/48kHz
can exceed the5.33ms callback deadline; normal tested single-note rendering
averaged about0.55ms. There is no hard realtime deadline guarantee.

The pitch frequency map uses a2200..8000Hz tonality limit based on
`character`. Low/mid body follows the requested interval; the upper spectrum
has a progressively smaller relative shift. This intentionally differs from
uniform transposition of every frequency. No fundamental-note or formant
estimate is invented for noisy material. Stereo phase relationships are linked.

This is an audio-thread-owned API. It is impure, invalid in GFX/serialization,
and forbidden inside both ordinary deferred tasks and arena tasks. Runtime
state lives outside guest memory; it is neither arena data nor serialized
preset data. Corpus records the pitch, character and natural duration itself.

Dependencies are vendored with pinned revisions and MIT notices in
`libs/signalsmith-stretch` and `libs/signalsmith-linear`. Production archives
must carry both license files.


## Sample Layers integration

The opt-in `sample_pitch_read2_ex(pool, sample, start, frames, semitones, age,
stream, character, playbackRate, outL, outR)` reader adds a bounded 0.0625..4
input/output time ratio. It consumes a fractional-rate input count per cached
output block; pitch and source timing are independent. `stream` encodes a
0..63 slot in its low six bits and a monotonic reuse token above it. The token
resets a reused slot even when the excerpt and first output age are identical.
The original ten-argument reader retains its rate=1 contract for existing users.
Both readers remain audio-only and forbidden in deferred tasks and GFX.

Non-Faust Sample preserves its working standard AOT/EEL graphics path. Its
Coherent mode owns 16 note-reader slots and 48 shared grain slots. A short grain,
a full grain budget, or an authored Hub source loop retains Original rendering;
the choice is latched for the grain. The UI reports grain fallback. Rendering
never shares a slot between sounding grains. Source discontinuities restart the
allowed excerpt. Exact zero-pitch single-layer playback bypasses the spectral
reader and uses the original path, including its Lively drift.

New Sample instances start with Coherent, one layer, zero pitch/timing/pan
spread. Older saved states missing the appended controls restore Original and
one centred layer. Existing parameter IDs/ranges/defaults are unchanged. Quick
starting buttons edit only this instance's saved parameters; there is no
personal-start or cross-plugin default mechanism.
