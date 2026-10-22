# DSP-JSFX Sample Pool API

`file_mem()` remains the compatibility path: it copies decoded file data into JSFX `mem[]`, where each cell is a double. That is correct for JSFX-style scripts, but it is wasteful for large sample banks.

`sample_pool_*` is the large-bank path. It decodes selected files into runtime-owned packed `float32` storage, keeps immutable generations for realtime reads, and exposes small accessors to DSP-JSFX.

## Core model

```eel
pool = sample_pool_from_slot(0, "main");
sample_pool_set_mode(pool, SAMPLE_MODE_RESIDENT);
sample_pool_set_budget_mb(pool, 4096); // 0 means unlimited
sample_pool_commit(pool);
```

The worker builds a new generation from the file-slot selection. DSP code reads from the current immutable generation:

```eel
id = sample_get(pool, i);
ok = sample_read2_interp(pool, id, phase, l, r);
```

`sample_get()` returns a sample ID, not a `mem[]` pointer. Raw decoded audio does not live in JSFX heap memory unless explicitly exported.

## Modes

```eel
SAMPLE_MODE_RESIDENT = 0; // load decoded audio into float32 pool
SAMPLE_MODE_BUDGETED = 1; // skip samples after RAM budget is reached
SAMPLE_MODE_LAZY     = 2; // reserved for future lazy/paged loading
SAMPLE_MODE_STREAM   = 3; // reserved for future streaming
```

## State

```eel
state    = sample_pool_state(pool);
selected = sample_pool_selected(pool);
loaded   = sample_pool_loaded(pool);
failed   = sample_pool_failed(pool);
ram_mb   = sample_pool_ram_mb(pool);
gen      = sample_pool_generation(pool);
```

State constants:

```eel
SAMPLE_POOL_EMPTY    = 0;
SAMPLE_POOL_SCANNING = 1;
SAMPLE_POOL_LOADING  = 2;
SAMPLE_POOL_READY    = 3;
SAMPLE_POOL_PARTIAL  = 4;
SAMPLE_POOL_FAILED   = 5;
```

## Metadata

```eel
id     = sample_get(pool, index);
frames = sample_len(pool, id);
chans  = sample_channels(pool, id);
sr     = sample_srate(pool, id);
peak   = sample_peak(pool, id);
rms    = sample_rms(pool, id);
ok     = sample_name(pool, id, #name);
```

## Opt-in source level analysis (API v2)

`sample_pool_set_level_analysis(pool, 1)` enables additional measurements on the
existing loader worker. Set it before committing; the option participates in
request deduplication. It defaults to off for existing clients. Both decoded
files and in-memory import results are analyzed before their immutable generation
is published. Original PCM, `sample_peak`, and `sample_rms` are unchanged.

`sample_level(pool, id, field)` reads cached scalar metadata without scanning PCM:

| Field | Measurement |
| --- | --- |
| 0 / 1 | Whole-file playback RMS, stereo / mono source |
| 2 / 3 | Playback loudness, stereo / mono source |
| 4 / 5 | Estimated true-peak amplitude, stereo / mono source |
| 6 / 7 | Loudness method: 0 invalid/silent, 1 integrated, 2 event estimate |

Playback stereo uses the first two channels and duplicates one-channel sources.
Mono source averages the first two channels and duplicates that result. Thus
mono cancellation does not invalidate a usable stereo measurement. Loudness uses
BS.1770 K-weighting and channel-energy summation. Sounds below 750 ms use gated
20 ms event blocks (absolute -70 LUFS, relative -20 LU); energy blends to standard
400 ms / 100 ms-hop integrated gating between 750 and 1250 ms. Method 2 is a
sampler estimate, not standards-compliant integrated LUFS. True peak uses the
BS.1770-5 Annex 2 four-phase FIR, includes sample peaks, and flushes its tail.
These are source measurements, not post-pitch, post-effects or summed-bus limits.
Nonfinite inputs produce invalid metadata; check the method/energy before
deriving a gain. Missing entries/invalid fields return zero.

## Audio reads

```eel
x = sample_read(pool, id, channel, frame);
x = sample_read_interp(pool, id, channel, phase);
ok = sample_read2(pool, id, phase, l, r);
ok = sample_read2_interp(pool, id, phase, l, r);
```

`sample_read2*` is the hot path for stereo playback.

## Compatibility export

```eel
n = sample_export_mem(pool, id, dst_base, src_frame, frame_count);
n = sample_export_mem2(pool, id, dst_base, src_frame, frame_count);
```

These copy pool audio into JSFX double `mem[]`. They are explicit, expensive compatibility escape hatches and are valid only in `@block`.

## GFX rule

`@gfx` should display metadata/previews mirrored from DSP or a declared native
publication contract rather than scanning full-resolution PCM. DSP accessors
`sample_preview_bins(pool, id)` and
`sample_preview_read(pool, id, bin, minimum, maximum, rms)` expose prepared
preview data; they do not authorize direct sample-pool calls from GFX.

## Deferred generation adoption

`sample_pool_set_deferred(pool, 1)` holds a completed worker generation for
script-controlled adoption. Call `sample_pool_adopt(pool)` from an allowed
setup section (`@init`, `@slider` or `@block`), normally at a block-safe model
boundary. Validate generation-dependent descriptors whenever the pool changes.
