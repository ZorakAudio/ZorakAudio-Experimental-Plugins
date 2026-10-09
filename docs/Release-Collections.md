# Release collections

New full catalog releases contain three ZIPs. **Each ZIP includes Windows,
macOS and Linux builds**, with both VST3 and CLAP in separate operating-system
folders. The standalone JIT Editor has its own GitHub releases and separate
Windows/Linux packages; it is not part of these collection ZIPs. macOS Editor
support still requires a platform port.

| Collection | Plugins | Intended use |
| --- | ---: | --- |
| Essentials | 13 | Focused non-Joep toolkit with one selected identity per overlapping product family. |
| All | 36 | Every distributable non-Joep plugin, including alternate and experimental variants. |
| JoepVanlier | 50 | The separate vendored Saike/Joep catalog, with upstream notices. |

Essentials is a subset of All. Choose one; installing both adds the same plugin
identities. JoepVanlier can be installed alongside either. “All” describes catalog
plugins, not the separately distributed JIT Editor or developer IPC probes.

## Essentials selection

| Plugin | Reason to include it |
| --- | --- |
| Sample Faust | Flagship load-and-play multisampler; faithful original character formulas and editor. |
| Corpus | Flagship corpus/granular instrument with deferred preparation and analysis. |
| Hyperreal 3D Panner (`3DPanner`) | Canonical original physical model with the faithful optimized renderer. |
| Anti-Salience MX | Background placement, adaptive surviving-anchor suppression and optional selective scatter. |
| CMD | Cross-mix decluttering and somatic bus processing. |
| SOMA | Perceptual/somatic limiting; a different job from decluttering. |
| ATTACK | Impact-oriented transient and body macros. |
| GTS | Gaussian FIR attack/sustain separation, distinct from ATTACK's macro design. |
| EasyExpander Faust | Compact ERB-weighted expansion with qualified native performance improvement. |
| ModTilt | Shapes slow versus fast envelope motion. |
| ERB Tilt | Shapes spectral balance; it does not duplicate ModTilt's envelope tilt. |
| Click-Be-Gone SG | Dedicated click restoration. |
| GesturePad | Draw-and-play MIDI gestures, notes and motion lanes. |

### Variant choices and evidence

EasyExpander Faust measured **15.821 s versus 3.476 s (4.55×)** for matched,
continuously active complete processing of the authorized recording, with
58,558,936 float values byte-identical. This is a qualified workload, not a
promise of the same ratio for every preset, host or platform. See
[FAUST qualification](FAUST-Qualification.md) and
[the sleep audit](validation/EasyExpander-Sleep-Audit.md).

Sample Faust saved **1.7%, 7.8% and 8.2%** in repeated active whole-plugin
comparisons at 48 kHz/64, 48 kHz/256 and 96 kHz/1024. All 58,368,000 compared float
values matched the cached original, including tested handoffs. It keeps that
original path as fallback. With character disabled, single pairs instead showed
0.7–3.5% more CPU from staged execution overhead. Essentials selects the faster
active character implementation requested for the flagship; **All retains Sample
for workloads where the original is cheaper**. The much larger isolated-kernel
speedup is not the full-plugin speedup. See [the detailed report](Sample-Faust-Character-Integration.md).

`3DPanner` already uses the faithful optimized original renderer also distributed
as HyperrealFast. Essentials uses `3DPanner`'s established identity. HyperrealFast
remains in All for existing sessions; HyperrealFaust and HyperrealHybrid are
different renderer designs and also remain in All. They are not presented as
faithful replacements or bundled alongside the canonical panner in Essentials.
See [the panner report](Hyperreal-Panner.md).

## What is left out

Essentials omits ADS, RTT, RED, VAR, Alias, BedRock, Contour, Texture, TexturePM,
TextureXY, TSEQ, SpectralStabilizer, 3DPannerManager, DDT, DOT, DPT, PsychoConvolver
and Roomalizer, plus the unselected Sample, EasyExpander and Hyperreal variants.
These remain in All. This is curation for focus and overlap, not a new claim that
each excluded plugin is broken or objectively inferior.

**IPCProbeA and IPCProbeB are excluded from normal builds**, including `--only`,
smoke and full/sharded builds. Their source stays available for developer work.
**SaliencePush is retired from new builds and collections**, replaced by
AntiSalienceMX. Its source also stays available. Anti-Salience MX has a new ID and
different parameters; it does not silently replace SaliencePush in saved sessions.
Keep older installed binaries when a project uses them.

## Install and build

For a tag such as `R1`, the three catalog assets are:

```text
ZorakAudio-Experimental-Plugins-R1-Essentials-all-platforms.zip
ZorakAudio-Experimental-Plugins-R1-All-all-platforms.zip
ZorakAudio-Experimental-Plugins-R1-JoepVanlier-all-platforms.zip
```

Inside a ZIP, open **only your operating system's folder**: `windows/`, `macos/`
or `linux/`. Copy the category folders under its `VST3/` or `CLAP/` directory to
your normal plugin location. Keep bundles intact. Plugin manuals remain embedded
behind `?`; Joep upstream license notices ship beside the corresponding bundles.
macOS is universal2 and ad-hoc signed, not notarized. Linux's baseline remains
Ubuntu 24.04. Root and per-platform manifests list exact membership and identities.

The policy lives in [release-collections.json](../release-collections.json).
The builder and merger use this same policy. DSP compiles once per platform;
Essentials packaging reuses exactly the bytes already built for All.

`scripts/build.py` still produces a platform-local archive for local builds or
one CI shard. Full CI uses four disjoint shards per platform. After all twelve
shards pass, a separate assembly job validates coverage and nonempty executable
payloads, then creates the three collection ZIPs. It preserves binary bytes,
bundle permissions, symlinks and AppleDouble resource metadata. Build receipts
identify the release-policy SHA-256; mismatched policy receipts fail assembly.

To assemble complete current-policy shard archives locally:

```sh
python scripts/merge_catalog_archives.py --input build/catalog-shards --output dist/catalog --tag R1 --collections
```

All three platforms are required. Missing shards, duplicate plugin receipts,
missing/empty binaries or a changed policy prevent publication. Tests use small
synthetic ZIP fixtures to exercise these gates without rebuilding the catalog:

```sh
python tests/build/test_catalog_merge.py
```

Catalog tags (`R*` or `v*`) publish only these three collections after their
catalog gates pass. JIT Editor tags (`jit-v*`) use a separate workflow and create
a separate prerelease with four Windows/Linux format packages. Catalog releases
do not wait for JIT builds, and JIT tags do not rebuild the catalog. See
[publishing instructions](Build-and-CI.md#independent-release-targets).

A manual full **Run workflow** also uploads the three ZIPs as `catalog-collections`, without
creating a GitHub release. Branch/PR smoke runs upload their five-plugin platform
archives; they do not masquerade as complete collections.
