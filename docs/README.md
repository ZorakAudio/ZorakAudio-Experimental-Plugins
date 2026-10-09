# Documentation

Start with [DSP-JSFX: consolidated guide](DSP-JSFX-Guide.md). It explains the current additions to stock JSFX with examples and limits. Plugin leaf READMEs are the operation manuals embedded in native help panels.

For release readers, use the [catalog overview and Essentials summaries](releases/Catalog.md)
or the separate [JIT Editor overview](releases/JIT-Editor.md). These Markdown
files also supply the bodies of new tagged releases. See
[release collections](Release-Collections.md) for downloads and
[independent release targets](Build-and-CI.md#independent-release-targets) for publishing.

## Author contracts

| Topic | Reference |
| --- | --- |
| FAUST binding, ordering and block streams | [FAUST sections](JSFX-Faust-Sections.md) |
| Tasks, cancellation, buffers and arenas | [Structured tasks](Structured-Tasks.md) |
| Sample banks and generation adoption | [Sample pool](DSP-JSFX-SamplePool.md) |
| Messages, text sliders and shared memory | [Communication](DSP-JSFX-Communication.md) |
| Import expansion, preprocessing and compatibility | [Compatibility/imports](Compatibility-and-Imports.md) |
| Live shared native graphics | [Native Legacy](Native-GFX-Legacy.md) |
| Graphics snapshots and commands | [Native publication](Native-GFX-Publication.md) |
| Automatic/cooperative sleep and wake | [Sleep](Cooperative-Sleep.md) |
| In-memory host import recipes | [File recipes](FileImportRecipes.md) |

## Current decisions and evidence

| Topic | Reference |
| --- | --- |
| Accepted FAUST migrations, rejected boundaries and sleep grants | [FAUST qualification](FAUST-Qualification.md) |
| Original CMD design with block FAUST | [CMD](CMD-Original-FAUST-Integration.md) |
| Optional Sample character variant | [Sample Faust](Sample-Faust-Character-Integration.md) |
| Hyperreal default, alternate models, idle and matched comparisons | [Hyperreal](Hyperreal-Panner.md) |
| Historical long-recording preparation timings and model checks | [Corpus preparation](validation/Corpus-Preparation.md) |
| JoepVanlier native coverage | [Joep checkpoint](JoepVanlier-Native-Compatibility.md) |
| JoepVanlier native WDL/EEL2 JIT versus LLVM DSP timings | [Joep performance](Joep-Performance.md) |
| Non-Joep editor/audio coverage | [Catalog checkpoint](Plugin-Catalog-Regression.md) |

Result matrices and raw evidence under `validation/`, `catalog-regression/` and `joep-native-qualification/` are historical checkpoints, not claims that every current source was requalified. Read their fingerprints, fixture scope and platform limits. Older measurements remain evidence for their pinned inputs; they do not override current contracts.

Keep the guide concise and contracts authoritative. Put operation instructions in plugin READMEs. When manifests change, update product status in the qualification summaries. Record benchmark reference identities, units, workloads, sleep settings and limitations rather than publishing chronological progress logs as the API.

This cleanup removed 17 obsolete/overlapping documents after incorporating useful material here. Early graphics restrictions, pre-native catalog failures, unimplemented-block-mode claims, cooperative-only sleep instructions and pre-promotion panner descriptions no longer define current behavior. CMD Flow remains an archived test experiment outside the catalogue. Code, test fixtures, licenses, manifests and raw validation assets were retained.
