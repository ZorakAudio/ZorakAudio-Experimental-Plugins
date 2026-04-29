# File Import Recipes / Drag-Drop Import

Direct-file patch contents:

- `src/JSFXJuceProcessor.cpp`
- `src/ZAAudioImportRecipe.h`
- `docs/FileImportRecipes.md`

Copy these over the same paths in the repository.

## Added behavior

- Root editor drag/drop via `juce::FileDragAndDropTarget`.
- Soundly-style landing pad overlay while dragging files over the plugin.
- Clipboard import for `file://` URI lists and raw local paths via Ctrl/Cmd+V.
- Post-drop / post-paste action chooser.
- Direct load.
- Raw append as one runtime file.
- Modify / preprocess existing files.
- One long file to many segmented entries.
- Multiple files to one Mega Texture.
- Segment then build Mega Texture.
- Before/after waveform preview for preprocessing actions.
- Silence trimming, internal silence stripping, silence segmentation, low-RMS pruning, near-duplicate pruning, novelty/spectral-flux ordering, crossfaded texture assembly, and final RMS normalization.
- Recipe/source fingerprint sidecar XML written beside generated cache audio.

## Non-destructive policy

Source files are not overwritten. Rendered outputs go to the OS temp directory under:

```text
ZorakAudio/FileRecipes/<timestamp_uuid>/
```

Each rendered recipe directory includes:

```text
recipe.zafileimport.xml
```

The current integration renders transformed assets to cache files, then feeds those paths into the existing `setFileSlotPathsWithMode` loader. That keeps the existing audio-thread cache path stable while adding the queued import/recipe workflow.
