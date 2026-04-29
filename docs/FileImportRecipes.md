# File Import Recipes

This patch adds direct file ingress and non-destructive recipe processing for the JSFX/JUCE file-slot bridge.

## Import actions

- **Load Directly**: source files are assigned to the slot unchanged.
- **Append Raw / Mega Texture**: multiple files are assembled into one logical in-memory texture.
- **Segment Long File**: one or more long files are split into logical in-memory samples using deterministic RMS/silence rules.
- **Modify / Preprocess**: files are trimmed/stripped/normalized/pruned, then loaded as in-memory results.
- **Segment Then Mega Texture**: source files are segmented first, then assembled into one in-memory texture.

## Fix 4 behavior

- Recipe output is kept in memory; no temporary WAV files are written for segmentation, modification, or mega-texture rendering.
- `sample_pool_*` users receive the rendered in-memory recipe output instead of accidentally reloading the original source paths.
- Segmentation preview reads the full source file and overlays proposed cut regions before apply.
- Segmentation detection now uses deterministic per-sample RMS across channels with `globalRMS * thresholdRatio`, matching the prototype behavior while rejecting only short false gaps through `minSilenceMs`.
- Recipe XML is stored for deterministic replay; source paths/fingerprints remain the replay inputs.

## Important limits

The rendered results are resident in memory. Very large source sets can consume large RAM. That is intentional for this feature and avoids disk-cache clutter.
