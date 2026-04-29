# File Import Recipes

This patch adds direct file ingress and non-destructive recipe processing for the JSFX/JUCE file-slot bridge.

## Import actions

- **Load Directly**: source files are assigned to the slot unchanged.
- **Append Raw / Mega Texture**: multiple files are assembled into one logical in-memory texture.
- **Segment Long File**: one or more long files are split into logical in-memory samples using deterministic RMS/silence rules.
- **Modify / Preprocess**: files are trimmed/stripped/normalized/pruned, then loaded as in-memory results.
- **Segment Then Mega Texture**: source files are segmented first, then assembled into one in-memory texture.

## Current behavior

- Recipe output is kept in memory; no temporary WAV files are written for segmentation, modification, or mega-texture rendering.
- `sample_pool_*` users receive the rendered in-memory recipe output instead of reloading the original source paths.
- Segmentation preview reads the full source file and overlays proposed cut regions before apply.
- Segmentation now uses a short RMS envelope plus an explicit **Silence threshold dBFS** control. The old RMS-ratio gate remains available as an optional secondary gate.
- Low-RMS pruning can now remove weak proposed segments during segmentation, not only whole files during preprocessing.
- Segment boundaries are hard-clamped at the chosen quiet cut point so post-roll/pre-roll cannot bleed into the next pseudo-file.
- Existing recipe-backed slots can be reopened with **Edit Current Import Recipe...** and re-rendered deterministically.
- Sliders support right-click or double-click reset. The preview also has a full Reset button.
- Recipe XML is stored for deterministic replay; source paths/fingerprints remain the replay inputs.

## Important limits

The rendered results are resident in memory. Very large source sets can consume large RAM. That is intentional for this feature and avoids disk-cache clutter.
