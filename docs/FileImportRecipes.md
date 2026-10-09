# File Loader and Sample Extractor

The shared File Loader can turn long recordings into separate playable samples
or a continuous texture. Preview and audition automatic cuts, refine them by
hand, and guide the extractor with examples. Processing creates in-memory
results without overwriting source recordings. For authors, see
[the sample-pool API](DSP-JSFX-SamplePool.md) for DSP access.

## Auto-segmentation workflow

1. In a plugin's file-slot menu, choose **Segment / auto-segment...** when loading
   recordings, or **Auto-Segment Current Selection...** for an existing selection.
2. In **Sample Extractor**, review the proposed regions on the waveform. Use
   **Fewer / more cuts**, **Events / whole gestures** and **Tail preservation** to
   steer detection. Adaptive background handling and the silence controls help
   distinguish events from the surrounding recording.
3. Audition regions or **Play context**. Adjust boundaries, create regions,
   **Split**, **Merge next**, or **Delete / restore** cuts. **Confirm** protects a
   region; **Re-analyse** preserves corrected, confirmed and rejected regions.
   **Unlock selected** lets automatic analysis replace that region again.
4. Optionally mark a selected event with **Add Example**, or mark an unwanted
   event with **Wrong Event**. Example matching guides proposals; **Only matching
   proposals** filters automatic regions while keeping them recoverable. Manual
   regions are protected. **Next Review** visits uncertain regions, and
   **Examples / Profile** saves or loads reusable example profiles.
5. Choose **Load separate samples** or **Build texture**, then **Apply**. Use
   **Edit Current Import Recipe...** to revisit an applied import.

Examples guide acoustic matching; they are not a guarantee that every event is
classified correctly. Audition and review the proposed cuts before applying.

## Import actions

- **Load Directly**: source files are assigned to the slot unchanged.
- **Append Raw / Mega Texture**: multiple files are assembled into one logical in-memory texture.
- **Segment / Auto-Segment**: one or more long files become logical in-memory
  samples using assisted event detection, optional example matching and reviewed
  cuts. The legacy RMS/silence detector remains available.
- **Modify / Preprocess**: files are trimmed/stripped/normalized/pruned, then loaded as in-memory results.
- **Segment Then Mega Texture**: source files are segmented first, then assembled into one in-memory texture.

## Current behavior

- Recipe output is kept in memory; no temporary WAV files are written for segmentation, modification, or mega-texture rendering.
- `sample_pool_*` users receive the rendered in-memory recipe output instead of reloading the original source paths.
- Segmentation preview reads the full source file and overlays proposed cut regions before apply.
- New segmentation imports enable the assisted detector, with controls for
  sensitivity, gesture grouping, tail preservation and adaptive background.
  Older recipes retain their legacy detector unless changed. The explicit
  **Silence threshold dBFS** and optional relative RMS gate remain available.
- Low-RMS pruning can now remove weak proposed segments during segmentation, not only whole files during preprocessing.
- Segment boundaries are hard-clamped at the chosen quiet cut point so post-roll/pre-roll cannot bleed into the next pseudo-file.
- Existing recipe-backed slots can be reopened with **Edit Current Import Recipe...** and re-rendered deterministically.
- Sliders support right-click or double-click reset. The preview also has a full Reset button.
- Recipe XML is stored for deterministic replay; source paths/fingerprints remain the replay inputs.
- Reviewed cuts and example profiles are included in the recipe. Changing
  detection controls replaces automatic proposals while preserving protected
  edits; **Reset controls** also retains manual cuts, examples and source removals.

## Important limits

The rendered results are resident in memory. Very large source sets can consume large RAM. That is intentional for this feature and avoids disk-cache clutter.

Saving a recipe does not embed its source recordings. Keep those files available
when reopening or moving a project. Segmentation previews analyse the full
recording, so long sources still require loading and analysis time.
