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

## Recordings and the workspace

New multi-file extractions use **One timeline**: recordings are concatenated in
the shown order before segmentation. There are no inserted gaps, fades, trimming
or gain changes. The first file supplies the timeline sample rate; mono stays
mono when every source is mono. Mixed mono/stereo sources use stereo and mixed
rates use the existing import resampler. Original source files are never altered.

Drop one or more files anywhere on the extractor, or use **Add...**, to append
recordings. Protected cuts keep their positions when audio is appended. **Files**
opens a scrollable source list, with **Replace...** and **Separate files** when
individual inspection is preferable. Changing between combined and separate
coordinates starts fresh cuts; Undo restores the prior files, mode and edits.
Apply includes newly dropped files. Saved recipes retain all source paths and
fingerprints, and reconstruct the same combined timeline on reopening.

The sidebar is split into **Detect**, **Edges**, **Match**, and **Output**.
Detect contains the preset, sensitivity, grouping, tail length and minimum
event duration. **Advanced detection...** opens detector/version and noise-gate
options. Edges contains start/end, padding, fades, snapping and auto-play.
Match contains examples and profiles; Output contains gain/pruning and reset.
Numeric fields support a single click to edit. The default Detect page fits at
the minimum supported window size; longer secondary sections scroll as needed.

**Balanced**, **Short hits**, and **Whole gestures** are local import presets.
They never change global or personal defaults. Whole gestures uses a longer
relative-energy trough to keep sub-actions together while separating the next
gesture, even when low background activity never reaches the absolute noise gate.
Use auditions and review: a preset does not establish ground-truth boundaries.

## Fast editing and navigation

Transport and the primary Keep/Delete/Split/Merge/Review actions remain above the waveform at every supported
window size. **Keys ?** (or `?`) opens the complete shortcut guide. Shortcuts
leave numeric text entry and dropdown selection alone.

| Action | Shortcut |
| --- | --- |
| Previous / next kept event | Left / Right, or Shift+Tab / Tab |
| Previous / next source (Separate files) | Page Up / Page Down |
| Play / pause; stop | Space; Esc |
| Play with surrounding context | C |
| Auto-play on navigation; snap edges | A; N |
| Confirm and advance; delete / restore and advance | Enter; Delete |
| Split at cursor; merge with next | S; M |
| Set start / end to cursor | [ / ] |
| Nudge start backward / forward | comma / period |
| Nudge end backward / forward | semicolon / apostrophe |
| Coarse boundary nudge | Hold Shift: 10 ms instead of 1 ms |
| Move cursor | Ctrl+Left / Right: 10 ms; add Shift for 1 ms |
| Cursor to selection edge / file edge | Home / End; Ctrl+Home / End |
| Fit selection; fit whole recording; zoom | F; 0; + / - |
| Pan waveform | Alt+Left / Right, Shift+wheel, middle-drag |
| Next review; positive / negative example | R; E / X |
| Unlock selection; manage profiles | U; P |
| Re-analyse automatic regions | Ctrl+R |
| Undo; redo | Ctrl+Z; Ctrl+Shift+Z or Ctrl+Y |

On macOS use Command where Ctrl is listed. Apply and Cancel remain explicit
buttons; Esc stops playback or closes the shortcut guide in the extractor.

Drag a boundary toward either edge of the zoomed waveform to pan continuously,
even while holding the pointer still. Ctrl+drag creates a new region and follows
the edges too. Alt bypasses snapping while dragging. Event navigation preserves
the current zoom and reveals the next event's start; use F to fit its full span.

## Detector versions and boundaries

The advanced **Detector** dropdown offers **Adaptive gestures**, the previous
**Adaptive boundaries**, **Assisted (original)** and **Silence / RMS**. New
segmentation imports use Adaptive gestures (version 3). Its Balanced settings
retain version 2 boundaries; higher Grouping settings evaluate longer relative
quiet dips. Version 1 and 2 behavior remains available. Saved
recipes without `boundaryDetectorVersion` retain original behavior, and saved
region snapshots replay exactly. Changing the detector explicitly re-analyses
automatic regions while preserving corrected, confirmed and rejected regions.
Example profiles keep their existing version and features.

Adaptive boundaries measures a 2 ms energy envelope, smoothed over 6 ms, alongside
the existing spectral features. It requires a sustained dip and renewed attack
to separate events, and groups brief, much quieter rebounds with the preceding
gesture. Tail preservation controls the audible decay threshold rather than
adding a fixed extension. Pre/post-roll stays bounded by adjacent cuts. Analysis
and auditions run on the owned background workers, outside the audio callback.

Lower **Events / whole gestures** to separate smaller actions; raise it to keep
textured actions together. **Min detected segment ms** rejects fragments;
**Tail preservation** retains more of a decay. None of these can reliably
separate overlapping sounds with no acoustic boundary. Uncertain omissions
remain available through **Next Review** and manual region creation.

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
- New segmentation imports enable Adaptive gestures, with controls for
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

## Input compatibility

Missing `joinSegmentationInputs` retains the original per-file workflow. Missing
`boundaryDetectorVersion` retains the original assisted detector; stored version
2 recipes continue to use version 2. Existing reviewed regions replay without
re-analysis. Joined recipes validate every member file before preview/export;
missing or changed audio is reported instead of applying stale coordinates.

The shared extractor changes apply to plugins rebuilt with these headers. No
cross-plugin preference system or persistent scratch-audio cache is introduced.
