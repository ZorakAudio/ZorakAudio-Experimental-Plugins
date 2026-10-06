# Hyperreal Hybrid Panner

The original Artistic renderer paired with a new block-based FAUST Physical renderer. Use the existing Render Model control: Artistic retains the original algorithm; Physical uses designed head-shadow, interaural/propagation delays, pinna coloration, early reflections and a late feedback-comb field. Physical intentionally sounds different from the original measured KEMAR renderer.

The canvas, automation controls and Mono/Stereo/Bed/Dual source modes remain. This current shell uses local controls; the historical manager link is absent from the maintained V7.1.2 source. This plugin has a separate native VST3/CLAP identity and does not replace the original.

Both audio engines stay warm. EEL processes the original Artistic signal first; one FAUST compute call processes the whole host buffer using captured raw stereo input and the original per-sample renderer crossfade. FAUST selects Artistic output directly when the blend reaches zero. The original Artistic late field has independent excitation. Mode changes do not freeze Physical buffers or replay a tail from an earlier activation. Keeping Physical warm adds CPU cost even while Artistic is selected. There is no playback-time JIT compilation or worker dispatch.

FAUST uses its own private controls and histories and does not export over the original Artistic smoothers. Physical controls smooth from zero at startup. The late field stays warm even when its return is disabled; enabling it can reveal recent excitation. Moving delays can bend pitch. Maximum delays are 65534 samples for propagation, 16382 for early reflections and 126 for pinna coloration; extreme behavior therefore depends on sample rate. This design does not promise physically exact localization or identical Physical output.

Build with `python scripts/build.py --only HyperrealHybrid`. The embedded source requires this repository's native compiler and FAUST LLVM at build time; it is not stock REAPER JSFX. Recreate it with `python tests/faust/panner_candidates.py`. Its Physical DSP derives from `tests/faust/panner_renderer.dsp`; integration deliberately removes the all-FAUST Artistic voicing and UI exports. Requalify generated edits with `tests/faust/profile_panner_hybrid.py`.

See the hybrid validation report for full-plugin timings, Artistic null tests and limitations. Headphone audition and an independent REAPER render remain necessary to judge subjective quality.
