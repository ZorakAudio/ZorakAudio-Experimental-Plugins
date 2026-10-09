# JoepVanlier: native WDL/EEL2 JIT versus LLVM DSP

This 2026-10-09 comparison covers all 50 buildable JoepVanlier entries without simplifying their JSFX algorithms. It compares the vendored **native x64 SSE WDL JIT** to the LLVM DSP kernel used by the native JUCE plugins. It does **not** compare complete REAPER and JUCE host callbacks.

48 entries matched float audio and MIDI exactly in all measured trials; 0 were within tolerance; 2 differed; 0 did not complete a comparable measurement. A mismatch is a correctness finding, not a validated performance improvement.

## Method

- Windows x64, AMD Ryzen 9 7940HS; clang version 21.1.8.
- 48 kHz; 64- and 512-frame buffers; default source controls.
- 5 trials per buffer, 4 seconds measured after one second of warmup.
- Serial, below-normal priority; alternating engine order per block and trial.
- Identical continuous synthetic input and short MIDI events; no external audio files loaded.
- Cached optimized LLVM IR is checked against current expanded JSFX/imports and re-emitted with `-O2 -Xclang -disable-llvm-passes`. This retains the existing IR optimizer/inliner decisions and optimizes final machine code without adding fast-math.
- WDL uses NOFPSTATE with a scoped FP environment, avoiding a floating-point environment transition per sample. LLVM calls its block entry point; WDL runs its original block/sample sections with required channel marshalling.
- Timed: the DSP audio sections and required sample/buffer marshalling. Excluded: compilation, initialization, generated signal construction, transport/MIDI/control setup, output checks, GFX, JUCE host callback and DAW.
- Timing uses C++ `steady_clock` elapsed durations, including any scheduling interruption; the raw `*_cpu_seconds` field names do not mean process CPU accounting. Paired order and trial ranges help expose measurement variability.

The ratio is the median of paired **WDL time / LLVM time**. Above 1 favours LLVM; below 1 favours WDL. Leaf READMEs report microseconds per frame and the trial range. Close ratios should be treated as similar performance, not a universal speed claim. Tiny kernels are more sensitive to timer/marshalling overhead.

## Results

| Plugin | Ratio, 64 frames | Ratio, 512 frames | Output and scope |
| --- | --- | --- | --- |
| [Amaranth](../plugins/JoepVanlier/Amaranth/README.md) | 1.01× | 1.03× | Exact audio/MIDI |
| [BandJoiner](../plugins/JoepVanlier/BandJoiner/README.md) | 3.99× | 4.06× | Exact audio/MIDI |
| [BandSplitter](../plugins/JoepVanlier/BandSplitter/README.md) | 2.15× | 2.15× | Exact audio/MIDI |
| [BandSplitter_phasematcher](../plugins/JoepVanlier/BandSplitter_phasematcher/README.md) | 2.20× | 2.23× | Exact audio/MIDI |
| [Filther](../plugins/JoepVanlier/Filther/README.md) | 1.28× | 1.42× | Exact audio/MIDI |
| [FM_Filter](../plugins/JoepVanlier/FM_Filter/README.md) | 1.46× | 1.48× | Exact audio/MIDI |
| [modizer](../plugins/JoepVanlier/modizer/README.md) | 2.01× | 2.01× | Exact audio/MIDI |
| [MS_20](../plugins/JoepVanlier/MS_20/README.md) | 1.50× | 1.51× | Exact audio/MIDI |
| [nott](../plugins/JoepVanlier/nott/README.md) | 1.58× | 1.64× | Exact audio/MIDI |
| [poprocks](../plugins/JoepVanlier/poprocks/README.md) | 1.59× | 1.56× | Exact audio/MIDI |
| [Ravager_MB](../plugins/JoepVanlier/Ravager_MB/README.md) | 1.53× | 1.56× | Exact audio/MIDI |
| [ReaBee](../plugins/JoepVanlier/ReaBee/README.md) | 1.13× | 1.12× | Exact audio/MIDI; GFX excluded |
| [Reflectosaurus](../plugins/JoepVanlier/Reflectosaurus/README.md) | 1.05× | 1.02× | Exact audio/MIDI |
| [ripple](../plugins/JoepVanlier/ripple/README.md) | 1.53× | 3.54× | Exact audio/MIDI; baseline only |
| [saike_abyss](../plugins/JoepVanlier/saike_abyss/README.md) | 1.17× | 1.21× | **Output mismatch; no speedup claim** |
| [saike_bric_a_brac](../plugins/JoepVanlier/saike_bric_a_brac/README.md) | 1.29× | 1.36× | Exact audio/MIDI; baseline only |
| [saike_duskverb](../plugins/JoepVanlier/saike_duskverb/README.md) | 1.06× | 1.15× | Exact audio/MIDI |
| [saike_final_boss](../plugins/JoepVanlier/saike_final_boss/README.md) | 1.38× | 1.41× | Exact audio/MIDI |
| [Saike_FMFilter2](../plugins/JoepVanlier/Saike_FMFilter2/README.md) | 1.46× | 1.47× | Exact audio/MIDI |
| [saike_lava](../plugins/JoepVanlier/saike_lava/README.md) | 1.16× | 1.20× | **Output mismatch; no speedup claim** |
| [saike_midi_arp](../plugins/JoepVanlier/saike_midi_arp/README.md) | 1.08× | 1.09× | Exact audio/MIDI |
| [Saike_Morph](../plugins/JoepVanlier/Saike_Morph/README.md) | 1.63× | 1.63× | Exact audio/MIDI |
| [saike_never_odd_or_even](../plugins/JoepVanlier/saike_never_odd_or_even/README.md) | 2.01× | 2.03× | Exact audio/MIDI |
| [saike_nostalgizer](../plugins/JoepVanlier/saike_nostalgizer/README.md) | 1.34× | 1.35× | Exact audio/MIDI |
| [saike_nuker](../plugins/JoepVanlier/saike_nuker/README.md) | 1.50× | 1.51× | Exact audio/MIDI |
| [saike_partials](../plugins/JoepVanlier/saike_partials/README.md) | 1.18× | 1.24× | Exact audio/MIDI |
| [saike_phase_mangler](../plugins/JoepVanlier/saike_phase_mangler/README.md) | 1.24× | 1.26× | Exact audio/MIDI |
| [Saike_Pitch_Shift](../plugins/JoepVanlier/Saike_Pitch_Shift/README.md) | 1.46× | 1.52× | Exact audio/MIDI |
| [saike_protosynth](../plugins/JoepVanlier/saike_protosynth/README.md) | 1.48× | 1.73× | Exact audio/MIDI |
| [Saike_Routing_Utility](../plugins/JoepVanlier/Saike_Routing_Utility/README.md) | 1.48× | 1.49× | Exact audio/MIDI; baseline only |
| [saike_smooth](../plugins/JoepVanlier/saike_smooth/README.md) | 1.89× | 1.91× | Exact audio/MIDI |
| [Saike_Stereo_Bub_II](../plugins/JoepVanlier/Saike_Stereo_Bub_II/README.md) | 1.94× | 1.95× | Exact audio/MIDI |
| [Saike_Stereo_Bub_III](../plugins/JoepVanlier/Saike_Stereo_Bub_III/README.md) | 1.86× | 1.87× | Exact audio/MIDI |
| [Saike_SuperSpreaderClone](../plugins/JoepVanlier/Saike_SuperSpreaderClone/README.md) | 1.26× | 1.30× | Exact audio/MIDI |
| [Saike_Yutani](../plugins/JoepVanlier/Saike_Yutani/README.md) | 1.36× | 1.39× | Exact audio/MIDI |
| [saikedrums](../plugins/JoepVanlier/saikedrums/README.md) | 1.55× | 1.58× | Exact audio/MIDI |
| [SaikeMultiSpectralAnalyzer](../plugins/JoepVanlier/SaikeMultiSpectralAnalyzer/README.md) | 2.04× | 2.12× | Exact audio/MIDI; GFX excluded |
| [SaikeMultiSpectralAnalyzer_MK2](../plugins/JoepVanlier/SaikeMultiSpectralAnalyzer_MK2/README.md) | 0.76× | 0.76× | Exact audio/MIDI; GFX excluded |
| [SaikeMultiSpectralAnalyzer_old](../plugins/JoepVanlier/SaikeMultiSpectralAnalyzer_old/README.md) | 0.72× | 0.71× | Exact audio/MIDI; GFX excluded |
| [SatanVerb](../plugins/JoepVanlier/SatanVerb/README.md) | 1.19× | 1.20× | Exact audio/MIDI |
| [SequencedFX](../plugins/JoepVanlier/SequencedFX/README.md) | 1.40× | 1.43× | Exact audio/MIDI; baseline only |
| [Squashman](../plugins/JoepVanlier/Squashman/README.md) | 1.10× | 1.12× | Exact audio/MIDI |
| [StereoManipulator](../plugins/JoepVanlier/StereoManipulator/README.md) | 2.73× | 2.78× | Exact audio/MIDI |
| [StereoSpectrumSplit](../plugins/JoepVanlier/StereoSpectrumSplit/README.md) | 5.86× | 5.63× | Exact audio/MIDI; GFX excluded |
| [Swellotron](../plugins/JoepVanlier/Swellotron/README.md) | 1.26× | 1.23× | Exact audio/MIDI |
| [Tanh_Saturator_AA](../plugins/JoepVanlier/Tanh_Saturator_AA/README.md) | 1.62× | 1.66× | Exact audio/MIDI |
| [Tight_Compressor](../plugins/JoepVanlier/Tight_Compressor/README.md) | 1.45× | 1.44× | Exact audio/MIDI |
| [ToneStacks](../plugins/JoepVanlier/ToneStacks/README.md) | 3.05× | 3.25× | Exact audio/MIDI |
| [Transience](../plugins/JoepVanlier/Transience/README.md) | 1.63× | 1.61× | Exact audio/MIDI |
| [wahriffic](../plugins/JoepVanlier/wahriffic/README.md) | 0.40× | 0.39× | Exact audio/MIDI |

## Interpretation and limits

Exact matching means the observed finite float values and short MIDI messages matched for these inputs and defaults; positive and negative zero are not distinguished. It is not qualification of every preset, rate, sample bank, automation path, custom serialization, interactive state or DAW. Silent output and empty default patterns/slots provide limited active-workload evidence. The arpeggiator fixture creates the same four active steps in both guests so MIDI generation is actually exercised.

The spectral analyzers and ReaBee do substantial work or drive audio state from `@gfx`; that work is absent here. Their ratios describe the measured audio path only. Bric-a-brac has no samples loaded, SEQS is at its default playback/pattern state, and Ripple's default pattern is empty. Instruments receive short note events, rather than an exhaustive maximum-polyphony arrangement.

Output discrepancies against native WDL (saike_abyss, saike_lava) are recorded rather than hidden behind their timings. The earlier Linux qualification used WDL's portable backend, so an exact result there is not evidence of equivalence to this native backend. This audit does not resolve which backend/semantic detail causes a discrepancy.

LLVM is not uniformly faster. At 512 frames, the measured ratios are 1.03× for Amaranth, 1.42× for Filther, 3.25× for ToneStacks and 2.78× for StereoManipulator. Wahriffic is 0.39×: its LLVM audio path takes 2.56 times the WDL time. These are workload-specific results, not a promise about every setting.

The previously reported **12.9× Amaranth complete-callback improvement** compares our old and optimized production callbacks. It includes removed host-variable lookup overhead. It is **not** a WDL speedup and is not interchangeable with this kernel comparison. Amaranth's source algorithm is unchanged.

Initial harness reached the default Windows stack limit (0xC00000FD). Rechecked with 8 MiB reserve; PE header only changed, all executable section bytes unchanged. Updated runner sets this reserve for all future fixtures.

## Evidence and reproduction

[All raw trials and fingerprints](validation/joep-performance/results.json) are retained in the repository. The runner is `tests/jsfx_showcase/benchmark_joep_native.py`, using `joep_native_benchmark.cpp` and the original shared numeric/slider/runtime helpers. It requires Windows x64 Clang, cached regular build IR/headers/source, and a native WDL static library (including the SSE assembly from `src/WDL/eel2/asm-nseel-x64-sse.asm`). A portable WDL library must not be substituted.

```text
python tests/jsfx_showcase/benchmark_joep_native.py --out build/joep-readme-performance --wdl-lib <native-wdl.lib> --trials 5 --seconds 4
python scripts/update_joep_performance.py build/joep-readme-performance/results.json
python scripts/check_plugin_readmes.py
```

The publication script requires complete, disjoint coverage of every JoepVanlier entry; it will not publish a partial matrix as the complete catalog. Documentation is embedded when plugins build. Updating the source README does not replace help in already installed binaries.
