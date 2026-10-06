# Original CMD: faithful FAUST integration

The production CMD now uses the original design with its audio graph implemented in `@faust block`. The experimental CMD Flow redesign has been moved out of the plugin catalogue into `tests/faust/experiments/CMDFlow`. Its large speedups came chiefly from algorithm changes and do not describe this implementation.

## What is preserved

The unchanged `CrossMixDeclutter.jsfx` remains the reference. The new `CrossMixDeclutterFaust.jsfx` preserves the 8–24 ERB band options, two cascaded bandpass biquads per ear per band, envelope and onset detection, breathing, cut and somatic smoothing, width, conditional nonlinear saturation, output gain, and emergency bounds. Disabled filter histories freeze; disabled gain states reset according to the original coefficient rebuild behavior.

The original shared-bus coordination, roles, TurnPulse, policy updates, and canvas remain in JSFX. Policy still runs at the original host-block cadence. FAUST receives coefficients and targets, processes a full host buffer once, and exports measurements for the next original policy update and smoothing values for the UI. There are no one-sample FAUST compute calls. The compiled stage has two audio inputs, 79 outputs including scalar exports, and 183 inferred controls.

The existing plugin name, IDs, and slider layout are retained. The CMD manifest selects the new source and the original canvas mode. The generator is `tests/faust/generate_cmd_original_faust.py`.

## Apples-to-apples performance

Measurements use two real JUCE plugin processors with synthetic stereo input, six seconds per instance including one second of silence. Both processors participate in the shared bus, use different roles, and receive matched parameter changes. Each row is the median of three interleaved original/FAUST trials; only complete processing callbacks are timed. Compilation, input generation, PCM dumping, editor rendering, and setup are excluded. Default 12-band quality is used.

| Sample rate / buffer | Original callbacks | FAUST callbacks | Speedup | Time saved |
| --- | ---: | ---: | ---: | ---: |
| 48 kHz / 64 | 3.484 s | 2.874 s | 1.21× | 17.5% |
| 48 kHz / 256 | 2.752 s | 1.892 s | 1.45× | 31.2% |
| 96 kHz / 1024 | 4.736 s | 3.559 s | 1.33× | 24.8% |

These are measured whole-processor improvements for the same algorithm. They are not REAPER render timings, an all-quality benchmark, or a guarantee of the same improvement on another machine. No recording was loaded for these tests.

## Numerical and lifecycle qualification

Eighteen paired workloads compare 27,648,000 floating-point output values. Nine steady workload pairs cover the benchmark configurations. Nine additional pairs exercise zero Manifold/Somatic controls, extreme controls with role changes and saturation, and live changes through 8, 12, 16, 20, and 24 bands at the same three sample-rate/buffer configurations.

The worst absolute sample difference across these tests is **2.842170943040401e-14**, below 3e-14. Several cases are byte-identical. This supports numerical equivalence in the tested cases; it is not a universal bit-for-bit guarantee.

The harness also checks finite bounded audio, active peer participation, full-block FAUST call counts with zero scalar calls, state restoration, offscreen editor rendering, and reset to a different sample rate and buffer size. The test fixture assigns deterministic instance IDs using a test-only slider; this shim is removed from production.

Machine-readable results are in `build/cmd-flow/original-port-results.json` and `build/cmd-flow/original-port-checks.json`. The final control-check log is `build/cmd-flow/original-port-checks.log`. An earlier fixture binary was stale during the first live-band check; it was rebuilt before the final passing checks.

## Limits

The Windows CLAP and VST3 release package built successfully with Ninja/Clang. The packaged CLAP passed module loading, initialization, an advertised-range Manifold parameter update, activation, and approximately one second of finite nonzero audio processing, followed by shutdown. The production expanded source matches the qualified fixture after removing its deterministic-instance test shim, and the generated FAUST stage identity matches. ZIP integrity and packaged CLAP byte identity were checked before delivery.

No manual listening or REAPER project test was performed. The numerical comparison uses the actual plugin processor through a dedicated harness, while the packaged CLAP has a separate module-load, parameter-update, and audio-processing smoke test. VST3 is built but has no separate VST3 host test. The package is provided for installation by the user; this work does not install it automatically.
