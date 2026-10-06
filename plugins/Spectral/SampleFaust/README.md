# Sample Faust

An optional native build of Sample with its five character bands implemented
in FAUST. The transistor and asymmetric diode formulas, harmonic enrichment,
parameters and interactive editor retain Sample's behavior. See
[Sample's manual](../Sample/README.md) for controls and loading.

Character processing runs in chunks of up to 256 samples. The original EEL
implementation handles configurations with active analyzers, EQ solo, Contour,
DeCrust, output fades or input-delay paths. Histories are synchronized when
ownership changes. FAUST is skipped completely while the fallback owns this
stage. This does not add latency or an automatic sleep policy.

This is a separately identified native plugin for comparison; the existing
Sample build remains available. Whole-plugin performance depends on settings
and buffer size. The isolated character kernel's speedup is not a promise of
the same improvement for the complete sampler. See
[the integration measurements](../../../docs/Sample-Faust-Character-Integration.md).

Build locally with `python scripts/build.py --only SampleFaust`. FAUST with its
LLVM backend is required at build time; no compiler or JIT runs in the plugin.
The embedded `@faust` source targets this repository's native compiler and is
not a stock REAPER JSFX effect.

The generated entry is `../Sample/src/SampleFaust.jsfx`. Its maintained source
is Sample's cached `Sample.jsfx` plus the generator:

```
python tests/faust/sample_character_candidate.py --output plugins/Spectral/Sample/src/SampleFaust.jsfx
```

The generator rejects changed integration boundaries rather than silently
producing a second character pass. Re-run qualification after regenerating.
