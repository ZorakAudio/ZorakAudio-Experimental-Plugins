# Non-Joep production regression checks

The configured catalog has 28 non-Joep JSFX and 5 Faust entries. These fixtures
exercise their default production processors and editors. They require Linux
development dependencies, llvmlite, CMake, Faust and a desktop/Xvfb display.

```sh
git submodule update --init --recursive
python tests/plugin_catalog/test_build_modes.py
python tests/plugin_catalog/qualify_catalog.py --out build/catalog --reuse
python tests/plugin_catalog/setup_host.py --out build/catalog --host build/catalog-host
python tests/plugin_catalog/qualify_editors.py --out build/catalog --host build/catalog-host --reuse
python tests/plugin_catalog/qualify_banks.py --out build/catalog --host build/catalog-host --reuse
cmake -S tests/jsfx_showcase -B build/catalog-wdl -DCMAKE_BUILD_TYPE=Release \
  -DSHOWCASE_GENERATED="$PWD/build/catalog/ADS"
cmake --build build/catalog-wdl --target showcase_eel showcase_numeric_runtime --parallel 2
python tests/plugin_catalog/qualify_numerics.py --out build/catalog --wdl-build build/catalog-wdl --reuse
python tests/plugin_catalog/collect_results.py --out build/catalog
```

Do not run editor/bank scripts concurrently against the same host build: they
replace generated guest headers and rebuild the production host. Numerical and
code-generation workers use separate directories. A successful subprocess must
produce its terminal JSON record and required artifacts; exit status alone is
insufficient. Cached records require matching source/runtime/artifact hashes.

The editor fixture reuses one Sample build identity and JUCE module cache.
Actual guest DSP objects use O2; the host uses O0. Each JSFX guest recompiles the
real JSFX processor/runtime; each Faust guest compiles the real Faust processor.
No mock graphics interpreter is used. This verifies guest/runtime behavior,
not every individually packaged VST3/CLAP wrapper, host, platform or control.

The WDL comparison uses the real project's WDL source, production numeric and
slider helper code, and unmodified generated guest objects. It mirrors slider
aliases as the default production host does. A short-message MIDI adapter
supplies identical deterministic events. It executes audio sections, not GFX.
Plugins requiring custom sample/file/IPC services are explicitly marked
NOT_COMPARABLE and use production-host tests instead of dummy WDL callbacks.

Loaded service gates read GFX-published readiness fields. Reading a private DSP
flag through a publication snapshot is not a valid readiness check. The bank
fixture checks slot zero for older file-based plugins; passing three file paths
does not imply that those plugins decode or play three files simultaneously.
Loaded Contour/Texture workers require a visible waveform, and TextureXY
requires a connected drawn path and audible playback after release. Their
source ownership/display declarations are part of the regression fixes.
