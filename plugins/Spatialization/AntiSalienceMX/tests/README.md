# Anti-Salience MX package check

`check_clap.cpp` loads the actual packaged CLAP through its public interface.
It checks the product identity, four source/reference inputs and two outputs,
parameter defaults, synthetic core/scatter transitions, finite processed audio,
peak containment, settled uncontained dry bypass and native Windows editor
creation/resizing/destruction. It loads no recordings. The GUI remains hidden.
This is a package smoke check, not a listening test or a complete GFX raster test.

Windows reproduction with Clang and a Windows SDK:

```powershell
clang++ -std=c++20 -O2 -Ilibs/clap-juce-extensions/clap-libs/clap/include plugins/Spatialization/AntiSalienceMX/tests/check_clap.cpp -o build/AntiSalienceMX-clap-check.exe -luser32 -lshell32
build/AntiSalienceMX-clap-check.exe PATH_TO_PACKAGED_AntiSalienceMX.clap
```

Local Windows Release CLAP and VST3 built successfully. The CLAP check passed;
the VST3 was built and its archive payload verified, but not rendered through a
VST3 host in this check. Native macOS/Linux qualification remains a CI/manual
step. The imported source is byte-identical to the supplied v1.1.0 text:
SHA-256 `da98e26ba77d2b4754c4f1e17318c3e8e7b08d16ce58b5880e4858444c98c0b6`.
