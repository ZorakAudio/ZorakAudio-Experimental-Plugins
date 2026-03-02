[![Build & Release](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml)

[![DSP-JSFX Null Test vs. Reaper JSFX](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/dsp-jsfx-nulltest.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/dsp-jsfx-nulltest.yml)

# ZorakAudio Experimental Plugins

This repository contains a collection of **experimental, open-source audio plugins** developed by **ZorakAudio**. It also currently houses (but will be refactored into its own repository later) DSP-JSFX - an extended subset of Cockos EEL2 that supports many features such as:

1. Comment Macros for tooltips `// #TOOLTIPS:` and help-sections `// #HELP:`
2. Automatic inferral of IN and OUT pins via read/write operations on `splN` in code (read of `spl3` and `spl4` is used to infer a 4 IN/2 OUT configuration)  
3. Support for a growing subset of `@gfx` to create custom UI as well as I/O interactable controls (Mouse and Keyboard)
4. Compilation of `@init`, `@sample`, `@block`, and `@slider` down to LLVM IR with `-O2` optimizations enabled.
5. Direct JUCE wrappers that enable direct compilation from JSFX into VST3 and CLAP plugin formats.

Each plugin in this repository is a **standalone exploration** of a specific DSP idea. While many tools are immediately usable in real-world projects, they are released with an experimental mindset: ideas are tested openly, refined iteratively, and shared early.

There is no single framework or unified system by design. Each plugin exists independently.

---

## What to expect

These plugins generally focus on:

- perceptual and psychoacoustic DSP approaches  
- safe, predictable behavior under real-world conditions  
- minimal control surfaces that reduce chance of error  
- alternatives to traditional dynamics and filtering techniques  
- robustness under automation, silence, and extreme input  

Some plugins may feel unconventional. That is intentional.

---

