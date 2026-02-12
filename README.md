[![Build & Release](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml)

[![DSP-JSFX Null Test vs. Reaper JSFX](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/dsp-jsfx-nulltest.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/dsp-jsfx-nulltest.yml)

**Note:** Currently working on a JSFX-to-JUCE solution. JSFX -> AST/CFG -> LLVM IR -> JIT/AOT Compilation

# ZorakAudio Experimental Plugins

This repository contains a collection of **experimental, open-source audio plugins** developed by **ZorakAudio**.

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

## Repository structure

Each subfolder represents **one plugin** and contains everything relevant to that plugin:

