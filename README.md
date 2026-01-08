[![Build & Release](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml)

**Note:** I am experimenting with CI for building Windows/Mac OS X/Linux VST3 + CLAP builds automatically - these plugins may differ from the codes seen [in R0.1.1](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/releases/tag/R0.1.1)! If something is wrong with these plugins, i.e. they fail to build, please fall back to R0.1.1 for the meanwhile as this repository undergoes testing. New plugins are already written and are waiting to be converted to FAUST but that is only after this is all situated. Please be patient!

**Note2:** Given difficulties with converting some JSFX to FAUST, additional work is required to auto-build C++ JUCE with FAUST JUCE via the CI. However, in the end, there will be a mixture of pure JUCE and FAUST-to-JUCE plugins.

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

