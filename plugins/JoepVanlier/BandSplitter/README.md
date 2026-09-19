# Saike 4-pole BandSplitter

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `Basics/BandSplitter.jsfx`
- Author: Joep Vanlier
- Version: 0.23
- Tags: bandsplitter
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# 4-pole Band Splitter
4-pole band splitter that preserves phase between the bands. It has a UI and uses much steeper crossover filters (24 dB/oct) than the default that ships with Reaper thereby providing sharper band transitions.
It also has an option for linear phase FIR crossovers instead of the default IIR filters. IIRs cost less CPU and introduce no preringing or latency. The linear phase FIRs however prevent phase distortion (which can be important in some mixing settings), but introduce latency compensation. Note that when using the linear phase filters, it is not recommended to modulate the crossover frequencies as this introduces crackles.
[Screenshot](https://i.imgur.com/nOhiaJB.png)
### Demos
You can find a tutorial of the plugin [here](https://www.youtube.com/watch?v=JU_7gIr5RTI).

## Attribution / license

See `LICENSE.upstream` and the original source header.
