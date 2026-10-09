# Saike Spectral Analyzer (beta)

The current multichannel spectral analyzer, with spectrum and time/sonogram displays.

## Quick start

1. Route the channels to be inspected into this instance, then enable the corresponding Ch buttons in the canvas.
2. Choose FFT size/window and set floor, integration and smoothing.
3. Select spectrum, sonogram or time views and compare mono or mid/side display modes where available.

## Controls and routing

Audio processing stages incoming samples; FFT, plotting and much of the analysis run in @gfx. DSP-only timing excludes those costs and cannot establish an open-editor analyzer speedup. Channel and view options differ between revisions.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **FFT size** (`slider1`): default `9`; declared range/choices `0,11,1{16,32,64,128,256,512,1024,2048,4096,8192,16384,32768}`. Canvas / hidden.
- **floor** (`slider2`): default `-90`; declared range/choices `-450,-12,6`. Canvas / hidden.
- **show phase** (`slider3`): default `0`; declared range/choices `0,1,1{disabled,enabled}`. Canvas / hidden.
- **window** (`slider4`): default `2`; declared range/choices `0,5,1{rectangular,hamming,blackman-harris,blackman,flat-top,gaussian}`. Canvas / hidden.
- **integration time (ms)** (`slider5`): default `200`; declared range/choices `0,2500,1`. Canvas / hidden.
- **scaling** (`slider6`): default `1`; declared range/choices `1,6,.2`. Canvas / hidden.
- **smoothing** (`slider7`): default `20`; declared range/choices `0,100,1`. Canvas / hidden.
- **colormap** (`slider8`): default `1`; declared range/choices `0,15,1{dark,intense,fluo,colorblind,pimp,shades,fancy,pastel,purple,dark2,dark3,dark4,dark5,colorblind2,smooth,user}`. Canvas / hidden.
- **mxchannels** (`slider9`): default `16`; declared range/choices `1,16,1`. Canvas / hidden.
- **colorBackground** (`slider10`): default `.15`; declared range/choices `0,1,.4`. Canvas / hidden.
- **Ch1** (`slider11`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch2** (`slider12`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch3** (`slider13`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch4** (`slider14`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch5** (`slider15`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch6** (`slider16`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch7** (`slider17`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch8** (`slider18`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch9** (`slider19`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch10** (`slider20`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch11** (`slider21`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch12** (`slider22`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch13** (`slider23`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch14** (`slider24`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch15** (`slider25`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Ch16** (`slider26`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Sum** (`slider27`): default `1`; declared range/choices `0,1,1`. Canvas / hidden.
- **Channel** (`slider28`): default `-1`; declared range/choices `-1,1,16`. Canvas / hidden.
- **Sonogram** (`slider29`): default `0`; declared range/choices `0,3,1{Sonogram,Sample,Sample all,Off}`. Canvas / hidden.
- **SonoScale** (`slider30`): default `5000`; declared range/choices `10,45000,100`. Canvas / hidden.
- **SignalScale** (`slider31`): default `.5`; declared range/choices `0.1,3,.03`. Canvas / hidden.
- **colormap2** (`slider32`): default `6`; declared range/choices `0,7,1{viridis,viridisinv,magma,magmainv,inferno,infernoinv,plasma,plasmainv}`. Canvas / hidden.
- **Sonolog** (`slider33`): default `1`; declared range/choices `0,1,1{Logarithmic,Linear}`. Canvas / hidden.
- **SonoBig** (`slider34`): default `0`; declared range/choices `0,1,1{Yes,No}`. Canvas / hidden.
- **Factor** (`slider35`): default `1`; declared range/choices `1,8,1`. Canvas / hidden.
- **Solo channel** (`slider36`): default `-1`; declared range/choices `-1,1,16`. Canvas / hidden.
- **Alpha** (`slider37`): default `.15`; declared range/choices `0,1,.4`. Canvas / hidden.
- **slope** (`slider38`): default `0`; declared range/choices `0,9,.25`. Canvas / hidden.
- **show grid** (`slider39`): default `1`; declared range/choices `0,1,1{disabled,enabled}`. Canvas / hidden.
- **gridAlpha** (`slider40`): default `.15`; declared range/choices `0,1,.4`. Canvas / hidden.
- **show channels** (`slider41`): default `0`; declared range/choices `0,1,1{disabled,enabled}`. Canvas / hidden.
- **show options** (`slider42`): default `0`; declared range/choices `0,1,1{disabled,enabled}`. Canvas / hidden.
- **show theme** (`slider43`): default `0`; declared range/choices `0,1,1{disabled,enabled}`. Canvas / hidden.
- **freeze spectrum** (`slider44`): default `0`; declared range/choices `0,1,1{disabled,enabled}`. Canvas / hidden.
- **Smoothing method** (`slider45`): default `4`; declared range/choices `0,4,1{Average,Maximum,Loess,Adaptive,Fast}`. Canvas / hidden.
- **Sonogram Grid Alpha** (`slider46`): default `0.4`; declared range/choices `0,1,0.001`. Canvas / hidden.
- **Initialized** (`slider50`): default `0`; declared range/choices `0,1,0`. Canvas / hidden.
- **Latency Compensation** (`slider51`): default `0`; declared range/choices `0,1,0`. Canvas / hidden.
- **Show A** (`slider53`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Show B** (`slider54`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Integration mode** (`slider55`): default `0`; declared range/choices `0,1,0`. Canvas / hidden.
- **Show side channel** (`slider56`): default `0`; declared range/choices `0,1,0`. Canvas / hidden.
- **Offscreen buffer** (`slider57`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **interpolate_lines** (`slider58`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Display mode** (`slider60`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Scale offset** (`slider61`): default `0`; declared range/choices `-48,48,1`. Canvas / hidden.
- **Sync** (`slider62`): default `0`; declared range/choices `0,4,1`. Canvas / hidden.
- **shift** (`slider63`): default `0`; declared range/choices `0,0.5,0.00001`. Canvas / hidden.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.0570 | 0.0280 | 2.04× | 2.00–2.08× |
| 512 | 0.0531 | 0.0257 | 2.12× | 2.04–2.18× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

**Graphics-dependent workload:** this omits the analysis/display or simulation in `@gfx`. No complete analyzer or interactive-effect speedup is established.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier, Trond-Viggo Melssen, Cockos, Feed The Cat. Vendored version: 5.0.42. Original path: `SpectrumAnalyzer/SaikeMultiSpectralAnalyzer.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
