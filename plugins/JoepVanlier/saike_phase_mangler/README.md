# Saike Phase Mangler (BETA)

An STFT effect with a drawn frequency-dependent phase curve and amount modulation.

## Quick start

1. Feed stereo audio and start with a gentle curve and low amount.
2. Choose Linked, Opposite or Mono-Opposite mode in the canvas.
3. Draw the phase curve and increase modulation while checking the stereo and mono result.

## Controls and routing

Opposite mode can cancel in mono; sharp phase changes can lose the intended allpass magnitude response. FFT/window settings affect cost and time behaviour.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **1X** (`slider1`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **1Y** (`slider2`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **2X** (`slider3`): default `0.1`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **2Y** (`slider4`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **3X** (`slider5`): default `0.2`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **3Y** (`slider6`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **4X** (`slider7`): default `0.3`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **4Y** (`slider8`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **5X** (`slider9`): default `0.4`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **5Y** (`slider10`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **6X** (`slider11`): default `0.5`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **6Y** (`slider12`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **7X** (`slider13`): default `0.6`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **7Y** (`slider14`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **8X** (`slider15`): default `0.7`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **8Y** (`slider16`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **9X** (`slider17`): default `0`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **9Y** (`slider18`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **10X** (`slider19`): default `0.1`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **10Y** (`slider20`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **11X** (`slider21`): default `0.2`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **11Y** (`slider22`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **12X** (`slider23`): default `0.3`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **12Y** (`slider24`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **13X** (`slider25`): default `0.4`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **13Y** (`slider26`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **14X** (`slider27`): default `0.5`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **14Y** (`slider28`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **15X** (`slider29`): default `0.6`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **15Y** (`slider30`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **16X** (`slider31`): default `0.7`; declared range/choices `0,1,0.000001`. Canvas / hidden.
- **16Y** (`slider32`): default `0`; declared range/choices `-0.5,0.5,0.000001`. Canvas / hidden.
- **Polarity** (`slider33`): default `0`; declared range/choices `0,1,{Same,Opposite,Opposite Compensated (not flat but effect disappears when summed to mono)}`. Canvas / hidden.
- **Mono compensation** (`slider34`): default `0.5`; declared range/choices `0,1,0.0001`. Canvas / hidden.
- **Test mono** (`slider35`): default `0`; declared range/choices `0,1,{Off,On}`. Canvas / hidden.
- **scale** (`slider36`): default `1`; declared range/choices `0,20,0.1`. Canvas / hidden.

Declared inputs: 1: left input, 2: right input.

Declared outputs: 1: left output, 2: right output.

## More background from the vendored source

### A plugin to manipulate audio phase in a frequency dependent manner
This plugin can be used to shift the phase of audio in a frequency dependent manner. One can
draw a phase distortion using a spline. This distortion can be modulated by scaling the amount
of phase distortion.

The plugin has a number of modes of operation.
  Linked - Left and right are distorted according to the same shape.
  Opposite - Left and right are phase shifted in an opposite manner. Good for making whoosy
  laser-like sounds, but can lead to loss of mono compatiblity.
  Mono-Opposite - This mode works like opposite, but computes the changes applied to the left and
  right channel and adds an opposite change to the other channel. This serves more as a 
  widener.
  
This plugin operates using an STFT. Sharp phase transitions can lead to a loss of allpass response.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.3942 | 0.3172 | 1.24× | 1.21–1.28× |
| 512 | 0.3851 | 0.3051 | 1.26× | 1.24–1.28× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.05. Original path: `PhaseMangler/saike_phase_mangler.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
