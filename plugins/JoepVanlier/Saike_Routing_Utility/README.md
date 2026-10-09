# Saike Monitor Routing Utility [ALPHA]

A monitor-routing utility with three stereo inputs, monitor selection, trims, delays and channel tests.

## Quick start

1. Route DAW, PC and miscellaneous sources to inputs 1/2, 3/4 and 5/6 as needed.
2. Map Monitor A/B/C to actual output pairs and enable the intended monitor.
3. Set source gains, trim/delay, Dim and mono/mid/side checks.

## Controls and routing

The plugin exposes six stereo output pairs. Selecting a pair does not create a hardware connection; configure the DAW output routing too.

The reference lists the current source declarations. Hidden controls belong to the custom canvas and saved automation state; they are not additional generic sliders. Normalized values are mapped by the source and may not use the units displayed in the canvas. Dummy/deprecated placeholders are omitted.

## Source parameter reference

- **gain 1 (dB)** (`slider1`): default `0`; declared range/choices `-45,15,.001`. Canvas / hidden.
- **gain 2 (dB)** (`slider2`): default `0`; declared range/choices `-45,15,.001`. Canvas / hidden.
- **gain 3 (dB)** (`slider3`): default `0`; declared range/choices `-45,15,.001`. Canvas / hidden.
- **Swap LR** (`slider4`): default `0`; declared range/choices `0,1,{Off,On}`. Canvas / hidden.
- **Mid side switch** (`slider5`): default `0`; declared range/choices `0,2,{Normal,Mid,Side}`. Canvas / hidden.
- **Dim** (`slider6`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Monitor A** (`slider7`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Delay Monitor A [ms]** (`slider8`): default `0`; declared range/choices `-10,20,0.001`. Canvas / hidden.
- **Output channel A** (`slider9`): default `0`; declared range/choices `0,6,1{1-2,3-4,5-6,7-8,9-10,11-12,None}`. Canvas / hidden.
- **Monitor A trim** (`slider10`): default `0`; declared range/choices `-12,0,.0001`. Canvas / hidden.
- **Monitor B** (`slider11`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Delay Monitor B [ms]** (`slider12`): default `0`; declared range/choices `-10,20,0.001`. Canvas / hidden.
- **Output channel B** (`slider13`): default `1`; declared range/choices `0,6,1{1-2,3-4,5-6,7-8,9-10,11-12,None}`. Canvas / hidden.
- **Monitor B trim** (`slider14`): default `0`; declared range/choices `-12,0,.0001`. Canvas / hidden.
- **Monitor C** (`slider15`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Delay Monitor C [ms]** (`slider16`): default `0`; declared range/choices `-10,20,0.001`. Canvas / hidden.
- **Output channel C** (`slider17`): default `2`; declared range/choices `0,6,1{1-2,3-4,5-6,7-8,9-10,11-12,None}`. Canvas / hidden.
- **Monitor C trim** (`slider18`): default `0`; declared range/choices `-12,0,.0001`. Canvas / hidden.
- **Headphones** (`slider19`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Headphones output channel** (`slider20`): default `3`; declared range/choices `0,6,1{1-2,3-4,5-6,7-8,9-10,11-12,None}`. Canvas / hidden.
- **Headphones trim** (`slider21`): default `0`; declared range/choices `-12,0,.0001`. Canvas / hidden.
- **Subwoofer** (`slider22`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Delay Sub [ms]** (`slider23`): default `0`; declared range/choices `-10,20,0.001`. Canvas / hidden.
- **Subwoofer crossover** (`slider24`): default `80`; declared range/choices `10,200,1`. Canvas / hidden.
- **Subwoofer output channel** (`slider25`): default `4`; declared range/choices `0,6,1{1-2,3-4,5-6,7-8,9-10,11-12,None}`. Canvas / hidden.
- **Subwoofer trim** (`slider26`): default `0`; declared range/choices `-12,0,.0001`. Canvas / hidden.
- **Main gain** (`slider27`): default `0`; declared range/choices `-60,5,.0001`. Canvas / hidden.
- **Mute all** (`slider28`): default `0`; declared range/choices `0,1,1{Off,On}`. Canvas / hidden.
- **Mute input 1** (`slider29`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Mute input 2** (`slider30`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Mute input 3** (`slider31`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Solo input 1** (`slider32`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Solo input 2** (`slider33`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.
- **Solo input 3** (`slider34`): default `0`; declared range/choices `0,1,1`. Canvas / hidden.

Declared inputs: 1: daw input left, 2: daw input right, 3: pc input left, 4: pc input right, 5: misc input left, 6: misc input right.

Declared outputs: 1: left output 1, 2: right output 1, 3: left output 2, 4: right output 2, 5: left output 3, 6: right output 3, 7: left output 4, 8: right output 4, 9: left output 5, 10: right output 5, 11: left output 6, 12: right output 6.

## Native build and help

This JSFX is packaged as a native VST3/CLAP using the LLVM compiler and shared JUCE runtime. Its custom canvas uses the native Legacy GFX path when present. Hidden slider identities remain available for state/automation. Audio and MIDI pin configuration is inferred at build time; connect the pins in your DAW. Imports and packaged images are included in the build.

The `?` button displays this README embedded at build time; no online manual or loose README is needed. A documentation update takes effect after rebuilding and replacing the plugin.

## Performance comparison

Measured 2026-10-09 on AMD Ryzen 9 7940HS (Windows x64), at 48 kHz with default controls. 5 serial trials per buffer, 4 seconds of generated audio after one second of warmup, alternating engine order. The reference is the vendored **native x64 SSE WDL/EEL2 JIT**, not its portable interpreter. LLVM uses the production optimized final backend.

These are **DSP-only** timings: audio sections and their required sample/buffer marshalling. Compilation, initialization, signal generation, control/MIDI/transport setup, output checks, GFX and the full JUCE/DAW callback are excluded. They do not predict total REAPER CPU or an installed plugin's complete callback time. No algorithm simplification was made.

**Observed output: identical finite float audio values and matching MIDI for this workload in every trial.** The comparison does not distinguish positive and negative zero. This covers the measured defaults, not every preset or interactive graphics state.

| Buffer | WDL µs/frame | LLVM µs/frame | WDL time / LLVM time | Trial ratio range |
| --- | --- | --- | --- | --- |
| 64 | 0.1590 | 0.1098 | 1.48× | 1.45–1.51× |
| 512 | 0.1556 | 0.1041 | 1.49× | 1.48–1.49× |

A frame includes all processed channels. Ratios are median paired WDL/LLVM times: above 1 favours LLVM; below 1 favours WDL. Ratios near 1 should be read as similar speed, considering the trial range.

**Workload limit:** Default monitor selection may output silence; active monitor routing needs a separate workload.

See the [complete catalog method and result matrix](../../../docs/Joep-Performance.md). The machine-readable evidence records source/IR/header/executable fingerprints and every trial.

## Attribution and source

Author: Joep Vanlier. Vendored version: 0.07. Original path: `Basics/Saike_Routing_Utility.jsfx` in JoepVanlier/JSFX. The DSP algorithm is not simplified for this packaging or benchmark pass.

See `LICENSE.upstream` and the original source headers for applicable licensing and additional credits; per-file LGPL declarations are retained.
