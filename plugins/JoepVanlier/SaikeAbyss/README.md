# Saike Abyss — ZSFX compatibility showcase

**Original effect: Abyss Reverb 0.06, by Joep Vanlier (Saike), MIT license.**
This is a third-party JSFX running through ZorakAudio's native DSP compiler and
GFX runtime, not a new reverb algorithm and not an endorsement by its author.

## Use

Put the effect on a stereo audio track. The host sliders control diffusion,
decay, modulation, damping, shimmer, drops, nonlinearity, and dry/wet. The custom
GFX pane is the original animated particle display; it is not a second control
panel. Defaults match the upstream source. Sample rates up to 96 kHz are the
range declared by this upstream version. Do not treat rates above that as tested.

Build from the repository root:

```bat
python scripts/build.py --only SaikeAbyss --config Release
```

No submodule or download is required. The four source files and dependency
subdirectory are already included. The full import chain feeds both native DSP
and GFX from the same generated `JSFXExpanded.jsfx`.

## Pinned snapshot

Repository: https://github.com/JoepVanlier/JSFX

ReaPack release: **0.06**, 28 October 2024.

Commit: `c4f6629d3661bcc0ea7f77a9c3fda13c9165f672`.

`upstream.json` records each original path, pinned URL, and the checksum of the
included local copy. This is a whitespace-normalized source snapshot, not a
byte-identical Git checkout. There are no intended DSP or GFX source edits.
`LICENSE.upstream` contains the original MIT license. Builds never update it.

## Wrapper settings

`plugin.json` enables EEL-style assignment filtering for this plugin. Its
pitch-shifter initialization reaches a nonfinite intermediate value in a compound
assignment. WDL's checked assignment clears it. The opt-in native mode filters
all assignment stores conservatively; it is not a claim of complete EEL numeric
compatibility or of duplicating every check elision in WDL's optimizer.
Other plugins do not gain these store checks unless they enable the option.

The wrapper also requests explicit GFX memory synchronization with no shared
ranges. Abyss's `particles` array is private animation state: DSP never reads it
after initialization. The GFX VM initializes and advances its own particle array;
the normal scalar snapshot still carries parameter and modulation values. This
avoids repeatedly overwriting the animation with audio-side particle memory.
Neither setting edits the upstream files.

## Validate

```bat
python scripts/verify_jsfx_snapshot.py plugins/JoepVanlier/SaikeAbyss
python scripts/test_jsfx_showcase.py
```

The optional `--upstream` verification fetches only the recorded commit and
compares executable tokens. It never overwrites the local files. Full plugin
build and host audition are distinct from the headless DSP/GFX regression tests.

## Upstream license

The build also stages this notice beside the VST3/CLAP artifacts.

```text
MIT License

Copyright (c) 2022 Joep Vanlier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
