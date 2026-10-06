# Native GFX publication mode: ownership and Sample reference

This mode compiles a declared graphics ownership contract, rather than exposing
live DSP RAM. It remains an opt-in backend, distinct from Native Legacy.
Sample's complete `@gfx` runs through the opt-in AOT compiler and C++ graphics
bridge in the actual JUCE editor. The native editor does not construct an EEL
interpreter or mirror the DSP heap. The Sample manifest has no native graphics override, so its normal build retains
EEL graphics; Sample's
parameter IDs, sample-pool loading path and audio heap allocator remain intact.

## Ownership and execution

| Data or action | Owner | Native graphics contract |
| --- | --- | --- |
| DSP `mem[]`, coefficients, voice/effect histories | Audio runtime | Never exposed as a live pointer |
| PCM sample pool | Existing pool worker/audio reader system | Loading/adoption unchanged; PCM is not published to graphics |
| Sample metadata, voice flags, spectra, MIDI log | Audio runtime | Bounded, immutable publications at about 30 Hz |
| Control previews, strings, fonts, interaction state | Graphics worker | Persist locally; decode parameter mirrors from effective sliders |
| Slider edits and gestures | Graphics/message/audio bridge | Immediate local preview; host notification on the message thread; `@slider` on the audio thread |
| Temporary EQ solo | Audio runtime | Latest-value mailbox, committed at a block boundary |
| MIDI-log open state and scroll anchor | Processor/editor | Explicitly retained across editor lifetimes |

The compiler validates the complete reachable graphics graph. Sample now has
68 reachable helpers, 31 indexed read sites, and **zero indexed write sites**.
Only the explicit `posteq_solo_target` command is exported to DSP state. The
`locals` declaration permits graphics-local copies of parameter mirrors and
scratch variables; it does not authorize writing those variables into DSP.

The `// za_native_gfx: {...}` source declaration opts into interactive APIs and
specifies record publications, preview ownership, commands and retained state.
It is ignored by normal JSFX/EEL builds. New plugins must declare their own
contract; enabling native compilation is not a blanket acceptance of arbitrary
heap or graphics calls.

### Compact publications

| Publication | Contents | Maximum records | Cells at maximum |
| --- | --- | ---: | ---: |
| Sample metadata | Root, brightness, sharpness, length in milliseconds | 16,384 | 65,536 |
| Voice records | Active flag only | 16 | 16 |
| Pre/post EQ spectra | Two arrays of 96 bins | 192 | 192 |
| MIDI Hub log | 100 records × 12 fields | 100 | 1,200 |
| **Total** | | | **66,944 doubles** |

Each of three snapshot slots allocates its capacity in the processor constructor.
The audio publisher gathers fields from the current DSP layout without resizing
or allocating publication storage. Layout bases, counts, parameter values and
published data are coherent within a pinned slot. The reader retains the slot
through graphics execution, then releases it before rasterization. A modal menu
may pin one slot; the other two remain available to the publisher.

Native indexed reads preserve the compiler's address conversion and resolve
only declared fields through `jsfx_native_read_mem`. The graphics state's `mem`
pointer stays null. Missing fields, exceeded capacities and invalid ranges fail
the graphics frame visibly instead of exposing DSP memory. Indexed writes and
`gmem` are rejected at compile time. Publication buffers are refreshed every
publication, including metadata; there is no generation-based cache whose
invalidation could hide a changing descriptor.

The DSP heap can still grow and its bases can relocate. The fixed maximum applies
to the published UI records, not the DSP allocator or PCM storage. This avoids
requiring a maximum allocation for the entire DSP heap merely to stabilize a
shared pointer.

### Parameter and command semantics

Color/Harmonics coefficient recalculations were removed from graphics. Existing
`@slider`/`@block` calculations now receive native UI edits through the audio
slider preview lane. Graphics initialization is a pure scalar helper, separate
from DSP initialization; the worker never runs DSP `@init`.

Hidden controls refresh their mirrors from current effective sliders. Their
commit helper sends only changed fields, avoiding unnecessary notifications and
overwriting unrelated host automation. Packed profile/Thrill edits preserve the
other field. Individual high slider masks remain exact; direct slider arguments
also support sliders above 64.

Local previews remain until their audio acknowledgement sequence is published.
They do not depend on exact floating-point equality with a rounded host value.
A runtime epoch invalidates previews and transient capture after re-preparation.
Native controls bracket host gestures even when their source uses only
`sliderchange`; explicit `slider_automate` requests are retained.

EQ solo uses a latest-value mailbox rather than the potentially full general
write queue. Focus changes and preparation invalidate old command epochs;
commands from frames started before focus loss cannot re-arm solo on focus
regain. Hidden/closed editors are gated out on the audio thread. Command changes
also wake the idle processor. Editor teardown cancels modal menu waits before
joining the worker and ends active host gestures.

## APIs implemented for full Sample

In addition to the earlier drawing subset, the native runtime implements ordered
mouse/wheel input, keyboard queues/window flags, `gfx_getchar`, modal
`gfx_showmenu`, `strcpy`, `strcat`, `strncpy`, byte-counted `strlen`, monotonic
`time_precise`, and slider change/automation hooks. Named strings are bounded to
16,383 bytes. String literals preserve UTF-8. Fonts and drawing commands continue
to use the existing JUCE/WDL raster backend.

Graphics-global lookup now uses cached indices rather than scanning Sample's
thousands of variables for each drawing operation.

## Validation

The checks link the actual processor/editor and run under JUCE/X11 with synthetic
audio callbacks. WAV fixtures load through the existing file-slot setter; this
is not an automation of the OS file chooser or a DAW-hosted test.

- Load three stereo samples; trigger offset MIDI and confirm finite, nonzero
  output. Both native and EEL runs report the same initial peak, 0.0591894.
- Edit playback mode on the canvas; automate Color/Harmonics from the host;
  verify the native display follows those parameters.
- Adjust hidden Formant by wheel; edit packed Thrill and select a profile through
  the real menu without losing the other packed field. Verify later host edits
  replace acknowledged previews and host gestures balance.
- Ctrl-drag an EQ node; verify solo commits and releases on focus loss, hiding,
  and editor close.
- Send valid and invalid MIDI Hub packets; render actual accepted/rejected log
  records, resize, and retain the log view after reopening.
- Grow the bank from 3 to 65 samples, forcing layout relocation; shrink to 1;
  re-prepare at 44.1 kHz with the editor open. Publication capacity stays fixed.
- Close while the worker is blocked in a modal menu; confirm teardown completes
  and graphics execution stops.
- Run four simultaneous native Sample editors with separate banks/publications
  and finite audio. Their combined three-slot publication capacity is
  803,328 doubles, about 6.13 MiB.
- Keep the existing EEL Sample baseline and EasyExpander native lifecycle checks
  passing. EasyExpander still allocates zero graphics heap snapshot storage.
- Pass 19 compiler ownership/capability tests and 10 existing native/EEL numeric
  comparisons. Native/EEL drawing qualification remains pixel-identical at
  1380×760 and 2070×1140 (3,317 and 5,197 commands respectively).
- Exercise runtime string aliasing/limits/UTF-8, input queues/window flags, high
  slider masks/gestures, and rejection of an unpublished memory read.
- Build Linux x86-64 VST3 and CLAP. Native objects are PIC; neither binary has
  `DT_TEXTREL`.

### Publication cost

The three-file bank comparison measures memory cells and snapshot publication,
not total process memory or complete DSP/graphics CPU.

| Measurement | EEL Sample | Native Sample |
| --- | ---: | ---: |
| Memory cells copied per publication | 747,826 | 1,420 |
| Memory bytes copied | 5.705 MiB | 11.094 KiB |
| Reserved memory cells per slot | 2,844,978 | 66,944 |
| Reserved memory per slot | 21.705 MiB | 0.511 MiB |
| Observed mean publication time | 0.69–0.89 ms | 0.017–0.022 ms |

This removes about **99.81% of the memory-copy volume** at this bank size and
about **97.65% of reserved publication memory**. Native additionally publishes
116 scalar variables, 256 sliders and their acknowledgement sequences. The
one-file publication contains 1,412 memory cells; the 65-file publication
contains 1,668. At the declared maximum it contains 66,944.

These are short local profiling runs, including test diagnostics, under
synthetic callbacks and varying system load. They are not a DAW deadline or
end-to-end latency guarantee. At a full 16,384-file bank, metadata gathering still
scales with the bank count; the prototype has not benchmarked that case.

## Visual evidence

Actual native editor captures:
[playing](native-gfx/sample-native-playing.png),
[profile menu](native-gfx/sample-native-menu.png),
[EQ solo](native-gfx/sample-native-solo.png),
[populated MIDI log](native-gfx/sample-native-log.png),
[65-file bank](native-gfx/sample-native-grown-bank.png),
[re-prepared editor](native-gfx/sample-native-reprepared.png), and
[reopened log](native-gfx/sample-native-reopened.png).

Inspection confirms sample-map bars, text, envelope and spectra, menu selection,
solo highlighting and populated log records. Existing low-contrast labels,
long knob-value text and the overflowing processing hint remain source-layout
limitations. The drawing parity test shares a raster backend with EEL; it
qualifies native execution/commands, not an independent raster implementation.

## Building and remaining gates

Initialize recursive submodules and use the existing build script:

```sh
git submodule update --init --recursive
python scripts/build.py --only Sample --native-gfx-prototype
python tests/jsfx_showcase/test_native_gfx_compiler.py
python tests/jsfx_showcase/sample_native_qualification.py --out build/sample-gfx-audit
```

For the actual Sample editor runner, generate native AOT artifacts first, then
configure `cmake/plugin` for Sample with `-DZA_SAMPLE_GFX_TEST_RUNNER=ON`. Build
`sample_editor_check` and pass an absolute capture directory. Set
`ZA_GFX_PROFILE=1` to collect publication timings. The same runner validates the
normal EEL build when native AOT is disabled. Test diagnostics are excluded when
the CMake test option is off.

This remains an opt-in prototype. Remaining release work includes actual DAW
loading/automation/state restoration and Windows/macOS validation. The shared
raster/types header still links WDL/EEL support; native execution has been
removed from EEL, not all library dependencies. Generic native loops do not yet
have a graphics instruction budget, so arbitrary third-party code needs an
additional execution-limit policy before broad enablement.

Plugins using image/blit APIs, native file/drop calls, or graphics-authored DSP
memory need separate capability and ownership work. In particular, a waveform
editor should commit edits through a designed transaction or immutable-buffer
adoption; this read-only publication contract deliberately does not turn into a
shared mutable heap. Keep migration per plugin until those behaviors are tested.

## Display-only canvases

EasyExpander provides the smaller display-only reference. Its native graphics owns local strings/fonts/scratch and reads selected scalar meter snapshots, with no exposed DSP heap. Opening an editor does not rerun DSP initialization. Six native/EEL captures at two sizes and three states matched pixels in the original Linux fixture, with independent text-presence checks. That is a historical raster/execution checkpoint, not Windows/macOS or all-DAW qualification. The Sample contract above extends the approach to interactive controls and bounded memory publications; the early display-only unsupported-API list is no longer the complete current interface.
