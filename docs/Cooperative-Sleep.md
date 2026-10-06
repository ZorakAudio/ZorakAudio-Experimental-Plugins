# Cooperative plugin sleep

Realtime sleep supports both automatic silence/event modes and explicit plugin
permission. The sleep badge's right-click menu selects Default/Auto, Never sleep,
Silence, Events, or Free-running; the choice is saved with the project. Existing
`idle`, `idle_hold_ms`, `idle_out_db`, and `za_idle_mode` controls work again.
`idle_in_db` is accepted for compatibility, but input wake uses exact zero: every
nonzero sample wakes DSP.

Default/Auto chooses cooperative sleep for plugins declaring `za_sleep_ready`.
Otherwise it uses the plugin's idle option and topology. Automatic silence sleep
waits for quiet output and the configured hold/tail interval. It freezes state
below the configured output threshold, so it is a CPU-saving heuristic, not a
bit-exact guarantee for every realtime recovery. Never sleep remains available.
Offline/non-realtime renders always advance DSP.

Hyperreal Panner uses `idle=input idle_hold_ms=1000 idle_out_db=-140`: a very low
output threshold and a one-second quiet hold preserve audible tails before
suspending DSP. Opening an unchanged canvas must not wake its audio processor.

## Interface

```eel
@init
za_sleep_ready = 0;

@block
// These are conditions defined and maintained by this plugin.
za_sleep_ready = no_pending_work && state_is_settled;
```

No option or new section is required. `za_sleep_ready` is an ordinary JSFX
scalar. The host clears it to zero immediately before every DSP block; only
an exactly finite `1` written during that block is a fresh permission. A grant
from `@init`, an earlier block, or a saved state does not authorize future
processed blocks. Audio plugins should compute readiness after advancing their
state (for example in `@sample`, where the final sample's flag is observed).
FAUST can export a matching signal into a JSFX-declared `za_sleep_ready` too.

The promise is stronger than "my input is quiet": while there are no external
events, skipping this DSP must leave both current output and future behavior
unchanged. The plugin must account for filter/delay state, envelopes, oscillators,
queued events, internal clocks and future scheduled work. A host cannot prove
that an arbitrary plugin's grant is truthful. Omitting the flag uses the configured automatic policy. EasyExpander does not grant sleep because its detector and
gain envelopes continue evolving through quiet input.

## Host conditions and wake behavior

The host accepts a fresh grant only after exact-zero output, with no incoming
activity, keep-awake request, outgoing MIDI/slider activity, or outstanding
structured-task obligations. `za_keep_awake` remains a veto. This adds no DSP
state ABI field and involves no lock, allocation or compiler operation in the
new decision path.

Sleeping persists until new activity. Any nonzero host audio sample wakes it,
including levels below the former -96 dB threshold; parameters, MIDI, transport
changes, file/data adoption and existing explicit host wake events also wake it.
Nonfinite audio is activity rather than a reason to discard a block. Revoking
the readiness flag or asserting `za_keep_awake` also prevents continued sleep.
The wake block executes normally and readiness must be granted again.

Pending/running tasks and completed-but-unreleased task results keep the audio
consumer awake. Use `task_release` after consuming a result; cancellation alone
is not acknowledgment. A private arena also keeps the consumer awake until
released/adopted. This prevents the consumer from sleeping through completion
or losing the opportunity to adopt results. This is deliberately conservative.

Hosts that report offline/non-realtime processing always execute DSP, even if
the plugin grants sleep. This preserves rendering and null comparisons. It
cannot reconstruct state already skipped before rendering; restart the render
from a fresh instance/state when comparing old binaries.

## Validation

`tests/faust/cooperative_idle_check.cpp` exercises no permission, stale grants,
fresh grants, exact-silence sleep, tiny-input wake, parameter wake, keep-awake,
deferred-result acknowledgment, restored selectors/state persistence and offline
processing in the actual JUCE processor. It uses numerical blocks in memory and
loads no audio file. `idle_null.cpp` compares the complete supplied recording
and a quiet/recovery sequence derived from it against continuously active DSP.
The task integration suite checks the pending-result veto in both state ABIs.

All existing plugins must be rebuilt to receive this host policy. Existing
binaries keep their previous sleep behavior.


A host buffer-length change also wakes processing and requires a fresh certificate. The catalog audit adds conservative grants to ADS, SaliencePush, DPT, and DDT; see `docs/FAUST-Qualification.md` for their state proofs and validation.
