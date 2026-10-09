# DSP-JSFX IPC Probe A

A diagnostic for native DSP-JSFX shared messages and memory. Probe A and B contain the same test program under separate identities. Either instance can send or receive.

## Quick start

1. Insert Probe A and B on tracks the host will process.
2. Set one **Role** to Sender and the other to Receiver.
3. Give both the same **Bus Name**, initially `ZorakAudio.IPC.Probe`.
4. Run processing. The receiver's sender ID, sequence and received-count display should advance.
5. Change one Bus Name to isolate it, then restore the matching name to reconnect.

## Controls and output

**Role** selects Sender or Receiver. **Bus Name** selects the communication domain; spelling must match. The sender publishes its instance ID, sequence and block length, and updates shared-memory diagnostics.

The sender outputs silence. The receiver outputs a quiet 440 Hz diagnostic tone following its received count, capped at 0.02 amplitude. This replaces incoming audio. Use a test track rather than expecting a transparent effect.

## Notes

Both instances need processing callbacks to send and observe messages. A stopped or host-suspended track may not advance the display. This tests native IPC extensions; stock WDL/EEL2 does not provide them.
