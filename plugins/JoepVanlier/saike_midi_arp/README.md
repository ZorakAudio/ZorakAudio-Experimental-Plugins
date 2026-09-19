# Saike MIDI ARP (beta)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `saike_midi_arp/saike_midi_arp.jsfx`
- Author: Joep Vanlier
- Version: 0.44
- Tags: midi arpeggiator
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A small utility JSFX to arpeggiate midi chords.
Program patterns and play chords. The JSFX will then play the notes according to that note pattern.

## Attribution / license

See `LICENSE.upstream` and the original source header.
