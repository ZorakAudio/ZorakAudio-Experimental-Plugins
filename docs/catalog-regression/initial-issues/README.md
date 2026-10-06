# Initial failed checks

IPCProbeA/B: the fixture set the primary to Receiver but checked the peer, which remained Sender. Corrected to set and check the peer Receiver.

Five Faust editor fixtures: the link dropped cached JUCE dependencies along with the JSFX archive. The actual Faust processor now resolves createPluginFilter first while the archive supplies the required JUCE modules. No JSFX processor is selected.

GTS format build: an empty cached JUCE GUI object caused unresolved symbols. Deleting that object and rebuilding completed both VST3 and CLAP wrappers. The cause of the empty object is undetermined.

TextureXY: point_count was incorrectly checked through a DSP publication snapshot even though it is UI-owned. Visual inspection then revealed a real default-mode bug: stale DSP heap publications overwrote UI gesture arrays. The original path had 4777 green pixels outside the dragged path bounds. Explicit directional ownership fixes this; the final bank fixture requires the expected connected path and nonzero post-release audio with silent input.

These initial failures are preserved as diagnostics. Only final completed workers count as PASS.

Contour/Texture: file-service scratch grew the logical heap, moving the automatic suffix away from waveform/metadata addresses. Their DSP produced audio but previews were flat. Explicit display/metadata ranges restore the preview. Final loaded workers require waveform pixels across more than 20 rows.
