# TexturePhase Surface PM

An input-driven texture effect. A loaded sample controls a small phase/time warp of the live stereo input. The texture acts as the modulator rather than replacing the sound with a separate instrument. No live input means no generated output; this prototype has no MIDI carrier or analog fallback.

## Quick start

1. Load a texture into the **Texture Sample** file slot (slot 0).
2. Feed stereo audio into the plugin and wait for the loader to become ready.
3. Start with low **Surface transfer** and **Max depth**, then raise them while listening.
4. Adjust **Texture motion** and **Stabilize**; choose **free** or **onset** locking.
5. Set **Output** to compare levels. The three waveform views show texture, input and processed signal.

## Controls

- **Rescan texture** requests a refresh of the selected texture.
- **Surface transfer** controls how much texture motion reaches the signal.
- **Max depth (ms)** bounds the time-warp depth, from 0.02 to 3 ms.
- **Texture motion** controls traversal of the texture.
- **Stabilize** restrains motion using input activity.
- **Texture lock** runs freely or locks to detected onsets.
- **Onset sensitivity (dB)** sets the input level for onset detection.
- **Output (dB)** is the final trim.
- **Scope gain** changes waveform display scale.

The declared sliders are hidden because the custom canvas owns these controls. Their automation and saved-state identities remain available.

## Routing and loading

Use stereo input/output 1/2. The sample-pool loader prepares the slot asynchronously; an empty or failed texture slot cannot provide the intended modulation. Replacing the texture changes its sample generation. Use Rescan when updating the material.

This source uses the repository's sample-pool extension and requires its native compiler/runtime. It is a development prototype, rather than stock REAPER JSFX with equivalent file-loading services.
