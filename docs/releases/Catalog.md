# ZorakAudio — sound, space, and perception

**Turn recordings into instruments. Give impacts physical weight. Let layers make room for each other.**

This major release brings together instruments and effects that explore how sound attracts attention, carries motion and occupies space.

**VST3 + CLAP · Windows, macOS and Linux · Custom interfaces · Built-in manuals**

## Pick your collection

| Collection | What's inside |
| --- | --- |
| **Essentials** | **13 selected instruments and effects.** The flagships and a focused toolkit for shaping sound. Start here. |
| **All** | **36 ZorakAudio plugins.** The complete non-Joep collection, including specialist tools and alternate designs. |
| **JoepVanlier** | **50 native builds from Joep Vanlier / Saike's JSFX catalog.** Available as a separate collection for use in VST3/CLAP hosts. |

Each ZIP includes **all three operating systems**. Essentials is included in All; choose one, then add JoepVanlier if you want it.

## File Loader — turn long takes into playable material

**Find the events. Refine the cuts. Play the bank.**

The shared File Loader includes **auto-segmentation** with waveform previews and auditioning. Adjust detection to preserve whole gestures and tails, split or merge cuts, and guide extraction with examples of wanted and unwanted events. Load the results as separate samples or assemble a continuous texture—all in memory, leaving the original recordings untouched.

[![File Loader auto-segmentation: previewing and refining cuts in Sample](https://raw.githubusercontent.com/ZorakAudio/ZorakAudio-Experimental-Plugins/main/docs/media/auto-segmentation.gif)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/raw/refs/heads/main/docs/media/auto-segmentation.mp4)

*30-second preview. Click for the longer demonstration with audio (3:06).*

[File Loader and sample-extraction workflow](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/docs/FileImportRecipes.md).

## Meet the Essentials

Click a name for its manual, or open **?** inside the plugin.

| Plugin | What makes it worth exploring |
| --- | --- |
| [Sample](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Spectral/SampleFaust/README.md) | **Make a folder of recordings playable.** A multisampler with coherent sample selection, tape, hybrid and granular playback, expressive envelopes, cleanup and harmonic character. Load a bank and perform. |
| [Corpus](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Spectral/Corpus/README.md) | **Explore a recording as an instrument.** Acoustic analysis organizes its material into a navigable map; MIDI, drawn paths and coherence controls guide new sequences through its textures. |
| [Hyperreal 3D Panner](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Spatialization/3DPanner/README.md) | **Place sound in a scene.** Artistic and Physical renderers combine direction, distance, elevation, obstruction and room cues for headphone spatialization. Drag sources, shape their motion, and explore listener geometry. |
| [Anti-Salience MX](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Spatialization/AntiSalienceMX/README.md) | **Shape what commands attention.** Adaptive processing targets persistent tonal and transient cues to help a layer recede, with identity protection, optional foreground guidance and selective spatial scatter. |
| [CMD — Cross-Mix Somatic Bus](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Spectral/CMD/README.md) | **A mix whose tracks cooperate.** Instances coordinate spectral turn-taking across a mix, while shared motion signals influence body, width and saturation. Designed for interacting layers. |
| [SOMA](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Dynamics/SOMA/README.md) | **Loudness with weight and identity.** A psychoacoustic limiter that combines perceptual preservation with body reinforcement driven by gain reduction. Explore punch, density and controlled aggression. |
| [ATTACK](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Dynamics/ATTACK/README.md) | **Sculpt the force of a hit.** Body, Edge, Punch, Crack and Tight macros reshape the weight and bite of drums, impacts, percussion and Foley. |
| [GTS — Gaussian Transient Shaper](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Dynamics/GTS/README.md) | **Pull attack and sustain apart.** Gaussian smoothing separates the signal into independently adjustable components, giving direct control over transient definition, body and bloom. |
| [EasyExpander](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Dynamics/EasyExpanderFaust/README.md) | **Control what lingers between events.** Perceptually weighted downward expansion reduces quiet spill, noise and tails, with adjustable depth and a contour ranging from gentle to gate-like. |
| [ModTilt](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Dynamics/ModTilt/README.md) | **Tilt the motion of sound.** Rebalance slow and fast envelope movement to shift a sound toward snap and urgency, or weight and lingering body. |
| [ERB Tilt](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Spectral/ERBTilt/README.md) | **Tonal balance shaped around human hearing.** Perceptually weighted tilt shifts use Equivalent Rectangular Bandwidth (ERB) filter banks, a movable pivot, and global loudness compensation. |
| [Click-Be-Gone SG](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Restoration/ClickBeGoneSG/README.md) | **Restore delicate textures.** Prediction and targeted replacement tackle needle-clicks, wet splats and granular spikes. Delta monitoring lets you hear exactly what the processor removes. |
| [GesturePad](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/Control/GesturePad/README.md) | **Draw a performance.** Turn gestures into MIDI notes and multiple control lanes, including speed and acceleration. Replay, loop or perform them to animate instruments and effects. |

## Joep Vanlier / Saike — 50 creative tools, beyond REAPER

A separate collection brings **50 of [Joep Vanlier / Saike's instruments and effects](https://github.com/JoepVanlier/JSFX/blob/master/README.md)** into VST3 and CLAP hosts, with their custom interfaces and built-in manuals. Explore expressive synthesis, resonant textures, atmospheric spaces and unapologetic distortion.

**The original algorithms, with measured performance.** A few highlights from Joep's catalog:

| Plugin | Explore | Processing speed vs WDL/EEL2 |
| --- | --- | --- |
| [Yutani](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/Saike_Yutani/README.md) | Bass synthesis with characterful nonlinear filters and modulation. | **1.36–1.39×** |
| [Protosynth](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/saike_protosynth/README.md) | Polyphonic synthesis with eight oscillators and unusual mixing possibilities. | **1.48–1.73×** |
| [FM Filter 2](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/Saike_FMFilter2/README.md) | Yutani's filter character applied to your own audio. | **1.46–1.47×** |
| [Partials](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/saike_partials/README.md) | Turn incoming sound into pitched, physically inspired resonances. | **1.18–1.24×** |
| [Filther](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/Filther/README.md) | Dynamic filtering, drawable waveshaping and feedback. | **1.28–1.42×** |
| [Squashman](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/Squashman/README.md) | Multiband distortion with extensive modulation. | **1.10–1.12×** |
| [Ravager](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/Ravager_MB/README.md) | Extreme upward compression for aggressive detail and density. | **1.53–1.56×** |
| [Dusk Verb](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/saike_duskverb/README.md) | Atmospheric reverb, granular resampling, shimmer and shifting. | **1.06–1.15×** |
| [Reflectosaurus](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/Reflectosaurus/README.md) | Connect delays, filters and reverb into evolving spaces. | Similar (**1.02–1.05×**) |
| [Amaranth](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/plugins/JoepVanlier/Amaranth/README.md) | Capture or load sound, then perform overlapping grains. | Similar (**1.01–1.03×**) |

These compare audio-processing speed against **WDL's native EEL2 JIT**, without simplifying the algorithms. Ranges cover 64- and 512-sample buffers at 48 kHz on Windows, using default controls; audio and MIDI matched in these tests. Interface and DAW overhead are excluded, and results vary with settings. [Full results for all 50 plugins, methods and exceptions](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/docs/Joep-Performance.md).

The wider update improves idle CPU behavior, host automation, MIDI/UI synchronization and built-in help. Authors also gain background analysis, sample banks and communication between instances. [Author guide](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/docs/DSP-JSFX-Guide.md).

## Start exploring

Download **Essentials**, choose your operating system's folder, install VST3 or CLAP, and rescan in your DAW. [Installation, platform requirements and upgrade notes](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/docs/Release-Collections.md).

This is an experimental release, and listening feedback is welcome. Known output differences in the Joep **Abyss and Lava** builds are recorded in the [comparison report](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/blob/main/docs/Joep-Performance.md). [Share feedback or report an issue](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/issues).

Thanks to **Joep Vanlier / Saike** and the JSFX/WDL, FAUST, JUCE and CLAP communities.
