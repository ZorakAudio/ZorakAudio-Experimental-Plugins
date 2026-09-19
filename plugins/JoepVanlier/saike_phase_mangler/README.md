# Saike Phase Mangler (BETA)

Vendored upstream JSFX source, packaged for the ZorakAudio JSFX-to-JUCE build system.

- Upstream source: `PhaseMangler/saike_phase_mangler.jsfx`
- Author: Joep Vanlier
- Version: 0.05
- Tags: phase shifting plugin
- Upstream license declaration: MIT
- DSP source: preserved as supplied; no algorithmic/DSP edits were made by this packaging pass.
- Dependency layout: preserved under `src/`; the current source resolver can resolve REAPER `provides:`-style bare imports.
- Image resources: mirrored under `Resources/` so the current build script can embed static `gfx_loadimg` assets.

## Upstream description

# A plugin to manipulate audio phase in a frequency dependent manner
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

## Attribution / license

See `LICENSE.upstream` and the original source header.
