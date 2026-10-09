# Anti-Salience MX

Anti-Salience MX v1.1.0 pushes a sound into the background by reducing the cues
that make it stand out. It combines the Salience Push core, adaptive suppression
of persistent perceptual anchors, and optional selective scatter. The supplied
JSFX algorithm is retained; this is a new plugin identity, not a parameter-compatible
upgrade of SaliencePush. Existing sessions can keep their installed SaliencePush.

## Quick start and routing

1. Insert it on the sound that should become less prominent. Inputs 1–2 are the
   stereo **Source**; outputs 1–2 are the processed sound.
2. Optionally send the foreground sound to inputs 3–4, **Foreground Ref**. These
   inputs guide suppression; they are not mixed into the output. With no reference,
   the processor uses the source's own prominence.
3. Choose **Neutral**, **Vocals**, **SFX**, **Foley**, or **BG** for the source.
4. Start with **Scatter Mix = 0%**. Adjust **Erasure**, **Transient Tame** and
   **Identity Retention** while listening in the full mix. Give the slow analysis
   time to settle before judging a change.
5. Add Scatter only when a change in texture and spatial character is welcome.
   Use **Overall Mix** for dry blending and level-match with **Output Trim**.

## Controls

| Control | Range / default | Purpose |
| --- | --- | --- |
| Profile | Neutral, Vocals, SFX, Foley, BG / Neutral | Changes the core policy and band priorities. |
| Erasure | 0–100% / 68% | Strength of the core and adaptive suppression. |
| Transient Tame | 0–100% / 48% | Suppresses prominent upper-band transients. |
| Identity Retention | 0–100% / 72% | Protects body and natural spectral hierarchy. |
| Lock Half-Life | 80–2500 ms / 520 ms | Speed and persistence of anchor detection. |
| Scatter Mix | 0–100% / 0% | Morphs targeted residue into its scattered replacement. |
| Scatter Spread | 0–100% / 48% | Diffuser separation and stereo disagreement. |
| Scatter Tail | 0–100% / 24% | Scatter recirculation and persistence. |
| Novelty Guard | 0–100% / 84% | Smooths target changes and restrains scatter prominence. |
| Flash Guard | 0–100% / 92% | Restrains abrupt broadband changes and onset scatter. |
| Adaptive Targeting | Off/On / On | Enables the post-core eight-band anchor tax. |
| PE2 Anchor Memory | Off/On / On | Retains debt for recurring anchors. |
| Modulation De-Binding | Off/On / On | Suppresses bands dominating coherent envelope motion. |
| Core Level Lock | Off/On / On | Slowly restores some clean-core level loss. |
| Overall Mix | 0–100% / 100% | Dry-to-processed blend after the scatter morph. |
| Output Trim | −18 to +12 dB / 0 dB | Final processed gain before peak containment. |
| Bypass | Active/Bypass / Active | Smoothed dry bypass; scatter continues draining. |

The custom interface owns these controls. All 17 sliders are hidden from the
generic slider panel but remain host parameters for automation. Drag horizontally;
Shift-drag gives finer movement, the wheel steps, and right-click restores a
control's default. Hover for help. **Transparent**, **Anti-Lock**, **Scattered**
and **Obliterate** set groups of controls; they leave the selected Profile alone.
Enlarge the window if the interface asks for more editing space.

## What the layers and meters mean

The first layer applies slow, capped attenuation to Form, Edge, Air and stereo
Side cues. The second detects surviving anchors across eight broad regions and
applies additional attenuation. Neither suppression layer creates ambience.
**Core Level Lock** is a separate gain restoration stage, capped internally at
1.55×; the final processed signal is therefore not guaranteed to be quieter.

Scatter replaces only the targeted portion of the core. At a settled **Scatter
Mix = 0%**, its return contributes nothing. Increasing it changes phase, texture
and stereo behavior; it is a sound-design choice. Parameter smoothing means a
return fades out when the control is moved back to zero.

**Input**, **Core**, **Final Lock**, **Strongest Survivor** and the band/history
charts are estimates from this model. They are not measured human audibility or
a guarantee that a sound will become imperceptible. Common Cut, Side Cut and
Anchor Tax show the processing effort; Ref Active shows reference detection.
"Engine Idle" is a display condition, not an explicit runtime sleep certificate.

## Latency, bypass and limits

The clean core declares zero PDC latency. Optional scatter uses short internal
delays and can leave a tail. A two-second tail hint is declared; it is not proof
that every internal memory has settled after two seconds. The processed branch
has linked 0.98 peak containment. Bypass is applied afterward and deliberately
passes the dry input without that containment or Output Trim.

This is intended for background placement and perceptual sound design, not
transparent mastering, source separation or a safety limiter. Use listening and
level matching to assess the result. Build with
`python scripts/build.py --only AntiSalienceMX`. Its native shared-state graphics
use the same production runtime as the other native Legacy plugins.
