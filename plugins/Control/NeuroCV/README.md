# NeuroCV

## What it is
NeuroCV is a **predictive CV field prototype** built around multi-lane audio analysis and lightweight adaptive behavior.

The current JSFX source exposes a family of read-only control signals such as:

- master unipolar activity
- master bipolar centered motion
- surprise
- uncertainty
- structure
- body
- confidence
- regime

It also supports routing one selected analysis lane out to an audio pair so the control field can be used elsewhere in a multichannel setup.

---

## Why use it
Use NeuroCV when you want modulation that feels **learned, predictive, or analysis-driven** instead of hand-drawn or cyclic.

It is aimed at experiments where the audio itself becomes the control source.

---

## Quick start
1. Treat this folder as a documented prototype first. It is currently parked in-tree with `plugin.json.bak`, not a live build-catalog entry.
2. Feed the source audio into 1/2.
3. Choose **Listen**, **Adaptive**, or **Perform** mode depending on whether you want observation, learning, or active use.
4. Watch the read-only CV lanes and tune **Range**, **Smooth**, **Surprise Blend**, **Learn Rate**, and **Grounding**.
5. Use **Audio Route Lane** and **Audio Route Pair** if you want one lane sent out as routable audio/CV.

---

## Main controls
### Mode
Listen observes, Adaptive learns, and Perform behaves like the active control stage.

### Master outputs
CV Uni Output and CV Bi Output are the main read-only summary lanes.

### Learning / stability
Range scales the overall output span. Smooth sets output smoothing. Surprise Blend and Learn Rate control how reactive and how adaptive the field feels. Grounding pulls it back toward stability.

### Analysis timing
Analysis frame sets the working time scale of the analysis pass.

### Routing
Audio Route Lane chooses which internal lane is exported. Audio Route Pair chooses where it is sent.

### Read-only feature lanes
Surprise, Uncertainty, Structure, Body, Confidence, and Regime expose the current internal field state for monitoring or routing.

---

## Routing / notes
The source currently declares many input and output lanes, with normal audio on **1/2** and additional routable pairs extending upward.

This is useful when you want to park CV-like analysis on channels such as 3/4, 5/6, 7/8, and beyond.

---

## Notes
NeuroCV is documented here because it is part of the tree, but this pass does **not** promote it into the normal build catalog.

Think of this README as documentation for the current source and current intent, not as a claim that the plugin is already a public/stable catalog entry.

---

## In one sentence
NeuroCV is an in-tree prototype that turns audio analysis into a predictive multi-lane control field.
