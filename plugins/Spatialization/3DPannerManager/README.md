# 3D Panner Manager

A scene/camera controller for older Hyperreal panner revisions that implement the manager IPC subscription. It publishes a named camera and optional SceneVerb metadata while each linked panner retains its own source position.

**The maintained V7.1.2 3DPanner, HyperrealFast, HyperrealFaust and HyperrealHybrid operate locally and do not subscribe to this manager.** Their retained manager parameters are compatibility identities. The manager cannot control those current builds.

## Quick start

1. Use a panner revision that supports manager linking.
2. Set the manager's **Name** and select that exact name in each compatible panner.
3. Enable **Manager Enable** and confirm peers in the linked-ID list.
4. Adjust **Camera Rotation**, **Distance Scale** and **Elevation**, or use the orbit canvas.
5. Enable **SceneVerb** only when linked panners support and opt into its metadata.

## Controls

- **Camera Rotation** and **Rotation Scale** set listener yaw.
- **Distance Scale** scales linked distances; **Elevation** sends a global height hint.
- **Third Person**, **Orbit Angle**, **Orbit Radius** and **Look At Pivot** control orbit/parallax.
- **Automation Smooth** publishes a smoothing hint; **Tracker Input** is reserved.
- **Linked List Scroll** or the wheel scrolls the linked-ID list.
- **SceneVerb Enable**, **Late Field**, **Scene Size**, **Scene Damping**, **Low Protect** and **Verb Duck** publish optional room settings.

## Routing and state

This is a UI/control utility with stereo passthrough. Visible controls live in the canvas; hidden sliders retain saved/automation state. The host must process it and compatible subscribers for IPC to advance. The manager does not force a panner to implement a missing subscription.
