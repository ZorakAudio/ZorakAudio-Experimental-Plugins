# Cross-Mix Somatic Bus (CMD v2)

CMD v2 is a distributed psychoacoustic mixer insert. Put one instance on every track that should coordinate with the others.

For this version, all instances join one global bus. String bus selection is intentionally isolated to the endpoint literals in the JSFX source so a host/string-input layer can replace them later without changing the DSP logic.

## What it does

CMD v2 has two engines:

1. **Declutter Engine** — ERB/gammatone-band turn taking. Each instance publishes its per-band activity into `gmem`; peers decide who owns contested auditory bands. Losing instances yield only in those bands and build fairness credit so ownership rotates instead of locking permanently.
2. **Somatic Coupling Engine** — peer-caused modulation. Each instance publishes pressure, contact, viscosity, thrust, arousal, and fatigue. Other instances translate those shared motion signals into bounded role-aware modulation: ERB-band sympathy, small width movement, and micro-saturation.

This is not sidechain ducking. It is multi-actor shared-state behavior.

## Use

Add CMD v2 to every related layer:

```text
Wet layer A     -> CMD v2
Wet layer B     -> CMD v2
Body layer      -> CMD v2
Impact layer    -> CMD v2
Air/tingle      -> CMD v2
Ambience bed    -> CMD v2
```

Set the **Role** per track. Start with these settings:

```text
Declutter:        35-55%
Somatic:          20-40%
Turn Fairness:    50-70%
Activity Floor:   -46 dB
Max Band Cut:     6-9 dB
Release:          100-180 ms
Influence Out:    50-80%
Influence In:     40-75%
Safety Governor:  65-85%
```

## Controls

### Role

How this layer participates in the shared body.

- **Neutral**: balanced send/receive behavior.
- **Lead**: protected, conservative modulation.
- **Body**: pressure/thrust driven low-mid warmth and subtle saturation.
- **Wet Detail**: contact/viscosity driven texture and air movement.
- **Impact**: transient/thrust dominant, lower receive movement.
- **Ambience**: yields salience, widens/darkens under foreground arousal.
- **Sub/Weight**: low pressure bloom with mono-safe width reduction.
- **Air/Tingle**: high-band contact excitation with fatigue limiting.

### Declutter

Amount of ERB-band masking management.

### Somatic

Amount of peer-caused liveliness injection.

### Turn Fairness

How quickly masked instances build credit and take turns owning contested bands.

### Activity Floor

Noise/activity threshold. Raise it if room tone or hiss keeps triggering the bus.

### Importance / Focus Bias

Priority for contested ERB bands.

### Max Band Cut

Maximum reduction in a single declutter band.

### Release

Envelope release for activity and ownership.

### Influence Out

How strongly this instance drives bus motion.

### Influence In

How strongly this instance reacts to bus motion from peers.

### Safety Governor

Higher values reduce modulation caps. Lower values allow stronger somatic movement.

### Output Trim

Post-process output trim.

## Implementation notes

The DSP follows the current DSP-JSFX communication model:

- `gmem` stores random-access shared state: slots, heartbeats, ERB band energy, fairness credit, and somatic motion scalars.
- `msg_*` is used only for block-rate heartbeat/membership traffic.
- The audio path never relies on same-sample inter-instance feedback.

The single global endpoint appears in only these call sites:

```js
msg_subscribe("ZorakAudio.CMSB.Global");
msg_advertise("ZorakAudio.CMSB.Global", 1);
msg_recv("ZorakAudio.CMSB.Global", ...);
msg_send("ZorakAudio.CMSB.Global", ...);
gmem_attach_size("ZorakAudio.CMSB.Global", GMEM_SIZE);
```

When string bus support is ready, replace that endpoint string with the resolved bus name/namespace. Do not thread bus-name conditionals through the processing code.

## Safety limits

The somatic engine is intentionally bounded:

- ERB-band injection is capped by **Somatic**, **Influence In**, and **Safety Governor**.
- Width movement is small and role-limited.
- Saturation is subtle and unipolar.
- No pitch, delay-time, Haas, or phase modulation is used in this version.

