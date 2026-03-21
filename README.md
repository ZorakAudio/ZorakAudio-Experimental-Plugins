[![Build & Release](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml)

[![DSP-JSFX Null Test vs. Reaper JSFX](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/dsp-jsfx-nulltest.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/dsp-jsfx-nulltest.yml)

# ZorakAudio Experimental Plugins

This repository is a **multi-plugin laboratory** for ZorakAudio DSP ideas.

It now uses a **category-first plugin tree** with **no central `plugins.json`**. Every plugin lives in its own leaf folder under `plugins/`, carries its own `plugin.json`, and is auto-discovered by the build system and CI.

The structure is intentionally simple:

- one top-level **category** folder
- one **plugin key** folder inside that category
- no extra X/Y subcategory layer

That keeps the repo easier to grow, and it lets release packages mirror the same category folders inside `VST3/` and `CLAP/`.

---

## Design rules for the plugin tree

The repository now standardizes on this layout:

```text
plugins/
  <Category>/
    <PluginKey>/
      plugin.json
      README.md
      src/
        <entry>.jsfx | <entry>.dsp
      tests/   # optional
      docs/    # optional
      assets/  # optional
```

Example:

```text
plugins/Spatialization/DDT/
  plugin.json
  README.md
  src/DDT.jsfx
  tests/DDT Null Test.rpp
```

A few conventions make this future-proof:

- `Category` is the only grouping layer under `plugins/`.
- `PluginKey` is a short, stable folder key, usually the same as the plugin slug.
- the human-facing plugin name lives in `plugin.json`, so display names can evolve without forcing deep path design
- if a category gets crowded, add another top-level category instead of introducing another nesting layer

See [`plugins/README.md`](plugins/README.md) for the conventions used by plugin leaves.

Every leaf `README.md` is also embedded into the plugin binary and shown in the in-plugin `?` help panel. The old JSFX `#HELP` comment macro is now deprecated; keep the leaf README as the canonical user-facing guide.

---

## What is in here

The repository currently contains:

- experimental **JSFX** plugins compiled through the DSP-JSFX/JUCE path into **VST3** and **CLAP**
- experimental **Faust** plugins compiled through JUCE into **VST3** and **CLAP**
- the DSP-JSFX AOT compiler, JUCE wrapper code, CMake glue, null-test tooling, and CI workflows needed to build the collection

This is not meant to be a single uniform product line. It is a structured place to explore, test, package, and ship many independent DSP ideas without losing organization as the catalog grows.

---

## Current catalog

### Ambience

- `ADS` — Ambience Discipline System (ADS) — JSFX

### Control

- `GesturePad` — GesturePad — JSFX

### Dynamics

- `ATTACK` — ATTACK — JSFX
- `EasyExpander` — EasyExpander — JSFX
- `GTS` — Gaussian Transient Shaper (GTS) — Faust
- `ModTilt` — ModTilt — Faust
- `RED` — Reverb Expanding Downwards (RED) — Faust
- `RTT` — Reverb Tail Tamer — JSFX

### Restoration

- `ClickBeGoneSG` — Click-Be-Gone (SG) — Faust
- `VAR` — Vocal Air Recovery (VAR) — Faust

### Spatialization

- `3DPanner` — Hyperreal 3D Panner — JSFX
- `DDT` — Designed Distance Topology (DDT) — JSFX
- `DOT` — Designed Occlusion Topology (DOT) — JSFX
- `DPT` — Designed Panning Topology (DPT) — JSFX
- `Roomalizer` — Roomalizer — JSFX
- `SaliencePush` — Salience Push — JSFX

### Spectral

- `BedRock` — BedRock — JSFX
- `ERBTilt` — ERB Tilt — JSFX
- `SpectralStabilizer` — Spectral Stabilizer — JSFX
- `TSEQ` — Temporal Structural EQ (TSEQ) — JSFX
- `Texture` — Texture — JSFX

---

## Build workflow

### 1. Get submodules

```bash
git submodule update --init --recursive
```

### 2. Install Python dependency

```bash
python -m pip install --upgrade pip llvmlite
```

For JSFX null tests you also need `numpy`:

```bash
python -m pip install numpy
```

### 3. Install external build tools

Typical requirements:

- **CMake**
- **Ninja** on macOS/Linux
- **Visual Studio + MSVC** on Windows
- **Faust** if you want to build the `.dsp` plugins

### 4. Inspect discovery

```bash
python scripts/build.py --list
```

### 5. Build everything

```bash
python scripts/build.py --config Release --tag dev --out dist
```

### 6. Build a single plugin

```bash
python scripts/build.py --config Release --tag dev --only DDT
```

`--only` matches category, plugin key, slug, plugin name, repository path, bundle id, and CLAP id.

---

## Release package layout

A packaged build now looks like this:

```text
dist/<tag>/<os>/ZorakAudio-Experimental-Plugins-<tag>-<os>/
  VST3/
    Spatialization/
      Designed Distance Topology (DDT).vst3
  CLAP/
    Spatialization/
      Designed Distance Topology (DDT).clap
  INSTALL.txt
  manifest.json
```

This is intentional: the CI/release artifacts are organized so users can copy the **category folders** from `VST3/` and `CLAP/` directly into their plugin directories.

Practical note: many hosts recurse through subfolders just fine, but not every host behaves the same way. If a specific host does not pick up category subfolders, flatten that host's install folder as a fallback.

Current format behavior in this repository:

- **VST3**: built on all supported platforms
- **CLAP**: currently packaged on Windows and Linux in this repository's build script

---

## JSFX null tests

The Windows null-test workflow uses portable REAPER and auto-discovers test projects from plugin leaf folders.

Typical layout:

```text
plugins/<Category>/<PluginKey>/tests/*.rpp
```

Run locally with:

```bash
python scripts/run_dsp-jsfx_nulltests.py
```

The null-test workflow watches the plugin tree plus the build/runtime scripts that affect discovery and packaging.

---

## Adding a new plugin

The fastest route is to scaffold a leaf folder and then replace the placeholder source.

### Option A: use the scaffold script

```bash
python scripts/new_plugin.py Dynamics NewIdea \
  --name "New Idea" \
  --plugin-code NIDE \
  --plugin-type jsfx
```

That creates:

```text
plugins/Dynamics/NewIdea/
  plugin.json
  README.md
  src/NewIdea.jsfx
```

### Option B: create the leaf manually

1. create `plugins/<Category>/<PluginKey>/`
2. add `plugin.json`
3. add the entry source file in `src/`
4. add `README.md`
5. optionally add `tests/`, `docs/`, or `assets/`
6. run `python scripts/build.py --list`
7. build

Minimal `plugin.json` example:

```json
{
  "name": "Designed Distance Topology (DDT)",
  "slug": "DDT",
  "pluginCode": "DDT2",
  "bundleId": "com.zorakaudio.experimental.ddt",
  "clapId": "com.zorakaudio.experimental.ddt",
  "clapFeatures": ["audio-effect"],
  "pluginType": "jsfx",
  "entry": "src/DDT.jsfx"
}
```

The leaf folder key is usually the same as `slug`, but the human-facing display name can be longer and live in `name`.

---

## Repository highlights

- no central plugin registry to maintain
- categories map cleanly to packaged install folders
- plugin metadata lives beside the source it describes
- CI auto-discovers plugins from the tree
- folder keys are stable and concise, while display names remain flexible

That makes the repository easier to maintain as an experimental plugin collection rather than a flat pile of unrelated folders.
