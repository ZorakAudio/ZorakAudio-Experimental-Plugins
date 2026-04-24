[![Build & Release](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml)

# ZorakAudio Experimental Plugins

ZorakAudio Experimental Plugins is a category-organized repository for building, validating, packaging, and shipping a growing catalog of experimental audio tools.

This repo is not a single-plugin project and it is not a loose pile of prototypes. It is a shared plugin platform where DSP-JSFX, Faust, JUCE, CMake, per-plugin metadata, embedded markdown help, and automated packaging all work together.

## What lives here

This repository currently brings together:

- experimental **JSFX** plugins compiled into **VST3** and **CLAP** through the DSP-JSFX/JUCE toolchain
- experimental **Faust** plugins packaged through the same JUCE-based infrastructure
- a category-first plugin tree under `plugins/`
- per-plugin metadata via leaf-local `plugin.json`
- per-plugin `README.md` files embedded into each plugin's `?` help panel
- shared build, validation, CI, and packaging tooling for the whole catalog

Current top-level categories:

- `Ambience`
- `Control`
- `Dynamics`
- `Restoration`
- `Spatialization`
- `Spectral`

## Repository shape

Every buildable plugin lives as a self-contained leaf:

```text
plugins/
  <Category>/
    <PluginKey>/
      plugin.json
      README.md
      src/
      tests/      # optional
      docs/       # optional
      assets/     # optional
```

That layout keeps the catalog scalable:

- categories stay readable and intentionally broad
- plugin metadata stays beside the source it describes
- display names can evolve without forcing path redesigns
- build discovery stays automatic as the tree grows
- documentation ships with the plugin instead of drifting into a wiki

The root README stays high level. Plugin-specific behavior, routing, controls, and workflow notes belong in each leaf `README.md`.

## Build and packaging model

The build discovers plugin leaves from the `plugins/` tree.

Useful entry points:

```bash
python scripts/build.py --list
python scripts/build.py --config Release --tag dev --out dist
```

Release artifacts are packaged by category so the output mirrors the repository structure inside `VST3/` and `CLAP/`.

## Correctness and validation

The scalable JSFX validation path is the built-in **WDL/EEL2 shadow runtime** enabled with `--correctness-check`:

```bash
python scripts/build.py --config Release --tag dev --out dist --correctness-check
```

Target a single plugin when needed:

```bash
python scripts/build.py --config Release --tag dev --out dist --only DDT --correctness-check
```

That mode checks the compiled DSP-JSFX path against a WDL/EEL2 reference execution path. This repo is no longer documented around the legacy REAPER/AHK null-test workflow.

## Documentation model

Each plugin leaf `README.md` is the canonical user-facing help page for that plugin.

The intent is:

- root docs explain the platform
- category docs explain the catalog shape
- leaf docs explain what the plugin actually does right now

That matters because the leaf README is what ends up inside the plugin help UI.

## Getting oriented

Good next places to look:

- `plugins/README.md`
- any individual `plugins/<Category>/<PluginKey>/README.md`
- `scripts/build.py`
- `scripts/new_plugin.py`

## Release-note intent for this pass

This documentation pass is focused on:

- syncing plugin READMEs to the current upstream JSFX and Faust sources
- cleaning up category indexes so they match the actual tree
- removing the outdated DSP-JSFX REAPER null-test workflow from CI
- drafting a release note for the next catalog refresh
