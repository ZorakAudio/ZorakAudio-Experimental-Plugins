[![Build & Release](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml)

# ZorakAudio Experimental Plugins

ZorakAudio Experimental Plugins is a category-organized repository for building, testing, packaging, and shipping a growing collection of experimental audio plugins.

It is not a single plugin project and it is not a loose dump of prototypes. It is a shared plugin platform: a place where DSP-JSFX, Faust, JUCE, CMake, per-plugin metadata, embedded documentation, and automated packaging all work together so new ideas can be added quickly without flattening the repository into chaos.

## What lives here

This repository currently brings together:

- experimental **JSFX** plugins compiled into **VST3** and **CLAP** through the DSP-JSFX/JUCE toolchain
- experimental **Faust** plugins packaged through the same JUCE-based infrastructure
- a category-first plugin tree under `plugins/`
- per-plugin metadata via leaf-local `plugin.json` files instead of a central `plugins.json`
- per-plugin `README.md` files that are embedded directly into each plugin's `?` help panel
- shared build, CI, packaging, and validation tooling for the entire catalog

The current top-level plugin categories include:

- `Ambience`
- `Control`
- `Dynamics`
- `Restoration`
- `Spatialization`
- `Spectral`

## Repository shape

Every plugin is a self-contained leaf:

```text
plugins/
  <Category>/
    <PluginKey>/
      plugin.json
      README.md
      src/
      tests/    # optional
      docs/     # optional
      assets/   # optional
```

That keeps the tree future-proof:

- categories stay readable and intentionally broad
- plugin metadata lives beside the source it describes
- display names can evolve without forcing path redesigns
- build and CI discovery stay automatic as the catalog grows

The root README is intentionally high-level. Plugin-specific details belong in each leaf `README.md`.

## Build and packaging model

The repository auto-discovers plugin leaves from the `plugins/` tree. There is no root plugin registry to maintain.

Useful entry points:

```bash
python scripts/build.py --list
python scripts/build.py --config Release --tag dev --out dist
```

Release artifacts are packaged by category so the output mirrors the repository structure inside `VST3/` and `CLAP/`. That keeps installs organized and makes the repository itself a close preview of the shipped layout.

## Correctness and validation

The repository is moving away from treating REAPER null tests as the primary validation story.

The scalable correctness path is the built-in **WDL/EEL2 shadow runtime** enabled with `--correctness-check`:

```bash
python scripts/build.py --config Release --tag dev --out dist --correctness-check
```

You can also target a single plugin:

```bash
python scripts/build.py --config Release --tag dev --out dist --only DDT --correctness-check
```

That mode builds JSFX plugins with shadow EEL2 instrumentation so the compiled DSP-JSFX path can be checked against a WDL/EEL2 reference execution path. For a repository designed to scale across many plugins, that is a better long-term fit than centering the workflow around REAPER project-based null tests.

## Why this repository matters

This repository is the experimental side of the ZorakAudio plugin ecosystem: a place to explore new DSP ideas seriously, with real packaging, real documentation, real correctness tooling, and a structure that can keep expanding without collapsing into manual bookkeeping.

If you are browsing the project for the first time, the best next places to look are `plugins/README.md`, any individual `plugins/<Category>/<PluginKey>/README.md`, and `scripts/new_plugin.py`.
