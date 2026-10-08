[![Build & Release](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml/badge.svg)](https://github.com/ZorakAudio/ZorakAudio-Experimental-Plugins/actions/workflows/release.yml)

# ZorakAudio Experimental Plugins

Start with the [consolidated DSP-JSFX guide](docs/DSP-JSFX-Guide.md) for the current
FAUST interface, background tasks, sample pools, communication, graphics, sleep,
and differences from stock JSFX. Historical qualification reports document their
tested configurations; the guide links the current contracts.

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

The normal AOT and JIT builds automatically check/apply the repository's host
wrapper patches, including when reusing a build directory. CI builds both formats
on Windows, universal2 macOS and Linux, plus the standalone JIT Editor on Windows.
See [building and CI](docs/Build-and-CI.md) for fresh checkout setup, artifact
downloads, full/smoke builds, dependency updates and platform qualification limits.

## Correctness and validation

Structured background tasks (`defer`, `defer_after`, `defer_for`,
`defer_reduce`, and completion joins) are available in the compiled DSP-JSFX
language. See [Structured tasks](docs/Structured-Tasks.md) for the API,
ownership rules, resource bounds, and current qualification limits.

Native shared-state JSFX graphics is available as an opt-in prototype:

```bash
python scripts/build.py --only Sample --native-gfx-legacy --config Release
```

This mode compiles `@gfx` and DSP against one preallocated `options:maxmem` heap.
See [Native GFX legacy mode](docs/Native-GFX-Legacy.md) for concurrency semantics,
performance measurements, supported APIs, and qualification limits.
The existing EEL graphics and bounded-publication native modes remain available.

The JoepVanlier catalog always builds in native Legacy mode, including when no
GFX flag is supplied or `--native-gfx-prototype` is requested:

```bash
python scripts/build.py --only JoepVanlier --config Release
```

Other JSFX retain EEL graphics by default; their native modes remain opt-in.
Faust plugins do not use either JSFX mode. Automatic Legacy is also applied by
the `build_jsfx_aot()` helper for sources under `plugins/JoepVanlier/`.
The shadow EEL correctness monitor rejects selections containing Joep plugins
before build or staging directories are changed.
See [native package coverage and qualification](docs/JoepVanlier-Native-Compatibility.md).
See [non-Joep catalog regression coverage](docs/Plugin-Catalog-Regression.md).

For eligible ordinary JSFX, the validation path is the built-in **WDL/EEL2 shadow runtime** enabled with `--correctness-check`:

```bash
python scripts/build.py --only DDT --config Release --tag dev --out dist --correctness-check
```

Target a single plugin when needed:

```bash
python scripts/build.py --config Release --tag dev --out dist --only DDT --correctness-check
```

That mode checks eligible ordinary DSP-JSFX against a WDL/EEL2 reference.
FAUST-mixed, task-enabled and native Legacy selections reject the shadow monitor;
use their dedicated fixtures and paired processor comparisons. This repo is no longer documented around the legacy REAPER/AHK null-test workflow.

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

## Platform references

Use the [documentation index](docs/README.md) to distinguish current API contracts,
product decisions and fingerprinted qualification checkpoints. The
[consolidated guide](docs/DSP-JSFX-Guide.md) is the starting point.
