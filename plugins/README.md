# Plugin Tree Conventions

Every buildable plugin in this repository now lives as a **leaf directory under `plugins/`**.

The structure is intentionally flat at the category level:

```text
plugins/<Category>/<PluginKey>/
  plugin.json
  README.md
  src/
    <entry>.jsfx | <entry>.dsp
  tests/        # optional
  docs/         # optional
  assets/       # optional
```

## No subcategories

There is **one category folder only**.

Use:

```text
plugins/Spatialization/DDT/
```

Not:

```text
plugins/Spatialization/Distance and Occlusion/DDT/
```

If a category becomes too broad, add a new top-level category instead of inserting another nesting layer.

## `PluginKey`

`PluginKey` is the stable leaf folder name.

In this repository it is usually the same as the plugin slug, for example:

- `plugins/Dynamics/RED/`
- `plugins/Spatialization/DOT/`
- `plugins/Spectral/TSEQ/`

The display name shown to users lives in `plugin.json` as `name`, so it can be longer and more descriptive than the folder key.

## `README.md`

Each leaf `README.md` is required. The build embeds it directly into the plugin, and the in-plugin `?` panel renders that markdown at runtime.

For JSFX plugins, the historical `// #HELP:` macro is now deprecated. Keep user-facing help in the leaf README instead.

## `plugin.json`

Minimal example:

```json
{
  "name": "Reverb Expanding Downwards (RED)",
  "slug": "RED",
  "pluginCode": "RED1",
  "bundleId": "com.zorakaudio.experimental.red",
  "clapId": "com.zorakaudio.experimental.red",
  "clapFeatures": ["audio-effect"],
  "pluginType": "faust",
  "entry": "src/Reverb Expanding Downwards (RED).dsp"
}
```

## Conventions

- `name` is the human-facing plugin name.
- `slug` is the stable build target identifier.
- `pluginCode` is the 4-character JUCE plugin code.
- `entry` points at the single build entry file for the leaf.
- `pluginType` is `jsfx` or `faust`.
- `tests/` is optional. Any `.rpp` files there are discovered by the JSFX null-test workflow.
- release packaging mirrors the top-level category into `VST3/` and `CLAP/`.

## Adding a plugin quickly

Use the scaffold helper:

```bash
python scripts/new_plugin.py Spatialization NewTool \
  --name "New Tool" \
  --plugin-code NTL1 \
  --plugin-type jsfx
```

Or create the leaf manually and then run:

```bash
python scripts/build.py --list
```

If discovery succeeds, the build and CI will pick the new plugin up automatically.
