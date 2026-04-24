# Plugin Tree Conventions

Every buildable plugin in this repository lives as a **leaf directory under `plugins/`**.

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

## No subcategories

There is **one category folder only**.

Use:

```text
plugins/Spatialization/DDT/
```

Not:

```text
plugins/Spatialization/Distance/DDT/
```

If a category becomes too broad, add a new top-level category instead of inserting another nesting layer.

## `PluginKey`

`PluginKey` is the stable leaf folder name. In most cases it matches the plugin slug.

Examples:

- `plugins/Dynamics/RED/`
- `plugins/Spatialization/DOT/`
- `plugins/Spectral/TSEQ/`

The user-facing display name lives in `plugin.json` as `name`.

## `README.md`

Each leaf `README.md` is required.

The build embeds it directly into the plugin, and the in-plugin `?` panel renders that markdown at runtime. Treat the leaf README as the canonical user help page for the current source, controls, routing, and workflow.

For JSFX plugins, historical `// #HELP:` comments are no longer the primary documentation target. Keep durable user-facing help in the leaf README.

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

Core conventions:

- `name` is the human-facing plugin name
- `slug` is the stable build target identifier
- `pluginCode` is the 4-character JUCE plugin code
- `entry` points at the single build entry file for the leaf
- `pluginType` is `jsfx` or `faust`

## `tests/`

`tests/` is optional.

Do not assume REAPER `.rpp` files are discovered by CI. The repo's documented JSFX correctness path is `scripts/build.py --correctness-check`. Keep `tests/` for future fixtures, manual sessions, targeted assets, or whatever a specific plugin genuinely needs.

## Adding a plugin quickly

Use the scaffold helper:

```bash
python scripts/new_plugin.py Spatialization NewTool   --name "New Tool"   --plugin-code NTL1   --plugin-type jsfx
```

Or create the leaf manually and then run:

```bash
python scripts/build.py --list
```

If discovery succeeds, the normal build and packaging flow will pick the plugin up automatically.
