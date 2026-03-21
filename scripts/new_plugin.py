from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pluginlib import DEFAULT_BUNDLE_ID_BASE, DEFAULT_CLAP_FEATURES, PLUGINS_ROOT_NAME


def slug_token(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", text.lower())
    return token or "plugin"


def sanitize_path_part(label: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise SystemExit(f"{label} must not be empty")
    if value in {".", ".."}:
        raise SystemExit(f"{label} must not be '.' or '..'")
    if any(ch in value for ch in r'\\/'):
        raise SystemExit(f"{label} must be a single path part, not a nested path: {value!r}")
    return value


def entry_filename(key: str, plugin_type: str, entry_base: str) -> str:
    stem = entry_base.strip() or key
    suffix = ".jsfx" if plugin_type == "jsfx" else ".dsp"
    return stem + suffix


def build_jsfx_template(name: str) -> str:
    return (
        f"desc:{name}\n"
        "\n"
        "slider1:0<0,1,0.01>Example\n"
        "\n"
        "@init\n"
        "\n"
        "@slider\n"
        "\n"
        "@sample\n"
        "spl0 = spl0;\n"
        "spl1 = spl1;\n"
    )


def build_faust_template(name: str) -> str:
    return (
        f'declare name "{name}";\n'
        'declare author "ZorakAudio";\n'
        "\n"
        "process = _, _;\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scaffold a new plugin leaf folder under plugins/<Category>/<PluginKey>/"
    )
    ap.add_argument("category", help="Top-level category folder, for example Spatialization or Dynamics")
    ap.add_argument("key", help="Stable plugin folder key, usually the slug")
    ap.add_argument("--name", required=True, help="Human-facing plugin name shown in hosts and packages")
    ap.add_argument("--plugin-code", required=True, help="Four-character JUCE plugin code")
    ap.add_argument("--plugin-type", choices=("jsfx", "faust"), required=True)
    ap.add_argument("--entry-base", default="", help="Entry filename stem inside src/; defaults to the key")
    ap.add_argument("--bundle-id", default="", help="Explicit bundle id; defaults to com.zorakaudio.experimental.<slug>")
    ap.add_argument("--clap-id", default="", help="Explicit CLAP id; defaults to the bundle id")
    ap.add_argument("--clap-feature", action="append", dest="clap_features", help="Repeatable CLAP feature flag")
    ap.add_argument("--force", action="store_true", help="Overwrite scaffold files if the leaf already exists")
    args = ap.parse_args()

    category = sanitize_path_part("category", args.category)
    key = sanitize_path_part("key", args.key)
    name = args.name.strip()
    plugin_code = args.plugin_code.strip()
    plugin_type = args.plugin_type.strip().lower()

    if not name:
        raise SystemExit("--name must not be empty")
    if len(plugin_code) != 4:
        raise SystemExit("--plugin-code must be exactly 4 characters")

    repo_root = Path(__file__).resolve().parents[1]
    plugins_root = repo_root / PLUGINS_ROOT_NAME
    leaf_dir = plugins_root / category / key
    src_dir = leaf_dir / "src"

    if leaf_dir.exists() and any(leaf_dir.iterdir()) and not args.force:
        raise SystemExit(f"Target leaf already exists and is not empty: {leaf_dir}\nUse --force to overwrite scaffold files.")

    entry_name = entry_filename(key=key, plugin_type=plugin_type, entry_base=args.entry_base)
    entry_rel = Path("src") / entry_name
    default_bundle_id = f"{DEFAULT_BUNDLE_ID_BASE}.{slug_token(key)}"
    bundle_id = args.bundle_id.strip() or default_bundle_id
    clap_id = args.clap_id.strip() or bundle_id
    clap_features = args.clap_features or list(DEFAULT_CLAP_FEATURES)

    leaf_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    plugin_json = {
        "name": name,
        "slug": key,
        "pluginCode": plugin_code,
        "bundleId": bundle_id,
        "clapId": clap_id,
        "clapFeatures": clap_features,
        "pluginType": plugin_type,
        "entry": str(entry_rel).replace("\\", "/"),
    }

    readme_text = (
        f"# {name}\n\n"
        f"- Category: `{category}`\n"
        f"- Folder key / slug: `{key}`\n"
        f"- Plugin type: `{plugin_type}`\n\n"
        "This README is embedded into the plugin and shown in the in-plugin `?` help panel at build time.\n\n"
        "## Overview\n\n"
        "Describe the DSP idea, intended use, and any design constraints here.\n"
    )

    source_text = build_jsfx_template(name) if plugin_type == "jsfx" else build_faust_template(name)

    (leaf_dir / "plugin.json").write_text(json.dumps(plugin_json, indent=2) + "\n", encoding="utf-8")
    (leaf_dir / "README.md").write_text(readme_text, encoding="utf-8")
    (src_dir / entry_name).write_text(source_text, encoding="utf-8")

    print(f"Created: {leaf_dir}")
    print(f"- plugin.json")
    print(f"- README.md")
    print(f"- {entry_rel}")


if __name__ == "__main__":
    main()
