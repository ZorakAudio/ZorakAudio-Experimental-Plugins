from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable

PLUGINS_ROOT_NAME = "plugins"
PLUGIN_METADATA_FILENAME = "plugin.json"
PLUGIN_README_FILENAME = "README.md"
DEFAULT_MANUFACTURER_NAME = "ZorakAudio"
DEFAULT_MANUFACTURER_CODE = "Zrak"
DEFAULT_BUNDLE_ID_BASE = "com.zorakaudio.experimental"
DEFAULT_CLAP_FEATURES = ("audio-effect",)
EXPECTED_PLUGIN_PATH_PARTS = 2  # <Category>/<PluginKey>


class PluginDiscoveryError(RuntimeError):
    pass


def _slug_token(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", text.lower())
    return token or "plugin"


@dataclass(frozen=True)
class PluginSpec:
    repo_root: Path
    root_dir: Path
    meta_path: Path
    rel_dir: Path
    install_rel_dir: Path
    category: str
    name: str
    slug: str
    plugin_code: str
    bundle_id: str
    clap_id: str
    clap_features: tuple[str, ...]
    plugin_type: str
    entry_rel: Path
    entry_path: Path
    manufacturer_name: str
    manufacturer_code: str
    readme_path: Path
    raw: dict

    @property
    def repo_rel_dir(self) -> Path:
        return self.root_dir.relative_to(self.repo_root)

    @property
    def key(self) -> str:
        return self.root_dir.name

    @property
    def leaf_name(self) -> str:
        return self.key

    @property
    def category_parts(self) -> tuple[str, ...]:
        return (self.category,)

    @property
    def source_root(self) -> Path:
        return self.entry_path.parent

    @property
    def install_display(self) -> str:
        return str(self.install_rel_dir)


def plugins_root(repo_root: Path) -> Path:
    return repo_root / PLUGINS_ROOT_NAME


def _infer_entry(root_dir: Path) -> Path:
    src_root = root_dir / "src"
    search_roots = [src_root] if src_root.exists() else [root_dir]
    candidates: list[Path] = []
    for base in search_roots:
        for ext in ("*.jsfx", "*.dsp"):
            candidates.extend(sorted(p for p in base.rglob(ext) if p.is_file()))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise PluginDiscoveryError(f"No .jsfx or .dsp entry file found under {root_dir}")
    raise PluginDiscoveryError(
        f"Multiple possible entry files found under {root_dir}. Set 'entry' in {PLUGIN_METADATA_FILENAME}."
    )


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PluginDiscoveryError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PluginDiscoveryError(f"Expected JSON object in {path}")
    return data


def load_plugin(repo_root: Path, meta_path: Path) -> PluginSpec:
    root = plugins_root(repo_root)
    if not meta_path.is_file():
        raise PluginDiscoveryError(f"Metadata file not found: {meta_path}")

    data = _load_json(meta_path)
    plugin_root = meta_path.parent
    try:
        rel_dir = plugin_root.relative_to(root)
    except ValueError as exc:
        raise PluginDiscoveryError(f"Plugin metadata must live under {root}: {meta_path}") from exc

    if len(rel_dir.parts) != EXPECTED_PLUGIN_PATH_PARTS:
        raise PluginDiscoveryError(
            "Plugin metadata must live at "
            f"{PLUGINS_ROOT_NAME}/<Category>/<PluginKey>/{PLUGIN_METADATA_FILENAME}. "
            f"Subcategory folders are no longer supported: {meta_path}"
        )
    category, key = rel_dir.parts

    name = str(data.get("name") or key).strip()
    slug = str(data.get("slug") or "").strip()
    plugin_code = str(data.get("pluginCode") or "").strip()
    manufacturer_name = str(data.get("manufacturerName") or DEFAULT_MANUFACTURER_NAME).strip()
    manufacturer_code = str(data.get("manufacturerCode") or DEFAULT_MANUFACTURER_CODE).strip()

    if not name:
        raise PluginDiscoveryError(f"Missing 'name' in {meta_path}")
    if not slug:
        raise PluginDiscoveryError(f"Missing 'slug' in {meta_path}")
    if not plugin_code:
        raise PluginDiscoveryError(f"Missing 'pluginCode' in {meta_path}")
    if len(plugin_code) != 4:
        raise PluginDiscoveryError(f"pluginCode must be exactly 4 characters in {meta_path}: {plugin_code!r}")
    if len(manufacturer_code) != 4:
        raise PluginDiscoveryError(
            f"manufacturerCode must be exactly 4 characters in {meta_path}: {manufacturer_code!r}"
        )

    readme_path = plugin_root / PLUGIN_README_FILENAME
    if not readme_path.is_file():
        raise PluginDiscoveryError(
            f"Missing {PLUGIN_README_FILENAME} in plugin leaf {plugin_root}. The embedded '?' help panel now renders the leaf README directly."
        )

    install_override = data.get("installPath")
    if install_override is not None:
        raise PluginDiscoveryError(
            f"installPath overrides are no longer supported; packaged folders mirror the repo category path now: {meta_path}"
        )
    install_rel_dir = Path(category)

    entry_value = str(data.get("entry") or "").strip()
    if entry_value:
        entry_rel = Path(entry_value)
        entry_path = plugin_root / entry_rel
    else:
        entry_path = _infer_entry(plugin_root)
        entry_rel = entry_path.relative_to(plugin_root)

    if not entry_path.exists():
        raise PluginDiscoveryError(f"Entry file listed in {meta_path} does not exist: {entry_rel}")

    plugin_type = str(data.get("pluginType") or "").strip().lower()
    if not plugin_type:
        if entry_path.suffix.lower() == ".dsp":
            plugin_type = "faust"
        elif entry_path.suffix.lower() == ".jsfx":
            plugin_type = "jsfx"
    if plugin_type not in {"faust", "jsfx"}:
        raise PluginDiscoveryError(
            f"Invalid pluginType in {meta_path}: {plugin_type!r} (expected 'faust' or 'jsfx')"
        )

    if plugin_type == "faust" and entry_path.suffix.lower() != ".dsp":
        raise PluginDiscoveryError(f"Faust plugin entry must be a .dsp file in {meta_path}")
    if plugin_type == "jsfx" and entry_path.suffix.lower() != ".jsfx":
        raise PluginDiscoveryError(f"JSFX plugin entry must be a .jsfx file in {meta_path}")

    bundle_id = str(data.get("bundleId") or f"{DEFAULT_BUNDLE_ID_BASE}.{_slug_token(slug)}").strip()
    clap_id = str(data.get("clapId") or bundle_id).strip()

    clap_features_raw = data.get("clapFeatures") or list(DEFAULT_CLAP_FEATURES)
    if not isinstance(clap_features_raw, list) or not all(isinstance(x, str) and x.strip() for x in clap_features_raw):
        raise PluginDiscoveryError(f"clapFeatures must be a non-empty list of strings in {meta_path}")
    clap_features = tuple(x.strip() for x in clap_features_raw)

    return PluginSpec(
        repo_root=repo_root,
        root_dir=plugin_root,
        meta_path=meta_path,
        rel_dir=rel_dir,
        install_rel_dir=install_rel_dir,
        category=category,
        name=name,
        slug=slug,
        plugin_code=plugin_code,
        bundle_id=bundle_id,
        clap_id=clap_id,
        clap_features=clap_features,
        plugin_type=plugin_type,
        entry_rel=entry_rel,
        entry_path=entry_path,
        manufacturer_name=manufacturer_name,
        manufacturer_code=manufacturer_code,
        readme_path=readme_path,
        raw=data,
    )


def discover_plugins(repo_root: Path) -> list[PluginSpec]:
    root = plugins_root(repo_root)
    if not root.exists():
        raise PluginDiscoveryError(f"Missing plugins root: {root}")

    specs = [load_plugin(repo_root, meta_path) for meta_path in sorted(root.rglob(PLUGIN_METADATA_FILENAME))]
    if not specs:
        raise PluginDiscoveryError(
            f"No {PLUGIN_METADATA_FILENAME} files found under {root}. Add plugins as leaf folders under plugins/<Category>/<PluginKey>/."
        )

    by_slug: dict[str, Path] = {}
    by_clap: dict[str, Path] = {}
    for spec in specs:
        if spec.slug in by_slug:
            raise PluginDiscoveryError(
                f"Duplicate slug {spec.slug!r} in {spec.meta_path} and {by_slug[spec.slug]}"
            )
        by_slug[spec.slug] = spec.meta_path
        if spec.clap_id in by_clap:
            raise PluginDiscoveryError(
                f"Duplicate clapId {spec.clap_id!r} in {spec.meta_path} and {by_clap[spec.clap_id]}"
            )
        by_clap[spec.clap_id] = spec.meta_path

    return specs


def plugin_matches(spec: PluginSpec, needle: str) -> bool:
    q = needle.strip().lower()
    if not q:
        return True
    haystacks = [
        spec.category,
        spec.slug,
        spec.name,
        spec.key,
        str(spec.rel_dir),
        str(spec.repo_rel_dir),
        spec.bundle_id,
        spec.clap_id,
    ]
    return any(q in h.lower() for h in haystacks)


def filter_plugins(specs: Iterable[PluginSpec], needle: str) -> list[PluginSpec]:
    if not needle.strip():
        return list(specs)
    return [spec for spec in specs if plugin_matches(spec, needle)]
