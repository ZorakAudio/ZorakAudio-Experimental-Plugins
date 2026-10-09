"""Merge CI catalog shards without changing plugin bytes or ZIP bundle metadata."""
import argparse
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import zipfile
from pluginlib import discover_plugins

ROOT = Path(__file__).resolve().parents[1]

def verify_plugin_payloads(archive, package, specs, platform):
    """Require both executable payloads, not just a plugin's manifest entry."""
    for spec in specs:
        prefix = f'{package}/{{format}}/{spec.install_rel_dir.as_posix()}/{spec.slug}'
        if platform == 'macos':
            clap = prefix.format(format='CLAP') + f'.clap/Contents/MacOS/{spec.slug}'
            vst3 = prefix.format(format='VST3') + f'.vst3/Contents/MacOS/{spec.slug}'
        else:
            clap = prefix.format(format='CLAP') + '.clap'
            architecture, suffix = ('x86_64-win', '.vst3') if platform == 'windows' else ('x86_64-linux', '.so')
            vst3 = prefix.format(format='VST3') + f'.vst3/Contents/{architecture}/{spec.slug}{suffix}'
        for name in (clap, vst3):
            try:
                info = archive.getinfo(name)
            except KeyError as error:
                raise ValueError('Missing plugin payload: ' + name) from error
            if info.is_dir() or info.file_size == 0:
                raise ValueError('Empty plugin payload: ' + name)

def merge(input_dir, output_dir, tag, platforms=('windows', 'macos', 'linux')):
    specs = discover_plugins(ROOT)
    expected = {spec.slug for spec in specs}
    grouped = defaultdict(list)
    for path in sorted(input_dir.rglob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            manifests = [name for name in archive.namelist() if name.count('/') == 1 and name.endswith('/manifest.json')]
            if len(manifests) != 1: raise ValueError('Expected one catalog manifest in ' + str(path))
            manifest = json.loads(archive.read(manifests[0]))
            platform = manifest['package'].rsplit('-', 1)[-1]
            if manifest['package'] != f'ZorakAudio-Experimental-Plugins-{tag}-{platform}':
                raise ValueError('Unexpected catalog package/tag: ' + manifest['package'])
            grouped[platform].append((path, manifest))
    if set(grouped) != set(platforms): raise ValueError('Missing or unexpected platform archives: ' + str(sorted(grouped)))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for platform, parts in sorted(grouped.items()):
        records = {}
        indices, counts = set(), set()
        for path, manifest in parts:
            shard = manifest.get('buildShard')
            if shard is None: raise ValueError('Missing shard receipt: ' + str(path))
            if shard['index'] in indices: raise ValueError('Duplicate shard index')
            indices.add(shard['index']); counts.add(shard['count'])
            for item in manifest['plugins']:
                if item['slug'] in records: raise ValueError('Duplicate plugin in shard archives: ' + item['slug'])
                records[item['slug']] = item
        if len(counts) != 1 or indices != set(range(next(iter(counts)))):
            raise ValueError('Missing catalog shards for ' + platform)
        if set(records) != expected: raise ValueError('Catalog membership mismatch for ' + platform)
        by_slug = {spec.slug: spec for spec in specs}
        for path, manifest in parts:
            with zipfile.ZipFile(path) as archive:
                verify_plugin_payloads(archive, manifest['package'],
                    [by_slug[item['slug']] for item in manifest['plugins']], platform)
        package = parts[0][1]['package']
        target = output_dir / (package + '.zip')
        written = {}
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as merged:
            for path, _ in parts:
                with zipfile.ZipFile(path) as source:
                    for info in source.infolist():
                        name = info.filename
                        relative = PurePosixPath(name)
                        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe ZIP path: ' + name)
                        if name in (package + '/manifest.json', package + '/INSTALL.txt'): continue
                        data = source.read(info)
                        if name in written:
                            if written[name] != (info.CRC, info.file_size): raise ValueError('Conflicting shard asset: ' + name)
                            continue
                        # Preserve symlinks, modes, timestamps and signing metadata.
                        merged.writestr(info, data)
                        written[name] = (info.CRC, info.file_size)
            manifest = dict(schemaVersion=2, package=package,
                            plugins=[records[spec.slug] for spec in specs],
                            mergedBuildShards=next(iter(counts)))
            merged.writestr(package + '/manifest.json', json.dumps(manifest, indent=2) + '\n')
            lines = ['ZorakAudio Experimental Plugins', '',
                     'Copy the category folders inside VST3/ and CLAP/ into the corresponding plugin folders.',
                     '', 'Plugins included in this package:']
            for spec in specs: lines.append(f'- [{spec.category}] {spec.key} -> {spec.name} [{spec.plugin_type}]')
            merged.writestr(package + '/INSTALL.txt', '\n'.join(lines) + '\n')
        results[platform] = dict(plugins=len(records), shards=len(parts), archive=str(target))
    print(json.dumps(results, indent=2))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--platform', action='append', choices=('windows','macos','linux'))
    args = parser.parse_args()
    merge(args.input, args.output, args.tag, args.platform or ('windows','macos','linux'))
