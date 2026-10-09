"""Merge CI catalog shards without changing plugin bytes or ZIP bundle metadata."""
import argparse
import copy
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile
from pluginlib import discover_plugins
from release_collections import build_catalog, collections_for, load_policy, policy_digest, PLATFORMS

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

def merge(input_dir, output_dir, tag, platforms=PLATFORMS, report=True):
    specs = build_catalog(discover_plugins(ROOT))
    expected = {spec.slug for spec in specs}
    grouped = defaultdict(list)
    for path in sorted(input_dir.rglob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            manifests = [name for name in archive.namelist() if name.count('/') == 1 and name.endswith('/manifest.json')]
            if len(manifests) != 1: raise ValueError('Expected one catalog manifest in ' + str(path))
            manifest = json.loads(archive.read(manifests[0]))
            digest = manifest.get('releasePolicySha256')
            if digest != policy_digest():
                raise ValueError('Release policy differs from build receipt: ' + str(path))
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
                            releasePolicySha256=policy_digest(),
                            plugins=[records[spec.slug] for spec in specs],
                            mergedBuildShards=next(iter(counts)))
            merged.writestr(package + '/manifest.json', json.dumps(manifest, indent=2) + '\n')
            lines = ['ZorakAudio Experimental Plugins', '',
                     'Copy the category folders inside VST3/ and CLAP/ into the corresponding plugin folders.',
                     '', 'Plugins included in this package:']
            for spec in specs: lines.append(f'- [{spec.category}] {spec.key} -> {spec.name} [{spec.plugin_type}]')
            merged.writestr(package + '/INSTALL.txt', '\n'.join(lines) + '\n')
        results[platform] = dict(plugins=len(records), shards=len(parts), archive=str(target))
    if report:
        print(json.dumps(results, indent=2))
    return results


def selected_member(name, package, specs):
    """Include selected bundles, adjacent license notices and their parents.

    ditto puts AppleDouble resource metadata under __MACOSX; map it to the
    corresponding ordinary member for selection without changing its bytes.
    """
    logical = name
    resource = name.startswith('__MACOSX/')
    if resource:
        parts = name[len('__MACOSX/'):].split('/')
        if parts[-1].startswith('._'):
            parts[-1] = parts[-1][2:]
        logical = '/'.join(parts)
    for spec in specs:
        for format_name, suffix in (('CLAP', '.clap'), ('VST3', '.vst3')):
            leaf = f'{package}/{format_name}/{spec.install_rel_dir.as_posix()}/'
            bundle = leaf + spec.slug + suffix
            if logical.rstrip('/') == bundle or logical.startswith(bundle + '/'):
                return True
            if logical == leaf + spec.slug + '.LICENSE.upstream.txt':
                return True
            if logical.endswith('/') and bundle.startswith(logical):
                return True
    return False


def assemble_collections(input_dir, output_dir, tag):
    """Validate all shards, then publish three ZIPs containing all three OSes.

    Collection selection never recompiles DSP. Signed bundles, executable modes,
    symlinks and resource metadata are copied verbatim from validated builds.
    """
    specs = discover_plugins(ROOT)
    collections = collections_for(specs)
    policy = load_policy()
    purposes = {item['slug']: item['purpose'] for item in policy['essentials']}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix='catalog-verified-', dir=output_dir) as directory:
        verified = merge(input_dir, Path(directory), tag, report=False)
        for collection, selected in collections.items():
            package = f'ZorakAudio-Experimental-Plugins-{tag}-{collection}-all-platforms'
            target = output_dir / (package + '.zip')
            temporary = Path(directory) / target.name
            platform_receipts = {}
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
                for platform in PLATFORMS:
                    with zipfile.ZipFile(verified[platform]['archive']) as source:
                        receipt_name = next(name for name in source.namelist() if name.count('/') == 1 and name.endswith('/manifest.json'))
                        receipt = json.loads(source.read(receipt_name))
                        old_package = receipt['package']
                        records = {record['slug']: record for record in receipt['plugins']}
                        verify_plugin_payloads(source, old_package, selected, platform)
                        for original in source.infolist():
                            if not selected_member(original.filename, old_package, selected):
                                continue
                            info = copy.copy(original)
                            resource = original.filename.startswith('__MACOSX/')
                            prefix = ('__MACOSX/' if resource else '') + old_package + '/'
                            if not original.filename.startswith(prefix):
                                raise ValueError('Unexpected selected member: ' + original.filename)
                            info.filename = ('__MACOSX/' if resource else '') + package + '/' + platform + '/' + original.filename[len(prefix):]
                            output.writestr(info, source.read(original))
                        platform_receipts[platform] = dict(
                            platform=platform, mergedBuildShards=receipt['mergedBuildShards'],
                            plugins=[records[spec.slug] for spec in selected])
                        output.writestr(package + '/' + platform + '/manifest.json',
                                        json.dumps(platform_receipts[platform], indent=2) + '\n')
                manifest = dict(schemaVersion=3, package=package, collection=collection,
                                releasePolicySha256=policy_digest(),
                                platforms=list(PLATFORMS), pluginCount=len(selected),
                                plugins=[dict(slug=spec.slug, name=spec.name, category=spec.category,
                                              installPath=spec.install_rel_dir.as_posix(),
                                              bundleId=spec.bundle_id, clapId=spec.clap_id,
                                              purpose=purposes.get(spec.slug, '')) for spec in selected])
                output.writestr(package + '/manifest.json', json.dumps(manifest, indent=2) + '\n')
                lines = [f'ZorakAudio {collection} — {tag}', '',
                         'This ZIP contains Windows x86-64, macOS universal2 and Linux x86-64 builds.',
                         'Open ONLY the folder for your operating system: windows/, macos/ or linux/.',
                         'Copy its category folders inside VST3/ and/or CLAP/ to the corresponding plugin folder.',
                         'Keep each .vst3 or macOS .clap bundle intact; do not copy just its inner binary.',
                         'macOS builds are ad-hoc signed, not notarized. Linux baseline: Ubuntu 24.04.',
                         'Use one format per track. The ? button opens the embedded plugin manual.',
                         '', 'Essentials is a subset of All. Installing both is unnecessary.',
                         'JoepVanlier is independent of both non-Joep collections.',
                         'The separately packaged JIT Editor is not part of these catalog ZIPs.',
                         'Keep older installed identities if existing DAW sessions use them.',
                         '', f'{len(selected)} plugins in this collection (each in all three OS folders):']
                for spec in selected:
                    lines.append(f'- [{spec.category}] {spec.name} ({spec.slug})')
                    if spec.slug in purposes:
                        lines.append('  ' + purposes[spec.slug])
                output.writestr(package + '/INSTALL.txt', '\n'.join(lines) + '\n')
            temporary.replace(target)
            results[collection] = dict(plugins=len(selected), platforms=list(PLATFORMS), archive=str(target))
    print(json.dumps(results, indent=2))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--platform', action='append', choices=('windows','macos','linux'))
    parser.add_argument('--collections', action='store_true', help='Publish JoepVanlier, Essentials and All; requires all three platforms')
    args = parser.parse_args()
    if args.collections:
        if args.platform:
            parser.error('--collections requires all three platforms; do not use --platform')
        assemble_collections(args.input, args.output, args.tag)
    else:
        merge(args.input, args.output, args.tag, args.platform or PLATFORMS)
