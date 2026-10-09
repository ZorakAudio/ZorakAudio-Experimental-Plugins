"""Catalog publication requires complete, disjoint shards and intact bundle bytes."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from merge_catalog_archives import merge, assemble_collections
from pluginlib import discover_plugins
from release_collections import build_catalog, collections_for, load_policy, policy_digest, PLATFORMS
import build as builder

class CatalogMergeTests(unittest.TestCase):
    def fixture(self, root, missing=None, duplicate=False, platform='linux', missing_payload=False, empty_payload=False):
        specs = build_catalog(discover_plugins(ROOT))
        package = 'ZorakAudio-Experimental-Plugins-test-' + platform
        for index in range(4):
            if index == missing: continue
            with zipfile.ZipFile(root / f'{index}.zip', 'w') as archive:
                records = [dict(slug=spec.slug) for position, spec in enumerate(specs) if position % 4 == index]
                if duplicate and index == 1: records.append(dict(slug=specs[0].slug))
                archive.writestr(package + '/manifest.json', json.dumps(dict(package=package, releasePolicySha256=policy_digest(), buildShard=dict(index=index,count=4),plugins=records)))
                for spec in (s for position, s in enumerate(specs) if position % 4 == index):
                    prefix = f'{package}/{{format}}/{spec.install_rel_dir.as_posix()}/{spec.slug}'
                    if platform == 'macos':
                        clap = prefix.format(format='CLAP') + f'.clap/Contents/MacOS/{spec.slug}'
                        vst3 = prefix.format(format='VST3') + f'.vst3/Contents/MacOS/{spec.slug}'
                    else:
                        clap = prefix.format(format='CLAP') + '.clap'
                        architecture, suffix = ('x86_64-win', '.vst3') if platform == 'windows' else ('x86_64-linux', '.so')
                        vst3 = prefix.format(format='VST3') + f'.vst3/Contents/{architecture}/{spec.slug}{suffix}'
                    archive.writestr(clap, b'ELF fixture bytes ' + spec.slug.encode())
                    if not (missing_payload and spec is specs[0]):
                        archive.writestr(vst3, b'' if empty_payload and spec is specs[0] else b'VST3 fixture bytes')
                    if platform == 'macos':
                        link = zipfile.ZipInfo(prefix.format(format='VST3') + '.vst3/Contents/Frameworks/link')
                        link.create_system=3;link.external_attr=(0o120777 << 16)
                        archive.writestr(link, b'../MacOS/' + spec.slug.encode())
                        archive.writestr('__MACOSX/' + prefix.format(format='CLAP') + '.clap/Contents/MacOS/._' + spec.slug, b'AppleDouble fixture')
                    if spec.category == 'JoepVanlier':
                        for format_name in ('CLAP','VST3'):
                            archive.writestr(f'{package}/{format_name}/{spec.install_rel_dir.as_posix()}/{spec.slug}.LICENSE.upstream.txt', b'Upstream license fixture')
                link = zipfile.ZipInfo(package + f'/VST3/plugin-{index}.vst3/link')
                link.create_system=3;link.external_attr=(0o120777 << 16)
                archive.writestr(link, b'Contents/MacOS/plugin')
        return package, specs

    def test_complete_catalog_preserves_bytes_and_symlink_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package,specs=self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()): merge(root,root/'merged','test',['linux'])
            with zipfile.ZipFile(root/'merged'/(package+'.zip')) as archive:
                self.assertEqual(len(json.loads(archive.read(package+'/manifest.json'))['plugins']),len(specs))
                self.assertEqual(archive.read(package+'/CLAP/'+specs[0].install_rel_dir.as_posix()+'/'+specs[0].slug+'.clap'), b'ELF fixture bytes '+specs[0].slug.encode())
                self.assertEqual(archive.getinfo(package+'/VST3/plugin-0.vst3/link').external_attr >> 16, 0o120777)

    def test_missing_group_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,missing=2)
            with self.assertRaisesRegex(ValueError,'Missing catalog shards'): merge(root,root/'merged','test',['linux'])
            self.assertFalse(list((root/'merged').glob('*.zip')))

    def test_duplicate_plugin_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,duplicate=True)
            with self.assertRaisesRegex(ValueError,'Duplicate plugin'): merge(root,root/'merged','test',['linux'])

    def test_missing_binary_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,missing_payload=True)
            with self.assertRaisesRegex(ValueError,'Missing plugin payload'): merge(root,root/'merged','test',['linux'])
            self.assertFalse(list((root/'merged').glob('*.zip')))

    def test_empty_binary_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,empty_payload=True)
            with self.assertRaisesRegex(ValueError,'Empty plugin payload'): merge(root,root/'merged','test',['linux'])
            self.assertFalse(list((root/'merged').glob('*.zip')))

    def test_windows_and_macos_bundle_payload_paths(self):
        for platform in ('windows','macos'):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);package,specs=self.fixture(root,platform=platform)
                with contextlib.redirect_stdout(io.StringIO()): merge(root,root/'merged','test',[platform])
                with zipfile.ZipFile(root/'merged'/(package+'.zip')) as archive:
                    self.assertEqual(len(json.loads(archive.read(package+'/manifest.json'))['plugins']),len(specs))

    def test_policy_is_reviewed_and_excludes_diagnostics_and_duplicate_essentials(self):
        all_specs = discover_plugins(ROOT)
        selected = collections_for(all_specs)
        eligible = build_catalog(all_specs)
        self.assertFalse({'IPCProbeA','IPCProbeB','SaliencePush'} & {s.slug for s in eligible})
        essential = {s.slug for s in selected['Essentials']}
        self.assertIn('AntiSalienceMX', essential)
        self.assertIn('SampleFaust', essential)
        self.assertIn('EasyExpanderFaust', essential)
        self.assertEqual(essential & {'3DPanner','HyperrealFast','HyperrealFaust','HyperrealHybrid'}, {'3DPanner'})
        self.assertFalse({'Sample','EasyExpander'} & essential)
        joep = {s.slug for s in selected['JoepVanlier']}
        nonjoep = {s.slug for s in selected['All']}
        self.assertFalse(joep & nonjoep)
        self.assertTrue(essential <= nonjoep)
        self.assertEqual(joep | nonjoep, {s.slug for s in eligible})
        self.assertEqual(essential | set(load_policy()['essentialsExcluded']), nonjoep)

    def test_actual_builder_excludes_probes_even_for_explicit_only_selection(self):
        with patch.object(sys,'argv',['build.py','--list']), contextlib.redirect_stdout(io.StringIO()) as output:
            builder.main()
        self.assertIn('AntiSalienceMX',output.getvalue())
        self.assertNotIn('IPCProbe',output.getvalue())
        self.assertNotIn('SaliencePush',output.getvalue())
        for name in ('IPCProbeA','IPCProbeB','SaliencePush'):
            with self.subTest(plugin=name), patch.object(sys,'argv',['build.py','--only',name]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    builder.main()
                self.assertEqual(error.exception.code,2)

    def test_three_collections_have_exact_membership_on_every_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for platform in PLATFORMS:
                folder=root/platform;folder.mkdir()
                self.fixture(folder,platform=platform)
            with contextlib.redirect_stdout(io.StringIO()):
                result=assemble_collections(root,root/'published','test')
            self.assertEqual(len(list((root/'published').glob('*.zip'))), 3)
            for collection, specs in collections_for(discover_plugins(ROOT)).items():
                with self.subTest(collection=collection), zipfile.ZipFile(result[collection]['archive']) as archive:
                    package=Path(result[collection]['archive']).stem
                    manifest=json.loads(archive.read(package+'/manifest.json'))
                    self.assertEqual(manifest['platforms'], list(PLATFORMS))
                    self.assertEqual([s.slug for s in specs], [r['slug'] for r in manifest['plugins']])
                    self.assertFalse(any('IPCProbe' in name or '/SaliencePush/' in name for name in archive.namelist()))
                    for platform in PLATFORMS:
                        prefix=package+'/'+platform
                        receipt=json.loads(archive.read(prefix+'/manifest.json'))
                        self.assertEqual([r['slug'] for r in receipt['plugins']], [s.slug for s in specs])
                        for spec in specs:
                            stem=prefix+'/CLAP/'+spec.install_rel_dir.as_posix()+'/'+spec.slug
                            clap=stem+('.clap/Contents/MacOS/'+spec.slug if platform=='macos' else '.clap')
                            self.assertEqual(archive.read(clap), b'ELF fixture bytes '+spec.slug.encode())
                            if platform=='macos':
                                link=prefix+'/VST3/'+spec.install_rel_dir.as_posix()+'/'+spec.slug+'.vst3/Contents/Frameworks/link'
                                self.assertEqual(archive.getinfo(link).external_attr >> 16, 0o120777)
                                self.assertEqual(archive.read(link), b'../MacOS/'+spec.slug.encode())
                                self.assertEqual(archive.read('__MACOSX/'+stem+'.clap/Contents/MacOS/._'+spec.slug), b'AppleDouble fixture')
                            if spec.category=='JoepVanlier':
                                self.assertEqual(archive.read(stem+'.LICENSE.upstream.txt'), b'Upstream license fixture')
                    # Payload directories, not just manifests, have exact membership.
                    actual={name.split('/')[4] for name in archive.namelist()
                            if name.startswith(package+'/windows/CLAP/') and name.endswith('.clap')}
                    self.assertEqual(actual, {s.slug+'.clap' for s in specs})

    def test_collections_require_all_platforms_before_publishing_any_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            with self.assertRaisesRegex(ValueError,'platform archives'):
                assemble_collections(root,root/'published','test')
            self.assertFalse(list((root/'published').glob('*.zip')))

    def test_changed_release_policy_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            path=root/'0.zip'
            with zipfile.ZipFile(path) as archive:
                members=[(info,archive.read(info)) for info in archive.infolist()]
            with zipfile.ZipFile(path,'w') as archive:
                for info,data in members:
                    if info.filename.endswith('/manifest.json'):
                        receipt=json.loads(data);receipt['releasePolicySha256']='wrong';data=json.dumps(receipt)
                    archive.writestr(info,data)
            with self.assertRaisesRegex(ValueError,'Release policy differs'):
                merge(root,root/'merged','test',['linux'])

if __name__=='__main__': unittest.main()
