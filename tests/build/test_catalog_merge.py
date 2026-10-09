"""Catalog publication requires complete, disjoint shards and intact bundle bytes."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from merge_catalog_archives import merge
from pluginlib import discover_plugins

class CatalogMergeTests(unittest.TestCase):
    def fixture(self, root, missing=None, duplicate=False, platform='linux', missing_payload=False, empty_payload=False):
        specs = discover_plugins(ROOT)
        package = 'ZorakAudio-Experimental-Plugins-test-' + platform
        for index in range(4):
            if index == missing: continue
            with zipfile.ZipFile(root / f'{index}.zip', 'w') as archive:
                records = [dict(slug=spec.slug) for position, spec in enumerate(specs) if position % 4 == index]
                if duplicate and index == 1: records.append(dict(slug=specs[0].slug))
                archive.writestr(package + '/manifest.json', json.dumps(dict(package=package, buildShard=dict(index=index,count=4),plugins=records)))
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

if __name__=='__main__': unittest.main()
