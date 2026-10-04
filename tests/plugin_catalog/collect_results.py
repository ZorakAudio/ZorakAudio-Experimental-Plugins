#!/usr/bin/env python3
"""Publish completed catalog evidence; refuse partial or mismatched artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pluginlib import discover_plugins


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); out = a.out.resolve(); destination = ROOT / 'docs/catalog-regression'
    specs = [p for p in discover_plugins(ROOT) if p.category != 'JoepVanlier']; rows = []
    for spec in specs:
        d = out / spec.slug; g = json.loads((d / 'generation.json').read_text()); e = json.loads((d / 'editor.json').read_text())
        assert g['status'] == e['status'] == 'PASS', spec.slug
        assert e['finite_audio'] and e['editor_lifetimes'] == 2 and e['host_parameter_state'], spec.slug
        assert spec.plugin_type != 'jsfx' or e['gfx_lifetimes'] == 2, spec.slug
        for f, h in g['files'].items(): assert sha(d / f) == h, spec.slug + '/' + f
        for f, h in e['screenshots'].items(): assert sha(d / 'screenshots' / f) == h, spec.slug + '/' + f
        numerical = json.loads((d / 'numerical.json').read_text()) if spec.plugin_type == 'jsfx' else None
        assert numerical is None or numerical['status'] in ['PASS', 'NOT_COMPARABLE'], spec.slug
        bank = json.loads((d / 'bank.json').read_text()) if spec.slug in ['PsychoConvolver', 'Contour', 'Texture', 'TexturePM', 'TextureXY'] else None
        if bank:
            assert bank['status'] == 'PASS' and bank['finite_audio'] and bank['peak'] > .000001, spec.slug
            if spec.slug in ['Contour', 'Texture']: assert bank['waveform_rows'] > 20, spec.slug
            if spec.slug == 'TextureXY': assert bank['ui_path_pixels'] > 50 and bank['ui_path_stray_pixels'] < 50, spec.slug
            for f, h in bank['screenshots'].items(): assert sha(d / 'bank-screenshots' / f) == h, spec.slug + '/' + f
        if spec.slug in ['Sample', 'Corpus']: assert e['bank_files'] == 3 and e['peak'] > .000001
        if spec.slug.startswith('IPCProbe'): assert e['ipc_received'] > 0 and e['ipc_peer_peak'] > .000001
        rows.append({'plugin': spec.slug, 'category': spec.category, 'type': spec.plugin_type,
                     'entry': str(spec.entry_path.relative_to(ROOT)), 'generation': g, 'editor': e,
                     'numerical': numerical, 'loaded': bank,
                     'executable_sha256': sha(d / 'catalog_editor_check')})
    assert len(rows) == 33 and sum(r['numerical'] is not None and r['numerical']['status'] == 'PASS' for r in rows) == 17
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        d = out / row['plugin']; target = destination / row['plugin']; target.mkdir(exist_ok=True)
        for f in ['generation.json', 'editor.json', 'numerical.json', 'bank.json', 'JSFXImports.json',
                  'compile.log', 'editor-run.log', 'reference-run.log', 'bank-run.log']:
            if (d / f).exists(): shutil.copyfile(d / f, target / f)
        for folder in ['screenshots', 'bank-screenshots']:
            if (d / folder).exists():
                images = target / folder; images.mkdir(exist_ok=True)
                for image in (d / folder).glob('*.png'): shutil.copyfile(image, images / image.name)
    for f in ['build-policy-tests.log', 'automatic-joep.log', 'automatic-joep-release.log', 'faust-release.log',
              'math-compile-tests.log', 'sample-pool-tests.log', 'comm-compile-tests.log', 'release-policy.json']:
        if (out / f).exists(): shutil.copyfile(out / f, destination / f)
    issues = out / 'initial-issues'
    if issues.exists(): shutil.copytree(issues, destination / 'initial-issues', dirs_exist_ok=True)
    for image in out.glob('contact-*.png'):
        if image.name != 'contact-partial.png': shutil.copyfile(image, destination / image.name)
    from datetime import datetime, timezone
    result = {'date_utc': datetime.now(timezone.utc).isoformat(),
              'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'compiler_sha256': sha(ROOT / 'dsp_jsfx_aot.py'), 'build_script_sha256': sha(ROOT / 'scripts/build.py'),
              'packages': len(rows), 'jsfx': 28, 'faust': 5, 'wdl_matches': 17, 'host_extension_plugins': 11,
              'records': rows}
    (destination / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    text = '# Non-Joep catalog result matrix\n\nActual production screenshots and completed worker records. See [tested scope](../Plugin-Catalog-Regression.md).\n\n'
    text += '| Plugin | Type/mode | Codegen | Editor/audio | WDL audio | Loaded service | Screenshot |\n|---|---|---|---|---|---|---|\n'
    for r in rows:
        n = 'Exact match' if r['numerical'] and r['numerical']['status'] == 'PASS' else 'Host extensions' if r['numerical'] else 'N/A'
        loaded = 'WAV/IR' if r['loaded'] else '3-file MIDI bank' if r['plugin'] in ['Sample', 'Corpus'] else 'Paired IPC' if r['plugin'].startswith('IPCProbe') else '—'
        text += f"| {r['plugin']} | {r['generation']['mode']} | PASS | PASS | {n} | {loaded} | [Open]({r['plugin']}/screenshots/open.png) |\n"
    (destination / 'README.md').write_text(text)
    print(json.dumps({k: result[k] for k in ['packages', 'jsfx', 'faust', 'wdl_matches', 'host_extension_plugins']}))


if __name__ == '__main__': main()
