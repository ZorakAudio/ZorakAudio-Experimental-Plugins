"""Real-CLAP Legacy compatibility and matching-mode state regression.

Pass an explicit authorized WAV bank and before/after modules. No installation,
source recording edits, or external services are used. Numerical measurement
edge cases live in tests/dsp-jsfx-sample-pool/level_analysis.cpp.
"""
from pathlib import Path
import argparse
import array
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def read_state(path):
    data = path.read_bytes()
    return ET.fromstring(data[8:8 + struct.unpack_from('<I', data, 4)[0]])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--host', type=Path)
    parser.add_argument('--seconds', type=float, default=6)
    args = parser.parse_args()
    if sys.platform != 'win32':
        raise SystemExit('This public-CLAP fixture currently uses the Windows loader')
    args.out.mkdir(parents=True, exist_ok=True)
    host = args.host or args.out / 'sample_matching_clap.exe'
    if args.host is None:
        compiler = shutil.which('clang++')
        if compiler is None:
            raise SystemExit('clang++ is required to build the CLAP test host')
        subprocess.run([compiler, '-std=c++17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
                        '-I' + str(ROOT / 'libs/clap-juce-extensions/clap-libs/clap/include'),
                        str(Path(__file__).with_name('sample_matching_clap.cpp')),
                        '-o', str(host), '-luser32'], check=True)
    for label, module in [('before', args.before), ('after', args.after)]:
        subprocess.run([str(host), 'inspect', str(module), str(args.out / label)], check=True)
    before = list(csv.DictReader((args.out / 'before.params.tsv').open(), delimiter='\t'))
    after = list(csv.DictReader((args.out / 'after.params.tsv').open(), delimiter='\t'))
    assert all(before[i]['id'] == after[i]['id'] and before[i]['default'] == after[i]['default']
               for i in range(64)), 'Existing parameter identity/default changed'
    assert before[-1]['id'] == after[-1]['id'], 'OS parameter identity changed'
    assert after[64]['default'] == '0', 'Fresh instances must still default to Legacy'
    paths = sorted(args.bank.glob('*.wav')) if args.bank.is_dir() else [args.bank]
    assert paths and all(p.is_file() for p in paths), 'No authorized WAV files'
    template = read_state(args.out / 'before.state')
    results = []

    def state(name, changes):
        root = ET.Element('PARAMS')
        # Deliberately exclude unrelated persistent file UI/history.
        for old in template.findall('PARAM'):
            ET.SubElement(root, 'PARAM', **old.attrib)
        files = ET.SubElement(root, 'FILES')
        slot = ET.SubElement(files, 'SLOT', index='0')
        for p in paths:
            ET.SubElement(slot, 'PATH', path=str(p.resolve()))
        for index, value in changes.items():
            id = 'slider' + str(index)
            p = root.find("PARAM[@id='" + id + "']")
            if p is None:
                p = ET.SubElement(root, 'PARAM', id=id)
            p.set('value', str(value))
        data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        path = args.out / (name + '.state')
        path.write_bytes(b'VC2!' + struct.pack('<I', len(data)) + data + b'\0')
        return path

    def render(name, module, config, restore=None):
        prefix = args.out / name
        env = dict(os.environ)
        for key in ['SAMPLE_RESTORE_STATE', 'SAMPLE_PARAM_CHANGE', 'SAMPLE_ALLOW_SILENCE']:
            env.pop(key, None)
        if restore:
            env['SAMPLE_RESTORE_STATE'] = str(restore)
        subprocess.run([str(host), 'render', str(module), str(config), str(prefix),
                        '48000', '256', str(args.seconds), 'sparse', '100'],
                       env=env, check=True)
        return prefix.with_suffix('.f32').read_bytes()

    def null(name, a, b):
        assert len(a) == len(b)
        u, v = array.array('f'), array.array('f')
        u.frombytes(a)
        v.frombytes(b)
        row = {'test': name, 'float_samples': len(u),
               'different': sum(x != y for x, y in zip(u, v)),
               'maximum_error': max(abs(x-y) for x, y in zip(u, v))}
        results.append(row)
        print(json.dumps(row), flush=True)
        assert a == b, name + ' did not cancel exactly'

    shot = {2: .1, 20: 337, 3: 110, 4: 1, 5: 180, 9: .22, 10: 0,
            13: 1, 14: 1, 15: 1, 22: .27}
    old_raw = None
    old_config = None
    for mode in range(4):
        config = state('legacy' + str(mode), shot | {17: mode})
        a = render('before' + str(mode), args.before, config)
        b = render('after' + str(mode), args.after, config)
        null('Legacy engine ' + str(mode), a, b)
        if mode == 0:
            old_raw, old_config = a, config
    for mode in [1, 2, 3]:
        config = state('matching' + str(mode), shot | {17: 0, 65: mode,
                       66: -23, 67: -18.4, 68: 6, 69: -3})
        a = render('matching' + str(mode), args.after, config)
        saved = args.out / ('matching' + str(mode) + '.loaded.state')
        b = render('roundtrip' + str(mode), args.after, saved)
        null('Mode ' + str(mode) + ' roundtrip', a, b)
        value = read_state(saved).find("PARAM[@id='slider65']").get('value')
        assert int(float(value)) == mode
        if mode == 3:
            b = render('old_over_loudness', args.after, config, restore=old_config)
            null('Missing mode restores Legacy over Loudness', old_raw, b)
            saved = read_state(args.out / 'old_over_loudness.loaded.state')
            assert float(saved.find("PARAM[@id='slider65']").get('value')) == 0
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    print('PASS: Legacy nulls, stable parameter IDs/defaults, matching state roundtrips, old-state migration')


if __name__ == '__main__':
    main()
