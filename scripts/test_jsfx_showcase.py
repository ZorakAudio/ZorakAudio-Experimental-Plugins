#!/usr/bin/env python3
"""Offline Abyss showcase verification. This does not build or launch a VST3 host.

Windows: use an x64 Visual Studio Developer Command Prompt for the headless C++
tests. The ordinary scripts/build.py plugin build has no new prerequisite.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from pluginlib import discover_plugins
from jsfx_source import resolve_source
from verify_jsfx_snapshot import verify
import build


def command(args: list[str], *, timeout=180, env=None) -> str:
    print('+ ' + ' '.join(map(str, args)), flush=True)
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            encoding='utf-8', errors='replace', capture_output=True, timeout=timeout)
    text = result.stdout + result.stderr
    print(text.rstrip(), flush=True)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {args[0]}\n{text[-6000:]}')
    return text


def existing_matrix(out: Path, baseline_zip: Path | None) -> list[dict]:
    import dsp_jsfx_aot as current
    baseline = None
    if baseline_zip:
        # Only explicitly supplied, trusted project code is executed here.
        with zipfile.ZipFile(baseline_zip) as archive:
            names = [n for n in archive.namelist() if n.split('/')[-1] == 'dsp_jsfx_aot.py']
            if len(names) != 1:
                raise ValueError('Baseline ZIP must contain exactly one dsp_jsfx_aot.py')
            source = archive.read(names[0])
        path = out/'baseline_aot.py';path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('showcase_baseline_aot', path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline;spec.loader.exec_module(baseline)
    result = []
    for plugin in discover_plugins(ROOT):
        if plugin.plugin_type != 'jsfx' or plugin.slug == 'SaikeAbyss':
            continue
        try:
            text = resolve_source(plugin.entry_path).text
            ir, _ = current.compile_jsfx_to_ir(text)
            rendered = str(ir)
            row = {'plugin': plugin.slug, 'status': 'PASS',
                   'llvm_sha256': hashlib.sha256(rendered.encode()).hexdigest()}
            if baseline:
                previous, _ = baseline.compile_jsfx_to_ir(text)
                row['identical_to_baseline_llvm'] = rendered == str(previous)
                if not row['identical_to_baseline_llvm']:
                    row['status'] = 'CHANGED'
            result.append(row)
        except Exception as exc:
            result.append({'plugin': plugin.slug, 'status': 'FAIL', 'error': str(exc)})
        print('Existing LLVM:', plugin.slug, result[-1]['status'], flush=True)
    (out/'existing_matrix.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if any(r['status'] != 'PASS' for r in result):
        raise AssertionError('Existing-plugin LLVM regression; see existing_matrix.json')
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'build/jsfx_showcase')
    ap.add_argument('--config', choices=['Release', 'Debug'], default='Release')
    ap.add_argument('--cxx', help='Native C++ compiler (cl in an x64 VS developer prompt on Windows)')
    ap.add_argument('--generator', help='Optional CMake generator; use a fresh --out directory when changing it')
    ap.add_argument('--juce-root', type=Path, help='Optional local JUCE checkout for the real JUCE headless test')
    ap.add_argument('--asan', action='store_true', help='Clang/Linux/macOS: instrument native LLVM, WDL and renderer')
    ap.add_argument('--skip-existing', action='store_true', help='Skip compilation of the other installed JSFX')
    ap.add_argument('--baseline-archive', type=Path, help='Trusted pre-patch repo ZIP for exact existing LLVM comparison')
    args = ap.parse_args()
    out = args.out.resolve();out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'status': 'RUNNING', 'python': sys.version, 'config': args.config,
              'asan': args.asan, 'gfx_backend': 'JUCE' if args.juce_root else 'JUCE contract double',
              'scope': 'Headless generated native DSP vs supplied portable WDL/EEL; real CPU LICE',
              'not_validated': ['full JUCE plugin build', 'MSVC plugin link', 'DAW/pluginval',
                                'REAPER process output', 'entire JoepVanlier repository']}
    try:
        if not shutil.which('cmake'):
            raise RuntimeError('CMake is required for headless tests')
        cxx = args.cxx or ('cl' if os.name == 'nt' else ('clang++' if args.asan else 'c++'))
        if not shutil.which(cxx):
            raise RuntimeError(f'{cxx} not found. Windows headless tests require an x64 Visual Studio Developer Command Prompt.')
        if args.asan and (os.name == 'nt' or not shutil.which('clang')):
            raise RuntimeError('This ASan test runner requires Clang on Linux/macOS')
        package = ROOT/'plugins/JoepVanlier/SaikeAbyss'
        report['snapshot'] = verify(package)
        for line in report['snapshot']:print(line, flush=True)
        report['source_tests'] = command([sys.executable, str(ROOT/'tests/jsfx_showcase/test_source_resolver.py')])
        plugin = next(s for s in discover_plugins(ROOT) if s.slug == 'SaikeAbyss')
        generated = out/'generated';generated.mkdir(exist_ok=True)
        build.build_jsfx_aot(ROOT, generated, plugin.slug, plugin.entry_path, plugin.raw['jsfxCompatibility'])
        source = (generated/'JSFXExpanded.jsfx').read_text(encoding='utf-8')
        # Both executable worlds must receive exactly the same expanded text.
        header = (generated/'JSFXSource.h').read_text(encoding='utf-8')
        chunks = re.findall(r'^".*"$', header, flags=re.M)
        embedded = ''.join(ast.literal_eval(c) for c in chunks)
        if embedded.encode('latin1').decode('utf-8') != source:
            raise AssertionError('Embedded GFX source differs from the DSP compiler input')
        manifest = json.loads((generated/'JSFXImports.json').read_text())
        meta = json.loads((generated/'JSFXDSP_meta.json').read_text())
        if len(manifest['files']) != 4 or meta['numeric_semantics'] != 'eel2-stores':
            raise AssertionError('Missing imported dependencies or checked-store setting')
        if '// @za:gfx_sync_policy EXPLICIT' not in source:
            raise AssertionError('Showcase lost its explicit/private GFX memory setting')
        report['same_dsp_gfx_source'] = True
        report['expanded_source_sha256'] = manifest['expanded_sha256']
        report['source_files'] = manifest['files']
        if args.asan:
            sys.path.insert(0, str(ROOT/'tests/jsfx_showcase'))
            from instrument_ir import instrument
            instrument(generated/'JSFXDSP.ll', generated/'JSFXDSP.o', shutil.which('clang'))
        native = out/'native'
        configure = ['cmake', '-S', str(ROOT/'tests/jsfx_showcase'), '-B', str(native),
                     '-DSHOWCASE_GENERATED='+str(generated), '-DCMAKE_BUILD_TYPE='+args.config,
                     '-DPython3_EXECUTABLE='+sys.executable]
        if args.generator:configure += ['-G', args.generator]
        if os.name == 'nt':configure += ['-A', 'x64']
        else:configure += ['-DCMAKE_CXX_COMPILER='+str(shutil.which(cxx))]
        if args.asan:configure += ['-DZA_SHOWCASE_ASAN=ON', '-DCMAKE_C_COMPILER='+str(shutil.which('clang'))]
        if args.juce_root:configure += ['-DZA_SHOWCASE_JUCE_ROOT='+str(args.juce_root.resolve())]
        command(configure)
        command(['cmake', '--build', str(native), '--config', args.config, '--parallel', '4'], timeout=300)
        def executable(name):
            filename = name + ('.exe' if os.name == 'nt' else '')
            for p in (native/args.config/filename, native/filename):
                if p.is_file():return str(p)
            raise FileNotFoundError(filename+' not built')
        env = dict(os.environ)
        if args.asan:env['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=1'
        cases = [(44100, 4, 0, 'stress', 'sample'), (48000, 4, 0, 'stress', 'sample'),
                 (48000, 4, 1, 'stress', 'sample'), (48000, 4, 2, 'stress', 'sample'),
                 (96000, 4, 0, 'stress', 'sample'), (48000, 4, 0, 'defaults', 'float'),
                 (48000, 4, 0, 'dry', 'float'), (48000, 4, 0, 'automation', 'float')]
        report['audio'] = []
        for case in cases:
            text = command([executable('dsp_reference'), str(generated/'JSFXExpanded.jsfx'), *map(str, case)], env=env)
            line = next(line for line in text.splitlines() if line.startswith('rate='))
            report['audio'].append(dict(part.split('=', 1) for part in line.split()))
        report['gfx'] = command([executable('gfx_showcase'), str(generated/'JSFXExpanded.jsfx')], env=env)
        micro = [sys.executable, str(ROOT/'tests/jsfx_showcase/test_native_semantics.py'),
                 '--out', str(out/'micro'), '--eel', executable('eel_eval'), '--cxx', cxx]
        if args.asan:micro += ['--asan']
        command(micro, env=env)
        report['native_micro_cases'] = json.loads((out/'micro/results.json').read_text())
        if not args.skip_existing:report['existing'] = existing_matrix(out, args.baseline_archive)
        if args.baseline_archive:report['baseline_sha256'] = hashlib.sha256(args.baseline_archive.read_bytes()).hexdigest()
        report['snapshot_after_tests'] = verify(package)
        report['status'] = 'PASS'
    except (Exception, SystemExit) as exc:
        report['status'] = 'FAIL';report['error'] = str(exc)
        print('FAIL:', exc, file=sys.stderr, flush=True)
    finally:
        report['elapsed_seconds'] = round(time.perf_counter()-started, 3)
        (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(report['status']+': '+str(out/'validation.json'), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
