"""Windows DSP timing against the actual WDL native x64 SSE JIT.

Re-emits verified cached, optimized IR through the production final Clang
backend. Does not change JSFX algorithms. Compilation and JUCE/GFX overhead
are excluded. Both engines receive identical controls, signal and MIDI.
Requires the native WDL static library built by the native benchmark fixture.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(Path(__file__).parent)]
from jsfx_source import resolve_source
from pluginlib import discover_plugins
from joep_legacy_qualification import config_header


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_name():
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()


def run(command, log, timeout=600, env=None):
    started = time.monotonic()
    result = subprocess.run([str(x) for x in command], capture_output=True,
                            text=True, timeout=timeout, env=env)
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}: {result.stderr[-1800:]}')
    return result, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wdl-lib', type=Path, required=True)
    parser.add_argument('--clang', default=r'C:\Program Files\LLVM\bin\clang++.exe')
    parser.add_argument('--cached-build', type=Path, default=REPO / 'build/windows')
    parser.add_argument('--plugins', nargs='+')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=4)
    parser.add_argument('--blocks', nargs='+', type=int, default=[64, 512])
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('This fixture requires Windows native x64 WDL, not the portable interpreter.')
    if args.trials < 3 or args.seconds <= 0 or any(b < 1 or b > 512 for b in args.blocks):
        parser.error('Use at least 3 trials, positive duration, and block sizes 1..512.')
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).with_name('joep_native_benchmark.cpp')
    bridge = (REPO / 'src/YSFXGfxInterpreter.h').read_text(encoding='utf-8')
    portable = '#ifndef EEL_TARGET_PORTABLE\n#define EEL_TARGET_PORTABLE 1\n#endif'
    if bridge.count(portable) != 1:
        raise RuntimeError('Native WDL fixture override no longer matches the graphics adapter.')
    (out / 'YSFXGfxInterpreter.h').write_text(bridge.replace(portable, '// Native WDL benchmark.'), encoding='utf-8')
    (out / 'juce_core').mkdir(exist_ok=True)
    (out / 'juce_core/juce_core.h').write_text('#pragma once\n// Graphics contract supplied by juce_contract_stub.h.\n')
    (out / 'numeric_runtime.inc').write_text('#include "WDL/fft.h"\n#include "JsfxNumericBuiltins.h"\n')
    include = ['-I' + str(p) for p in (out, REPO / 'src', driver.parent)]
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-DNOMINMAX=1', '-DJOEP_FULL_NATIVE=1',
             '-DWDL_FFT_REALSIZE=8', '-D_CRT_SECURE_NO_WARNINGS=1']
    runtime_names = ('DspJsfxRuntime', 'DspJsfxRuntimeBuiltins', 'DspJsfxGmem', 'DspJsfxMessageBus', 'DspJsfxSharedMemory')
    runtime_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((REPO / 'src').glob('*')) if p.is_file())).hexdigest()
    version = subprocess.check_output([args.clang, '--version'], text=True).splitlines()[0]
    specs = [s for s in discover_plugins(REPO) if s.category == 'JoepVanlier' and
             (not args.plugins or s.slug in args.plugins)]
    report = dict(date=date.today().isoformat(), method=dict(
        cpu=cpu_name(), platform='Windows x64', compiler=version,
        backend_flags=['-O2', '-Xclang', '-disable-llvm-passes'],
        source='Verified cached optimized LLVM IR; final production machine-code flags',
        reference='Vendored WDL native x64 SSE JIT; NOFPSTATE with a scoped FP environment',
        sample_rate=48000, blocks=args.blocks, trials=args.trials,
        seconds_per_trial=args.seconds, warmup_seconds=1,
        execution='Serial, below normal priority, alternating engine order per block and trial',
        includes='DSP audio sections and required sample/buffer marshalling',
        timer='C++ steady_clock elapsed time; *_cpu_seconds field names do not indicate process CPU accounting',
        excludes='Compilation, initialization, input generation, MIDI/transport setup, output validation, GFX, JUCE host callback, DAW',
        workload='Default controls, synthetic continuous input, repeated note on/off, modulation wheel and pitch bend; no external sample files',
        wdl_library_sha256=sha(args.wdl_lib), driver_sha256=sha(driver)), plugins=[])
    report_path = out / 'results.json'
    for spec in specs:
        d = out / spec.slug
        d.mkdir(exist_ok=True)
        cache = args.cached_build / spec.slug
        row = dict(plugin=spec.slug, name=spec.name, trials=[])
        try:
            source = cache / 'JSFXExpanded.jsfx'
            resolved = resolve_source(spec.entry_path)
            if [s for s in source.read_text(encoding='utf-8').splitlines() if s.strip()] != [s for s in resolved.text.splitlines() if s.strip()]:
                raise RuntimeError('Cached source no longer matches current source/import resolution.')
            row.update(source_sha256=sha(source), ir_sha256=sha(cache / 'JSFXDSP.ll'), header_sha256=sha(cache / 'JSFXDSP.h'))
            fingerprint = hashlib.sha256((row['ir_sha256'] + row['header_sha256'] + sha(driver) + sha(args.wdl_lib) + runtime_hash + 'stack-8388608').encode()).hexdigest()
            obj, exe = d / 'optimized.obj', d / 'benchmark.exe'
            stamp = d / 'build.sha256'
            if not args.reuse or not exe.exists() or not stamp.exists() or stamp.read_text() != fingerprint:
                (d / 'JoepTestConfig.h').write_text(config_header(source.read_text(encoding='utf-8'), spec.slug), encoding='utf-8')
                (d / 'host_slider_runtime.inc').write_text('#include "JsfxSliderBuiltins.h"\n')
                common = []
                for name in runtime_names:
                    runtime_source = REPO / 'src' / (name + '.cpp')
                    runtime_obj = d / (name + '.obj')
                    run([args.clang, '-c', runtime_source, *flags, *include, '-I' + str(cache), '-o', runtime_obj], d / (name + '.log'))
                    common.append(runtime_obj)
                _, elapsed = run([args.clang, '-c', cache / 'JSFXDSP.ll', '-O2', '-Xclang', '-disable-llvm-passes', '-o', obj], d / 'emit.log', timeout=1200)
                row['emit_seconds'] = elapsed
                run([args.clang, *flags, *include, '-I' + str(d), '-I' + str(cache),
                     driver, obj, *common, args.wdl_lib, '-fuse-ld=lld', '-o', exe,
                     '-Wl,/stack:8388608', '-luser32', '-lgdi32', '-ladvapi32', '-lwinmm', '-lshell32', '-lcomdlg32'], d / 'build.log')
                stamp.write_text(fingerprint)
            row['executable_sha256'] = sha(exe)
            for block in args.blocks:
                case = dict(block_size=block, trials=[])
                for trial in range(args.trials):
                    env = dict(os.environ, ZA_JOEP_SECONDS=str(args.seconds), ZA_BENCH_BLOCK=str(block))
                    if trial % 2:
                        env['ZA_BENCH_REVERSE'] = '1'
                    else:
                        env.pop('ZA_BENCH_REVERSE', None)
                    result, _ = run([exe, source, '48000', 'defaults', 'float'], d / f'{block}-{trial}.log', timeout=120, env=env)
                    data = json.loads(result.stdout)
                    if data['backend'] != 'WDL native x64 SSE JIT':
                        raise RuntimeError('Wrong WDL backend.')
                    case['trials'].append(data)
                values = case['trials']
                case.update(speedup_median=statistics.median(v['speedup'] for v in values),
                    speedup_min=min(v['speedup'] for v in values), speedup_max=max(v['speedup'] for v in values),
                    llvm_us_per_sample=statistics.median(v['native_cpu_seconds'] / v['frames'] * 1e6 for v in values),
                    wdl_us_per_sample=statistics.median(v['wdl_cpu_seconds'] / v['frames'] * 1e6 for v in values),
                    exact_differences=max(v['exact_differences'] for v in values),
                    differing_samples=max(v['differing_samples'] for v in values), max_error=max(v['max_error'] for v in values),
                    relative_rms_error=max(v['relative_rms_error'] for v in values),
                    midi_differences=max(v['midi_differences'] for v in values), peak=min(v['peak'] for v in values))
                case['status'] = 'EXACT' if not case['exact_differences'] and not case['midi_differences'] else ('WITHIN_TOLERANCE' if not case['differing_samples'] and not case['midi_differences'] else 'DIVERGES')
                row['trials'].append(case)
                print(spec.slug, block, case['status'], f"{case['speedup_median']:.3f}x", flush=True)
            row['status'] = 'MEASURED'
        except Exception as error:
            row.update(status='NOT_COMPARABLE', error=str(error))
            print(spec.slug, row['status'], row['error'][:220], flush=True)
        (d / 'result.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        report['plugins'].append(row)
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
