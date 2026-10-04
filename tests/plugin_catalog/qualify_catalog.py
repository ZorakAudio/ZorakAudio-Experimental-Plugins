#!/usr/bin/env python3
"""Fresh production codegen gate for the 33 configured non-Joep plugins."""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]
from build import build_jsfx_aot, native_gfx_modes_for_plugin
from pluginlib import discover_plugins
from jsfx_source import resolve_source, apply_host_options
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(spec, d):
    if spec.plugin_type == 'jsfx':
        prototype, legacy = native_gfx_modes_for_plugin(spec)
        build_jsfx_aot(ROOT, d, spec.slug, spec.entry_path, spec.raw.get('jsfxCompatibility'), native_gfx_prototype=prototype, native_gfx_legacy=legacy)
        meta = json.loads((d / 'JSFXDSP_meta.json').read_text()); obj = d / 'JSFXDSP.o'
        assert obj.read_bytes()[:4] == b'\x7fELF' and bool(meta.get('native_gfx_legacy')) == legacy
        files = ['JSFXDSP.h', 'JSFXDSP.o', 'JSFXDSP_meta.json', 'JSFXSource.h', 'JSFXResources.h', 'JSFXExpanded.jsfx', 'JSFXImports.json']
        result = {'mode': 'legacy' if legacy else 'eel-publication', 'object_bytes': obj.stat().st_size,
                  'has_gfx': bool(re.search(r'(?m)^\s*@gfx\b', (d / 'JSFXExpanded.jsfx').read_text())), 'heap_cells': meta['memtop_slots']}
    else:
        faust = shutil.which('faust'); assert faust, 'Faust unavailable'
        subprocess.run([faust, '-lang', 'cpp', '-i', '-cn', 'mydsp', '-o', str(d / 'FaustDSP.h'), str(spec.entry_path)], check=True)
        files = ['FaustDSP.h']; result = {'mode': 'faust', 'faust_version': subprocess.check_output([faust, '--version'], text=True).strip()}
    print(json.dumps({'worker_complete': True, **result, 'files': {f: sha(d / f) for f in files}}), flush=True)
def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plugins', nargs='+'); ap.add_argument('--reuse', action='store_true'); ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args(); specs = [s for s in discover_plugins(ROOT) if s.category != 'JoepVanlier' and (not a.plugins or s.slug in a.plugins)]
    if a.worker:
        assert len(specs) == 1; worker(specs[0], a.out); return 0
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True); rows = []
    code = b''.join((ROOT / f).read_bytes() for f in ['dsp_jsfx_aot.py', 'scripts/build.py', 'scripts/jsfx_source.py', 'scripts/embed_gfx_resources.py', 'scripts/pluginlib.py'])
    for spec in specs:
        start = time.monotonic(); d = out / spec.slug; d.mkdir(exist_ok=True)
        row = {'plugin': spec.slug, 'type': spec.plugin_type, 'entry': str(spec.entry_path.relative_to(ROOT))}
        try:
            source = apply_host_options(resolve_source(spec.entry_path), spec.raw.get('jsfxCompatibility')).text if spec.plugin_type == 'jsfx' else spec.entry_path.read_text()
            fingerprint = hashlib.sha256(code + source.encode() + spec.plugin_type.encode()).hexdigest(); row['fingerprint'] = fingerprint
            old = json.loads((d / 'generation.json').read_text()) if a.reuse and (d / 'generation.json').exists() else {}
            if old.get('status') == 'PASS' and old.get('fingerprint') == fingerprint and all((d / f).exists() and sha(d / f) == h for f, h in old['files'].items()): rows.append(old); continue
            fresh = Path(tempfile.mkdtemp(prefix=spec.slug + '-', dir=out))
            with (d / 'compile.log').open('w') as log:
                p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out', str(fresh), '--plugins', spec.slug, '--worker'], stdout=log, stderr=subprocess.STDOUT,
                                   timeout=1200, env=dict(os.environ, JSFX_AOT_TRACE_PHASES='1', JSFX_AOT_OPT_LEVEL='2'))
            if p.returncode: raise RuntimeError(f'Worker exit {p.returncode}: {(d / "compile.log").read_text()[-2000:]}')
            terminal = json.loads((d / 'compile.log').read_text().strip().splitlines()[-1]); assert terminal.pop('worker_complete', False)
            for f, h in terminal['files'].items():
                assert sha(fresh / f) == h; shutil.copyfile(fresh / f, d / f)
            # IR is reproducible and large. Retain verified objects and source manifests.
            shutil.rmtree(fresh); row.update(terminal, status='PASS')
        except Exception as exc: row.update(status='FAIL', error=str(exc))
        row['seconds'] = round(time.monotonic() - start, 3); (d / 'generation.json').write_text(json.dumps(row, indent=2) + '\n'); rows.append(row)
        (out / 'generation.json').write_text(json.dumps(rows, indent=2) + '\n'); print('GENERATE', spec.slug, row['status'], row.get('error', '')[:180], flush=True)
    (out / 'generation.json').write_text(json.dumps(rows, indent=2) + '\n'); return int(not rows or any(r['status'] != 'PASS' for r in rows))
if __name__ == '__main__': raise SystemExit(main())
