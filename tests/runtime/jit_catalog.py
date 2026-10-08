"""Execute each JSFX through the actual JIT processor; record bounded coverage.

This is a defaults/GFX/state smoke gate, not an audio equivalence oracle.
Every failed or timed-out entry remains visible in the result matrix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from pluginlib import discover_plugins

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plugins',nargs='+')
    ap.add_argument('--frontend',choices=['python-reference','cpp-frontend'],default='python-reference')
    ap.add_argument('--exe',type=Path,default=ROOT/'build/jit-editor/jit_editor_check.exe')
    ap.add_argument('--out',type=Path,default=ROOT/'build/runtime-unification/jit-catalog')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    specs=[p for p in discover_plugins(ROOT) if p.plugin_type=='jsfx' and (not a.plugins or p.slug in a.plugins)]
    if a.plugins:assert set(a.plugins)=={p.slug for p in specs},'Unknown/non-JSFX catalog selection'
    executable=a.exe.resolve();rows=[]
    record=dict(frontend=a.frontend,executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                scope='initialize, native link, default DSP with varied inputs/MIDI/block sizes, GFX and state save; no full preset/audio equivalence claim',rows=rows)
    for spec in specs:
        folder=a.out/spec.slug;folder.mkdir(exist_ok=True)
        command=[str(executable)]+(['--cpp-frontend'] if a.frontend=='cpp-frontend' else [])+['--catalog',str(spec.entry_path)]
        began=time.monotonic();row=dict(plugin=spec.slug,source=str(spec.entry_path),source_sha256=hashlib.sha256(spec.entry_path.read_bytes()).hexdigest())
        try:
            p=subprocess.run(command,cwd=folder,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=660)
            (folder/'run.log').write_text(p.stdout+p.stderr,encoding='utf-8')
            row['exit_code']=p.returncode;row['passed']=p.returncode==0
            if row['passed']:row['result']=json.loads(p.stdout.strip().splitlines()[-1])
            else:row['diagnostic']=(p.stdout+p.stderr)[-5000:]
        except subprocess.TimeoutExpired:
            row.update(passed=False,diagnostic='Test process exceeded 660 seconds; terminated by test harness')
        row['wall_seconds']=time.monotonic()-began;rows.append(row)
        (a.out/'results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print(spec.slug,'PASS' if row['passed'] else 'FAIL',round(row['wall_seconds'],2),flush=True)
    if not all(r['passed'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
