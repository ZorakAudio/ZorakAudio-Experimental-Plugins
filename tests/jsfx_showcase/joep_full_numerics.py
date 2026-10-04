#!/usr/bin/env python3
"""WDL comparison of original audio with full native objects (GFX not run).

Short MIDI messages use a deterministic host adapter for both guests.
The production editor harness separately exercises the JUCE MIDI path.
The file adapter uses the native worker backend on this offline test thread.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'tests/jsfx_showcase')]
from joep_legacy_qualification import config_header,host_helpers
from joep_test_fingerprints import numeric_fingerprint

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--wdl-build',type=Path,required=True);ap.add_argument('--plugins',nargs='+');ap.add_argument('--seconds',default='2');ap.add_argument('--reuse',action='store_true');a=ap.parse_args();rows=[]
 for d in sorted(a.out.iterdir()):
  if not (d/'generation.json').exists() or a.plugins and d.name not in a.plugins:continue
  generated=json.loads((d/'generation.json').read_text())
  if generated.get('status')!='PASS' or not (d/'JSFXDSP_meta.json').exists() or not (d/'JSFXDSP.o').exists():
   row={'plugin':d.name,'status':'BLOCKED','error':'Native generation has not passed'}
   (d/'numerical.json').write_text(json.dumps(row,indent=2));rows.append(row);continue
  fingerprint=numeric_fingerprint(REPO,d,a.wdl_build,a.seconds)
  meta=json.loads((d/'JSFXDSP_meta.json').read_text());row={'plugin':d.name,'scope':'original audio with full native object; no GFX execution','midi':meta['midi'],'fixture_sha256':fingerprint}
  if a.reuse and (d/'numerical.json').exists():
   cached=json.loads((d/'numerical.json').read_text())
   if cached.get('status')=='PASS' and cached.get('fixture_sha256')==fingerprint:rows.append(cached);continue
  try:
   src=(d/'expanded.jsfx').read_text();(d/'JoepTestConfig.h').write_text(config_header(src,d.name));host_helpers(d)
   cmd=['c++','-std=c++20','-O1','-DNDEBUG','-DJOEP_FULL_NATIVE=1','-DEEL_TARGET_PORTABLE=1','-DWDL_FFT_REALSIZE=8','-I'+str(REPO/'src'),'-I'+str(d),'-I'+str(a.wdl_build),REPO/'tests/jsfx_showcase/joep_dsp_reference.cpp',d/'JSFXDSP.o',REPO/'src/DspJsfxRuntime.cpp',REPO/'src/DspJsfxRuntimeBuiltins.cpp',REPO/'src/DspJsfxGmem.cpp',REPO/'src/DspJsfxMessageBus.cpp',REPO/'src/DspJsfxSharedMemory.cpp',a.wdl_build/'libshowcase_eel.a','-lpthread','-lrt','-lm','-o',d/'full_reference']
   r=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,timeout=180);(d/'reference-build.log').write_text(r.stdout+r.stderr)
   if r.returncode:raise RuntimeError(r.stderr[-1500:])
   env=dict(os.environ,ZA_JOEP_SECONDS=a.seconds,ZA_JOEP_DIAG='1');r=subprocess.run([str(d/'full_reference'),str(d/'expanded.jsfx'),'48000','defaults','float'],capture_output=True,text=True,timeout=600,env=env)
   row.update(exit_code=r.returncode,stderr=r.stderr,status='PASS' if not r.returncode else 'FAIL');
   if r.stdout.strip():row.update(json.loads(r.stdout))
  except subprocess.TimeoutExpired as e:row.update(status='BLOCKED',error=str(e),stderr=(e.stderr or b'').decode(errors='replace') if isinstance(e.stderr,bytes) else e.stderr)
  except Exception as e:row.update(status='BLOCKED',error=str(e))
  (d/'numerical.json').write_text(json.dumps(row,indent=2));rows.append(row);print('NUMERIC',d.name,row['status'],row.get('max_error',row.get('error',''))[:100] if isinstance(row.get('max_error',row.get('error','')),str) else row.get('max_error'),flush=True);(a.out/'full-numerics.json').write_text(json.dumps(rows,indent=2))
 (a.out/'full-numerics.json').write_text(json.dumps(rows,indent=2))
 return int(not rows or any(r['status']!='PASS' for r in rows))
if __name__=='__main__':sys.exit(main())
