#!/usr/bin/env python3
"""Generate unchanged Joep native fixtures and exercise the production editor.

The editor host reuses one stable JUCE build identity to avoid rebuilding JUCE
modules for every source. This tests production processor control flow, not
50 separate VST3 identities or a DAW's plugin discovery.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,sys,time
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from pluginlib import discover_plugins
from jsfx_source import resolve_source
from build import write_embedded_text_header
from embed_gfx_resources import write_gfx_resources

def call(cmd,log,timeout):
 with log.open('w') as stream:
  r=subprocess.run([str(x) for x in cmd],stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
 if r.returncode:raise RuntimeError(f'{r.returncode}: {log.read_text()[-2000:]}')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--plugins',nargs='+');ap.add_argument('--jobs',type=int,default=1);ap.add_argument('--editor-build',type=Path);ap.add_argument('--cmake',default='cmake');ap.add_argument('--reuse',action='store_true');ap.add_argument('--editors-only',action='store_true');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 specs=[s for s in discover_plugins(REPO) if s.category=='JoepVanlier' and (not a.plugins or s.slug in a.plugins)]
 compilerHash=hashlib.sha256((REPO/'dsp_jsfx_aot.py').read_bytes()).hexdigest()
 runtimeHash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((REPO/'src').glob('*')) if p.is_file())+(REPO/'tests/jsfx_showcase/joep_editor_check.cpp').read_bytes()+(REPO/'cmake/plugin/CMakeLists.txt').read_bytes()).hexdigest()
 def generate(spec):
  d=out/spec.slug;d.mkdir(exist_ok=True);start=time.monotonic();r={'plugin':spec.slug,'entry':str(spec.entry_path.relative_to(REPO)),'compiler_sha256':compilerHash}
  try:
   resolved=resolve_source(spec.entry_path);text=resolved.text;r['expanded_sha256']=hashlib.sha256(text.encode()).hexdigest();(d/'expanded.jsfx').write_text(text);(d/'source-manifest.json').write_text(json.dumps(resolved.manifest(REPO),indent=2))
   cached=json.loads((d/'generation.json').read_text()) if a.reuse and (d/'generation.json').exists() else {}
   if not (cached.get('compiler_sha256')==compilerHash and cached.get('expanded_sha256')==r['expanded_sha256'] and cached.get('status')=='PASS' and (d/'JSFXDSP.o').exists()):
    call([sys.executable,REPO/'dsp_jsfx_aot.py',d/'expanded.jsfx','--native-gfx-legacy','--opt','2','--out-h',d/'JSFXDSP.h','--meta',d/'JSFXDSP_meta.json','--out-obj',d/'JSFXDSP.o','--out-ll',d/'JSFXDSP.ll'],d/'generate.log',1200)
    write_embedded_text_header(text=text,variable_name='kJsfxSourceText',out_header=d/'JSFXSource.h',banner='unchanged Joep fixture')
    write_gfx_resources(spec.entry_path,text,d/'JSFXResources.h')
   meta=json.loads((d/'JSFXDSP_meta.json').read_text());r.update(status='PASS',heap_cells=meta['memtop_slots'],vars=meta['var_cap'],has_gfx=meta['sections_present']['gfx'],object_bytes=(d/'JSFXDSP.o').stat().st_size)
  except Exception as e:r.update(status='BLOCKED',error=str(e))
  r['seconds']=round(time.monotonic()-start,2);(d/'generation.json').write_text(json.dumps(r,indent=2));return r
 rows=[]
 if a.editors_only:
  rows=[json.loads(p.read_text()) for p in sorted(out.glob('*/generation.json'))]
  if a.plugins:rows=[r for r in rows if r['plugin'] in a.plugins]
 else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
   futures={pool.submit(generate,s):s for s in specs}
   for f in concurrent.futures.as_completed(futures):
    r=f.result();rows.append(r);print('GENERATE',r['plugin'],r['status'],r.get('error','')[:120],flush=True);(out/'native-builds.json').write_text(json.dumps(rows,indent=2))
 if a.editor_build:
  b=a.editor_build.resolve();editorRows=[]
  for r in sorted(rows,key=lambda r:r['plugin']):
   if r['status']!='PASS':continue
   d=out/r['plugin'];fixtureHash=hashlib.sha256(runtimeHash.encode()+b''.join((d/name).read_bytes() for name in ['JSFXDSP.h','JSFXDSP.o','JSFXSource.h','JSFXResources.h'])).hexdigest();entry={'plugin':r['plugin'],'scope':'production JUCE editor fixture; common host identity','fixture_sha256':fixtureHash};start=time.monotonic()
   try:
    if a.reuse and (d/'editor.json').exists():
     cached=json.loads((d/'editor.json').read_text())
     if cached.get('status')=='PASS' and cached.get('fixture_sha256')==fixtureHash:editorRows.append(cached);continue
    for name in ['JSFXDSP.h','JSFXDSP.o','JSFXSource.h','JSFXResources.h']:shutil.copyfile(d/name,b/name)
    call([a.cmake,'--build',b,'--target','joep_editor_check','-j2'],d/'editor-build.log',240)
    call([b/'joep_editor_check',d/'screenshots','gfx' if r['has_gfx'] else 'no-gfx','drop' if 'gfx_getdropfile' in (d/'expanded.jsfx').read_text().lower() else 'no-drop',r['plugin']],d/'editor-run.log',90)
    line=d.joinpath('editor-run.log').read_text().strip().splitlines()[-1];entry.update(json.loads(line));entry['status']='PASS'
   except Exception as e:entry.update(status='FAIL',error=str(e))
   entry['seconds']=round(time.monotonic()-start,2);editorRows.append(entry);(d/'editor.json').write_text(json.dumps(entry,indent=2));print('EDITOR',r['plugin'],entry['status'],entry.get('error','')[:120],flush=True);(out/'editor-runs.json').write_text(json.dumps(editorRows,indent=2))
 return int(any(r['status']!='PASS' for r in rows) or bool(a.editor_build) and any(r['status']!='PASS' for r in editorRows))
if __name__=='__main__':sys.exit(main())
