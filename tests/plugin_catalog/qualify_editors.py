#!/usr/bin/env python3
"""Actual production processors/editors in one reusable Linux JUCE test host.

DSP objects retain production O2; the shared host uses O0 to limit rebuild cost.
This intentionally does not qualify every individually packaged format wrapper.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shlex, shutil, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from pluginlib import discover_plugins
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def flags(host):
    t=(host/'CMakeFiles/sample_editor_check.dir/flags.make').read_text()
    return sum((shlex.split(re.search(r'^'+name+r' = (.*)$',t,re.M)[1]) for name in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']),[])
def link(host,obj,exe,faust=None):
    cmd=shlex.split((host/'CMakeFiles/sample_editor_check.dir/link.txt').read_text()); result=[]; i=0
    while i<len(cmd):
        token=cmd[i]
        if token=='-o': result+=['-o',str(exe)];i+=2;continue
        if 'sample_editor_check.cpp.o' in token:token=str(obj)
        if token.endswith('libSample_SharedCode.a') and faust:
            # Keep the cached JUCE archive as a dependency source. The real Faust
            # processor resolves createPluginFilter first; no JSFX processor is selected.
            result.append(str(faust))
        if not token.startswith('-Wl,--dependency-file='):result.append(token)
        i+=1
    return result
def run(cmd,log,cwd=None,timeout=600):
    with log.open('w') as f:p=subprocess.run([str(x) for x in cmd],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:raise RuntimeError(f'exit {p.returncode}: {log.read_text()[-3000:]}')
def prepare(spec,d,host,cmake,driver,driver_name='catalog_editor_check'):
    base=flags(host); opts=[v for v in base if not re.fullmatch(r'-O\w+',v)]+['-O0','-std=c++20']
    obj=host/(driver_name+('-jsfx' if spec.plugin_type=='jsfx' else '-faust')+'.o')
    # Compile each worker against the same host ABI; cache only exact source/flags.
    digest=hashlib.sha256(driver.read_bytes()+json.dumps(opts).encode()).hexdigest()
    stamp=obj.with_suffix('.sha')
    if not obj.exists() or not stamp.exists() or stamp.read_text()!=digest:
        run(['c++',*opts,f'-DCATALOG_JSFX={int(spec.plugin_type=="jsfx")}','-c',driver,'-o',obj],d/(driver_name+'-driver-build.log'));stamp.write_text(digest)
    faust=None
    if spec.plugin_type=='jsfx':
        for name in ['JSFXDSP.h','JSFXDSP.o','JSFXSource.h','JSFXResources.h']:shutil.copyfile(d/name,host/name)
        # Header changes rebuild the real processor and runtime, preserving cached JUCE objects.
        hostFlags=(host/'CMakeFiles/Sample.dir/flags.make').read_text()
        hostOpts=shlex.split(re.search(r'^CXX_FLAGS = (.*)$',hostFlags,re.M)[1]); hostOpts=[v for v in hostOpts if not re.fullmatch(r'-O\w+',v)]+['-O0']
        run([cmake,'--build',host,'--target','Sample','--','-j2','CXX_FLAGS='+' '.join(hostOpts)],d/(driver_name+'-host-build.log'),timeout=900)
    else:
        faust=d/'catalog_faust_processor.o'
        run(['c++',*opts,'-I'+str(d),'-I'+str(ROOT/'src'),'-c',ROOT/'src/FaustJuceProcessor.cpp','-o',faust],d/'faust-host-build.log')
    exe=d/driver_name;run(link(host,obj,exe,faust),d/(driver_name+'-link.log'),host);return exe
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--host',type=Path,required=True);ap.add_argument('--cmake',default='cmake');ap.add_argument('--plugins',nargs='+');ap.add_argument('--reuse',action='store_true');a=ap.parse_args()
    out=a.out.resolve();host=a.host.resolve();driver=Path(__file__).with_name('catalog_editor_check.cpp');rows=[]
    for spec in discover_plugins(ROOT):
        if spec.category=='JoepVanlier' or (a.plugins and spec.slug not in a.plugins):continue
        start=time.monotonic();d=out/spec.slug;g=json.loads((d/'generation.json').read_text());assert g['status']=='PASS'
        code=b''.join(p.read_bytes() for p in sorted((ROOT/'src').glob('*')) if p.is_file())+driver.read_bytes()+Path(__file__).read_bytes()
        fp=hashlib.sha256(code+json.dumps(g['files'],sort_keys=True).encode()+json.dumps(flags(host)).encode()).hexdigest();row={'plugin':spec.slug,'fingerprint':fp}
        old=json.loads((d/'editor.json').read_text()) if a.reuse and (d/'editor.json').exists() else {}
        if old.get('status')=='PASS' and old.get('fingerprint')==fp and all((d/'screenshots'/f).exists() and sha(d/'screenshots'/f)==h for f,h in old['screenshots'].items()):rows.append(old);continue
        try:
            exe=prepare(spec,d,host,a.cmake,driver);screens=d/'screenshots';screens.mkdir(exist_ok=True)
            for p in screens.glob('*.png'):p.unlink()
            run([exe,screens,spec.slug,'gfx' if g.get('has_gfx') else 'no-gfx'],d/'editor-run.log',host,timeout=120)
            result=json.loads((d/'editor-run.log').read_text().strip().splitlines()[-1]);assert result.pop('worker_complete',False)
            assert result['editor_lifetimes']==2 and (not g.get('has_gfx') or result['gfx_lifetimes']==2)
            assert all((screens/f).exists() for f in ['open.png','resized.png','reopened.png'])
            row.update(result,status='PASS',screenshots={p.name:sha(p) for p in screens.glob('*.png')},executable_sha256=sha(exe),host_optimization='O0',dsp_optimization='O2')
        except Exception as e:row.update(status='FAIL',error=str(e))
        row['seconds']=round(time.monotonic()-start,3);(d/'editor.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        (out/'editors.json').write_text(json.dumps(rows,indent=2)+'\n');print('EDITOR',spec.slug,row['status'],row.get('error','')[:240],flush=True)
    (out/'editors.json').write_text(json.dumps(rows,indent=2)+'\n');return int(any(r['status']!='PASS' for r in rows))
if __name__=='__main__':raise SystemExit(main())
