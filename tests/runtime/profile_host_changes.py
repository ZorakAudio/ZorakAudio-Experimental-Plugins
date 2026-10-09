"""Compare HEAD and working runtime, including the corrected final backend.

Windows probe links cached release JUCE/WDL, compiling every state-dependent
source for each side. No plugin installation or DAW interaction is involved.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(Path(__file__).parent)]
import dsp_jsfx_aot as compiler
from compare_frozen_host import commands,split
from pluginlib import discover_plugins
from jsfx_source import resolve_source,apply_host_options
from build import write_embedded_text_header
from embed_gfx_resources import write_gfx_resources

def run(args,**kw):
    p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,check=True,**kw)
    return p.stdout

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plugin',default='joep_amaranth');args=ap.parse_args()
    if os.name!='nt':raise SystemExit('This probe uses cached Windows release libraries')
    out=ROOT/'build/runtime-performance/production'/args.plugin;out.mkdir(parents=True,exist_ok=True)
    snapshot=out/'snapshot';(snapshot/'src').mkdir(parents=True,exist_ok=True)
    for name in run(['git','ls-files','src']).splitlines():
        source=ROOT/name
        if source.is_file():
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT))
    spec=next(s for s in discover_plugins(ROOT) if s.slug==args.plugin)
    source=apply_host_options(resolve_source(spec.entry_path),spec.raw.get('jsfxCompatibility')).text
    ir,meta=compiler.compile_jsfx_to_ir(source,native_gfx_legacy=True)
    # Do not add Faust/task approximations to this probe. Shared helpers are
    # compiled normally; the input IR is identical on both sides.
    if meta['has_faust'] or meta['has_tasks']:raise ValueError('Choose a plain Joep DSP for this backend/host comparison')
    compiler._aot_opt_and_emit(ir,2,str(out/'after.obj'),None,target_triple='x86_64-pc-windows-msvc',out_ll_opt=str(out/'optimized.ll'),native_gfx_legacy=True)
    run(['clang++','--target=x86_64-pc-windows-msvc','-c',out/'optimized.ll','-o',out/'before.obj'])
    cached=ROOT/'build/windows/CMD'
    cmd=split(next(row['command'] for row in commands(cached) if row.get('file','').endswith('JSFXJuceProcessor.cpp')))
    raw=cmd[1:cmd.index('-o')];flags=[];i=0
    while i<len(raw):
        if raw[i] in ('-MT','-MF'):i+=2;continue
        if raw[i]=='-MD':i+=1;continue
        if raw[i].startswith(('-DZA_PLUGIN_NAME=','-DJucePlugin_Name=')):flags.append(raw[i].split('=',1)[0]+'="PerformanceProbe"')
        elif not raw[i].startswith('-O'):flags.append(raw[i])
        i+=1
    flags+=['-O2','-std=c++20']
    libraries=[cached/'CMD_artefacts/Release/CMD_SharedCode.lib',cached/'clap_juce_extensions/clap_juce_extensions.lib']
    libraries+=['-l'+name for name in ['kernel32','user32','gdi32','winspool','shell32','ole32','oleaut32','uuid','comdlg32','advapi32','oldnames']]
    results={'plugin':args.plugin,'baseline_commit':run(['git','rev-parse','HEAD']).strip(),'timing':'complete production callback, default controls, 48 kHz, stereo input; warmup + 5 trials'}
    checks=[]
    for label,source_root in [('before',snapshot),('after',ROOT)]:
        folder=out/label;folder.mkdir(exist_ok=True)
        (folder/'JSFXDSP.h').write_text(compiler._emit_header(meta))
        write_embedded_text_header(text=source,variable_name='kJsfxSourceText',out_header=folder/'JSFXSource.h',banner='performance probe')
        write_gfx_resources(spec.entry_path,source,folder/'JSFXResources.h')
        includes=['-I'+str(folder),'-I'+str(source_root/'src')]
        objects=[out/(label+'.obj')]
        for name in ['JSFXJuceProcessor','DspJsfxRuntime','DspJsfxRuntimeBuiltins','DspJsfxMessageBus','DspJsfxGmem','DspJsfxSamplePool','DspJsfxSharedMemory']:
            obj=folder/(name+'.obj');run(['clang++',*includes,*flags,'-c',source_root/'src'/(name+'.cpp'),'-o',obj]);objects.append(obj)
        for driver in ['host_bit_check','host_performance_check']:
            obj=folder/(driver+'.obj');exe=folder/(driver+'.exe')
            run(['clang++',*includes,*flags,'-c',Path(__file__).with_name(driver+'.cpp'),'-o',obj])
            run(['clang++','-nostartfiles','-nostdlib','-fuse-ld=lld-link','-Xlinker','/subsystem:console',obj,*objects,*libraries,'-o',exe])
            text=run([exe],timeout=120);(folder/(driver+'.txt')).write_text(text)
            if driver=='host_bit_check':checks.append(text)
            else:
                rows=[line.split() for line in text.splitlines()]
                results[label]={str(frames):{'trials_us':[float(row[2]) for row in rows if int(row[0])==frames]} for frames in (64,512)}
                for value in results[label].values():value['median_us']=statistics.median(value['trials_us'])
        print(label+' production processor verified and profiled',flush=True)
    if checks[0]!=checks[1]:raise AssertionError('Audio bits, MIDI, parameters, latency or state size changed')
    results['exact_production_output']=True
    results['speedup']={str(n):results['before'][str(n)]['median_us']/results['after'][str(n)]['median_us'] for n in (64,512)}
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
