"""Windows before/after real AOT host check using cached, unchanged JUCE/WDL.

Every generated guest and state-dependent production source is recompiled.
The oracle is the frozen source/compiler snapshot, never a newly frozen build.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import subprocess
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from compare_frozen_aot import compiler,BASE
from pluginlib import discover_plugins
from build import native_gfx_modes_for_plugin,write_embedded_text_header
from jsfx_source import resolve_source,apply_host_options
from embed_gfx_resources import write_gfx_resources
OUT=ROOT/'build/runtime-unification/aot-host-comparison'
ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)
shell=ctypes.windll.shell32
shell.CommandLineToArgvW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_int)]
shell.CommandLineToArgvW.restype=ctypes.POINTER(ctypes.c_wchar_p)
def split(cmd):
    count=ctypes.c_int();ptr=shell.CommandLineToArgvW(cmd,ctypes.byref(count))
    result=[ptr[n] for n in range(count.value)];ctypes.windll.kernel32.LocalFree(ptr);return result
def run(args,cwd=ROOT,timeout=600):
    p=subprocess.run([str(a) for a in args],cwd=cwd,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    return p.stdout
def commands(folder):
    cache=(folder/'CMakeCache.txt').read_text()
    ninja=next(line.split('=',1)[1] for line in cache.splitlines() if line.startswith('CMAKE_MAKE_PROGRAM:'))
    return json.loads(run([ninja,'-C',folder,'-t','compdb']))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plugins',nargs='+',default=['EasyExpander','CMD','HyperrealHybrid','joep_amaranth','Sample','Corpus']);ap.add_argument('--banks',action='store_true');a=ap.parse_args();comparison_out=ROOT/'build/runtime-unification/aot-bank-comparison' if a.banks else OUT
    for name,expected in json.loads((BASE/'source-sha256.json').read_text()).items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==expected,'Frozen baseline changed: '+name
    specs={s.slug:s for s in discover_plugins(ROOT)}
    cmds=commands(ROOT/'build/windows/CMD')
    command=split(next(row['command'] for row in cmds if row.get('file','').endswith('JSFXJuceProcessor.cpp')))
    flags=command[1:command.index('-o')]
    # Dependency output belongs to the original build; remove it from the probe.
    clean=[];i=0
    while i<len(flags):
        if flags[i] in ('-MT','-MF'):i+=2;continue
        if flags[i]=='-MD':i+=1;continue
        if flags[i].startswith('-DZA_PLUGIN_NAME='):clean.append('-DZA_PLUGIN_NAME="SharedRuntimeVerification"')
        elif flags[i].startswith('-DJucePlugin_Name='):clean.append('-DJucePlugin_Name="SharedRuntimeVerification"')
        elif not flags[i].startswith('-O'):clean.append(flags[i])
        i+=1
    flags=clean+['-O0','-std=c++20']+(['-DZA_SAMPLE_GFX_TEST_RUNNER=1'] if a.banks else [])
    # Match the production host's CRT/configuration. Explicit guest-dependent
    # objects above resolve first; this archive supplies unchanged JUCE/WDL.
    host=ROOT/'build/windows/CMD'
    libs=[host/'CMD_artefacts/Release/CMD_SharedCode.lib',host/'clap_juce_extensions/clap_juce_extensions.lib']
    libs+=['-l'+name for name in ['kernel32','user32','gdi32','winspool','shell32','ole32','oleaut32','uuid','comdlg32','advapi32','oldnames']]
    comparison_out.mkdir(parents=True,exist_ok=True);results=[]
    for slug in a.plugins:
        spec=specs[slug];source=apply_host_options(resolve_source(spec.entry_path),spec.raw.get('jsfxCompatibility')).text
        prototype,legacy=native_gfx_modes_for_plugin(spec)
        if spec.category=='JoepVanlier':prototype,legacy=False,True
        outputs=[]
        for label,source_root in [('before',BASE),('after',ROOT)]:
            folder=comparison_out/slug/label;folder.mkdir(parents=True,exist_ok=True)
            runtime_fingerprint=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((source_root/'src').iterdir()) if p.is_file())).digest()
            c=compiler('host_'+label,source_root)
            ir,meta=c.compile_jsfx_to_ir(source,native_gfx_prototype=prototype,native_gfx_legacy=legacy)
            (folder/'JSFXDSP.h').write_text(c._emit_header(meta));(folder/'guest.ll').write_text(str(ir))
            write_embedded_text_header(text=source,variable_name='kJsfxSourceText',out_header=folder/'JSFXSource.h',banner='verification')
            write_gfx_resources(spec.entry_path,source,folder/'JSFXResources.h')
            guest=folder/'guest.obj';run(['clang++','-O2','-c',folder/'guest.ll','-o',guest])
            objects=[guest]
            includes=['-I'+str(folder),'-I'+str(source_root/'src')]
            for name in ['JSFXJuceProcessor','DspJsfxRuntime','DspJsfxRuntimeBuiltins','DspJsfxMessageBus','DspJsfxGmem','DspJsfxSamplePool','DspJsfxSharedMemory']:
                source_file=source_root/'src'/(name+'.cpp')
                if label=='before' and name=='JSFXJuceProcessor':
                    # Frozen source has a pre-existing non-native GFX build
                    # error: its callback has three parameters with defaults,
                    # whereas Interpreter needs a one-parameter callable. Adapt
                    # only that callback in a probe copy; retain the raw oracle.
                    text=source_file.read_text().replace('jsfx_gfx_resources::loadImage,','[](const juce::String& name){return jsfx_gfx_resources::loadImage(name);},').replace('=jsfx_gfx_resources::loadImage;','= [](const juce::String& name){return jsfx_gfx_resources::loadImage(name);};')
                    source_file=folder/'frozen-host-callback-adapter.cpp';source_file.write_text(text)
                obj=folder/(name+'.obj')
                fingerprint=hashlib.sha256(runtime_fingerprint+source_file.read_bytes()+(folder/'JSFXDSP.h').read_bytes()+(folder/'JSFXSource.h').read_bytes()+(folder/'JSFXResources.h').read_bytes()+json.dumps(includes+flags).encode()).hexdigest()
                stamp=obj.with_suffix('.sha')
                if not obj.exists() or not stamp.exists() or stamp.read_text()!=fingerprint:
                    run(['clang++',*includes,*flags,'-c',source_file,'-o',obj]);stamp.write_text(fingerprint)
                objects.append(obj)
            driver=folder/'driver.obj';run(['clang++',*flags,'-c',ROOT/('tests/runtime/host_bank_check.cpp' if a.banks else 'tests/runtime/host_bit_check.cpp'),'-o',driver])
            exe=folder/'check.exe';run(['clang++','-nostartfiles','-nostdlib','-fuse-ld=lld-link','-Xlinker','/subsystem:console',driver,*objects,*libs,'-o',exe]);text=run([exe,slug] if a.banks else [exe]);(folder/'result.txt').write_text(text);outputs.append(text)
            print('HOST RAN',slug,label,flush=True)
        if not a.banks and outputs[0]!=outputs[1]:raise AssertionError(slug+': real host bits/parameters/latency/state-size differ\n'+outputs[0]+'\n'+outputs[1])
        results.append(dict(plugin=slug,status='LOADED PASS (no audio-null claim)' if a.banks else 'EXACT PASS'));print('BANK PASS' if a.banks else 'HOST EXACT PASS',slug,flush=True)
        (comparison_out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
