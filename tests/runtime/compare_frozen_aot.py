"""Compare exact audio/state bits before and after the shared-runtime migration.

Uses the frozen compiler and production Faust engine as the reference. Each
side is separately compiled and linked; different state ABIs cannot mask a
layout mismatch by sharing an object. This is a DSP gate, not a host/UI gate.
"""
from pathlib import Path
import ast
import importlib.util
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'build/runtime-unification/aot-bit-comparison'
BASE=ROOT/'build/runtime-unification/baseline'
sys.path.insert(0,str(ROOT))
if os.name=='nt':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value);return value

def compiler(name,root):
    import scripts
    faust=module('scripts.jsfx_faust_compiler',root/'scripts/jsfx_faust_compiler.py')
    scripts.jsfx_faust_compiler=faust
    return module(name,root/'dsp_jsfx_aot.py')

def run(args):
    p=subprocess.run([str(a) for a in args],cwd=ROOT,capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    return p.stdout

DRIVER=r'''
#include "JSFXDSP.h"
#include <algorithm>
#include <bit>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>
#define JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED 1
#include <juce_core/juce_core.h>
#define WDL_FFT_REALSIZE 8
#include "WDL/fft.h"
#include "JsfxNumericBuiltins.h"
#define JSFX_FAUST_IMPLEMENTATION
#include "JsfxFaust.h"
extern "C" void jsfx_ensure_mem(DSPJSFX_State* s,int64_t n){if(n>s->memN)s->memoryFault=1;}
int main(){
 DSPJSFX_State s{};
#if DSPJSFX_DYNAMIC_VARIABLES
 std::vector<DSPJSFX_Cell> variables(DSPJSFX_VARS_COUNT);s.vars=variables.data();s.varsN=variables.size();
#endif
 std::vector<DSPJSFX_Cell> heap(65536);s.mem=heap.data();s.memN=heap.size();s.srate=48000;s.sliders[0]=.5;
#if DSPJSFX_HAS_FAUST
 jsfx_faust::Engine engine;engine.prepare(48000,4096);s.faustContext=&engine;
#endif
 jsfx_init(&s);jsfx_slider(&s);
 int position=0;
 for(int count:{1,7,32,64,256,1024,4096}){
   std::vector<float> left(count),right(count),a(count),b(count);
   for(int f=0;f<count;++f,++position){left[f]=float((position*17)%251-125)/128.f;right[f]=float((position*31)%239-119)/128.f;}
   const float* in[]={left.data(),right.data()};float* out[]={a.data(),b.data()};
   jsfx_process_block(&s,in,out,2,count);
   for(int f=0;f<count;++f)std::cout<<std::bit_cast<uint32_t>(a[f])<<' '<<std::bit_cast<uint32_t>(b[f])<<'\n';
 }
 for(auto& v:DSPJSFX_VARS)std::cout<<v.name<<' '<<std::bit_cast<uint64_t>(double(s.vars[v.index]))<<'\n';
 for(int n=0;n<256;++n)std::cout<<std::bit_cast<uint64_t>(double(s.sliders[n]))<<'\n';
 for(int n=0;n<256;++n)std::cout<<std::bit_cast<uint64_t>(double(s.mem[n]))<<'\n';
 std::cout<<"FAULT "<<s.memoryFault<<'\n';return s.memoryFault?1:0;
}
'''

def main():
    for name,expected in json.loads((BASE/'source-sha256.json').read_text()).items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=expected:
            raise AssertionError('Frozen baseline changed: '+name)
    reference=compiler('frozen_compiler',BASE)
    current=compiler('current_compiler',ROOT)
    OUT.mkdir(parents=True,exist_ok=True)
    fft=OUT/'wdl-fft.obj'
    run(['clang','-O2','-DWDL_FFT_REALSIZE=8','-c',ROOT/'src/WDL/fft.c','-o',fft])
    core=ROOT/'build/jit-editor/CMakeFiles/JITEditor.dir/D_/Dev/ZorakAudio-Experimental-Plugins/libs/JUCE/modules/juce_core/juce_core.cpp.obj'
    coreTime=core.with_name('juce_core_CompilationTime.cpp.obj')
    if not core.exists():raise RuntimeError('Build the production JITEditor target before this gate')
    cases={
      'stateful_eel': '@init\nz=0;gain=.5;\n@block\nblocks+=1;\n@sample\nz=.97*z+.03*spl0;0[0]=z;spl0=max(-1,min(1,z*3))*gain;spl1=.8*spl1+.2*z;',
      'eel_memory_fft': '@init\nmemset(0,0,128);0[0]=1;fft(0,16);fft_permute(0,16);fft_ipermute(0,16);ifft(0,16);gain=0[0]/16;\n@sample\nspl0*=gain;spl1*=gain;',
      'eel_functions': '@init\nfunction band(x) instance(z,g)(g=x;z=0;);function tick(x) instance(z,g)(z=.99*z+.01*x;z*g;);left.band(.5);right.band(.25);\n@sample\nspl0=left.tick(spl0);spl1=right.tick(spl1);',
    }
    tree=ast.parse((ROOT/'tests/faust/test_mixed.py').read_text())
    fixtures=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets))
    cases.update({name:value[1] for name,value in fixtures.items()})
    results=[]
    for legacy in (False,True):
      for name,text in cases.items():
        streams=[]
        for label,c,headers in [('before',reference,BASE/'src'),('after',current,ROOT/'src')]:
          folder=OUT/('native' if legacy else 'publication')/name/label;folder.mkdir(parents=True,exist_ok=True)
          ir,meta=c.compile_jsfx_to_ir(text,native_gfx_legacy=legacy)
          ll=folder/'program.ll';ll.write_text(str(ir));(folder/'JSFXDSP.h').write_text(c._emit_header(meta));(folder/'driver.cpp').write_text(DRIVER)
          obj=folder/'program.obj';exe=folder/'check.exe'
          run(['clang++','-O2','-c',ll,'-o',obj])
          run(['clang++','-std=c++20','-O2','-DNDEBUG','-I'+str(folder),'-I'+str(headers),'-I'+str(ROOT/'src'),'-I'+str(ROOT/'libs/JUCE/modules'),folder/'driver.cpp',obj,fft,core,coreTime,'-lkernel32','-luser32','-lshell32','-ladvapi32','-lole32','-loleaut32','-luuid','-lwinmm','-lws2_32','-lversion','-lshlwapi','-lbcrypt','-o',exe])
          output=run([exe]);(folder/'bits.txt').write_text(output);streams.append(output)
        if streams[0]!=streams[1]:
          a=streams[0].splitlines();b=streams[1].splitlines();i=next((n for n,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
          raise AssertionError(f'{name} legacy={legacy}: first bit mismatch at {i}: {a[i:i+1]} / {b[i:i+1]}')
        results.append(dict(case=name,native=legacy,status='EXACT PASS'))
        print('EXACT PASS',name,'native' if legacy else 'publication',flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'{len(results)} exact before/after AOT DSP and state comparisons passed')

if __name__=='__main__':main()
