"""Compare focused compiler cases to the actual WDL oracle, not hand-written DSP."""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CASES = [
 ('while_name_newlines', '''
historyX=64;historyY=80;historyX[0]=2;historyY[0]=3;viewScale=2;coord=0;
while
// A line comment and blank line between the name and group are whitespace.

(
  offsetX=(historyX[coord]*viewScale);
  offsetY=(historyY[coord]*viewScale);
  coord+=1;
  coord<1;
);
while /* block comment */
(coord<3)
(coord+=1;);
x=while
(coord+=1;coord<5;);
''', ['offsetx','offsety','coord','x']),
 ('receiver_namespace_members', '''
function band(x) instance(child,gain)(gain=x;child.left=gain;'''+16*'child.left+=gain/100;'+'''child.left;);
a=left.band(2);b=right.band(4);c=left.child.left;d=right.child.left;
''', ['a','b','c','d']),
 ('receiver_heap_reads', '''
table=64;table[0]=3;table[1]=5;
function coeff(x) local(n) instance(value) (n+=1;value=table[x]+n;'''+16*'value+=1;'+'''value;);
a=left.coeff(0);b=right.coeff(1);c=left.value;d=right.value;
''', ['a','b','c','d']),
 ('receiver_cells', '''
function coeff(x) local(n) instance(a,b) (n+=1;a=x+n;b=a*2;''' +16*'b+=a/100;'+'''b;);
v=left.coeff(2);w=right.coeff(4);z=left.coeff(8);
x=left.a;y=right.a;aa=left.b;bb=right.b;
''', ['v','w','z','x','y','aa','bb']),
 ('eel_loop_budgets', "i=0;x=while(i+=1;1;);y=i;i=0;z=loop(2000000,i+=1);u=i;v=loop(10^100,i+=1);w=i;", ['x','y','z','u','v','w']),

 ('eel_boolean_threshold', 'x=0.000001 ? 1 :0;y=!0.000001;z=-0.000001 ? 1 :0;a=0.00001 ? 1 :0;b=!(0/0);c=(0/0) ? 1 :0;d=(1/0) ? 1 :0;', ['x','y','z','a','b','c','d']),

 ('qualified_function_declarations', "function scope.setup(v) instance(w)(w=v;);function scope.read() instance(w)(w;);scope.setup(5);x=scope.read();y=scope.w;", ['x','y']),

 ('zero_parameter_overload', "function f()(7);x=f(3);function f(a)(a+1);y=f();z=f(4);", ['x','y','z']),

 ('overloaded_function_binding', 'function f(a,b)(a+b);function f(a)(a*2);x=f(3,4);y=f(3);z=f();a.x=f(7);', ['x','y','z','a.x']),
 ('empty_conditional_in_while', 'i=0;x=0;while(i<3 ? x+=1; i+=1;i<5;);y=i;', ['x','y']),

 ('redefined_function_binding', '''
function f(v) instance(value) (value=v+1;value;);
x=a.f(3);
function f(v) instance(child) (child.f(v)*2;);
y=b.f(7);z=b.child.value;xx=a.value;
''', ['x','y','z','xx']),
 ('eel_operator_precedence', '''
a=3;b=5;c=2;x=(a+b*=c);y=b;b=5;z=(a*b+=c);xx=(2+3/a+=b);
yy=24*8/4;zz=1||0&&0;aa=32>>1+2;bb=4|1==5;
''', ['a','x','y','z','xx','yy','zz','aa','bb']),
 ('eel_literals_and_declarations', '''
function f(v) local(a,,b) globals(x) (a=v;b=a+1;x=b;);f(3);
y='b';z='abcd';xx=$'a';yy=$~64;zz=0x10;aa=$xFF;
''', ['x','y','z','xx','yy','zz','aa']),
 ('sqrt_absolute_input', '''
v=-9;x=sqrt(-4);y=sqrt(v);z=sqrt(-0);xx=sqrt(16);yy=sqrt(-4)+1;
''', ['x','y','z','xx','yy']),
 ('instance_children', '''
function child_set(v) instance(value) (value=v;);
function set(v) instance(child) (child.child_set(v); child.twice=v*2;);
a.set(3); b.set(7);
x=a.child.value; y=b.child.value; xx=a.child.twice; yy=b.child.twice;
''', ['x','y','xx','yy']),
 ('relative_this', '''
function leaf(v)(this.value=v;);
function set(v)(this.child.leaf(v);this.other=v+1;);
a.set(4); b.set(9);x=a.child.value;y=b.child.value;xx=a.other;yy=b.other;
''', ['x','y','xx','yy']),
 ('persistent_locals', '''
function counter(v) local(n) instance(value) (n+=1;value=v;n;);
x=a.counter(2);y=b.counter(4);z=a.counter(7);xx=a.value;yy=b.value;
''', ['x','y','z','xx','yy']),
 ('nested_prefix', '''
function leaf(v) instance(value) (value=v;);
function mid(v) instance(child) (child.leaf(v););
function top(v) instance(sub) (sub.mid(v););
a.top(12);b.top(23);x=a.sub.child.value;y=b.sub.child.value;
''', ['x','y']),
 ('parameter_shadows_instance', '''
function f(x) instance(x,y) (x+=1;y=x;);
a.f(3);b.f(8);x=a.x;y=b.x;xx=a.y;yy=b.y;
''', ['x','y','xx','yy']),
 ('division_store_recovery', '''
d=0;x=1;x*=8/d;a=x;x=9;x/=d;b=x;c=0/d;v=1/d;w=v+1;
''', ['a','b','c','v','w']),
 ('memory_store_recovery', '''
d=0;mem[0]=1/d;mem[1]=2;mem[1]*=1/d;x=mem[0];y=mem[1];
''', ['x','y']),
 ('subnormal_store', '''
x=10^-300;x*=10^-20;y=x;a=10^-100;a*=10^-100;b=a;
''', ['x','y','a','b']),
 ('assignment_results', '''
d=0;x=(y=1/d)+2;z=(w=0;w/=d);q=3;
''', ['x','y','z','w','q']),
 ('independent_delay_buffers', '''
function init(ptr) instance(buffer,head)(buffer=ptr;head=0;);
function write(v) instance(buffer,head)(buffer[head]=v;head+=1;);
a.init(32);b.init(64);a.write(1);b.write(2);a.write(3);
x=32[0];y=64[0];z=32[1];ah=a.head;bh=b.head;
''', ['x','y','z','ah','bh']),
]


def run(cmd: list[str], timeout=120):
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(' '.join(cmd) + '\n' + result.stdout + result.stderr)
    return result.stdout


def test_cases(output: Path, eel_oracle: Path, cxx: str, *, sanitizer=False, legacy=False) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, code, queries in CASES:
        folder = output / name; folder.mkdir(exist_ok=True)
        # Force WDL's checked assignment path. Its default optimiser sometimes
        # elides checks; the opt-in native mode deliberately checks every store.
        # The full Abyss audio test separately uses default WDL optimisation.
        (folder / 'input.eel').write_text('//#eel-no-optimize:8\n' + code, encoding='utf-8')
        (folder / 'input.jsfx').write_text('options:za_eel2_stores=1\n@init\n' + code, encoding='utf-8')
        obj = folder / ('case.obj' if os.name == 'nt' else 'case.o')
        command = [sys.executable, str(ROOT/'dsp_jsfx_aot.py'), str(folder/'input.jsfx'),
                   '--out-h', str(folder/'case.h'), '--meta', str(folder/'case.json'),
                   '--out-ll', str(folder/'case.ll'), '--out-obj', str(obj)]
        if os.name == 'nt':command += ['--target','x86_64-pc-windows-msvc']
        if legacy: command += ['--native-gfx-legacy']
        run(command)
        if sanitizer:
            from instrument_ir import instrument
            clang = shutil.which('clang')
            if not clang:
                raise RuntimeError('ASan requires clang to instrument the native LLVM object')
            instrument(folder/'case.ll', obj, clang)
        meta = json.loads((folder/'case.json').read_text())
        if meta['numeric_semantics'] != 'eel2-stores':raise AssertionError('Store option not enabled')
        native_source = '''#include "case.h"
#include "JsfxStateVariables.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
extern "C" void jsfx_ensure_mem(DSPJSFX_State*st,int64_t n){
 if(n<0||n>st->memN)throw std::runtime_error("test memory bound");}
int main(){DSPJSFX_State st{};za::jsfx::StateVariables variables;variables.bind(st,DSPJSFX_VARS_COUNT);std::vector<DSPJSFX_Cell> memory(65536);st.mem=memory.data();st.memN=memory.size();st.srate=48000;
jsfx_init(&st);std::cout<<std::setprecision(17);
'''
        for q in queries:
            native_source += f'std::cout<<"{q}="<<st.vars[{meta["vars"][q]}]<<"\\n";\n'
        native_source += '}\n'
        (folder/'native.cpp').write_text(native_source)
        exe = folder / ('native.exe' if os.name == 'nt' else 'native')
        if os.name == 'nt':
            command=[cxx,'/nologo','/std:c++20' if legacy else '/std:c++17','/I'+str(ROOT/'src'),'/EHsc',str(folder/'native.cpp'),str(obj),'/Fe:'+str(exe)]
        else:
            command=[cxx,'-std=c++20' if legacy else '-std=c++17','-I'+str(ROOT/'src'),str(folder/'native.cpp'),str(obj),'-lm','-o',str(exe)]
            if sys.platform.startswith('linux'):command+=['-no-pie']
            if sanitizer:command+=['-fsanitize=address','-fno-omit-frame-pointer']
        run(command)
        native = run([str(exe)])
        reference = run([str(eel_oracle),str(folder/'input.eel'),*queries])
        def values(s):return {k:float(v) for k,v in (line.split('=',1) for line in s.splitlines())}
        a,b=values(native),values(reference)
        if a.keys()!=b.keys():raise AssertionError(f'{name}: result keys differ')
        for q in queries:
            if not math.isclose(a[q],b[q],rel_tol=1e-12,abs_tol=1e-300):
                raise AssertionError(f'{name}/{q}: native={a[q]!r}, EEL={b[q]!r}')
        results.append({'name':name,'values':a,'status':'PASS'})
        print('PASS native/EEL: '+name,flush=True)
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--eel',type=Path,required=True);ap.add_argument('--cxx',default='c++');ap.add_argument('--asan',action='store_true');ap.add_argument('--legacy',action='store_true')
    args=ap.parse_args()
    result=test_cases(args.out.resolve(),args.eel.resolve(),args.cxx,sanitizer=args.asan,legacy=args.legacy)
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
