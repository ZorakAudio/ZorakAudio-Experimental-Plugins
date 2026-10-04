"""Native shared string/match behavior against the vendored WDL oracle."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
CASES=[
 ('string_boundaries', "x=str_setchar(3,0,65);y=str_setchar(3,100,70);str_setlen(3,4);z=strcmp(3,\"A   \");strcpy(#a,\"abc\");strncpy(#b,#a,0);u=strcmp(#b,\"abc\");strncpy(#b,\"xyz\",0);v=strlen(#b);str_setchar(4,0,65);str_setchar(4,0,65535,'us');w=strlen(4);",['x','y','z','u','v','w']),

 ('string_edits','''#a="hello";#a+=" world";strncpy(#b,#a,7);strcpy_substr(#c,#a,-5);str_insert(#b,"-",2);str_delsub(#b,1,3);x=strlen(#a);y=strcmp(#b,"hllo w");z=stricmp(#c,"WORLD");u=str_getchar(#a,-1);''',['x','y','z','u']),
 ('binary_strings','''str_setchar(3,0,65535,'us');str_setchar(3,2,-120,'c');str_setchar(3,3,1.25,'f');str_setchar(3,7,4096,'I');x=strlen(3);y=str_getchar(3,0,'us');z=str_getchar(3,2,'c');u=str_getchar(3,3,'f');v=str_getchar(3,7,'I');''',['x','y','z','u','v']),
 ('matching','''x=match("%f","-21.5",v);y=matchi("note %d*?=*?%f*","NOTE 12 = 440.25",note,freq);z=match("*%0-16{dest}s*.*","C:\\kick.wav");u=strcmp(dest,"kick");dest=14;z=match("*\\\\%0-16{dest}s*.*","C:\\\\kick.wav");u=strcmp(14,"kick");a=match("%s","alpha",15);b=strcmp(15,"alpha");''',['x','v','y','note','freq','z','u','a','b']),
 ('capture_alias_and_named','''#a="foo";x=match("%s",#a,#a);y=strcmp(#a,"foo");z=match("%{#out}s","bar");u=strcmp(#out,"bar");a=match("*?x*","axbxx");b=match("%D%S","123abc",number,7);c=strcmp(7,"23abc");''',['x','y','z','u','a','b','number','c']),
 ('anonymous_strings','''a=sprintf(#,"foo");b=sprintf(#,"bar");x=strcmp(a,"foo");y=strcmp(b,"bar");z=(a!=b);''',['x','y','z']),
 ('formatting','''sprintf(#a,"%d %04x %.2f %s",-8,15,1.25,"foo");x=strcmp(#a,"-8 000f 1.25 foo");strcpy(2.6,"z");y=strcmp(3,"z");''',['x','y']),
]
def run(cmd):
 r=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 return r.stdout

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--eel',required=True,type=Path);ap.add_argument('--wdl-build',required=True,type=Path);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);rows=[]
 for name,code,queries in CASES:
  d=args.out/name;d.mkdir(exist_ok=True);(d/'input.eel').write_text(code);(d/'input.jsfx').write_text('@init\n'+code)
  run([sys.executable,ROOT/'dsp_jsfx_aot.py',d/'input.jsfx','--native-gfx-legacy','--out-h',d/'JSFXDSP.h','--out-obj',d/'JSFXDSP.o','--out-ll',d/'JSFXDSP.ll','--meta',d/'meta.json'])
  meta=json.loads((d/'meta.json').read_text());cpp='''#include "JSFXDSP.h"
#include "juce_contract_stub.h"
#include "YSFXGfxInterpreter.h"
#include "NativeGfxPrototype.h"
#include <iostream>
#include <iomanip>
extern "C" void jsfx_ensure_mem(DSPJSFX_State*,int64_t){}
int main(){DSPJSFX_State st{};std::vector<DSPJSFX_Cell> mem(65536);st.mem=mem.data();st.memN=mem.size();jsfx_native_gfx::Frame frame;st.nativeStrings=&frame.ownedStrings;frame.bindLegacy(st,640,400);jsfx_init(&st);std::cout<<std::setprecision(17);
'''
  cpp+=''.join(f'std::cout<<"{q}="<<double(st.vars[{meta["vars"][q]}])<<"\\n";\n' for q in queries)+'}\n';(d/'test.cpp').write_text(cpp)
  run(['c++','-std=c++20','-O0','-DEEL_TARGET_PORTABLE=1','-DWDL_FFT_REALSIZE=8','-I'+str(ROOT/'src'),'-I'+str(ROOT/'tests/jsfx_showcase'),'-I'+str(d),d/'test.cpp',d/'JSFXDSP.o',args.wdl_build/'libshowcase_eel.a','-lpthread','-lm','-o',d/'test'])
  def values(s):return {k:float(v) for k,v in (line.split('=',1) for line in s.splitlines())}
  native=values(run([d/'test']));reference=values(run([args.eel,d/'input.eel',*queries]));row={'name':name,'native':native,'reference':reference,'status':'PASS' if native==reference else 'FAIL'};rows.append(row);print(row,flush=True)
 (args.out/'results.json').write_text(json.dumps(rows,indent=2));return int(any(r['status']!='PASS' for r in rows))
if __name__=='__main__':sys.exit(main())
