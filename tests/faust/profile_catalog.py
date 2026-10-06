"""Quick native-kernel comparisons; RAM-generated input, never opens audio files."""
from pathlib import Path
import sys, subprocess, json, struct, re, math
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as c
DPT_FAUST='--dpt-faust' in sys.argv
if DPT_FAUST:sys.argv.remove('--dpt-faust')
SLEEP='--sleep' in sys.argv
if SLEEP:sys.argv.remove('--sleep')
LONG='--long' in sys.argv
if LONG:sys.argv.remove('--long')

def run(key):
    base=ROOT/'build/catalog-faust-audit'/key
    source=(base/'baseline.jsfx').read_text(encoding='utf-8')
    sliders=re.findall(r'(?m)^slider(\d+):(?:(\w+)=)?([\d.-]+)<([^,>]+)',source)
    defaults=[0.0]*max(int(a[0]) for a in sliders)
    aliases=['']*len(defaults)
    for index,alias,value,minimum in sliders:
        defaults[int(index)-1]=float(value);aliases[int(index)-1]=alias or ''
    alternate=float(sliders[0][3]) if float(sliders[0][3])!=defaults[0] else defaults[0]+1
    header='#define CATALOG_SLIDERS '+str(len(defaults))+'\n'
    header+='#define CATALOG_DEFAULTS {'+','.join(map(str,defaults))+'}\n'
    header+='#define CATALOG_ALIASES {'+','.join(json.dumps(a) for a in aliases)+'}\n'
    header+='#define CATALOG_ALTERNATE '+str(alternate)+'\n'
    cpp=(ROOT/'tests/faust/easy_kernel_profile.cpp').read_text(encoding='utf-8')
    if LONG:cpp=cpp.replace('rate*4+block','rate*40+block')
    cpp=cpp.replace('#include "JSFXDSP.h"','#include "catalog_controls.h"\n#include "JSFXDSP.h"')
    cpp=cpp.replace('const double defaults[]={-40,24,50,0,20000};','const double defaults[]=CATALOG_DEFAULTS;')
    cpp=cpp.replace('const char* aliases[]={"thresh_db","depth_db","contour","det_hpf_hz","det_lpf_hz"};','const char* aliases[]=CATALOG_ALIASES;').replace('c<5','c<CATALOG_SLIDERS')
    cpp=cpp.replace('s.sliders[3]=200;s.sliders[4]=5000;','s.sliders[0]=CATALOG_ALTERNATE;').replace('s.sliders[0]=-35;s.sliders[2]=75;','s.sliders[0]=defaults[0];')
    cpp=cpp.replace('const float* in[]={left.data()+i,right.data()+i};float* out[]={outputL.data()+i,outputR.data()+i};jsfx_process_block(&s,in,out,2,block);','const float* in[]={left.data()+i,right.data()+i,left.data()+i,right.data()+i};float* out[]={outputL.data()+i,outputR.data()+i,discardL.data(),discardR.data()};jsfx_process_block(&s,in,out,4,block);')
    cpp=cpp.replace('std::vector<double> trials;','std::vector<float> discardL(block),discardR(block); std::vector<double> trials;')
    cases=range(4) if key in ['ERBTilt','SpectralStabilizer'] else range(2) if DPT_FAUST else range(1)
    if DPT_FAUST:
        cpp=cpp.replace('for(int c=0;c<CATALOG_SLIDERS;++c)s.sliders[c]=defaults[c];sync();','for(int c=0;c<CATALOG_SLIDERS;++c)s.sliders[c]=defaults[c]; s.sliders[2]=argc>4?std::stoi(argv[4]):0; sync();')
    if key=='ERBTilt':
        cpp=cpp.replace('for(int c=0;c<CATALOG_SLIDERS;++c)s.sliders[c]=defaults[c];sync();',
            'for(int c=0;c<CATALOG_SLIDERS;++c)s.sliders[c]=defaults[c]; '
            'int scenario=argc>4?std::stoi(argv[4]):0; '
            'if(scenario==1){s.sliders[1]=20;s.sliders[2]=0;s.sliders[3]=0;} '
            'if(scenario==2){s.sliders[1]=20000;s.sliders[3]=100;} '
            'if(scenario==3){s.sliders[1]=8000;s.sliders[2]=50;s.sliders[3]=100;} sync();')
    if key=='SpectralStabilizer':
        cpp=cpp.replace('for(int c=0;c<CATALOG_SLIDERS;++c)s.sliders[c]=defaults[c];sync();',
            'for(int c=0;c<CATALOG_SLIDERS;++c)s.sliders[c]=defaults[c]; '
            'int scenario=argc>4?std::stoi(argv[4]):0; '
            'if(scenario==1){s.sliders[1]=0;s.sliders[2]=0;} '
            'if(scenario==2){s.sliders[1]=100;s.sliders[2]=100;} '
            'if(scenario==3){s.sliders[0]=2;s.sliders[1]=100;s.sliders[2]=0;} sync();')
    for version in ['baseline','candidate']:
        out=base/(('dpt-faust-' if DPT_FAUST else 'sleep-active-' if SLEEP else '')+version);out.mkdir(exist_ok=True)
        filename='sleep-candidate.jsfx' if SLEEP and version=='candidate' and key in ['ADS','SaliencePush'] else version+'.jsfx'
        if DPT_FAUST and version=='candidate':filename='faust-candidate.jsfx'
        module,meta=c.compile_jsfx_to_ir((base/filename).read_text(encoding='utf-8'),native_gfx_legacy=True)
        (out/'JSFXDSP.h').write_text(c._emit_header(meta),encoding='utf-8')
        (out/'catalog_controls.h').write_text(header,encoding='utf-8')
        (out/'profile.cpp').write_text(cpp,encoding='utf-8')
        (out/'dsp.ll').write_text(str(module),encoding='utf-8')
        (out/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        subprocess.run(['clang++','-O2','-c',str(out/'dsp.ll'),'-o',str(out/'dsp.obj')],check=True)
        subprocess.run(['clang++','-std=c++20','-O2','-UNDEBUG','-I'+str(out),'-I'+str(ROOT/'src'),str(out/'profile.cpp'),str(out/'dsp.obj'),'-o',str(out/'profile.exe')],check=True)
        for rate,block in [(48000,64),(48000,256),(96000,1024)]:
            for case in cases:
                subprocess.run([str(out/'profile.exe'),str(out/f'{rate}-{block}-{case}.json'),str(block),str(rate),str(case)],check=True)
    rows=[]
    for rate,block in [(48000,64),(48000,256),(96000,1024)]:
        for case in cases:
            name=f'{rate}-{block}-{case}.json'
            a=base/(('dpt-faust-' if DPT_FAUST else 'sleep-active-' if SLEEP else '')+'baseline')/name;b=base/(('dpt-faust-' if DPT_FAUST else 'sleep-active-' if SLEEP else '')+'candidate')/name
            x=struct.unpack('<'+'f'*(a.with_suffix('.json.bin').stat().st_size//4),a.with_suffix('.json.bin').read_bytes())
            y=struct.unpack('<'+'f'*len(x),b.with_suffix('.json.bin').read_bytes())
            if not all(math.isfinite(v) for v in x+y): raise RuntimeError('Non-finite output')
            error=max(abs(p-q) for p,q in zip(x,y))
            before=json.loads(a.read_text());after=json.loads(b.read_text())
            rows.append({'scenario':case,'before':before,'after':after,'speedup':before['median_seconds']/after['median_seconds'],'maximum_sample_error':error,'different_samples':sum(p!=q for p,q in zip(x,y))})
    (base/('dpt-faust-results.json' if DPT_FAUST else 'sleep-active-results.json' if SLEEP else 'results.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(key,json.dumps(rows),flush=True)

if __name__=='__main__':
    for key in sys.argv[1:]:run(key)
