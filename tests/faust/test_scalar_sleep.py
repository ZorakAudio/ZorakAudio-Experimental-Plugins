"""ADS/SaliencePush default detector certificates and audio/key wake null checks."""
from pathlib import Path
import subprocess,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as c
key=sys.argv[1];base=ROOT/'build/catalog-faust-audit'/key;rows=[]
cpp=(ROOT/'tests/faust/dpt_sleep_profile.cpp').read_text()
cpp=cpp.replace('rate*12','rate*124').replace('rate*10/block','rate*122/block')
cpp=cpp.replace('t<2 || t>=11','(t>=120 && t<121) || t>=123').replace('t>=10','t>=122')
if key=='SaliencePush':
 cpp=cpp.replace('s.sliders[0]=0;s.sliders[1]=70;s.sliders[2]=mode;s.sliders[3]=0;', 'const double defaults[]={0,45,35,60,0};for(int i=0;i<5;++i)s.sliders[i]=defaults[i];')
 cpp=cpp.replace('s.sliders[0]=73;s.sliders[2]=1;', 's.sliders[0]=2;s.sliders[1]=75;')
else:
 cpp=cpp.replace('s.sliders[0]=0;s.sliders[1]=70;s.sliders[2]=mode;s.sliders[3]=0;', 'const double defaults[]={65,75,70,35,20,40,0,60};for(int i=0;i<8;++i)s.sliders[i]=defaults[i];')
 cpp=cpp.replace('s.sliders[0]=73;s.sliders[2]=1;', 's.sliders[0]=90;s.sliders[1]=100;')
cpp=cpp.replace('const float* in[]={l.data(),r.data()};float* out[]={ol.data(),orr.data()};jsfx_process_block(&s,in,out,2,block);', 'const float* in[]={l.data(),r.data(),l.data(),r.data()};float* out[]={ol.data(),orr.data(),keyL.data(),keyR.data()};jsfx_process_block(&s,in,out,4,block);')
cpp=cpp.replace('bool sleeping=false;', 'std::vector<float> keyL(block),keyR(block);bool sleeping=false;')
(base/'scalar_sleep_profile.cpp').write_text(cpp)
for version in ['baseline','candidate']:
    out=base/('sleep-'+version);out.mkdir(exist_ok=True)
    m,meta=c.compile_jsfx_to_ir((base/('baseline.jsfx' if version=='baseline' else 'sleep-candidate.jsfx')).read_text(encoding='utf-8'),native_gfx_legacy=True)
    (out/'JSFXDSP.h').write_text(c._emit_header(meta));(out/'dsp.ll').write_text(str(m))
    subprocess.run(['clang++','-O2','-c',str(out/'dsp.ll'),'-o',str(out/'dsp.obj')],check=True)
    subprocess.run(['clang++','-O2','-std=c++20','-I'+str(out),'-I'+str(ROOT/'src'),str(base/'scalar_sleep_profile.cpp'),str(out/'dsp.obj'),'-o',str(out/'sleep.exe')],check=True)
    for rate,block in [(48000,256),(96000,1024)]:
        for mode in [0]:
            for coop in ([0] if version=='baseline' else [0,1]):
                p=out/f'{rate}-{block}-{mode}-{coop}.json'
                subprocess.run([str(out/'sleep.exe'),str(p),str(rate),str(block),str(coop),str(mode)],check=True)
for rate,block in [(48000,256),(96000,1024)]:
    for mode in [0]:
        a=base/'sleep-baseline'/f'{rate}-{block}-{mode}-0.json';original=a.with_suffix('.json.bin').read_bytes()
        for coop in [0,1]:
            b=base/'sleep-candidate'/f'{rate}-{block}-{mode}-{coop}.json';actual=b.with_suffix('.json.bin').read_bytes()
            if actual!=original:raise RuntimeError(f'Null failure: {rate}/{block}/{mode}/{coop}')
            row={'rate':rate,'block':block,'mode':mode,'cooperative':coop,'bit_identical':True,'before':json.loads(a.read_text()),'after':json.loads(b.read_text())}
            if coop and not row['after']['skipped_blocks']:raise RuntimeError('Fixture did not actually sleep')
            rows.append(row)
(base/'sleep-results.json').write_text(json.dumps(rows,indent=2));print('ALL 4 NULL COMPARISONS PASS')
