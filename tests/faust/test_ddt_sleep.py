"""Original versus new DPT, awake versus cooperative skip, without audio files."""
from pathlib import Path
import subprocess,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as c
base=ROOT/'build/catalog-faust-audit/DDT';rows=[]
cpp=(ROOT/'tests/faust/dpt_sleep_profile.cpp').read_text()
cpp=cpp.replace('s.sliders[0]=0;s.sliders[1]=70;s.sliders[2]=mode;s.sliders[3]=0;', 'const double defaults[]={30,50,40,55,2,100,0,0,50};for(int i=0;i<9;++i)s.sliders[i]=defaults[i];s.sliders[4]=mode;')
cpp=cpp.replace('s.sliders[0]=73;s.sliders[2]=1;', 's.sliders[0]=73;s.sliders[4]=4;s.sliders[8]=80;')
(base/'ddt_sleep_profile.cpp').write_text(cpp)
for version in ['baseline','candidate']:
    out=base/version;out.mkdir(exist_ok=True)
    m,meta=c.compile_jsfx_to_ir((base/(version+'.jsfx')).read_text(encoding='utf-8'),native_gfx_legacy=True)
    (out/'JSFXDSP.h').write_text(c._emit_header(meta));(out/'dsp.ll').write_text(str(m))
    subprocess.run(['clang++','-O2','-c',str(out/'dsp.ll'),'-o',str(out/'dsp.obj')],check=True)
    subprocess.run(['clang++','-O2','-std=c++20','-I'+str(out),'-I'+str(ROOT/'src'),str(base/'ddt_sleep_profile.cpp'),str(out/'dsp.obj'),'-o',str(out/'sleep.exe')],check=True)
    for rate,block in [(48000,64),(48000,256),(96000,1024)]:
        for mode in [0,2,4]:
            for coop in ([0] if version=='baseline' else [0,1]):
                p=out/f'{rate}-{block}-{mode}-{coop}.json'
                subprocess.run([str(out/'sleep.exe'),str(p),str(rate),str(block),str(coop),str(mode)],check=True)
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
    for mode in [0,2,4]:
        a=base/'baseline'/f'{rate}-{block}-{mode}-0.json';original=a.with_suffix('.json.bin').read_bytes()
        for coop in [0,1]:
            b=base/'candidate'/f'{rate}-{block}-{mode}-{coop}.json';actual=b.with_suffix('.json.bin').read_bytes()
            if actual!=original:raise RuntimeError(f'Null failure: {rate}/{block}/{mode}/{coop}')
            row={'rate':rate,'block':block,'mode':mode,'cooperative':coop,'bit_identical':True,'before':json.loads(a.read_text()),'after':json.loads(b.read_text())}
            if coop and not row['after']['skipped_blocks']:raise RuntimeError('Fixture did not actually sleep')
            rows.append(row)
(base/'sleep-results.json').write_text(json.dumps(rows,indent=2));print('ALL 18 NULL COMPARISONS PASS')
