"""Numerical integration qualification of the actual Sample/Corpus DSP islands."""
from pathlib import Path
import sys,subprocess,json,struct,math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));import dsp_jsfx_aot as c
BLOCK='--block' in sys.argv
if BLOCK:sys.argv.remove('--block')
PREFIX='block-' if BLOCK else ''
BASE=ROOT/'build/faust-sections/sampler-corpus'
def values(path,fmt):
 data=path.read_bytes();return struct.unpack('<'+fmt*(len(data)//struct.calcsize(fmt)),data)
def compare(a,b,fmt):
 x=values(a,fmt);y=values(b,fmt);assert len(x)==len(y);assert all(math.isfinite(z) for z in x) and all(math.isfinite(z) for z in y)
 return {'count':len(x),'different':sum(p!=q for p,q in zip(x,y)),'maximum_error':max(abs(p-q) for p,q in zip(x,y))}
for key in sys.argv[1:] or ['Corpus','Sample']:
 folder=BASE/key
 for version in ['before','after']:
  out=folder/(PREFIX+version);out.mkdir(exist_ok=True)
  module,meta=c.compile_jsfx_to_ir((folder/(PREFIX+'kernel-'+version+'.jsfx')).read_text(encoding='utf-8'),native_gfx_legacy=True)
  (out/'JSFXDSP.h').write_text(c._emit_header(meta),encoding='utf-8');(out/'dsp.ll').write_text(str(module),encoding='utf-8');(out/'meta.json').write_text(json.dumps(meta,indent=2))
  subprocess.run(['clang++','-O2','-c',str(out/'dsp.ll'),'-o',str(out/'dsp.obj')],check=True)
  subprocess.run(['clang++','-O2','-std=c++20','-UNDEBUG','-DPRIVATE_CORPUS='+str(int(BLOCK)),'-DCORPUS_TEST='+str(int(key=='Corpus')),'-I'+str(out),'-I'+str(ROOT/'src'),str(ROOT/'tests/faust/sampler_corpus_kernel.cpp'),str(out/'dsp.obj'),'-o',str(out/'kernel.exe')],check=True)
  for rate,block in [(48000,64),(48000,256),(96000,1024)]:
   for case in range(3 if key=='Corpus' else 4):subprocess.run([str(out/'kernel.exe'),str(out/f'{rate}-{block}-{case}.json'),str(rate),str(block),str(case)],check=True)
 rows=[]
 for rate,block in [(48000,64),(48000,256),(96000,1024)]:
  for case in range(3 if key=='Corpus' else 4):
   a=folder/(PREFIX+'before')/f'{rate}-{block}-{case}.json';b=folder/(PREFIX+'after')/a.name
   old=json.loads(a.read_text());new=json.loads(b.read_text());audio=compare(a.with_suffix('.json.bin'),b.with_suffix('.json.bin'),'f');state=compare(a.with_suffix('.json.states.bin'),b.with_suffix('.json.states.bin'),'d')
   row={'before':old,'after':new,'speedup':old['median_seconds']/new['median_seconds'],'audio':audio,'state':state};rows.append(row);print(key,rate,block,case,json.dumps(row),flush=True)
 (folder/(PREFIX+'kernel-results.json')).write_text(json.dumps(rows,indent=2)+'\n')
 assert max(x['audio']['maximum_error'] for x in rows)<=1e-7, key+' audio divergence'
 assert max(x['state']['maximum_error'] for x in rows)<=1e-10, key+' state divergence'
 print(key,'all finite audio/state comparisons passed',flush=True)
