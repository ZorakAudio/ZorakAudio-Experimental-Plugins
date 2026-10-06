"""Compare generated qualification dumps; never opens a source audio recording."""
from pathlib import Path
import sys,json,struct,math
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/faust-sections/sampler-corpus'
for key in sys.argv[1:] or ['Corpus','Sample']:
 base=BASE/key;rows=[]
 for rate,block in [(48000,64),(48000,256),(96000,1024)]:
  name=f'{rate}-{block}.json';a=base/'host-before'/name;b=base/'host-after'/name
  before=json.loads(a.read_text());after=json.loads(b.read_text());p=a.with_suffix('.json.pcm.bin');q=b.with_suffix('.json.pcm.bin');assert p.stat().st_size==q.stat().st_size
  count=different=0;maximum=squares=0
  with p.open('rb') as x,q.open('rb') as y:
   while data:=x.read(1048576):
    other=y.read(len(data));u=struct.unpack('<'+'f'*(len(data)//4),data);v=struct.unpack('<'+'f'*len(u),other)
    assert all(math.isfinite(z) for z in u+v)
    for z,w in zip(u,v):
     count+=1;e=abs(z-w);maximum=max(maximum,e);different+=z!=w;squares+=e*e
  state_equal=a.with_suffix('.json.state.csv').read_bytes()==b.with_suffix('.json.state.csv').read_bytes()
  row={'before':before,'after':after,'speedup':before['process_seconds']/after['process_seconds'],'audio':{'count':count,'different':different,'maximum_error':maximum,'rms_error':math.sqrt(squares/count)},'sampled_state_csv_exact':state_equal};rows.append(row);print(key,rate,block,json.dumps(row),flush=True)
 (base/'host-results.json').write_text(json.dumps(rows,indent=2)+'\n');assert max(x['audio']['maximum_error'] for x in rows)<=1e-6,key+' loaded output divergence'
