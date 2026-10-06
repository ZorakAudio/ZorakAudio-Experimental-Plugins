from pathlib import Path
import subprocess,sys,json,struct,math,os
# Compare only generated result dumps; opens no source audio.
r=Path.cwd()
b=Path('build/faust-sections/sampler-corpus/Sample')
prefix=sys.argv[1] if len(sys.argv)>1 else 'private'
rows=[]
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
 for case in range(4):
  a=b/(prefix+'-before')/f'{rate}-{block}-{case}.json';z=b/(prefix+'-after')/a.name
  def vals(p):d=p.with_suffix('.json.bin').read_bytes();return struct.unpack('<'+'f'*(len(d)//4),d)
  x=vals(a);y=vals(z);assert len(x)==len(y);assert any(q!=0 for q in x);assert all(math.isfinite(q) for q in x+y);old=json.loads(a.read_text());new=json.loads(z.read_text());row={'rate':rate,'block':block,'case':case,'before':old,'after':new,'speedup':old['median_seconds']/new['median_seconds'],'maximum_error':max(abs(q-w) for q,w in zip(x,y))};rows.append(row);print(row,flush=True)
(b/(prefix+'-results.json')).write_text(json.dumps(rows,indent=2));assert max(x['maximum_error'] for x in rows)<1e-7
