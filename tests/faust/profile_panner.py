"""Whole-plugin panner timing and preserved-renderer null comparisons."""
from pathlib import Path
import argparse,array,json,statistics,subprocess,math
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/faust-sections/panner'
RECORDING=r'C:\Users\Louis\OneDrive\Sound Effects\NSFW\Mouth Noises\Files to Cut\Slapping dirty slut BJ.flac'
p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');a=p.parse_args()
rows=[]
for rate,block in ([(48000,256)] if a.quick else [(48000,64),(48000,256),(96000,1024)]):
 for case in ([0,1,2] if a.quick else range(9)):
  samples=None
  for trial in range(1 if a.quick or case>=3 else 3):
   results={};audio={}
   for label in (['baseline','fast','faust'] if trial%2==0 else ['faust','fast','baseline']):
    f=BASE/label/f'{rate}-{block}-{case}-{trial}.json'
    subprocess.run([str(BASE/label/'host.exe'),RECORDING,str(f),str(rate),str(block),str(case)],check=True)
    results[label]=json.loads(f.read_text());audio[label]=array.array('f');audio[label].frombytes(Path(str(f)+'.pcm.bin').read_bytes())
    assert results[label]['faust_scalars']==0
    if label=='faust':assert results[label]['faust_blocks']==results[label]['callbacks'] and results[label]['faust_frames']==results[label]['frames']
   assert len(audio['baseline'])==len(audio['fast'])==len(audio['faust'])
   error=max(abs(x-y) for x,y in zip(audio['baseline'],audio['fast']))
   assert error<1e-7,('Preserved renderer mismatch',rate,block,case,error)
   newError=math.sqrt(sum((x-y)**2 for x,y in zip(audio['baseline'],audio['faust']))/len(audio['baseline']))
   rows.append({'rate':rate,'block':block,'scenario':case,'trial':trial,'compared_values':len(audio['baseline']),'fast_max_error':error,'new_rms_difference':newError,'measurements':results})
   (BASE/('quick-results.json' if a.quick else 'results.json')).write_text(json.dumps(rows,indent=2))
  print('SUMMARY',rate,block,case,{label:statistics.median(r['measurements'][label]['process_seconds'] for r in rows if r['rate']==rate and r['block']==block and r['scenario']==case) for label in ['baseline','fast','faust']},flush=True)
print('PASS',len(rows),'paired scenarios',flush=True)

if not a.quick:
 def planar(path,block):
  data=array.array('f');data.frombytes(path.read_bytes());channels=[array.array('f'),array.array('f')]
  for at in range(0,len(data),block*2):
   channels[0].extend(data[at:at+block]);channels[1].extend(data[at+block:at+block*2])
  return channels
 x=planar(BASE/'faust/48000-64-1-0.json.pcm.bin',64);y=planar(BASE/'faust/48000-256-1-0.json.pcm.bin',256)
 error=max(abs(v-w) for c,d in zip(x,y) for v,w in zip(c,d));assert error<1e-7
 (BASE/'partition-check.json').write_text(json.dumps({'rate':48000,'blocks':[64,256],'values':sum(map(len,x)),'maximum_error':error},indent=2))
 print('STATIC BLOCK PARTITION PASS',error,flush=True)
