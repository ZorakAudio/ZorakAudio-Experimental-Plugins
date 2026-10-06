"""Qualify hybrid Artistic preservation and full-plugin Physical performance."""
from pathlib import Path
import array,json,statistics,subprocess
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/faust-sections/panner'
RECORDING=r'C:\Users\Louis\OneDrive\Sound Effects\NSFW\Mouth Noises\Files to Cut\Slapping dirty slut BJ.flac'
rows=[]
def run(label,rate,block,scenario,tag,controls):
 f=BASE/label/f'hybrid-{tag}.json'
 cmd=[str(BASE/label/'host.exe'),RECORDING,str(f),str(rate),str(block),str(scenario)]
 for slider,value in controls.items():cmd += [str(slider),str(value)]
 subprocess.run(cmd,check=True)
 result=json.loads(f.read_text());audio=array.array('f');audio.frombytes(Path(str(f)+'.pcm.bin').read_bytes())
 assert result['faust_scalars']==0
 if label=='hybrid':assert result['faust_blocks']==result['callbacks'] and result['faust_frames']==result['frames']
 return result,audio
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
 for case in [0,1,2]:
  for trial in range(3):
   values={};audio={};tag=f'{rate}-{block}-{case}-{trial}'
   for label in (['baseline','hybrid'] if trial%2==0 else ['hybrid','baseline']):values[label],audio[label]=run(label,rate,block,case,tag,{})
   error=max(abs(x-y) for x,y in zip(audio['baseline'],audio['hybrid']))
   if case==0:assert error==0,('Artistic mismatch',tag,error)
   rows.append(dict(rate=rate,block=block,scenario=case,trial=trial,controls={},compared_values=len(audio['baseline']),maximum_difference=error,measurements=values))
  print('SUMMARY',rate,block,case,{label:statistics.median(r['measurements'][label]['process_seconds'] for r in rows if r['rate']==rate and r['block']==block and r['scenario']==case) for label in ['baseline','hybrid']},flush=True)
# Original Artistic output across source modes, wet late field, and moving controls.
for i,(case,controls) in enumerate([(2,{36:0,27:.8}),*( (c,{36:0,27:.8}) for c in [3,4,5])]):
 values={};audio={}
 for label in ['baseline','hybrid']:values[label],audio[label]=run(label,48000,256,case,f'artistic-{i}',controls)
 error=max(abs(x-y) for x,y in zip(audio['baseline'],audio['hybrid']))
 assert error==0,('Wet/automated Artistic mismatch',case,error)
 rows.append(dict(rate=48000,block=256,scenario=case,controls=controls,compared_values=len(audio['baseline']),maximum_difference=error,measurements=values))
for case in range(3,9):
 value,audio=run('hybrid',48000,256,case,f'physical-{case}',{})
 rows.append(dict(rate=48000,block=256,scenario=case,controls={},measurements={'hybrid':value}))
(BASE/'hybrid-results.json').write_text(json.dumps(rows,indent=2))
print('PASS',len(rows),'workloads',flush=True)
