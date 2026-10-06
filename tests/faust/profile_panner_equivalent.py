"""Same Physical expression graph through EEL2 vs FAUST; full plugin timings."""
from pathlib import Path
import array,json,math,statistics,subprocess
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/faust-sections/panner'
RECORDING=r'C:\Users\Louis\OneDrive\Sound Effects\NSFW\Mouth Noises\Files to Cut\Slapping dirty slut BJ.flac'
rows=[]
def run(label,rate,block,case,tag):
 p=BASE/label/f'equivalent-{tag}.json'
 subprocess.run([str(BASE/label/'host.exe'),RECORDING,str(p),str(rate),str(block),str(case)],check=True)
 a=array.array('f');a.frombytes(Path(str(p)+'.pcm.bin').read_bytes());m=json.loads(p.read_text())
 if label=='equivalent':assert m['faust_blocks']==0 and m['faust_scalars']==0
 else:assert m['faust_blocks']==m['callbacks'] and m['faust_scalars']==0 and m['faust_frames']==m['frames']
 return m,a
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
 for case in [1,2]:
  for trial in range(3):
   values={};audio={};tag=f'{rate}-{block}-{case}-{trial}'
   for label in (['equivalent','hybrid'] if trial%2==0 else ['hybrid','equivalent']):values[label],audio[label]=run(label,rate,block,case,tag)
   assert len(audio['equivalent'])==len(audio['hybrid'])
   error=max(abs(x-y) for x,y in zip(audio['equivalent'],audio['hybrid']));rms=math.sqrt(sum((x-y)**2 for x,y in zip(audio['equivalent'],audio['hybrid']))/len(audio['hybrid']));signal=math.sqrt(sum(x*x for x in audio['hybrid'])/len(audio['hybrid']))
   row=dict(rate=rate,block=block,scenario=case,trial=trial,values=len(audio['hybrid']),maximum_error=error,rms_error=rms,relative_error_db=20*math.log10(max(1e-300,rms/signal)),measurements=values);rows.append(row)
   (BASE/'equivalent-results.json').write_text(json.dumps(rows,indent=2))
   assert error<2e-6 and rms<2e-7,('Equivalent DSP mismatch',tag,error,rms)
  print('SUMMARY',rate,block,case,{label:statistics.median(r['measurements'][label]['process_seconds'] for r in rows if r['rate']==rate and r['block']==block and r['scenario']==case) for label in ['equivalent','hybrid']},flush=True)
for case in [0,3,4,5,6,7,8]:
 values={};audio={}
 for label in ['equivalent','hybrid']:values[label],audio[label]=run(label,48000,256,case,f'smoke-{case}')
 error=max(abs(x-y) for x,y in zip(audio['equivalent'],audio['hybrid']));rms=math.sqrt(sum((x-y)**2 for x,y in zip(audio['equivalent'],audio['hybrid']))/len(audio['hybrid']))
 rows.append(dict(rate=48000,block=256,scenario=case,values=len(audio['hybrid']),maximum_error=error,rms_error=rms,measurements=values));(BASE/'equivalent-results.json').write_text(json.dumps(rows,indent=2))
 assert error<2e-6 and rms<2e-7,('Equivalent DSP mismatch',case,error,rms)
print('PASS',len(rows),'same-algorithm pairs; maximum error',max(r['maximum_error'] for r in rows),flush=True)
