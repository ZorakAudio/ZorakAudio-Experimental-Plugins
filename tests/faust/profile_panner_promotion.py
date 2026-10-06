"""Verify promotion under the established identity against pre-Fast output."""
from pathlib import Path
import array,json,statistics,subprocess
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/faust-sections/panner';RECORDING=r'C:\Users\Louis\OneDrive\Sound Effects\NSFW\Mouth Noises\Files to Cut\Slapping dirty slut BJ.flac'
rows=[]
settings=[(48000,256,c,t) for c in range(9) for t in range(3 if c in [1,2] else 1)]+[(r,b,c,0) for r,b in [(48000,64),(96000,1024)] for c in [0,1,2]]
for rate,block,case,trial in settings:
 data={};audio={};tag=f'{rate}-{block}-{case}-{trial}'
 for label in (['baseline','promoted'] if trial%2==0 else ['promoted','baseline']):
  file=BASE/label/f'promotion-{tag}.json';subprocess.run([str(BASE/label/'host.exe'),RECORDING,str(file),str(rate),str(block),str(case)],check=True)
  data[label]=json.loads(file.read_text());audio[label]=array.array('f');audio[label].frombytes(Path(str(file)+'.pcm.bin').read_bytes());assert data[label]['faust_blocks']==0 and data[label]['faust_scalars']==0
 assert audio['baseline'].tobytes()==audio['promoted'].tobytes(),('Promotion changed output bits',tag)
 assert len(audio['baseline'])==len(audio['promoted']);error=max(abs(a-b) for a,b in zip(audio['baseline'],audio['promoted']));assert error==0,('Promotion changed audio',tag,error)
 rows.append(dict(rate=rate,block=block,scenario=case,trial=trial,values=len(audio['baseline']),maximum_error=error,measurements=data));(BASE/'promotion-results.json').write_text(json.dumps(rows,indent=2));print('NULL PASS',tag,error,flush=True)
print('PASS',len(rows),'pairs;',sum(r['values'] for r in rows),'bit-identical values',flush=True)
