from pathlib import Path
import subprocess,sys,json,statistics,array,math
root=Path(__file__).resolve().parents[2];base=root/'build/cmd-flow';rows=[]
if '--checks-only' not in sys.argv:
 for rate,block in [(48000,64),(48000,256),(96000,1024)]:
  trials=[];maximum=0;values=0
  for trial in range(3):
   data={};audio={}
   for label in (['original','original-faust'] if trial%2==0 else ['original-faust','original']):
    out=base/label/f'{rate}-{block}-1-{trial}.json';subprocess.run([str(base/label/'host.exe'),str(out),str(rate),str(block),'1','original'],check=True,cwd=root)
    data[label]=json.loads(out.read_text());a=array.array('f');a.frombytes(Path(str(out)+'.bin').read_bytes());audio[label]=a
   assert len(audio['original-faust'])==len(audio['original'])
   error=max(abs(x-y) for x,y in zip(audio['original-faust'],audio['original']));assert error<1e-6,('equivalent engine mismatch',rate,block,error)
   maximum=max(maximum,error);values+=len(audio['original-faust']);trials.append(data);print('EQUIVALENCE PASS',rate,block,trial,error,flush=True)
  med={name:statistics.median(t[name]['seconds'] for t in trials) for name in ['original','original-faust']}
  rows.append(dict(rate=rate,block=block,median_seconds=med,speedup=med['original']/med['original-faust'],maximum_sample_difference=maximum,compared_values=values,trials=trials))
  (base/'original-port-results.json').write_text(json.dumps(rows,indent=2))
checks=[]
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
 for scenario in [0,2,3]:
  checked={}
  for label in ['original-faust','original']:
   out=base/label/f'{rate}-{block}-{scenario}-check.json';subprocess.run([str(base/label/'host.exe'),str(out),str(rate),str(block),str(scenario),'original'],check=True,cwd=root)
   a=array.array('f');a.frombytes(Path(str(out)+'.bin').read_bytes());checked[label]=a
  err=max(abs(a-b) for a,b in zip(checked['original'],checked['original-faust']));assert err<1e-6,('control/band transition mismatch',rate,block,scenario,err)
  checks.append(dict(rate=rate,block=block,scenario=scenario,maximum_sample_difference=err,values=len(checked['original'])))
  (base/'original-port-checks.json').write_text(json.dumps(checks,indent=2))
  print('CONTROL/BAND NULL PASS',rate,block,scenario,err,flush=True)
print('ALL PASSED',json.dumps(rows),flush=True)
