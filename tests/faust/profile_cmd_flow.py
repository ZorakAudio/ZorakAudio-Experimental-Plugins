from pathlib import Path
import subprocess,sys,json,statistics,array,math
root=Path(__file__).resolve().parents[2];base=root/'build/cmd-flow';rows=[]
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
 trials=[];maximum=0;values=0
 for trial in range(3):
  data={};audio={}
  for label in (['original','eel','faust'] if trial%2==0 else ['faust','eel','original']):
   out=base/label/f'{rate}-{block}-1-{trial}.json';subprocess.run([str(base/label/'host.exe'),str(out),str(rate),str(block),'1'],check=True,cwd=root)
   data[label]=json.loads(out.read_text());a=array.array('f');a.frombytes(Path(str(out)+'.bin').read_bytes());audio[label]=a
  assert len(audio['faust'])==len(audio['eel'])
  error=max(abs(x-y) for x,y in zip(audio['faust'],audio['eel']));assert error<1e-6,('equivalent engine mismatch',rate,block,error)
  maximum=max(maximum,error);values+=len(audio['faust']);trials.append(data);print('EQUIVALENCE PASS',rate,block,trial,error,flush=True)
 med={name:statistics.median(t[name]['seconds'] for t in trials) for name in ['original','eel','faust']}
 rows.append(dict(rate=rate,block=block,median_seconds=med,faust_vs_equivalent_eel=med['eel']/med['faust'],combined_redesign_speedup=med['original']/med['faust'],maximum_sample_difference=maximum,compared_values=values,trials=trials))
 (base/'results.json').write_text(json.dumps(rows,indent=2))
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
 for scenario in [0,2]:
  for label in ['faust','eel']:
   out=base/label/f'{rate}-{block}-{scenario}-check.json';subprocess.run([str(base/label/'host.exe'),str(out),str(rate),str(block),str(scenario)],check=True,cwd=root)
print('ALL PASSED',json.dumps(rows),flush=True)
