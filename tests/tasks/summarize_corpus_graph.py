from pathlib import Path
import csv,json,hashlib
root=Path.cwd();folder=root/'build/tasks/corpus-profile';settings=json.loads((folder/'run-settings.json').read_text());equivalence=json.loads((folder/'model-equivalence.json').read_text())
def groups(name):
 rows=list(csv.DictReader((folder/name).open()));out={}
 for r in rows:
  key=r['stage'].split('/')[0];g=out.setdefault(key,{'wall_s':0.,'process_cpu_s':0.,'deadline_overruns':0.,'max_ms':0.})
  for k in ['wall_s','process_cpu_s','deadline_overruns']:
   if r.get(k):g[k]+=float(r[k])
  if r.get('max_ms'):g['max_ms']=max(g['max_ms'],float(r['max_ms']))
 return out
runs={'original_paced':groups('before.csv'),'recurrence_paced':groups('after.csv'),'graph_paced':groups('graph-paced.csv'),'original_unpaced':groups('before-unpaced.csv'),'graph_unpaced':groups('graph-unpaced.csv')}
comparison=[]
for baseline,graph in [('original_paced','graph_paced'),('recurrence_paced','graph_paced'),('original_unpaced','graph_unpaced')]:
 a=runs[baseline]['TOTAL']['wall_s'];b=runs[graph]['TOTAL']['wall_s'];comparison.append({'baseline':baseline,'graph':graph,'before_s':a,'after_s':b,'saved_s':a-b,'saved_percent':100*(a-b)/a,'speedup':a/b})
with Path(settings['input']).open('rb') as f:unchanged=hashlib.file_digest(f,'sha256').hexdigest()==settings['sha256']
result={'settings':settings,'runs':runs,'comparison':comparison,'model_equivalence':equivalence,'model_cells_checked':sum(x['cells'] for x in equivalence),'all_tested_cells_bit_exact':all(x['different']==0 for x in equivalence),'input_unchanged':unchanged,'checks':['AOT/runtime compiler and arena checks in standard and Legacy modes','Immutable generation binding, pending-request invalidation and reclamation','512-region matrix, restart, 12 rebuilds and exhausted-buffer fallback','Supplied FLAC reload during PE and during publication; latest request completes once without stale model publication','MIDI playback and editor lifecycle; no memory faults or sample-read errors','Native Legacy build-policy tests'], 'notes':['Single runs, not repeated-trial means. Same defaults, sample rate, block size and native Legacy mode; no editor during timing.','Only the user-supplied FLAC was loaded. Other tests used numeric data without audio loading.','The graph contains four dependent single-writer arena stages and uses a two-worker instance scheduler. It does not parallelize every loop iteration.','Initial indexing and bounded heap transfers still run in processBlock. Worker analysis itself runs independently of host audio pacing.','Arena adds at most 192 MiB of private analysis storage per active instance.','Callbacks use a normal-priority standalone host harness; durations include OS preemption and are not a DAW deadline certification.','Original/recurrence timing records are from the preceding profiling pass.']}
(folder/'background-analysis-results.json').write_text(json.dumps(result,indent=2))
with (folder/'background-analysis-results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=comparison[0].keys());w.writeheader();w.writerows(comparison)
print(json.dumps({'comparison':comparison,'cells':result['model_cells_checked'],'exact':result['all_tested_cells_bit_exact'],'input_unchanged':unchanged,'graph_overruns':sum(g['deadline_overruns'] for k,g in runs['graph_paced'].items() if k!='TOTAL')},indent=2))
outputs=Path(r'C:\Users\LouisJenkinsCS\Documents\Codex\2026-10-05\can-x20\outputs');outputs.mkdir(parents=True,exist_ok=True)
for suffix in ['json','csv']:(outputs/('Corpus-background-analysis-results.'+suffix)).write_bytes((folder/('background-analysis-results.'+suffix)).read_bytes())
