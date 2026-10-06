"""Summarize the paired benchmark logs without loading audio."""
import csv,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'build/tasks/corpus-profile'
settings=json.loads((folder/'run-settings.json').read_text())
result={'settings':settings,'runs':{},'comparison':[]}
for suffix in ['', '-unpaced']:
    runs={}
    for mode in ['before','after']:
        rows=list(csv.DictReader((folder/(mode+suffix+'.csv')).open()))
        groups={}
        for row in rows:
            name=row['stage'].split('/')[0]
            if name=='TOTAL':
                groups[name]={k:float(row[k]) for k in ['wall_s','audio_clock_s','process_cpu_s']}
                continue
            g=groups.setdefault(name,{k:0. for k in ['wall_s','audio_clock_s','process_cpu_s','callback_time_s','callbacks','deadline_overruns']})
            for k in g:g[k]+=float(row.get('callback_time_s',row.get('callback_cpu_s','0')) if k=='callback_time_s' else row[k])
        groups['Recurrence pairs']={k:float(next(row for row in rows if row['stage']=='Features/4')[k]) for k in ['wall_s','audio_clock_s','process_cpu_s']}
        runs[mode]=groups
        result['runs'][mode+suffix]={'groups':groups,'stages':rows,'log':(folder/(mode+suffix+'-run.log')).read_text()}
    for name in ['TOTAL','Decode','Index','Features','Recurrence pairs','Structure','Grammar','Map','Focus','PE','Prune']:
        a,b=runs['before'][name],runs['after'][name]
        result['comparison'].append({'mode':'paced' if not suffix else 'unpaced','stage':name,
            'before_s':a['wall_s'],'after_s':b['wall_s'],'saved_s':a['wall_s']-b['wall_s'],
            'saved_percent':100*(a['wall_s']-b['wall_s'])/a['wall_s'],
            'before_process_cpu_s':a['process_cpu_s'],'after_process_cpu_s':b['process_cpu_s']})
with Path(settings['input']).open('rb') as stream:
    result['input_unchanged']=hashlib.file_digest(stream,'sha256').hexdigest()==settings['sha256']
result['notes']=[
    'One paced pair and one unpaced pair; not repeated-trial means.',
    'Both builds use native Legacy graphics, no editor, default script settings, 48 kHz / 256 samples.',
    'Recurrence pairs is a subset of Features, not an additional stage.',
    'Per-stage wall times omit small profiler bookkeeping costs; TOTAL is measured end-to-end.',
    'The raw callback_cpu_s column is summed callback wall duration, not thread CPU time.',
    'Process CPU time includes all instance threads; unpaced audio-clock time is not observed elapsed time.',
    'Callback durations include scheduling/preemption; this normal-priority harness is not a DAW real-time-priority thread.'
]
(folder/'comparison.json').write_text(json.dumps(result,indent=2))
with (folder/'comparison.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(result['comparison'][0]));writer.writeheader();writer.writerows(result['comparison'])
print(json.dumps(result['comparison'][:6],indent=2))
