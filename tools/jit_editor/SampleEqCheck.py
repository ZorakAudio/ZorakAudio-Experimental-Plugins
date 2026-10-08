"""Compare actual Sample display equations in native GFX against WDL EEL."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def check(source, exe, oracle, output):
    text = source.read_text(encoding='utf-8-sig')
    starts = list(re.finditer(r'(?m)^function\s+(\w+)\s*\(', text))
    functions = {m.group(1): text[m.start():starts[i+1].start() if i+1<len(starts) else len(text)] for i,m in enumerate(starts)}
    names = ('clamp','clamp01','lerp','posteq_cut_q_norm','posteq_cut_res_gain_db','posteq_cut_res_resp_db','proc_band_resp','proc_cut_slope_db','proc_eq_resp_db')
    equations = '\n'.join(functions[name] for name in names)
    queries=[]; statements=['srate=48000;LOG10_INV=1/log(10);']
    cases = [
        dict(name='flat', gain=(0,0,0), cutoff=(20,20000), q=(.707,.707), slopes=(0,0)),
        dict(name='peaks', gain=(6,-4,9), cutoff=(20,20000), q=(.707,.707), slopes=(0,0)),
        dict(name='resonant_cuts', gain=(6,-4,9), cutoff=(250,6000), q=(2.8,3.5), slopes=(4,3)),
    ]
    for index,case in enumerate(cases):
        controls={32:case['gain'][0],33:case['gain'][1],34:case['gain'][2],35:120,36:950,37:6500,38:case['cutoff'][0],39:case['cutoff'][1],40:1.15,41:1.15,42:1.15,43:case['slopes'][0],44:case['slopes'][1],45:case['q'][0],46:case['q'][1]}
        statements.extend(f'slider{slot}={value};' for slot,value in controls.items())
        for point in range(129):
            name=f'eq_{index}_{point}';queries.append(name)
            statements.append(f'{name}=proc_eq_resp_db(exp(log(20)+({point}/128)*(log(20000)-log(20))));')
    code=equations+'\n'+'\n'.join(statements)
    output.mkdir(parents=True,exist_ok=True)
    eel=output/'equations.eel';eel.write_text(code,encoding='utf-8')
    jsfx=output/'equations.jsfx';jsfx.write_text('@gfx\n'+code,encoding='utf-8')
    query_file=output/'queries.json';query_file.write_text(json.dumps(queries),encoding='utf-8')
    reference=subprocess.check_output([str(oracle),str(eel),*queries],text=True)
    expected={k:float(v) for k,v in (line.split('=',1) for line in reference.splitlines())}
    rows=[]
    for frontend in ('python-reference','cpp-frontend'):
        command=[str(exe),'--graphics-values',str(jsfx),str(query_file)]
        if frontend=='cpp-frontend':command.append('--cpp-frontend')
        actual=json.loads(subprocess.check_output(command,text=True))
        assert actual.keys()==expected.keys()
        maximum=max(abs(actual[name]-expected[name]) for name in queries)
        assert maximum<1e-9, (frontend,maximum)
        rows.append(dict(frontend=frontend,points=len(queries),maximumDbError=maximum,passed=True))
    result=dict(source=str(source),sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference='WDL EEL',cases=cases,rows=rows,passed=True)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--oracle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();check(*(p.resolve() for p in (args.source,args.exe,args.oracle,args.out)))
