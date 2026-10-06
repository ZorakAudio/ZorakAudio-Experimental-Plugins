from pathlib import Path
import json,struct,math,sys
ROOT=Path(__file__).resolve().parents[2]
for key in sys.argv[1:]:
    p=ROOT/'build/catalog-faust-audit'/key/'host';total=different=0;maximum=rms=0
    with (p/'before.json.pcm.bin').open('rb') as a,(p/'after.json.pcm.bin').open('rb') as b:
        while x:=a.read(1048576):
            y=b.read(len(x))
            if len(y)!=len(x):raise RuntimeError('Length mismatch')
            xx=struct.unpack('<'+'f'*(len(x)//4),x);yy=struct.unpack('<'+'f'*len(xx),y)
            for l,r in zip(xx,yy):
                if not math.isfinite(l) or not math.isfinite(r):raise RuntimeError('Non-finite output')
                e=abs(l-r);maximum=max(maximum,e);rms+=e*e;different+=l!=r
            total+=len(xx)
        if b.read(1):raise RuntimeError('Length mismatch')
    before=json.loads((p/'before.json').read_text());after=json.loads((p/'after.json').read_text())
    result={'before':before,'after':after,'speedup':before['processBlock_seconds']/after['processBlock_seconds'],'samples':total,'different':different,'maximum_error':maximum,'rms_error':math.sqrt(rms/total)}
    (p/'comparison.json').write_text(json.dumps(result,indent=2));print(key,json.dumps(result),flush=True)
    if maximum>2e-6:raise RuntimeError('Migration failed the declared output-error limit')
