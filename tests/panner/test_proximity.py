"""Deterministic full-Artistic DSP checks; uses synthetic audio only."""
from pathlib import Path
import array, json, math, subprocess, statistics

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/panner-proximity'

import argparse,sys,hashlib
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reuse',action='store_true',help='Reuse previously compiled test objects')
parser.add_argument('--reference',type=Path,default=ROOT/'tests/faust/reference/3DPanner-before-fast.jsfx')
args=parser.parse_args()
if not args.reuse:
    sys.path.insert(0,str(ROOT));import dsp_jsfx_aot as compiler
    for label,source in [('before',args.reference),('after',ROOT/'plugins/Spatialization/3DPanner/src/3DPanner.jsfx')]:
        folder=BASE/label;folder.mkdir(parents=True,exist_ok=True)
        ir,meta=compiler.compile_jsfx_to_ir(source.read_text(encoding='utf-8'),native_gfx_legacy=True)
        (folder/'JSFXDSP.h').write_text(compiler._emit_header(meta))
        (folder/'dsp.ll').write_text(str(ir))
        (folder/'meta.json').write_text(json.dumps(meta))
        (folder/'source.sha256').write_text(hashlib.sha256(source.read_bytes()).hexdigest())
        subprocess.run(['clang++','-O2','-c',str(folder/'dsp.ll'),'-o',str(folder/'dsp.obj')],check=True)
        subprocess.run(['clang++','-O2','-std=c++20','-UNDEBUG','-I'+str(folder),'-I'+str(ROOT/'src'),str(Path(__file__).parent/'proximity_kernel.cpp'),str(folder/'dsp.obj'),'-o',str(folder/'kernel.exe')],check=True)
        print('COMPILED',label,flush=True)

OUT=BASE/'results';OUT.mkdir(exist_ok=True)
checks=[];cache={}
def run(label='after',fixture=0,mode=0,distance=1,amount=1,anti=.65,level=1,motion=0,room=.3,late=0,trim=0,lateral=0,model=0,rate=48000,block=256,seconds=3,reset=0):
    args=[rate,block,fixture,mode,distance,amount,anti,level,motion,room,late,trim,lateral,model,seconds,reset]
    key=(label,*args)
    if key in cache:return cache[key]
    prefix=OUT/f'{len(cache):03d}-{label}'
    subprocess.run([str(BASE/label/'kernel.exe'),str(prefix),*map(str,args)],check=True)
    samples=array.array('f');samples.frombytes(prefix.with_suffix('.f32').read_bytes())
    data=json.loads(prefix.with_suffix('.json').read_text());n=len(samples)//2
    # Measure settled sound separately from startup and motion.
    end=min(rate,n);tail=samples[n-end:n]+samples[2*n-end:]
    data['settled_rms']=math.sqrt(math.fsum(x*x for x in tail)/len(tail))
    data['settled_dbfs']=20*math.log10(max(1e-30,data['settled_rms']))
    data['arguments']=args;data['file']=str(prefix.with_suffix('.f32'))
    cache[key]=(samples,data);return cache[key]
def check(name,ok,**evidence):
    row=dict(check=name,passed=bool(ok),**evidence);checks.append(row)
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
    print(('PASS ' if ok else 'FAIL ')+name+' '+str(evidence),flush=True)
    assert ok,name

for mode in range(4):
    for fixture in [0,4,8]:
        before,_=run('before',mode=mode,fixture=fixture,amount=0,late=.5 if fixture==0 else 0)
        after,_=run(mode=mode,fixture=fixture,amount=0,late=.5 if fixture==0 else 0)
        check(f'legacy-null-mode{mode}-fixture{fixture}',before.tobytes()==after.tobytes(),values=len(after))
for rate,block in [(44100,64),(96000,1024)]:
    before,_=run('before',amount=0,rate=rate,block=block)
    after,_=run(amount=0,rate=rate,block=block)
    check(f'legacy-null-{rate}-{block}',before.tobytes()==after.tobytes(),values=len(after))

for mode in range(4):
    curve=[run(mode=mode,distance=d)[1]['settled_dbfs'] for d in [0,.25,.5,.75,1]]
    check(f'distance-monotonic-mode{mode}',all(y<=x+.25 for x,y in zip(curve,curve[1:])) and curve[-1]<curve[0]-8,dbfs=curve)
    old=run(mode=mode,amount=0)[1];new=run(mode=mode)[1]
    check(f'far-recedes-mode{mode}',new['settled_dbfs']<old['settled_dbfs']-5,reduction_db=old['settled_dbfs']-new['settled_dbfs'])

for fixture in [0,1,2,3,4,5,7,8,9,10]:
    plain=run(fixture=fixture,mode=2,anti=0)[1]
    adapted=run(fixture=fixture,mode=2,anti=1)[1]
    check(f'anti-salience-attenuation-fixture{fixture}',adapted['settled_rms']<=plain['settled_rms']+1e-7 and adapted['peak']<4,
          reduction_db=plain['settled_dbfs']-adapted['settled_dbfs'],restraint_db=adapted['restraint_db'])
    if fixture in [0,1,10]:check(f'dense-source-restraint-fixture{fixture}',adapted['restraint_db']>1 and adapted['restraint_db']<=12.001,restraint_db=adapted['restraint_db'])

for level in [0,1e-9,.01,.05,1,4]:
    samples,data=run(mode=2,fixture=0,anti=1,level=level)
    check(f'level-stability-{level}',data['peak']<max(1e-20,level*2.0) if level else all(x==0 for x in samples),peak=data['peak'])
    if level<=.01:check(f'quiet-not-raised-{level}',data['restraint_db']<.001,restraint_db=data['restraint_db'])

silent,_=run(fixture=6,late=.8,seconds=5)
check('silence-no-generated-audio',all(x==0 for x in silent))
antiphase,data=run(fixture=8,mode=2,anti=1)
check('antiphase-detected',data['density']>.3 and data['settled_rms']>1e-5,density=data['density'])
center,_=run(fixture=1,room=0,mode=0)
n=len(center)//2
check('center-image-linked',max(abs(x-y) for x,y in zip(center[:n],center[n:]))<1e-7)
for rate,block in [(22050,64),(44100,64),(96000,1024),(192000,1024)]:
    _,data=run(rate=rate,block=block,mode=2,seconds=1)
    check(f'sample-rate-{rate}',0<data['peak']<2,peak=data['peak'])
a,_=run(fixture=1,room=0,block=64)
b,_=run(fixture=1,room=0,block=1024)
check('block-partition-invariant',a.tobytes()==b.tobytes(),maximum_difference=max(abs(x-y) for x,y in zip(a,b)))
transition,data=run(fixture=1,motion=1,seconds=4,room=0,mode=2)
check('distance-and-enable-automation',data['max_step']<.12,max_step=data['max_step'])
tail,data=run(fixture=0,motion=2,seconds=5,late=.5,mode=2)
check('tail-drains-with-proximity',data['settled_rms']<1e-7,tail_rms=data['settled_rms'])
untrimmed,_=run(fixture=0,mode=2)
trimmed,_=run(fixture=0,mode=2,trim=6)
scale=10**(6/20)
error=max(abs(y-x*scale) for x,y in zip(untrimmed,trimmed))
check('output-trim-is-final',error<2e-7,maximum_error=error)
near_off,_=run(fixture=1,distance=0,lateral=.82,room=0,amount=0)
near_on,_=run(fixture=1,distance=0,lateral=.82,room=0,amount=1)
def ild(samples):
    n=len(samples)//2;a=samples[n-48000:n];b=samples[2*n-48000:]
    return 10*math.log10(max(1e-30,math.fsum(x*x for x in b))/max(1e-30,math.fsum(x*x for x in a)))
check('near-ear-contrast',ild(near_on)>ild(near_off)+1,original_db=ild(near_off),proximity_db=ild(near_on))
disabled,disabled_data=run(fixture=1,mode=2,motion=3,seconds=5,room=0)
legacy,_=run('before',fixture=1,mode=2,amount=0,seconds=5,room=0)
n=len(disabled)//2
off_error=max(abs(disabled[ch*n+i]-legacy[ch*n+i]) for ch in range(2) for i in range(n-48000,n))
check('disable-settles-to-exact-legacy',off_error==0,maximum_error=off_error,restraint_db=disabled_data['restraint_db'])
summary=dict(checks=checks,renders=[dict(label=k[0],**v[1]) for k,v in cache.items()])
(OUT/'report.json').write_text(json.dumps(summary,indent=2))
print('ALL PASSED',len(checks),'checks;',len(cache),'renders',flush=True)
