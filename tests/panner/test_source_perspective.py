"""Synthetic regression for Artistic source directivity and internal head/body POV."""
from pathlib import Path
import array,hashlib,json,math,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/panner-source-perspective'
sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as compiler
for label,source in [('before',ROOT/'tests/panner/reference/3DPanner-before-source-perspective.jsfx'),('after',ROOT/'plugins/Spatialization/3DPanner/src/3DPanner.jsfx')]:
    folder=BASE/label;folder.mkdir(parents=True,exist_ok=True)
    ir,meta=compiler.compile_jsfx_to_ir(source.read_text(encoding='utf-8'),native_gfx_legacy=True)
    (folder/'JSFXDSP.h').write_text(compiler._emit_header(meta))
    (folder/'dsp.ll').write_text(str(ir))
    (folder/'source.sha256').write_text(hashlib.sha256(source.read_bytes()).hexdigest())
    subprocess.run(['clang++','-O2','-c',str(folder/'dsp.ll'),'-o',str(folder/'dsp.obj')],check=True)
    subprocess.run(['clang++','-O2','-std=c++20','-UNDEBUG','-I'+str(folder),'-I'+str(ROOT/'src'),str(ROOT/'tests/panner/proximity_kernel.cpp'),str(folder/'dsp.obj'),'-o',str(folder/'kernel.exe')],check=True)
    print('COMPILED',label,flush=True)

OUT=BASE/'results';OUT.mkdir(exist_ok=True);cache={};checks=[]
def run(label='after',fixture=0,mode=0,distance=.25,proximity=.75,anti=.65,level=1,motion=0,room=.3,late=0,trim=0,lateral=0,model=0,rate=48000,block=256,seconds=2,facing=0,beam=0,internal=0,body=.5):
    args=[rate,block,fixture,mode,distance,proximity,anti,level,motion,room,late,trim,lateral,model,seconds,0,facing,beam,internal,body]
    key=(label,*args)
    if key in cache:return cache[key]
    prefix=OUT/f'{len(cache):03d}-{label}'
    subprocess.run([str(BASE/label/'kernel.exe'),str(prefix),*map(str,args)],check=True)
    samples=array.array('f');samples.frombytes(prefix.with_suffix('.f32').read_bytes())
    data=json.loads(prefix.with_suffix('.json').read_text());n=len(samples)//2
    tail=samples[n-rate//2:n]+samples[2*n-rate//2:]
    data['settled_rms']=math.sqrt(math.fsum(x*x for x in tail)/len(tail));data['settled_dbfs']=20*math.log10(max(1e-30,data['settled_rms']))
    data['arguments']=args;cache[key]=(samples,data);return cache[key]
def check(name,ok,**evidence):
    checks.append(dict(check=name,passed=bool(ok),**evidence))
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
    print(('PASS ' if ok else 'FAIL ')+name+' '+str(evidence),flush=True)
    assert ok,name
def same(a,b):return a.tobytes()==b.tobytes()

for mode in range(4):
    for distance in [0,.25,1]:
        for proximity in [0,.75]:
            args=dict(mode=mode,distance=distance,proximity=proximity,fixture=7,late=.4)
            a,_=run('before',**args);b,_=run(**args,facing=147,body=1)
            check(f'neutral-exact-v722-mode{mode}-distance{distance}-proximity{proximity}',same(a,b))

for mode in range(4):
    rows={}
    for fixture in [1,2,3]:
        args=dict(mode=mode,fixture=fixture,room=0,proximity=0,anti=0)
        front,_=run(**args);enabled,fd=run(**args,beam=1)
        side,sd=run(**args,beam=1,facing=90);back,bd=run(**args,beam=1,facing=180)
        check(f'on-axis-unity-mode{mode}-tone{fixture}',same(front,enabled))
        check(f'turn-monotonic-mode{mode}-tone{fixture}',fd['settled_rms']>=sd['settled_rms']>=bd['settled_rms'])
        rows[fixture]=fd['settled_dbfs']-bd['settled_dbfs']
    check(f'frequency-dependent-beaming-mode{mode}',rows[3]>rows[1]+5,attenuation_db=rows)

_,front=run(beam=1);_,back=run(beam=1,facing=180)
check('reflection-departure-separate-from-direct',front['beam_direct_high']>.999 and front['beam_front_high']<.14 and back['beam_direct_high']<.14 and back['beam_front_high']>.999,front=front['beam_front_high'],turned_away=back['beam_front_high'])
a,_=run(beam=1,facing=180,room=0);b,_=run(beam=1,facing=-180,room=0)
check('facing-wrap-equal-direct',same(a,b))

for mode in range(4):
    a,_=run(internal=1,mode=mode,fixture=7)
    b,_=run(internal=1,mode=mode,fixture=7,distance=1,lateral=.9,room=1,late=.8,proximity=1,anti=1,beam=1,facing=180)
    check(f'internal-independent-of-external-cues-mode{mode}',same(a,b))

clean,cd=run(internal=1,body=0,fixture=3);dark,dd=run(internal=1,body=1,fixture=3)
check('body-tone-rolls-off-highs',cd['settled_dbfs']-dd['settled_dbfs']>10,reduction_db=cd['settled_dbfs']-dd['settled_dbfs'])
mono,_=run(internal=1,fixture=7,lateral=1);n=len(mono)//2
check('internal-mono-head-centered',mono[:n].tobytes()==mono[n:].tobytes())
stereo,sd=run(internal=1,body=1,mode=2,fixture=8)
check('internal-antiphase-retained',sd['settled_rms']>.01,rms=sd['settled_rms'])
silent,_=run(internal=1,beam=1,fixture=6,late=.8)
check('no-generated-sound',all(x==0 for x in silent))
for rate,block in [(22050,64),(44100,64),(96000,1024),(192000,1024)]:
    _,data=run(beam=1,facing=140,internal=.6,mode=2,rate=rate,block=block,seconds=1)
    check(f'rate-stability-{rate}',0<data['peak']<2,peak=data['peak'])

a,_=run(internal=.7,beam=1,facing=120,mode=2,block=64)
b,_=run(internal=.7,beam=1,facing=120,mode=2,block=1024)
check('partition-invariant',same(a,b))
base,_=run(internal=1,mode=2);trim,_=run(internal=1,mode=2,trim=6)
error=max(abs(y-x*10**(.3)) for x,y in zip(base,trim))
relative=max(abs(y-x*10**(.3))/max(1,abs(y)) for x,y in zip(base,trim))
check('internal-final-trim',relative<1.5e-7,maximum_error=error,normalized_error=relative)
_,data=run(beam=1,motion=4,fixture=1,seconds=4)
check('facing-wrap-and-internal-automation',data['max_step']<.12,max_step=data['max_step'])
tail,data=run(internal=1,motion=2,late=.8,seconds=3)
check('internal-no-external-room-tail',data['settled_rms']<1e-8,tail_rms=data['settled_rms'])
a,_=run('before',model=1,mode=2,seconds=1,late=.4)
b,_=run(model=1,mode=2,seconds=1,late=.4,beam=1,facing=180,internal=1,body=1)
check('physical-unchanged',same(a,b))

summary=dict(checks=checks,renders=[dict(label=k[0],**v[1]) for k,v in cache.items()])
(OUT/'report.json').write_text(json.dumps(summary,indent=2))
print('ALL PASSED',len(checks),'checks;',len(cache),'renders',flush=True)
