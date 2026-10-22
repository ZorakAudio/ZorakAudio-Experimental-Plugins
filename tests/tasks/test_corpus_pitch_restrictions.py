from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[2]
here=root/'build/tasks/corpus-pitch-restrictions';here.mkdir(parents=True,exist_ok=True)
for builtin,call in [('sample_pitch_read2','sample_pitch_read2(1,1,0,48000,24,0,0,0,l,r);'),('sample_pitch_read2_ex','sample_pitch_read2_ex(1,1,0,48000,24,0,0,0,1,l,r);')]:
    cases={
        'ordinary':('@init\nt=defer('+call+');','Unsafe or unsupported deferred builtin: '+builtin),
        'arena':('@init\nt=defer_arena(1,0,'+call+');','Unsafe or unsupported deferred builtin: '+builtin),
        'gfx':('@gfx\n'+call,'only valid in'),
        'gfx_helper':('@init\nfunction audio_reader() ('+call+');\n@gfx\naudio_reader();','only valid in'),
        'serialize':('@serialize\n'+call,'only valid in'),
    }
    for legacy in [False,True]:
        for name,(source,error) in cases.items():
            f=here/(builtin+'-'+name+str(legacy)+'.jsfx');f.write_text(source)
            cmd=[sys.executable,str(root/'dsp_jsfx_aot.py'),str(f),'--out-ll',str(f.with_suffix('.ll'))]
            if legacy:cmd+=['--native-gfx-legacy']
            p=subprocess.run(cmd,capture_output=True,text=True)
            assert p.returncode and error in p.stdout+p.stderr,(name,legacy,p.returncode,p.stdout,p.stderr)
            print('PASS spectral reader rejects',builtin,name,'legacy=',legacy)
        f=here/(builtin+'-audio'+str(legacy)+'.jsfx');f.write_text('@init\nfunction audio_reader() ('+call+');\n@sample\naudio_reader();\n@gfx\ngfx_x=0;')
        cmd=[sys.executable,str(root/'dsp_jsfx_aot.py'),str(f),'--out-ll',str(f.with_suffix('.ll'))]
        if legacy:cmd+=['--native-gfx-legacy']
        p=subprocess.run(cmd,capture_output=True,text=True)
        assert p.returncode==0,(legacy,p.stdout,p.stderr)
        print('PASS spectral reader accepts audio',builtin,'legacy=',legacy)
