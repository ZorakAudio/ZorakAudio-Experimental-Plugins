"""Exercise the packaged helper with developer tool paths removed."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import tempfile


def check(runtime, frontend="python-reference"):
    python = runtime / ("python/python.exe" if os.name == "nt" else "python/bin/python3")
    worker = runtime / "compiler/compiler_worker.py"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PATH"] = ""
    paths = json.loads(subprocess.check_output([str(python), "-I", "-c", "import sys,json;print(json.dumps(sys.path))"], env=env, text=True))
    assert all(Path(p).resolve().is_relative_to(runtime.resolve()) for p in paths), paths
    cases = [
        ("scalar math", "jsfx", "@sample\nspl0=sin(spl0);spl1=cos(spl1);", True),
        ("function and persistent state", "jsfx", "@init\nv=0;\nfunction gain(x) (x*0.5);\n@sample\nv+=1;spl0=gain(spl0);spl1=gain(spl1);", True),
        ("mixed bulk EEL and Faust", "jsfx", "@sample\nspl0*=0.5;spl1*=0.5;\n@faust block\nprocess=spl0*0.5,spl1*0.5;", True),
        ("pure Faust filter", "faust", 'import("stdfaust.lib");process=fi.lowpass(2,1200),fi.lowpass(2,1200);', True),
        ("native visibility", "jsfx", "slider1:g=0.5<0,1,0.01>Gain\n@gfx\nslider_show(slider1,0);", True),
        ("task reduction", "jsfx", "@init\nt=defer_reduce(i,16,SUM,0,i+1);", True),
        ("serialization", "jsfx", "@serialize\nfile_var(0,x);file_mem(0,0,16);file_string(0,#s);", True),
        ("MIDI", "jsfx", "@block\nwhile(midirecv(off,msg,d1,d2))(midisend(off,msg,d1,d2));", True),
        ("legacy Faust sample", "jsfx", "@faust\nprocess=spl0*.5,spl1*.5;", True),
        ("metadata wildcard", "jsfx", "provides: resources/*\n@sample\nspl0*=.5;spl1*=.5;", True),
        ("Faust sample", "jsfx", "@faust sample\nprocess=spl0*.5,spl1*.5;", True),
        ("Faust table", "jsfx", '@faust block\nimport("stdfaust.lib");process=spl0*rdtable(16,1,0),spl1*rdtable(16,1,0);', True),
        ("syntax error", "jsfx", "@sample\nspl0*=;", False),
        ("heap", "jsfx", "@sample\nspl0=0[2];", True),
        ("native clock", "jsfx", "@sample\nspl0=time_precise();", True),
        ("native graphics", "jsfx", "@gfx\ngfx_rect(0,0,1,1);", True),
        ("invalid task reducer", "jsfx", "@init\nt=defer_reduce(i,10,BOGUS,0,i;);", False),
        ("unsupported foreign Faust", "faust", 'x=ffunction(float x(float),"x.h","");process=x;', False),
        ("unsupported Faust mode", "jsfx", "@faust banana\nprocess=_,_;", False),
        ("slider17", "jsfx", "slider17:0.5<0,1,0.01>Gain\n@sample\nspl0*=slider17;", True),
        ("named slider alias", "jsfx", "slider1:g=0.5<0,1,0.01>Gain\n@sample\nspl0*=g;", True),
        ("newline while", "jsfx", "@init\ni=0;while\n(/* comment */ i+=1;i<4;);", True),
        ("semicolon while is not whitespace", "jsfx", "@init\nwhile;(1);", False),
        ("Faust binary metadata", "faust", 'g=hslider("Gain",.5,0,1,.01); b=hslider("Binary",0,0,1,1); n=nentry("Entry",0,0,1,1); e=checkbox("Enable"); t=button("Trigger"); process=*(g+b+n+e+t),*(g+b+n+e+t);', True),
    ]
    results = []
    for name, mode, source, expected in cases:
        with tempfile.TemporaryDirectory(prefix="za-jit-check-") as directory:
            job = Path(directory)
            (job / "request.json").write_text(json.dumps(dict(source=source, mode=mode, frontend=frontend)), encoding="utf-8")
            result = subprocess.run([str(python), "-I", str(worker), str(job)], env=env, text=True, capture_output=True, timeout=45)
            ok = result.returncode == 0
            assert ok == expected, (name, result.stdout, result.stderr)
            if ok:
                descriptor=json.loads((job/"program.json").read_text(encoding="utf-8"))
                assert descriptor["frontend"] == ("faust" if mode=="faust" else frontend), name
                if name == "Faust binary metadata":
                    controls={control['label']:control for control in descriptor['metadata']['editor_sliders']}
                    assert controls['Gain']['step']==.01 and controls['Gain']['type']=='hslider'
                    assert controls['Binary']['step']==1 and controls['Entry']['type']=='nentry'
                    assert controls['Enable']['type']=='checkbox' and controls['Trigger']['type']=='button'
            results.append(dict(name=name, expected=expected, passed=True, diagnostic=result.stderr.strip() if not ok else ""))
    return dict(passed=True, frontend=frontend, isolatedPaths=paths, pathEnvironment="empty", cases=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--frontend", choices=("python-reference","cpp-frontend"),default="python-reference")
    args = parser.parse_args()
    print(json.dumps(check(args.runtime.resolve(),args.frontend), indent=2))
