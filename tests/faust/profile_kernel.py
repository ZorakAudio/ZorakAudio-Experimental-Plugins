from pathlib import Path
import sys,subprocess,json,struct,math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));import dsp_jsfx_aot as c
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,default=ROOT/'plugins/Dynamics/EasyExpander/src/EasyExpander.jsfx');parser.add_argument('--out',type=Path,default=ROOT/'build/faust-sections/kernel');args=parser.parse_args()
paths={'before':args.baseline,'after':ROOT/'plugins/Dynamics/EasyExpanderFaust/src/EasyExpanderFaust.jsfx'}
for name,path in paths.items():
 out=args.out/name;out.mkdir(parents=True,exist_ok=True)
 m,meta=c.compile_jsfx_to_ir(path.read_text(encoding='utf-8'),native_gfx_legacy=True);(out/'JSFXDSP.h').write_text(c._emit_header(meta));(out/'dsp.ll').write_text(str(m));(out/'meta.json').write_text(json.dumps(meta,indent=2))
 subprocess.run(['clang++','-O2','-c',str(out/'dsp.ll'),'-o',str(out/'dsp.obj')],check=True)
 subprocess.run(['clang++','-std=c++20','-O2','-UNDEBUG','-I'+str(out),'-I'+str(ROOT/'src'),str(ROOT/'tests/faust/easy_kernel_profile.cpp'),str(out/'dsp.obj'),'-o',str(out/'profile.exe')],check=True)
 for rate,block in [(48000,64),(48000,256),(96000,1024)]:subprocess.run([str(out/'profile.exe'),str(out/f'{rate}-{block}.json'),str(block),str(rate)],check=True)
rows=[]
for rate,block in [(48000,64),(48000,256),(96000,1024)]:
 stem=f'{rate}-{block}.json';a=args.out/'before'/stem;b=args.out/'after'/stem
 before=json.loads(a.read_text());after=json.loads(b.read_text());x=struct.unpack('<'+'f'*(a.with_suffix('.json.bin').stat().st_size//4),a.with_suffix('.json.bin').read_bytes());y=struct.unpack('<'+'f'*len(x),b.with_suffix('.json.bin').read_bytes());diff=[abs(p-q) for p,q in zip(x,y)];error=max(diff);print('MAX ERROR',error,'RMS',math.sqrt(sum(d*d for d in diff)/len(diff)));assert error<2e-6
 rows.append({'before':before,'after':after,'speedup':before['median_seconds']/after['median_seconds'],'maximum_sample_error':error,'rms_sample_error':math.sqrt(sum(d*d for d in diff)/len(diff))})
(args.out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
