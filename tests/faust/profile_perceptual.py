from pathlib import Path
import sys,json,subprocess,struct,math
R=Path.cwd();sys.path.insert(0,str(R));import dsp_jsfx_aot as c
B=R/'build/faust-sections/perceptual'
for label in sys.argv[1:] or ['before','cache']:
 p=B/label;p.mkdir(exist_ok=True);ir,m=c.compile_jsfx_to_ir((B/('kernel-'+label+'.jsfx')).read_text(encoding='utf-8'),native_gfx_legacy=True);(p/'JSFXDSP.h').write_text(c._emit_header(m));(p/'dsp.ll').write_text(str(ir));(p/'meta.json').write_text(json.dumps(m));subprocess.run(['clang++','-O2','-c',str(p/'dsp.ll'),'-o',str(p/'dsp.obj')],check=True);subprocess.run(['clang++','-O2','-std=c++20','-UNDEBUG','-I'+str(p),'-I'+str(R/'src'),str(R/'tests/faust/perceptual_kernel.cpp'),str(p/'dsp.obj'),'-o',str(p/'kernel.exe')],check=True)
 for rate,block in [(48000,64),(48000,256),(96000,1024)]:
  for case in range(5):subprocess.run([str(p/'kernel.exe'),str(p/f'{rate}-{block}-{case}.json'),str(rate),str(block),str(case)],check=True)
