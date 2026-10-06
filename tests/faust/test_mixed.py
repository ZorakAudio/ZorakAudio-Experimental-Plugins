from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));import dsp_jsfx_aot as c
sources={
 'conditional_block':(14,'@init\ngate=0;\n@faust block when gate\nprocess=(_:(+~_)),_;\n'),
 'block_quantum':(13,'options:za_faust_quantum=8\n@init\ngain=0.5;blocks=0;boundaries=0;\n@block\nblocks+=1;\n@block\nboundaries+=1;\n@sample\ngain+=0.01;\n@block\nmarker=0;\n@faust block\nprocess=spl0*gain,spl1*gain;\n'),
 'block_stream':(12,'@init\ngain=0.5;\n@sample\ngain+=0.01;\n@block\ngain=100;\n@faust block\nprocess=spl0*gain,spl1*gain;\n'),
 'block_helper_stream':(12,'@init\ngain=0.5;function advance()(gain+=0.01);\n@sample\nadvance();\n@block\ngain=100;\n@faust block\nprocess=spl0*gain,spl1*gain;\n'),
 'controls':(0,'@init\ngain=0.5;\n@faust\nprocess=_,_:*(gain),*(gain);\n'),
 'ordered':(1,'@init\ngain=0.5;\n@faust\nprocess=_,_:*(gain),*(gain);\n@block\ngain=0.25;\n@faust\nprocess=_,_:*(gain),*(gain);\n'),
 'fused':(2,'@init\ngain=0.5;\n@sample\ngain+=0.01;\n@faust\nprocess=_,_:*(gain),*(gain);\n'),
 'export':(3,'@init\ngain=0.5;meter=0;\n@faust\nmeter=abs(spl0);process=spl0*gain,spl1*gain;\n'),
 'sample_block':(4,'@init\ngain=1;\n@block\ngain=0.5;\n@sample\nspl0*=gain;spl1*=gain;\n@faust\nprocess=_,_:*(2),*(2);\n'),
 'state':(5,'@init\ngain=0.5;\n@faust\nprocess=(_:(+~_)),_;\n'),
 'output_sample':(6,'@init\nmeter=0;\n@faust\nmeter=abs(spl0);process=spl0,spl1;\n@sample\nspl0=meter;\n'),
 'faust_chain':(7,'@init\nmeter=0;\n@faust\nmeter=abs(spl0);process=spl0,spl1;\n@faust\nprocess=_,_:*(meter),*(meter);\n'),
 'slider_alias':(8,'slider1:level=0.5<0,1,0.1>Level\n@init\ngain=0.5;\n@sample\nlevel+=0.01;\n@faust\nprocess=_,_:*(slider1),*(slider1);\n'),
 'slider_output':(9,'slider1:level=0.5<0,1,0.1>Level\n@init\nlevel=0.5;\n@faust\nlevel=0.25;process=_,_;\n'),
 'oscillator_table':(10,'@init\ngain=0.5;\n@faust\nimport("stdfaust.lib");process=os.osc(440),os.osc(440);\n'),
 'rate_table':(11,'@init\ngain=0.5;\n@faust\nimport("stdfaust.lib");process=rdtable(16,float(ma.SR),int(_)%16),_;\n'),
 'namespace_input':(0,'@init\nparams.gain=0.5;\n@faust\nprocess=_,_:*(params.gain),*(params.gain);\n'),
 'namespace_output':(3,'@init\ngain=0.5;meter=0;params.meter=0;\n@faust\nparams=environment{meter=abs(spl0);};meter=params.meter;process=spl0*gain,spl1*gain;\n'),
 'case_input':(0,'@init\nGain=0.5;\n@faust\nprocess=_,_:*(Gain),*(Gain);\n'),
}
for name,(kind,text) in sources.items():
 out=ROOT/'build/faust-sections/tests'/name;out.mkdir(parents=True,exist_ok=True)
 module,meta=c.compile_jsfx_to_ir(text);(out/'mixed.ll').write_text(str(module));(out/'JSFXDSP.h').write_text(c._emit_header(meta));(out/'meta.json').write_text(json.dumps(meta,indent=2))
 subprocess.run(['clang++','-O2','-c',str(out/'mixed.ll'),'-o',str(out/'mixed.obj')],check=True)
 subprocess.run(['clang++','-std=c++17','-O2','-UNDEBUG','-DTEST_KIND='+str(kind),'-I'+str(out),'-I'+str(ROOT/'src'),str(ROOT/'tests/faust/mixed_runtime.cpp'),str(out/'mixed.obj'),'-o',str(out/'test.exe')],check=True)
 subprocess.run([str(out/'test.exe')],check=True)
print('All mixed fixtures passed')
