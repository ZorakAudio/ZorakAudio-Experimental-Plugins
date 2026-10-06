"""Generate an EEL2 reference of exactly the hybrid Physical expression graph.
Uses Faust's scalar C++ lowering solely as a checked translation intermediate.
The resulting DSP executes through the ordinary EEL2 AOT/runtime, without Faust.
"""
from pathlib import Path
import json,re,sys,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from jsfx_faust_compiler import injections_for,split
BASE=ROOT/'build/faust-sections/panner';OUT=BASE/'equivalent';OUT.mkdir(exist_ok=True)
hybrid=(ROOT/'plugins/Spatialization/HyperrealHybrid/src/HyperrealHybrid.jsfx').read_text()
meta=json.loads((BASE/'hybrid/JSFXDSP_meta.json').read_text());stage=next(x for x in meta['faust_stages'] if x['kind']=='faust')
source=next(x['source'] for x in split(hybrid)['stages'] if x['kind']=='faust')
signals=stage['signals'];controls=[n for n in stage['imports'] if n not in signals]
injections=[injections_for(controls)]+[f'{n}=__za_signal_{i};' for i,n in enumerate(signals)]
params=','.join(f'__za_signal_{i}' for i in range(len(signals)))
code=f'process({params})=__za_env.process with {{ __za_env=environment {{\n'+source+'\n'+'\n'.join(injections)+'\n}; };\n'
(OUT/'physical.dsp').write_text(code)
subprocess.run(['faust','-lang','cpp','-double','-mdd','2147483647','-cn','EquivalentPhysical','-o',str(OUT/'physical.cpp'),str(OUT/'physical.dsp')],check=True)
cpp=(OUT/'physical.cpp').read_text()
private=cpp.split('class EquivalentPhysical : public dsp {',1)[1].split(' public:',1)[0]
arrays=re.findall(r'\b(?:int|double)\s+(\w+)\[(\d+)\];',private)
zones=dict((member,var) for var,member in re.findall(r'addHorizontalSlider\("__za_in_(\w+)", &(\w+)',cpp))
def body(signature):
 at=cpp.index(signature);begin=cpp.index('{',at)+1;depth=1;end=begin
 while depth:
  depth += (cpp[end]=='{')-(cpp[end]=='}');end+=1
 return cpp[begin:end-1]
def eel(text):
 for name,size in arrays:
  if int(size)==2:
   for index in [0,1]:text=text.replace(f'{name}[{index}]',f'eq_{name}_{index}')
 text=re.sub(r'\b(?:double|int) (\w+) =',r'\1 =',text)
 text=re.sub(r'std::(min|max)<(?:double|int)>',r'\1',text)
 for old,new in [('std::fabs','abs'),('std::pow','pow'),('std::',''),('double(','('),('FAUSTFLOAT(','('),('int(','floor(')]:text=text.replace(old,new)
 for member,var in zones.items():text=re.sub(r'\b'+member+r'\b',var,text)
 # Prefix all generated state, temporaries, and constants.
 text=re.sub(r'\b(f(?:Const|Rec|Vec|Slow|Temp)\d+|i(?:Vec|Temp)\d+|IOTA0|fSampleRate|ftbl0EquivalentPhysicalSIG0)\b',r'eq_\1',text)
 if any(x in text for x in ['std::','{','}','RESTRICT','FAUSTFLOAT','double','int(']):raise ValueError('Untranslated C++ construct')
 return text.strip()
init='// Double-precision reference; exact scalar graph, state and interpolation.\neq_start=audio_memory_end;\neq_next=eq_start;\n'
for name,size in arrays:
 if int(size)==2:init+=f'eq_{name}_0=0;eq_{name}_1=0;\n'
 else:init+=f'eq_{name}=eq_next;eq_next+={size};\n'
init+='eq_ftbl0EquivalentPhysicalSIG0=eq_next;eq_next+=65536;\nmemset(eq_start,0,eq_next-eq_start);audio_memory_end=eq_next;\n'
init+=eel(body('virtual void instanceConstants(').replace('sample_rate','srate'))+'\n'
init+='eq_table_index=0;loop(65536,eq_ftbl0EquivalentPhysicalSIG0[eq_table_index]=sin(9.587379924285257e-05*eq_table_index);eq_table_index+=1;);\neq_IOTA0=0;\n'
compute=body('virtual void compute(')
compute=re.sub(r'FAUSTFLOAT\* (?:input|output)\d+ = (?:inputs|outputs)\[\d+\];','',compute)
loop='for (int i0 = 0; i0 < count; i0 = i0 + 1) {'
slow,sample=compute.split(loop,1);sample=sample.rsplit('}',1)[0]
for i,var in enumerate(signals):sample=sample.replace(f'input{i}[i0]',f'eq_input{i}')
sample=sample.replace('output0[i0]','eq_output0').replace('output1[i0]','eq_output1')
sample='\n'.join(f'eq_input{i}={var};' for i,var in enumerate(signals))+'\n'+eel(sample)+'\nspl0=eq_output0;spl1=eq_output1;\n'
# Multiple ordinary EEL sections are combined by the native compiler: controls
# run once per block, then original Artistic and equivalent Physical per sample.
a=hybrid.index('@slider');hybrid=hybrid[:a]+init+'\n'+hybrid[a:]
a=hybrid.index('@faust block');b=hybrid.index('@gfx',a)
equivalent=hybrid[:a]+'@block\n'+eel(slow)+'\n@sample\n'+sample+'\n'+hybrid[b:]
equivalent=equivalent.replace(hybrid.splitlines()[0],'desc:Hyperreal Equivalent EEL Physical (Benchmark Fixture)',1)
# Ordinary JSFX has one @block and @sample; combine the reference stages.
headers=list(re.finditer(r'(?m)^@(init|slider|block|sample|gfx)\b[^\n]*',equivalent))
sections={};order=[]
for i,h in enumerate(headers):
 name=h[1]
 if name not in sections:sections[name]=[h[0],[]];order.append(name)
 sections[name][1].append(equivalent[h.end():headers[i+1].start() if i+1<len(headers) else len(equivalent)])
equivalent=equivalent[:headers[0].start()]+'\n'.join(sections[k][0]+'\n'+'\n'.join(sections[k][1]) for k in order)
assert '@faust' not in equivalent
(ROOT/'tests/faust/equivalent_physical.jsfx').write_text(equivalent,encoding='utf-8')
(OUT/'translation.json').write_text(json.dumps({'source_sha256':stage['source_sha256'],'signals':signals,'controls':controls,'state_arrays':arrays,'precision':'double','memory_slots':sum(int(n) for _,n in arrays if int(n)!=2)+65536,'scalar_history_slots':sum(2 for _,n in arrays if int(n)==2),'note':'Same scalar expression graph lowered to ordinary EEL2; not a hand-written algorithm change.'},indent=2))
print('Equivalent EEL source generated',flush=True)
