from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[2]
for label,source in [('faust','tests/faust/experiments/CMDFlow/src/CMDFlow.jsfx'),('eel','tests/faust/fixtures/CMDFlowEquivalent.jsfx'),('original','plugins/Spectral/CMD/src/CrossMixDeclutter.jsfx'),('original-faust','plugins/Spectral/CMD/src/CrossMixDeclutterFaust.jsfx')]:
 if len(sys.argv)>1 and label not in sys.argv[1:]:continue
 fixture=root/'build/cmd-flow/inputs'/f'{label}.jsfx';fixture.parent.mkdir(parents=True,exist_ok=True)
 text=(root/source).read_text();text=text.replace('\n@init\n','\nslider12:1<1,2,1>-Test Identity\n\n@init\n',1).replace('iid = instance_id();','iid = slider12;').replace('\n@block\n','\n@block\niid=slider12;\n',1);fixture.write_text(text,newline='\n')
 subprocess.run([sys.executable,str(root/'tests/faust/build_example.py'),'--cmd-flow','--source',str(fixture),'--reports',str(root/'build/cmd-flow'/label)],check=True,cwd=root)
 print('READY',label,flush=True)
