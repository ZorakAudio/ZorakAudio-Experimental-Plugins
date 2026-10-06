"""Build the task fixture with the same AOT/JUCE/VST3/CLAP path as plugins."""
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import build

out=ROOT/'build/tasks/plugin-ninja'
out.mkdir(parents=True,exist_ok=True)
build.build_jsfx_aot(ROOT,out,'TaskGraphProbe',ROOT/'tests/tasks/processor.jsfx',native_gfx_legacy=True)
build.write_plugin_readme_header(out,ROOT/'tests/tasks/README.md')
args=['cmake','-S',str(ROOT/'cmake/plugin'),'-B',str(out)]
if os.name=='nt':
    vs=build.find_vs_installation_path()
    # Avoid MSBuild's inherited Path/PATH collision in sandboxed Windows shells.
    ninja=Path(vs)/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe' if vs else Path('ninja')
    args+=['-G','Ninja','-DCMAKE_MAKE_PROGRAM='+str(ninja),
           '-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang++','-DCMAKE_BUILD_TYPE=Release']
options={'PLUGIN_NAME':'Task Graph Probe','PLUGIN_SLUG':'TaskGraphProbe','PLUGIN_CODE':'TGP1',
         'MANUFACTURER_NAME':'ZorakAudio','MANUFACTURER_CODE':'ZrAu',
         'BUNDLE_ID':'com.zorakaudio.taskgraphprobe','PLUGIN_VERSION':'0.0.0',
         'PLUGIN_TYPE':'jsfx','PLUGIN_JSFX_OBJ':str(out/('JSFXDSP.obj' if os.name=='nt' else 'JSFXDSP.o')),
         'ZA_ROOT':str(ROOT),'ZA_ENABLE_CLAP':'ON','CLAP_ID':'com.zorakaudio.taskgraphprobe',
         'CLAP_FEATURES':'audio-effect','ZA_TASK_TEST_RUNNER':'ON'}
args+=['-D'+k+'='+v for k,v in options.items()]
# Explicit mapping also normalizes duplicate Path/PATH inherited on Windows.
environment=dict(os.environ)
if os.name=='nt' and 'PATH' in environment:
    environment['Path']=environment.pop('PATH')
subprocess.run(args,check=True,cwd=ROOT,env=environment)
with (out/'build.log').open('w',encoding='utf-8') as log:
    built=subprocess.run(['cmake','--build',str(out),'--config','Release','--target',
                          'task_processor_check','TaskGraphProbe_VST3','TaskGraphProbe_CLAP','--parallel','4'],
                         cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
if built.returncode:
    print('\n'.join((out/'build.log').read_text(errors='replace').splitlines()[-60:]))
    built.check_returncode()
print('Built task fixture, VST3 and CLAP; checking processor/editor lifecycle.',flush=True)
exe=out/('task_processor_check.exe' if os.name=='nt' else 'task_processor_check')
subprocess.run([str(exe)],check=True,cwd=ROOT,env=environment,timeout=60)
