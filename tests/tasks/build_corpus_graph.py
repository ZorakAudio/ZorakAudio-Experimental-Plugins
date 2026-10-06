"""Build Corpus's supplied-file profiler or package its production graph build.
No fixture recordings are loaded by this script.
"""
from pathlib import Path
import os,sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import build
out=ROOT/'build/tasks/corpus';reports=ROOT/'build/tasks/corpus-profile'
reports.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
build.build_jsfx_aot(ROOT,out,'Corpus',ROOT/'plugins/Spectral/Corpus/src/Corpus.jsfx',native_gfx_legacy=True)
build.write_plugin_readme_header(out,ROOT/'plugins/Spectral/Corpus/README.md')
environment=dict(os.environ)
if 'PATH' in environment:environment['Path']=environment.pop('PATH')
package='--package' in sys.argv
configure=['cmake','-S',str(ROOT/'cmake/plugin'),'-B',str(out)]
if not (out/'CMakeCache.txt').exists() and os.name=='nt':
    vs=build.find_vs_installation_path()
    ninja=Path(vs)/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe' if vs else Path('ninja')
    configure+=['-G','Ninja','-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang++','-DCMAKE_BUILD_TYPE=Release']
options={'PLUGIN_NAME':'Corpus','PLUGIN_SLUG':'Corpus','PLUGIN_CODE':'Corp',
         'MANUFACTURER_NAME':'ZorakAudio','MANUFACTURER_CODE':'ZrAu',
         'BUNDLE_ID':'com.zorakaudio.experimental.corpus','PLUGIN_VERSION':'0.0.0',
         'PLUGIN_TYPE':'jsfx','PLUGIN_JSFX_OBJ':str(out/('JSFXDSP.obj' if os.name=='nt' else 'JSFXDSP.o')),
         'ZA_ROOT':str(ROOT),'ZA_ENABLE_CLAP':'ON','CLAP_ID':'com.zorakaudio.experimental.corpus',
         'CLAP_FEATURES':'audio-effect','ZA_CORPUS_TASK_TEST_RUNNER':'OFF',
         'ZA_CORPUS_PROFILE_RUNNER':'OFF' if package else 'ON'}
configure+=['-D'+key+'='+value for key,value in options.items()]
subprocess.run(configure,check=True,env=environment,cwd=ROOT)
targets=['Corpus_VST3','Corpus_CLAP'] if package else ['corpus_profile']
logfile=reports/('graph-release.log' if package else 'graph-build.log')
with logfile.open('w') as log:
    result=subprocess.run(['cmake','--build',str(out),'--target',*targets,'--parallel','4'],env=environment,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    print('\n'.join(logfile.read_text(errors='replace').splitlines()[-60:]));result.check_returncode()
if package:
    destination=ROOT/'dist/Corpus-Tasks';destination.mkdir(parents=True,exist_ok=True)
    artefacts=out/'Corpus_artefacts/Release'
    for bundle in build.collect_stageable_vst3_artifacts(artefacts):
        shutil.copytree(bundle,destination/bundle.name,dirs_exist_ok=True)
    for plugin in artefacts.rglob('*.clap'):shutil.copy2(plugin,destination/plugin.name)
    shutil.copy2(ROOT/'plugins/Spectral/Corpus/README.md',destination/'README.md')
    print('Packaged production Corpus at',destination,flush=True)
else:
    shutil.copy2(out/'corpus_profile.exe',reports/'graph.exe')
    shutil.copy2(out/'JSFXDSP_meta.json',reports/'graph-meta.json')
    print('Built supplied-file Corpus graph profiler',flush=True)
