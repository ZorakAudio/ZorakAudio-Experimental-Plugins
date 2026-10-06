"""Build only the local qualification runner, with a test-only plugin identity.
Never packages, stages, or installs VST3/CLAP binaries.
The shared cache requires sequential builds; copied runners are independent.
"""
from pathlib import Path
import os,sys,subprocess,shutil,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));import build
parser=argparse.ArgumentParser()
choice=parser.add_mutually_exclusive_group(required=True)
choice.add_argument('--source',type=Path);choice.add_argument('--prebuilt',type=Path)
parser.add_argument('--reports',type=Path,required=True);cli=parser.parse_args()
out=ROOT/'build/faust-sections/juce';out.mkdir(parents=True,exist_ok=True)
if cli.prebuilt:
 for item in cli.prebuilt.iterdir():
  if item.is_file():shutil.copy2(item,out/item.name)
else:build.build_jsfx_aot(ROOT,out,'SamplerCorpusFixture',cli.source,native_gfx_legacy=True)
build.write_plugin_readme_header(out,ROOT/'plugins/Dynamics/EasyExpanderFaust/README.md')
env=dict(os.environ)
if 'PATH' in env:env['Path']=env.pop('PATH')
args=['cmake','-S',str(ROOT/'cmake/plugin'),'-B',str(out)]
if not (out/'CMakeCache.txt').exists():
 vs=build.find_vs_installation_path();ninja=Path(vs)/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
 args+=['-G','Ninja','-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang++','-DCMAKE_BUILD_TYPE=Release']
# Keep the historical fixture identity/cache. DSP/control source comes from the
# supplied plugin; plugin identity is irrelevant to these standalone runner tests.
options={'PLUGIN_NAME':'EasyExpander Faust','PLUGIN_SLUG':'EasyExpanderFaust','PLUGIN_CODE':'EeFa','MANUFACTURER_NAME':'ZorakAudio','MANUFACTURER_CODE':'ZrAu','BUNDLE_ID':'com.zorakaudio.experimental.easyexpanderfaust','PLUGIN_VERSION':'0.0.0','PLUGIN_TYPE':'jsfx','PLUGIN_JSFX_OBJ':str(out/'JSFXDSP.obj'),'ZA_ROOT':str(ROOT),'ZA_ENABLE_CLAP':'ON','CLAP_ID':'com.zorakaudio.experimental.easyexpanderfaust','CLAP_FEATURES':'audio-effect','ZA_FAUST_PROFILE_RUNNER':'ON'}
args+=['-D'+k+'='+v for k,v in options.items()];subprocess.run(args,check=True,env=env)
logfile=out/'build.log'
with logfile.open('w') as log:r=subprocess.run(['cmake','--build',str(out),'--target','sampler_corpus_host','--parallel','4'],env=env,stdout=log,stderr=subprocess.STDOUT)
if r.returncode:print('\n'.join(logfile.read_text(errors='replace').splitlines()[-60:]));r.check_returncode()
cli.reports.mkdir(parents=True,exist_ok=True)
shutil.copy2(out/'sampler_corpus_host.exe',cli.reports/'host.exe');shutil.copy2(out/'JSFXDSP_meta.json',cli.reports/'meta.json')
print('Built test-only loaded-bank qualification runner',flush=True)
