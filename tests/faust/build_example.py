"""Build the motivating plugin or its original baseline, without loading audio."""
from pathlib import Path
import os,sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));import build
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);parser.add_argument('--before',action='store_true');parser.add_argument('--package',action='store_true');parser.add_argument('--baseline',type=Path,default=ROOT/'plugins/Dynamics/EasyExpander/src/EasyExpander.jsfx');parser.add_argument('--reports',type=Path,default=ROOT/'build/faust-sections/profile');cli=parser.parse_args()
before=cli.before;package=cli.package
out=ROOT/'build/faust-sections/juce';out.mkdir(parents=True,exist_ok=True)
source=cli.source or (cli.baseline if before else ROOT/'plugins/Dynamics/EasyExpanderFaust/src/EasyExpanderFaust.jsfx')
build.build_jsfx_aot(ROOT,out,'EasyExpanderFaust',source,native_gfx_legacy=True)
build.write_plugin_readme_header(out,ROOT/'plugins/Dynamics/EasyExpanderFaust/README.md')
env=dict(os.environ)
if 'PATH' in env:env['Path']=env.pop('PATH')
args=['cmake','-S',str(ROOT/'cmake/plugin'),'-B',str(out)]
if not (out/'CMakeCache.txt').exists():
 vs=build.find_vs_installation_path();ninja=Path(vs)/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
 args+=['-G','Ninja','-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang++','-DCMAKE_BUILD_TYPE=Release']
options={'PLUGIN_NAME':'EasyExpander Faust','PLUGIN_SLUG':'EasyExpanderFaust','PLUGIN_CODE':'EeFa','MANUFACTURER_NAME':'ZorakAudio','MANUFACTURER_CODE':'ZrAu','BUNDLE_ID':'com.zorakaudio.experimental.easyexpanderfaust','PLUGIN_VERSION':'0.0.0','PLUGIN_TYPE':'jsfx','PLUGIN_JSFX_OBJ':str(out/'JSFXDSP.obj'),'ZA_ROOT':str(ROOT),'ZA_ENABLE_CLAP':'ON','CLAP_ID':'com.zorakaudio.experimental.easyexpanderfaust','CLAP_FEATURES':'audio-effect','ZA_FAUST_PROFILE_RUNNER':'OFF' if package else 'ON'}
args+=['-D'+k+'='+v for k,v in options.items()];subprocess.run(args,check=True,env=env)
targets=['EasyExpanderFaust_VST3','EasyExpanderFaust_CLAP'] if package else ['faust_profile','faust_idle_null','cooperative_idle_check']
logfile=out/('package.log' if package else 'build.log')
with logfile.open('w') as log:r=subprocess.run(['cmake','--build',str(out),'--target',*targets,'--parallel','4'],env=env,stdout=log,stderr=subprocess.STDOUT)
if r.returncode:print('\n'.join(logfile.read_text(errors='replace').splitlines()[-60:]));r.check_returncode()
if package:
 dest=ROOT/'dist/EasyExpander-Faust';dest.mkdir(parents=True,exist_ok=True)
 for bundle in build.collect_stageable_vst3_artifacts(out/'EasyExpanderFaust_artefacts/Release'):shutil.copytree(bundle,dest/bundle.name,dirs_exist_ok=True)
 for plugin in (out/'EasyExpanderFaust_artefacts/Release').rglob('*.clap'):shutil.copy2(plugin,dest/plugin.name)
 shutil.copy2(ROOT/'plugins/Dynamics/EasyExpanderFaust/README.md',dest/'README.md')
else:
 reports=cli.reports;reports.mkdir(parents=True,exist_ok=True);shutil.copy2(out/'faust_profile.exe',reports/('before.exe' if before else 'after.exe'));shutil.copy2(out/'JSFXDSP_meta.json',reports/('before-meta.json' if before else 'after-meta.json'));shutil.copy2(out/'faust_idle_null.exe',reports/('before-idle.exe' if before else 'after-idle.exe'));shutil.copy2(out/'cooperative_idle_check.exe',reports/'cooperative-idle.exe')
print('Built',('baseline' if before else 'Faust'),('production' if package else 'profiler'),flush=True)
