"""Build complete JUCE panner comparison hosts using one isolated cache."""
from pathlib import Path
import os,sys,subprocess,shutil,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));import build
p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');p.add_argument('--idle-check',action='store_true');p.add_argument('labels',nargs='*',default=['baseline','fast','faust']);a=p.parse_args()
BASE=ROOT/'build/faust-sections/panner';host=BASE/'juce';host.mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
if 'PATH' in env:env['Path']=env.pop('PATH')
env['CMAKE_BUILD_PARALLEL_LEVEL']='4'
vs=Path(build.find_vs_installation_path());ninja=vs/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
for label in a.labels:
 slug={'baseline':'3DPanner','fast':'HyperrealFast','faust':'HyperrealFaust','hybrid':'HyperrealHybrid','equivalent':'HyperrealEquivalentEEL','promoted':'3DPanner'}[label]
 folder=BASE/label;folder.mkdir(exist_ok=True)
 source=ROOT/'tests/faust/equivalent_physical.jsfx' if label=='equivalent' else ROOT/f'plugins/Spatialization/{slug}/src/{slug}.jsfx'
 if label=='baseline' and (ROOT/'tests/faust/reference/3DPanner-before-fast.jsfx').exists():source=ROOT/'tests/faust/reference/3DPanner-before-fast.jsfx'
 if not a.reuse:build.build_jsfx_aot(ROOT,folder,slug,source,native_gfx_legacy=True)
 for name in ['JSFXDSP.h','JSFXDSP.obj','JSFXSource.h','JSFXResources.h']:shutil.copyfile(folder/name,host/name)
 cmd=['cmake','-S',str(ROOT/'cmake/plugin'),'-B',str(host),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang++','-DZA_ROOT='+str(ROOT),'-DPLUGIN_NAME=Panner comparison fixture','-DPLUGIN_SLUG=PannerFixture','-DPLUGIN_CODE=PnFx','-DMANUFACTURER_NAME=ZorakAudio','-DMANUFACTURER_CODE=Zrak','-DBUNDLE_ID=com.zorakaudio.test.panner','-DPLUGIN_VERSION=0.0.0','-DPLUGIN_TYPE=jsfx','-DPLUGIN_JSFX_OBJ='+str(host/'JSFXDSP.obj'),'-DPLUGIN_IS_SYNTH=OFF','-DPLUGIN_NEEDS_MIDI_INPUT=OFF','-DPLUGIN_NEEDS_MIDI_OUTPUT=OFF','-DZA_NATIVE_GFX_LEGACY=ON','-DZA_ENABLE_CLAP='+('ON' if a.idle_check else 'OFF'),'-DCLAP_ID=com.zorakaudio.test.panner','-DCLAP_FEATURES=audio-effect','-DZA_PANNER_PROFILE_RUNNER=ON']
 subprocess.run(cmd,check=True,env=env);subprocess.run(['cmake','--build',str(host),'--target','panner_profile',*(['panner_idle','panner_clap_notifications'] if a.idle_check else [])],check=True,env=env)
 shutil.copy2(host/'panner_profile.exe',folder/'host.exe');
 if a.idle_check:
  for target in ['panner_idle','panner_clap_notifications']:shutil.copy2(host/(target+'.exe'),folder/(target+'.exe'))
 print('HOST READY',label,flush=True)
