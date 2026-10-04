#!/usr/bin/env python3
"""Configure the reusable Linux production editor fixture after catalog codegen."""
from pathlib import Path
import argparse, shutil, subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--host',type=Path,required=True);p.add_argument('--cmake',default='cmake');a=p.parse_args()
    host=a.host.resolve();host.mkdir(parents=True,exist_ok=True)
    for name in ['JSFXDSP.h','JSFXDSP.o','JSFXSource.h','JSFXResources.h']:shutil.copyfile(a.out.resolve()/'Sample'/name,host/name)
    cmd=[a.cmake,'-S',str(ROOT/'cmake/plugin'),'-B',str(host),'-G','Unix Makefiles','-DCMAKE_BUILD_TYPE=Release',f'-DZA_ROOT={ROOT}','-DPLUGIN_NAME=Sample','-DPLUGIN_SLUG=Sample','-DPLUGIN_CODE=Samp','-DMANUFACTURER_NAME=ZorakAudio','-DMANUFACTURER_CODE=Zrak','-DBUNDLE_ID=com.zorakaudio.experimental.sample','-DPLUGIN_VERSION=0.0.0','-DPLUGIN_TYPE=jsfx',f'-DPLUGIN_JSFX_OBJ={host}/JSFXDSP.o','-DZA_SAMPLE_GFX_TEST_RUNNER=ON','-DZA_NATIVE_GFX_LEGACY=OFF','-DZA_ENABLE_CLAP=ON','-DCLAP_ID=com.zorakaudio.experimental.sample','-DCLAP_FEATURES=audio-effect']
    subprocess.run(cmd,check=True)
    subprocess.run([a.cmake,'--build',str(host),'--target','sample_editor_check','--','-j2','CXX_FLAGS=-O0 -DNDEBUG -std=gnu++17 -fPIC'],check=True)
if __name__=='__main__':main()
