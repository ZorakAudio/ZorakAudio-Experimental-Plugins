"""Build original and migrated Corpus profilers without loading any fixture audio."""
from pathlib import Path
import os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import build
out=ROOT/'build/tasks/corpus'
reports=ROOT/'build/tasks/corpus-profile';reports.mkdir(parents=True,exist_ok=True)
baseline=reports/'before.jsfx'
baseline.write_bytes(subprocess.check_output(['git','show','HEAD:plugins/Spectral/Corpus/src/Corpus.jsfx'],cwd=ROOT))
environment=dict(os.environ)
if 'PATH' in environment:environment['Path']=environment.pop('PATH')
modes=[('before',baseline),('after',ROOT/'plugins/Spectral/Corpus/src/Corpus.jsfx')]
if '--before-only' in sys.argv:modes=modes[:1]
for mode,source in modes:
    build.build_jsfx_aot(ROOT,out,'Corpus',source,native_gfx_legacy=True)
    subprocess.run(['cmake','-S',str(ROOT/'cmake/plugin'),'-B',str(out),
                    '-DZA_CORPUS_TASK_TEST_RUNNER=OFF','-DZA_CORPUS_PROFILE_RUNNER=ON'],check=True,cwd=ROOT,env=environment)
    with (reports/(mode+'-build.log')).open('w') as log:
        result=subprocess.run(['cmake','--build',str(out),'--target','corpus_profile','--parallel','4'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join((reports/(mode+'-build.log')).read_text(errors='replace').splitlines()[-60:]));result.check_returncode()
    shutil.copy2(out/'corpus_profile.exe',reports/(mode+'.exe'))
    shutil.copy2(out/'JSFXDSP_meta.json',reports/(mode+'-meta.json'))
    print('Built',mode,'profiler',flush=True)
