param([switch]$WholeFilter)
$ErrorActionPreference='Stop'
$taskPython='C:\Users\LouisJenkinsCS\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$taskPrefix=if($WholeFilter){'privatefull'}else{'private'}
$env:ZA_SAMPLE_PROFILE_PREFIX=$taskPrefix
@'
from pathlib import Path
import os,sys,json
sys.path.insert(0,str(Path.cwd()));import dsp_jsfx_aot as c
b=Path('build/faust-sections/sampler-corpus/Sample');prefix=os.environ['ZA_SAMPLE_PROFILE_PREFIX']
for version in ['before','after']:
 p=b/(prefix+'-'+version);p.mkdir(exist_ok=True);ir,m=c.compile_jsfx_to_ir((b/(prefix+'-kernel-'+version+'.jsfx')).read_text(encoding='utf-8'),native_gfx_legacy=True);(p/'JSFXDSP.h').write_text(c._emit_header(m),encoding='utf-8');(p/'meta.json').write_text(json.dumps(m,indent=2),encoding='utf-8');(p/'dsp.ll').write_text(str(ir),encoding='utf-8')
'@ | & $taskPython -
if($LASTEXITCODE){throw 'DSP generation failed'}
foreach($version in @('before','after')) {
 $taskDir="build/faust-sections/sampler-corpus/Sample/$taskPrefix-$version"
 clang++ -O2 -c "$taskDir/dsp.ll" -o "$taskDir/dsp.obj"
 if($LASTEXITCODE){throw 'Object compilation failed'}
 clang++ -O2 -std=c++20 -UNDEBUG -DPRIVATE_CORPUS=0 -DPRIVATE_SAMPLE_BLOCK=1 -DCORPUS_TEST=0 "-I$taskDir" -Isrc tests/faust/sampler_corpus_kernel.cpp "$taskDir/dsp.obj" -o "$taskDir/eq_profile.exe"
 if($LASTEXITCODE){throw 'Runner compilation failed'}
 foreach($setting in @(@(48000,64),@(48000,256),@(96000,1024))) {foreach($scenario in 0..3) {
  & "./$taskDir/eq_profile.exe" "./$taskDir/$($setting[0])-$($setting[1])-$scenario.json" $setting[0] $setting[1] $scenario
  if($LASTEXITCODE){throw 'Kernel test failed'}
 }}
}
& $taskPython tests/faust/profile_private_sample.py $taskPrefix
if($LASTEXITCODE){throw 'Audio comparison failed'}
