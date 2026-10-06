"""Package the completed experiment and measured results, without loading audio."""
from pathlib import Path
import json,hashlib,shutil,zipfile,csv,subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(r'C:\Users\LouisJenkinsCS\Documents\Codex\2026-10-05\can-x20\outputs');OUT.mkdir(parents=True,exist_ok=True)
PROFILE=ROOT/'build/faust-sections/profile'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
before=json.loads((PROFILE/'before-final.json').read_text());after=json.loads((PROFILE/'after-final.json').read_text())
a=PROFILE/'before-final.json.pcm.bin';b=PROFILE/'after-final.json.pcm.bin'
assert a.stat().st_size==b.stat().st_size==before['frames']*2*4
assert sha(a)==sha(b)
speed=before['processBlock_seconds']/after['processBlock_seconds'];saved=100*(1-1/speed)
report={'example':'EasyExpander Faust (non-JoepVanlier EasyExpander baseline)','platform':'Windows x64','faust_version':'2.81.2','faust_llvm_version':17,'clang_version':21,'input_sha256':'1f44c88399234d13891fb9bd29eac84108262334c7e041111dabc89267c7e8c5','settings':'Identical defaults; 48 kHz stereo, 256-frame callbacks','before':before,'after':after,'speedup':speed,'processing_time_saved_percent':saved,'equivalence':{'float_samples':before['frames']*2,'different_samples':0,'maximum_absolute_error':0,'pcm_sha256':sha(a)},'kernel_profiles':json.loads((ROOT/'build/faust-sections/kernel/results.json').read_text()),'validation':{'mixed_runtime_fixtures':15,'compiler_contract_tests':8,'structured_task_regression_tests':11,'build_mode_regression_tests':7,'editor_and_rate_resets':True},'limitations':'Measured locally with these fixtures and supplied recording; no universal speedup, deadline or cross-platform guarantee. Native Legacy host lifecycle locking remains. Unsupported table initializer control bindings fail compilation.'}
(OUT/'Faust-in-JSFX-results.json').write_text(json.dumps(report,indent=2))
with (OUT/'Faust-in-JSFX-results.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['case','sample_rate','block_size','before_seconds','after_seconds','speedup','maximum_sample_error'])
 w.writerow(['full_JUCE_processor',48000,256,before['processBlock_seconds'],after['processBlock_seconds'],speed,0])
 for r in report['kernel_profiles']:w.writerow(['kernel',r['before']['sample_rate'],r['before']['block_size'],r['before']['median_seconds'],r['after']['median_seconds'],r['speedup'],r['maximum_sample_error']])
notes=f"\nMeasured on the supplied 609.989-second recording with identical defaults, 48 kHz\nstereo and 256-frame callbacks: original {before['processBlock_seconds']:.3f} s of\nprocessing time, Faust {after['processBlock_seconds']:.3f} s ({speed:.2f}x, {saved:.1f}% saved).\nAll {before['frames']*2:,} float output samples were bit-identical. File decoding\nand output dumping are excluded. Kernel profiles use three-trial medians across\n48/96 kHz and 64/256/1024 frames. Full-processor timings are one final paired run;\nearlier runs varied with machine scheduling but showed a similar ratio.\nValidation: 15 mixed-runtime fixtures, 8 compiler contracts, 11 defer regression\ntests, and complete processor editor/rate reset checks. These are local Windows\nx64 results, not universal performance or compatibility guarantees.\n"
p=ROOT/'plugins/Dynamics/EasyExpanderFaust/README.md';s=p.read_text(encoding='utf-8');p.write_text(s+notes,encoding='utf-8')
p=ROOT/'docs/JSFX-Faust-Sections.md';s=p.read_text(encoding='utf-8');p.write_text(s+notes,encoding='utf-8');shutil.copy2(p,OUT/'JSFX-Faust-Sections.md')
dest=ROOT/'dist/EasyExpander-Faust';shutil.copy2(ROOT/'plugins/Dynamics/EasyExpanderFaust/README.md',dest/'README.md');shutil.copy2(p,dest/'JSFX-Faust-Sections.md');shutil.copy2(OUT/'Faust-in-JSFX-results.json',dest/'results.json');shutil.copy2(ROOT/'plugins/Dynamics/EasyExpanderFaust/src/EasyExpanderFaust.jsfx',dest/'EasyExpanderFaust.jsfx')
imports={}
for binary in dest.rglob('*'):
 if binary.is_file() and binary.suffix in ('.clap','.vst3'):
  text=subprocess.run(['llvm-readobj','--coff-imports',str(binary)],capture_output=True,text=True,check=True).stdout
  import re
  names=re.findall(r'^  Name: (.+)$',text,re.M);assert not any('faust' in n.lower() or 'llvm' in n.lower() for n in names)
  imports[binary.relative_to(dest).as_posix()]=names
manifest={'dll_imports':imports,'files':{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},'build_dependencies':{'Faust':'2.81.2 LLVM 17','Clang':'21'},'runtime':'Ahead-of-time; no Faust compiler/libfaust required'}
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=OUT/'EasyExpander-Faust-Windows-x64.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in dest.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(dest))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'speedup':speed,'before_seconds':before['processBlock_seconds'],'after_seconds':after['processBlock_seconds'],'bit_identical_samples':before['frames']*2,'archive_bytes':archive.stat().st_size},indent=2))
