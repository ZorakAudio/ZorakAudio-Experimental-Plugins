# Package the validated current example without including recordings or PCM dumps.
from pathlib import Path
import shutil,json,hashlib,subprocess,re,zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=Path(r'C:\Users\LouisJenkinsCS\Documents\Codex\2026-10-05\can-x20\outputs');dest=ROOT/'dist/EasyExpander-Faust'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
for source,target in [(ROOT/'docs/Cooperative-Sleep.md','Cooperative-Sleep.md'),(ROOT/'docs/JSFX-Faust-Sections.md','JSFX-Faust-Sections.md'),(ROOT/'docs/validation/EasyExpander-Sleep-Audit.md','EasyExpander-Sleep-Audit.md'),(ROOT/'plugins/Dynamics/EasyExpanderFaust/src/EasyExpanderFaust.jsfx','EasyExpanderFaust.jsfx'),(ROOT/'tests/faust/fixtures/EasyExpander-user.jsfx','EasyExpander-original-user-version.jsfx'),(OUT/'EasyExpander-user-baseline-and-idle-results.json','results.json')]:shutil.copy2(source,dest/target)
notes='\nUse the VST3 or CLAP for the Faust implementation. EasyExpander-original-user-version.jsfx is the exact supplied stock JSFX baseline. EasyExpanderFaust.jsfx is extension source for this repository compiler and is not supported by stock REAPER JSFX. Copy the VST3 bundle or CLAP to a directory REAPER scans and reload/rescan to use the new binary.\n'
p=dest/'README.md';s=p.read_text(encoding='utf-8');p.write_text(s+notes,encoding='utf-8')
imports={}
for binary in dest.rglob('*'):
 if binary.is_file() and binary.suffix in ('.clap','.vst3'):
  text=subprocess.run(['llvm-readobj','--coff-imports',str(binary)],capture_output=True,text=True,check=True).stdout;names=re.findall(r'^  Name: (.+)$',text,re.M);assert names and not any('faust' in n.lower() or 'llvm' in n.lower() for n in names);imports[binary.relative_to(dest).as_posix()]=names
assert len(imports)==2
manifest={'platform':'Windows x64','sleep_policy':'Explicit fresh plugin permission only; no mode selector or threshold sleep; offline always active','build_dependencies':{'Faust':'2.81.2 LLVM 17','Clang':'21'},'dll_imports':imports,'files':{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='manifest.json'}}
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=OUT/'EasyExpander-Faust-cooperative-idle-Windows-x64.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in dest.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(dest))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,digest in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
shutil.copy2(ROOT/'docs/JSFX-Faust-Sections.md',OUT/'JSFX-Faust-Sections.md')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(manifest['files'])+1,'binary_import_check':'No libfaust/LLVM runtime','manifest_and_zip_validation':'passed'},indent=2))
