"""Experimental block-layout retries; does not modify production plugins.
Run after preparing and profiling the original ADS/SaliencePush candidates.
Then profile_catalog.py ADSBlockRetry SaliencePushBlockRetry.
"""
from pathlib import Path
import re,sys,shutil,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));import dsp_jsfx_aot as c
for key in ['ADS','SaliencePush']:
 old=ROOT/'build/catalog-faust-audit'/key;new=old.parent/(key+'BlockRetry');new.mkdir(exist_ok=True)
 source=(old/'candidate.jsfx').read_text(encoding='utf-8');baseline=(old/'baseline.jsfx').read_text(encoding='utf-8');sections=c.extract_sections(source)
 # Original candidates export every assigned name mentioned in @init, including
 # private histories initialized there. Only retain definitions externally read.
 outside=sections['gfx'][0]  # This experiment retains UI exports only.
 faust_start=source.index('@faust');faust_end=source.find('\n@',faust_start+1)
 if faust_end<0:faust_end=len(source)
 body=source[faust_start:faust_end]
 assigned=json.loads((old/'candidate/meta.json').read_text(encoding='utf-8'))['faust_stages'][0]['exports']
 for name in assigned:
  if name.startswith('spl') or re.search(r'\b'+re.escape(name)+r'\b',outside,re.I):continue
  body=re.sub(r'(?m)^'+re.escape(name)+r'\s*=', 'retry_private_'+name+' =',body)
 shutil.copyfile(old/'baseline.jsfx',new/'baseline.jsfx');(new/'candidate.jsfx').write_text(source[:faust_start]+body+source[faust_end:],encoding='utf-8')
 print(key,'experimental block candidate prepared')
