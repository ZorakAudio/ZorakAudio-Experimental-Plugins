#!/usr/bin/env python3
"""Loaded production DSP services; serializes mutations of the cached host."""
from pathlib import Path
import argparse, hashlib, json, time
from qualify_editors import ROOT, discover_plugins, prepare, run, sha, flags
SLUGS=['PsychoConvolver','Contour','Texture','TexturePM','TextureXY']
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--host',type=Path,required=True);ap.add_argument('--cmake',default='cmake');ap.add_argument('--plugins',nargs='+',default=SLUGS);ap.add_argument('--reuse',action='store_true');a=ap.parse_args()
    out=a.out.resolve();host=a.host.resolve();driver=Path(__file__).with_name('catalog_bank_check.cpp');rows=[]
    for spec in discover_plugins(ROOT):
        if spec.slug not in a.plugins:continue
        start=time.monotonic();d=out/spec.slug;g=json.loads((d/'generation.json').read_text());assert g['status']=='PASS'
        code=b''.join(p.read_bytes() for p in sorted((ROOT/'src').glob('*')) if p.is_file())+driver.read_bytes()+driver.with_name('catalog_editor_check.cpp').read_bytes()+Path(__file__).read_bytes()+driver.with_name('qualify_editors.py').read_bytes()
        fp=hashlib.sha256(code+json.dumps(g['files'],sort_keys=True).encode()+json.dumps(flags(host)).encode()).hexdigest();row={'plugin':spec.slug,'fingerprint':fp}
        old=json.loads((d/'bank.json').read_text()) if a.reuse and (d/'bank.json').exists() else {}
        if old.get('status')=='PASS' and old.get('fingerprint')==fp and all((d/'bank-screenshots'/f).exists() and sha(d/'bank-screenshots'/f)==h for f,h in old['screenshots'].items()):rows.append(old);continue
        try:
            # prepare's driver cache must also include the included lifecycle source.
            for obj in host.glob('catalog_bank_check*.o'):obj.unlink()
            exe=prepare(spec,d,host,a.cmake,driver,'catalog_bank_check');screens=d/'bank-screenshots';screens.mkdir(exist_ok=True)
            for p in screens.glob('*.png'):p.unlink()
            run([exe,screens,spec.slug],d/'bank-run.log',host,timeout=120)
            result=json.loads((d/'bank-run.log').read_text().strip().splitlines()[-1]);assert result.pop('worker_complete',False)
            assert result['finite_audio'] and result['peak']>1e-6 and (screens/'loaded.png').exists()
            row.update(result,status='PASS',screenshots={p.name:sha(p) for p in screens.glob('*.png')},executable_sha256=sha(exe))
        except Exception as e:row.update(status='FAIL',error=str(e))
        row['seconds']=round(time.monotonic()-start,3);(d/'bank.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row);(out/'banks.json').write_text(json.dumps(rows,indent=2)+'\n');print('BANK',spec.slug,row['status'],row.get('error','')[:240],flush=True)
    (out/'banks.json').write_text(json.dumps(rows,indent=2)+'\n');return int(any(r['status']!='PASS' for r in rows))
if __name__=='__main__':raise SystemExit(main())
