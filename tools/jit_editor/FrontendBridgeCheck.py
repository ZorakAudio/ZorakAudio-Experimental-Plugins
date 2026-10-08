"""Gate native AST reconstruction through the unchanged production lowering passes."""
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as compiler
from scripts.jsfx_source import resolve_source
from native_frontend import run_native, parser_from_result


def normalize(value):
    if isinstance(value,dict): return {k:normalize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [normalize(v) for v in value]
    if is_dataclass(value): return dict(type=type(value).__name__,**{f.name:normalize(getattr(value,f.name)) for f in fields(value) if f.name!='id'})
    return value


def run(runtime, output):
    rows=[]
    for config in sorted((ROOT/'plugins').rglob('plugin.json')):
        settings=json.loads(config.read_text(encoding='utf-8-sig'))
        if settings.get('pluginType','jsfx')!='jsfx': continue
        path=(config.parent/settings['entry']).resolve();source=path.read_text(encoding='utf-8-sig')
        began=time.perf_counter()
        with tempfile.TemporaryDirectory(dir=output.parent) as folder:
            job=Path(folder).resolve();native=run_native(runtime/'compiler',job,source,path)
            expanded=native['resolution']['text']
            assert expanded==resolve_source(path,text=source).text,settings['slug']+': source mismatch'
            plan=compiler.faust_compiler.split(expanded)
            if plan: native=run_native(runtime/'compiler',job,plan['text'])
            cpp=compiler.prepare_jsfx_pipeline(expanded,native_gfx_legacy=True,include_serialize=True,parse_program=parser_from_result(compiler,native))
            standard=compiler.prepare_jsfx_pipeline(expanded,native_gfx_legacy=True,include_serialize=True)
            for field in ['programs','fn_defs','faust_plan']:
                assert normalize(cpp[field])==normalize(standard[field]),settings['slug']+': '+field+' mismatch'
            cv=compiler.collect_user_vars(cpp['programs'],cpp['fn_defs']);pv=compiler.collect_user_vars(standard['programs'],standard['fn_defs'])
            assert cv==pv,settings['slug']+': scalar binding/layout mismatch'
            assert compiler.infer_spl_io(cpp['programs'],cpp['fn_defs'])==compiler.infer_spl_io(standard['programs'],standard['fn_defs']),settings['slug']+': pin mismatch'
        rows.append(dict(plugin=settings['slug'],passed=True,variables=len(cv),seconds=time.perf_counter()-began))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(passed=True,cases=rows,compilerSha256=hashlib.sha256((ROOT/'dsp_jsfx_aot.py').read_bytes()).hexdigest())
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('runtime',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    run(args.runtime.resolve(),args.output.resolve())
