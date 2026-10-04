#!/usr/bin/env python3
"""Compare unmodified default AOT DSP with actual WDL/EEL2 audio sections.

Custom IPC/sample/file callbacks require production host tests, not fake WDL services.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests/jsfx_showcase')]
from pluginlib import discover_plugins
from joep_legacy_qualification import config_header, host_helpers
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,log,**kwargs):
    with log.open('w') as f:p=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,**kwargs)
    if p.returncode:raise RuntimeError(f'exit {p.returncode}: {log.read_text()[-2500:]}')
def adapted():
    text=(ROOT/'tests/jsfx_showcase/joep_dsp_reference.cpp').read_text()
    text=text.replace('#include "JSFXDSP.h"','#include "JSFXDSP.h"\n#include "JsfxSharedCells.h"\n#include <array>\nstatic const auto DSPJSFX_LEGACY_SLIDER_ALIASES=[] { std::array<int,std::size(DSPJSFX_VARS)> a; a.fill(-1); return a; }();')
    text=text.replace(' st.atomicContext = &atomicMutex;','')
    marker='for (const auto &s : joepSliders) st.sliders[s.index] = s.value;'
    text=text.replace(marker,marker+'\n  auto syncAliases=[&] { for(const auto& slider:joepSliders) for(const auto& var:DSPJSFX_VARS) if(std::string(var.name)==slider.alias) st.vars[var.index]=st.sliders[slider.index]; };\n  syncAliases();')
    text=text.replace('jsfx_slider(&st); NSEEL_code_execute(handles["slider"]);','syncAliases(); jsfx_slider(&st); NSEEL_code_execute(handles["slider"]);')
    text=text.replace('auto nativeStart=std::chrono::steady_clock::now();','syncAliases(); auto nativeStart=std::chrono::steady_clock::now();')
    return text
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--wdl-build',type=Path,required=True);ap.add_argument('--plugins',nargs='+');ap.add_argument('--reuse',action='store_true');a=ap.parse_args()
    out=a.out.resolve();wdl=a.wdl_build.resolve();driver=out/'catalog_dsp_reference.cpp';driver.write_text(adapted());rows=[]
    for spec in discover_plugins(ROOT):
        if spec.category=='JoepVanlier' or spec.plugin_type!='jsfx' or (a.plugins and spec.slug not in a.plugins):continue
        start=time.monotonic();d=out/spec.slug;g=json.loads((d/'generation.json').read_text());assert g['status']=='PASS'
        callbacks=subprocess.check_output(['nm','-u',str(d/'JSFXDSP.o')],text=True)
        required=sorted(set(re.findall(r'\bjsfx_(?:sample|file|gmem|msg|instance)_\w+',callbacks)))
        row={'plugin':spec.slug,'object_sha256':sha(d/'JSFXDSP.o'),'source_sha256':sha(d/'JSFXExpanded.jsfx')}
        if required:row.update(status='NOT_COMPARABLE',reason='Custom host extensions; covered by real production processor tests',required_callbacks=required)
        else:
            dep=[driver,Path(__file__),wdl/'numeric_runtime.inc',wdl/'libshowcase_eel.a',ROOT/'src/YSFXGfxInterpreter.h',ROOT/'src/JsfxSharedCells.h',ROOT/'src/JsfxLegacyAtomics.h',ROOT/'src/JSFXJuceProcessor.cpp',ROOT/'tests/jsfx_showcase/juce_contract_stub.h',ROOT/'tests/jsfx_showcase/joep_legacy_qualification.py',d/'JSFXDSP.h',d/'JSFXDSP.o',d/'JSFXExpanded.jsfx']
            fp=hashlib.sha256(b''.join(p.read_bytes() for p in dep)).hexdigest();row['fingerprint']=fp
            old=json.loads((d/'numerical.json').read_text()) if a.reuse and (d/'numerical.json').exists() else {}
            if old.get('status')=='PASS' and old.get('fingerprint')==fp:rows.append(old);continue
            try:
                source=(d/'JSFXExpanded.jsfx').read_text();source=re.sub(r'(?m)^\s*slider\d+:\s*#[^\n]*<string>[^\n]*','',source)
                (d/'JoepTestConfig.h').write_text(config_header(source,spec.slug));host_helpers(d)
                processor=(ROOT/'src/JSFXJuceProcessor.cpp').read_text();startText=processor.index('extern "C" double jsfx_slider_show');end=processor.index('\n}',startText)+2
                with (d/'host_slider_runtime.inc').open('a') as f:f.write(processor[startText:end]+'\n')
                exe=d/'catalog_dsp_reference'
                run(['c++','-O2','-std=c++20','-DEEL_TARGET_PORTABLE=1','-DWDL_FFT_REALSIZE=8',*[f'-I{p}' for p in [ROOT/'src',ROOT/'tests/jsfx_showcase',d,wdl]],driver,d/'JSFXDSP.o',wdl/'libshowcase_eel.a','-pthread','-lm','-o',exe],d/'reference-build.log',timeout=240)
                run([exe,d/'JSFXExpanded.jsfx','48000','defaults','float'],d/'reference-run.log',timeout=120,env=dict(os.environ,ZA_JOEP_SECONDS='2',ZA_JOEP_DIAG='1'))
                terminal=json.loads((d/'reference-run.log').read_text().strip().splitlines()[-1]);row.update(status='PASS',result=terminal,executable_sha256=sha(exe))
            except Exception as e:row.update(status='FAIL',error=str(e))
        row['seconds']=round(time.monotonic()-start,3);(d/'numerical.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row);(out/'numerics.json').write_text(json.dumps(rows,indent=2)+'\n');print('REFERENCE',spec.slug,row['status'],row.get('error','')[:240],flush=True)
    (out/'numerics.json').write_text(json.dumps(rows,indent=2)+'\n');return int(any(r['status']=='FAIL' for r in rows))
if __name__=='__main__':raise SystemExit(main())
