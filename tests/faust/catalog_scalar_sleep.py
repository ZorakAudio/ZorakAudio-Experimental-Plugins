"""Exact fixed-point certificates for the monotone scalar detector recurrences.

Restricted to ADS and SaliencePush: zero-driven one-poles and attack/release
followers with fixed controls, no clocks, random generators or RAM state.
This is NOT a general state-comparison sleep optimizer.
"""
from pathlib import Path
from catalog_scalar_candidate import candidate
ROOT=Path(__file__).resolve().parents[2]
for key in ['ADS','SaliencePush']:
    base=ROOT/'build/catalog-faust-audit'/key
    source=(base/'baseline.jsfx').read_text(encoding='utf-8')
    _,states,_=candidate(source)
    init='za_sleep_ready=0;za_scalar_valid=0;za_scalar_zero=0;\n'
    init+='\n'.join('za_prev_'+n+'=0;' for n in states)+'\n'
    source=source.replace('@init\n','@init\n'+init,1)
    source=source.replace('@slider\n','@slider\n// New controls require a new zero-input fixed-point certificate.\nza_scalar_valid=0;\n',1)
    block='''@block
// Under fixed controls and zero input these scalar one-poles are monotone.
// Equality over the entire preceding silent block certifies a fixed point.
// This includes every persistent sample recurrence, not just audible output.
za_sleep_ready = za_scalar_valid && za_scalar_zero &&
'''+ ' &&\n'.join('  '+n+' === za_prev_'+n for n in states)+';\n'
    block+='\n'.join('za_prev_'+n+'='+n+';' for n in states)+'\nza_scalar_zero=1;za_scalar_valid=1;\n\n'
    sample='''@sample
// Observe original source AND key channels before sample code overwrites them.
za_scalar_zero = za_scalar_zero && spl0 === 0 && spl1 === 0 && spl2 === 0 && spl3 === 0;
'''
    source=source.replace('@sample\n',block+sample,1)
    (base/'sleep-candidate.jsfx').write_text(source,encoding='utf-8')
    (base/'sleep-states.txt').write_text('\n'.join(states),encoding='utf-8')
    print(key,'certified states',len(states),states)
