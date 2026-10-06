"""Build reviewable FAUST candidates for scalar-only sample programs.

Offline migration aid, deliberately rejects RAM, loops, calls with side effects,
and conditional assignments. Candidates are never installed by this script.
"""
from pathlib import Path
import sys, re, dataclasses
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dsp_jsfx_aot as c

def candidate(source):
    sections = c.extract_sections(source)
    sample = sections['sample'][0]
    nodes = c.Parser(sample).parse_program()
    assigned = {n.target.name.lower() for n in nodes if isinstance(n, c.Assign) and isinstance(n.target, c.Var)}
    env = {f'spl{i}':f'audio{i}' for i in range(4)}
    definitions, states = [], {}
    def read(name):
        name = name.lower()
        if name in env: return env[name]
        if name in assigned and not name.startswith('spl'):
            states.setdefault(name, 'old_' + name)
            return states[name]
        return name
    def expr(n):
        if isinstance(n, c.Num): return repr(float(n.value))
        if isinstance(n, c.Var): return read(n.name)
        if isinstance(n, c.Unary): return '(' + n.op + expr(n.a) + ')'
        if isinstance(n, c.Binary):
            if n.op not in ['+', '-', '*', '/', '<', '>', '<=', '>=', '==', '!=']: raise ValueError(n.op)
            return '(' + expr(n.l) + n.op + expr(n.r) + ')'
        if isinstance(n, c.Call) and n.fn.lower() in ['min','max','abs','sqrt','exp','log','pow']:
            return n.fn.lower() + '(' + ','.join(expr(a) for a in n.args) + ')'
        if isinstance(n, c.Ternary): return 'select2(' + expr(n.cond) + ',' + expr(n.els) + ',' + expr(n.then) + ')'
        raise ValueError('Unsupported expression: ' + type(n).__name__)
    for n in nodes:
        if not isinstance(n,c.Assign) or not isinstance(n.target,c.Var): raise ValueError('Non-scalar statement')
        name=n.target.name.lower()
        rhs=expr(n.value)
        if n.op!='=': rhs='('+read(name)+n.op[0]+rhs+')'
        token='v'+str(len(definitions))+'_'+name
        definitions.append('        '+token+' = '+rhs+';')
        env[name]=token
    outside='\n'.join(v[0] for k,v in sections.items() if k!='sample')
    exported=[n for n in sorted(assigned) if not n.startswith('spl') and re.search(r'\b'+re.escape(n)+r'\b', outside,re.I)]
    outputs=['spl0','spl1']+exported
    ns=len(states)
    values=[env[n] for n in states]+[env[n] for n in outputs]
    faust='@faust\n// Scalar recurrence translated from the original @sample, preserving statement order.\n'
    faust+='catalog_result = (spl0,spl1,spl2,spl3) : (step ~ ('+','.join('_' for _ in states)+')) with {\n'
    faust+='    step('+','.join(list(states.values())+[f'audio{i}' for i in range(4)])+') = '+','.join(values)+' with {\n'
    faust+='\n'.join(definitions)+'\n    };\n};\n'
    for i,n in enumerate(outputs):
        picks=['_' if j==ns+i else '!' for j in range(len(values))]
        faust+=n+' = catalog_result : ('+','.join(picks)+');\n'
    faust+='process = spl0,spl1;\n'
    # splN definitions shadow automatic audio imports; use separate input aliases.
    faust=faust.replace('spl0 = catalog_result','catalog_left = catalog_result').replace('spl1 = catalog_result','catalog_right = catalog_result').replace('process = spl0,spl1;','process = catalog_left,catalog_right;')
    return re.sub(r'(?ms)^@sample[^\n]*\n.*?(?=^@|\Z)',lambda _:faust+'\n',source),list(states),exported

def salience(source):
    """Explicit signal recurrences for SaliencePush; no general migration claim."""
    sections=c.extract_sections(source)
    nodes=c.Parser(sections['sample'][0]).parse_program()
    env={};lines=[];cuts={}
    def expr(n):
        if isinstance(n,c.Num):return repr(float(n.value))
        if isinstance(n,c.Var):return env.get(n.name.lower(),n.name.lower())
        if isinstance(n,c.Binary):return '('+expr(n.l)+n.op+expr(n.r)+')'
        if isinstance(n,c.Unary):return '('+n.op+expr(n.a)+')'
        if isinstance(n,c.Call):return n.fn.lower()+'('+','.join(expr(a) for a in n.args)+')'
        if isinstance(n,c.Ternary):return 'select2('+expr(n.cond)+','+expr(n.els)+','+expr(n.then)+')'
        raise ValueError(type(n).__name__)
    for n in nodes:
        if not isinstance(n,c.Assign):raise ValueError(type(n).__name__)
        name=n.target.name.lower();token='sp_'+str(len(lines))+'_'+name
        if isinstance(n.value,c.Ternary) and isinstance(n.value.cond,c.Binary) and n.value.cond.op=='>' and isinstance(n.value.cond.r,c.Var) and n.value.cond.r.name.lower() not in env:
            state=n.value.cond.r.name.lower()
            cuts[state]=(expr(n.value.then),expr(n.value.els),expr(n.value.cond.l))
            continue
        if n.op=='+=' and isinstance(n.value,c.Binary) and n.value.op=='*' and isinstance(n.value.r,c.Binary) and isinstance(n.value.r.r,c.Var) and n.value.r.r.name.lower()==name and name not in env:
            rhs='sp_pole('+expr(n.value.l)+','+expr(n.value.r.l)+')'
        elif name in cuts:rhs='sp_cut('+','.join(cuts[name])+')'
        elif name in ['cc','cs']:continue
        else:
            rhs=expr(n.value)
            if n.op!='=':rhs='('+expr(n.target)+n.op[0]+rhs+')'
        lines.append(token+' = '+rhs+';');env[name]=token
    outside='\n'.join(v[0] for k,v in sections.items() if k!='sample')
    exports=[n for n in env if not n.startswith('spl') and re.search(r'\b'+re.escape(n)+r'\b',outside,re.I)]
    text='''@faust
// Preserve the original scalar recurrence and operation order.
sp_pole(c,x) = x : (step ~ _) : (!,_) with {
    step(previous,value) = next,next with { next=previous+c*(value-previous); };
};
sp_cut(atk,rel,x) = x : (step ~ _) : (!,_) with {
    step(previous,value) = next,next with {
        next=value+select2(value>previous,rel,atk)*(previous-value);
    };
};
'''+ '\n'.join(lines)+'\n'
    text+='\n'.join(n+' = '+env[n]+';' for n in exports)+'\n'
    text+='process = '+env['spl0']+','+env['spl1']+';\n'
    return re.sub(r'(?ms)^@sample[^\n]*\n.*?(?=^@|\Z)',lambda _:text+'\n',source),[],exports

if __name__=='__main__':
    for key in sys.argv[1:]:
        base=ROOT/'build/catalog-faust-audit'/key
        try:
            text,states,exports=(salience if key in ['SaliencePush','ADS'] else candidate)((base/'baseline.jsfx').read_text(encoding='utf-8'))
            (base/'candidate.jsfx').write_text(text,encoding='utf-8')
            print(key,'states',len(states),'exports',len(exports))
        except Exception as e: print(key,'REJECTED',str(e))
