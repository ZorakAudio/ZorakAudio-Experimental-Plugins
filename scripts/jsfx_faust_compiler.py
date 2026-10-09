"""Build-time Faust LLVM sections and deterministic mixed section plans.
Faust remains the language parser: bridge imports are inferred only from its
undefined-symbol diagnostics, never by rewriting arbitrary Faust expressions.
"""
from __future__ import annotations
import hashlib,json,os,re,shutil,subprocess,tempfile
from pathlib import Path
from llvmlite import binding as llvm

MARKER='__za_faust_stage_'
SECTION=re.compile(r'^\s*@([A-Za-z_][\w]*)\b[^\n]*',re.M)

def split(text):
    matches=list(SECTION.finditer(text))
    if not any(m[1].lower()=='faust' for m in matches): return None
    stages=[]; parts=[text[:matches[0].start()]]
    for i,m in enumerate(matches):
        kind=m[1].lower();body=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
        if kind in ('faust','sample','block'):
            mode=m[0].strip().split()[1:] if kind=='faust' else []
            if mode and not (mode in (['block'],['sample']) or (len(mode)==3 and mode[0] in ('block','sample') and mode[1]=='when' and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*',mode[2]))):raise ValueError('Supported explicit Faust modes are @faust block/sample [when variable]')
            index=len(stages);stages.append({'kind':kind,'source':body,'line':text.count('\n',0,m.start())+2,'index':index,'block_mode':bool(mode and mode[0]=='block'),'condition':mode[2] if len(mode)==3 else None})
            if kind!='faust':parts.append('@'+kind+'\n'+MARKER+str(index)+'();\n'+body)
        else:parts.append(m[0]+body)
    return {'text':'\n'.join(parts),'stages':stages}

def mask(text):
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',lambda m:' '*len(m[0]),text,flags=re.S)

def definitions(source):
    tokens=re.finditer(r'[A-Za-z_]\w*|[{}();=]',mask(source));depth=0;out=[];previous=None
    for t in tokens:
        value=t[0]
        if value=='=' and depth==0 and previous and re.fullmatch(r'[A-Za-z_]\w*',previous):out.append(previous)
        if value in ('{','('):depth+=1
        if value in ('}',')'):depth-=1
        previous=value
    return set(out)

def definition_paths(source):
    clean=mask(source);out=set();roots=definitions(source)
    for root in roots:
        match=re.search(r'\b'+re.escape(root)+r'\s*=\s*environment\s*\{',clean)
        if not match:out.add(root);continue
        begin=match.end();depth=1;end=begin
        while end<len(clean) and depth:
            depth+=(clean[end]=='{')-(clean[end]=='}');end+=1
        out.update(root+'.'+path for path in definition_paths(source[begin:end-1]))
    return out

def injections_for(names):
    tree={}
    for n in names:
        node=tree;pieces=n.split('.')
        for piece in pieces[:-1]:
            node=node.setdefault(piece,{})
            if not isinstance(node,dict):raise ValueError('Conflicting scalar/namespace import: '+n)
        if isinstance(node.get(pieces[-1]),dict):raise ValueError('Conflicting scalar/namespace import: '+n)
        node[pieces[-1]]='hslider("__za_in_'+n+'",0,-1e300,1e300,0)'
    def emit(node):
        return '\n'.join(key+'='+('environment{'+emit(value)+'}' if isinstance(value,dict) else value)+';' for key,value in node.items())
    return emit(tree)

def parse_bitcode(data):
    try:return llvm.parse_bitcode(data)
    except RuntimeError:
        # Faust 2.81.2's Windows CLI writes bitcode through a text-mode stream.
        # Try only the known CRLF translation, and require a valid LLVM module.
        if os.name!='nt' or not data.startswith(b'BC\xc0\xde'):raise
        return llvm.parse_bitcode(data.replace(b'\r\n',b'\n'))

def metadata(module):
    candidates=[g.name for g in module.global_variables if g.name.startswith('{"name"')]
    if len(candidates)!=1:raise ValueError('Faust LLVM backend did not provide its DSP layout JSON')
    text=candidates[0]
    # Windows Faust 2.81.2 fails to JSON-escape its directory lists. These fields
    # are diagnostic paths; DSP layout and labels still use strict JSON parsing.
    text=re.sub(r'"(?:library_list|include_pathnames)"\s*:\s*\[[^\]]*\]',lambda m:m[0].replace('\\','/'),text)
    return json.loads(text)

def isolate_tables(module,name,version):
    # Faust LLVM puts read-only generated tables in mutable module globals.
    # Redirect their addresses through thread-local *pointers*, bound to the
    # current instance before initialization/compute. The actual tables remain
    # instance-owned, so moving an audio callback between threads is safe.
    llvm.initialize_native_target();llvm.initialize_native_asmprinter()
    data=llvm.Target.from_default_triple().create_target_machine().target_data
    globals=[g for g in module.global_variables if re.search(r'=.*\bglobal\b',str(g))]
    sizes=[data.get_abi_size(g.global_value_type) for g in globals]
    text=str(module);lines=[];counter=0;inside=False;generator_type=None
    generator_types={t.name:t for t in module.struct_types if ('struct.dsp'+name+'SIG') in t.name}
    # Faust 2.81.2 LLVM omits fSamplingFreq initialization in SIG generators.
    # Its private SIG layout starts with that i32 field. Initialize it before
    # generator constants are evaluated; otherwise SR-dependent tables use 1 Hz.
    generator_types={n:t for n,t in generator_types.items() if list(t.elements) and str(list(t.elements)[0])=='i32'}
    for line in text.splitlines():
        if line.startswith('define '):
            inside=True;generator_type=None
            if version=='2.81.2':
                match=re.search(r'@instanceInit('+re.escape(name)+r'SIG\d+)\(',line)
                if match:generator_type=next((n for n in generator_types if n=='struct.dsp'+match[1] or n.startswith('struct.dsp'+match[1]+'.')),None)
        if inside:
            for i,g in enumerate(globals):
                symbol='@'+g.name
                if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9.]*',g.name):raise ValueError('Unsupported Faust global table symbol')
                if re.search(re.escape(symbol)+r'(?![A-Za-z_0-9.])',line):
                    local='%__za_table_'+str(counter);counter+=1
                    lines.append('  '+local+' = load ptr, ptr @__za_tls_'+name+'_'+str(i))
                    line=re.sub(re.escape(symbol)+r'(?![A-Za-z_0-9.])',local,line)
                    # Generated Faust uses instruction GEPs. Reject a changed
                    # constant-expression form rather than guessing its type.
                    if 'getelementptr (' in line or 'getelementptr inbounds (' in line:
                        raise ValueError('Unsupported Faust table constant expression; regenerate with a supported LLVM backend')
        lines.append(line)
        if generator_type and re.match(r'entry_block:',line):
            lines+=['  %__za_sig_rate = getelementptr %'+generator_type+', ptr %dsp, i32 0, i32 0',
                    '  store i32 %sample_rate, ptr %__za_sig_rate, align 4']
        if line=='}':inside=False
    for i in range(len(globals)):lines.append('@__za_tls_'+name+'_'+str(i)+' = internal thread_local global ptr null')
    lines+=['define void @bindTables'+name+'(ptr %slots) {','entry:']
    for i in range(len(globals)):
        lines+=['  %slot'+str(i)+' = getelementptr ptr, ptr %slots, i64 '+str(i),
                '  %table'+str(i)+' = load ptr, ptr %slot'+str(i),
                '  store ptr %table'+str(i)+', ptr @__za_tls_'+name+'_'+str(i)]
    lines+=['  ret void','}']
    if globals and '@llvm.memcpy.p0.p0.i64(' not in text:lines.append('declare void @llvm.memcpy.p0.p0.i64(ptr,ptr,i64,i1)')
    lines+=['define void @seedTables'+name+'(ptr %slots) {','entry:']
    for i,g in enumerate(globals):
        lines+=['  %slot'+str(i)+' = getelementptr ptr, ptr %slots, i64 '+str(i),
                '  %table'+str(i)+' = load ptr, ptr %slot'+str(i),
                '  call void @llvm.memcpy.p0.p0.i64(ptr %table'+str(i)+', ptr @'+g.name+', i64 '+str(sizes[i])+', i1 false)']
    lines+=['  ret void','}']
    result=llvm.parse_assembly('\n'.join(lines));result.verify();return result,sizes

def compile_stage(stage,known,aliases,faust_cmd=None,include_paths=()):
    executable=faust_cmd or os.environ.get('JSFX_FAUST_COMPILER') or shutil.which('faust')
    if not executable:raise ValueError('@faust requires the Faust compiler with its LLVM backend on PATH (or JSFX_FAUST_COMPILER)')
    if Path(executable).is_file():executable=str(Path(executable).resolve())
    source=stage['source'];defs=definitions(source)
    exports=sorted(n for n in definition_paths(source) if n.lower() in known or n.lower() in aliases)
    if any(n.startswith('__za_') for n in defs):raise ValueError('@faust reserves the __za_ namespace for its bridge')
    key=hashlib.sha256(source.encode()).hexdigest()[:12];name='ZaFaust'+str(stage['index'])+'_'+key
    imported=[];module=None
    with tempfile.TemporaryDirectory(prefix='jsfx-faust-') as scratch:
        p=Path(scratch)/'section.dsp';bc=p.with_suffix('.bc')
        # The Windows Faust executable uses narrow argv for file operations.
        # Keep source/output arguments relative to its Unicode-aware process cwd.
        # Mirror only include roots that it cannot address, without relying on
        # NTFS short names being enabled on the user's installation volume.
        compiler_includes=[]
        for index,directory in enumerate(include_paths):
            directory=Path(directory).resolve()
            if os.name=='nt' and not str(directory).isascii():
                include=Path(scratch)/('include_'+str(index))
                shutil.copytree(directory,include)
                compiler_includes.append(include.name)
            else:compiler_includes.append(str(directory))
        for attempt in range(257):
            signals=[n for n in imported if re.fullmatch(r'spl\d+',n.lower()) or n.lower() in stage.get('stream_vars',set())]
            controls=[n for n in imported if n not in signals]
            injections=[injections_for(controls)]
            injections+=['%s=__za_signal_%d;'%(n,i) for i,n in enumerate(signals)]
            params=','.join('__za_signal_'+str(i) for i in range(len(signals)))
            selection=','.join(['__za_env.process']+['__za_env.'+n for n in exports])
            code='process'+('('+params+')' if signals else '')+'=('+selection+') with { __za_env=environment {\n'+source+'\n'+'\n'.join(injections)+'\n}; };\n'
            p.write_text(code,encoding='utf-8')
            env=dict(os.environ);env['FAUST_DEBUG']='FAUST_LLVM_NO_FM'
            try:
                result=subprocess.run([executable,'-lang','llvm','-double','-mdd','2147483647','-cn',name,*[item for directory in compiler_includes for item in ('-I',directory)],'-o',bc.name,p.name],cwd=scratch,capture_output=True,text=True,env=env,timeout=120)
            except subprocess.TimeoutExpired as exc:
                raise ValueError('@faust at line %d exceeded the 120-second build limit '
                                 '(inferred imports=%d, scalar exports=%d). '
                                 'Reduce or partition this expression graph; no replacement DSP was generated.' %
                                 (stage['line'], len(imported), len(exports))) from exc
            if result.returncode==0:
                module=parse_bitcode(bc.read_bytes());module.verify();break
            missing=re.search(r'undefined symbol\s*:\s*([A-Za-z_]\w*)',result.stderr)
            if not missing or missing[1] in imported:
                raise ValueError('@faust at line %d: %s'%(stage['line'],result.stderr.strip()))
            n=missing[1]
            namespace=[field for field in re.findall(r'\b'+re.escape(n)+r'(?:\.[A-Za-z_]\w*)+',mask(source)) if field.lower() in known]
            if namespace:
                additional=[field for field in dict.fromkeys(namespace) if field not in imported]
                if not additional:raise ValueError('Conflicting Faust namespace binding: '+n)
                imported.extend(additional);continue
            valid=n.lower() in known or n.lower() in aliases or n.lower() in ('srate','samplesblock') or re.fullmatch(r'(?:slider(?:[1-9]\d{0,2})|spl(?:[0-9]|[1-5][0-9]|6[0-3]))',n.lower())
            if not valid:raise ValueError('@faust at line %d: undefined Faust/JSFX variable %s'%(stage['line'],n))
            if n.lower().startswith('slider') and int(n[6:])>256:raise ValueError('@faust slider index exceeds 256')
            imported.append(n)
        if module is None:raise ValueError('@faust import limit is 256 symbols')
    info=metadata(module);module,table_sizes=isolate_tables(module,name,info.get('version'));audio_outputs=int(info['outputs'])-len(exports)
    if audio_outputs<0 or audio_outputs>64 or info['inputs']-len(signals)>64 or info['inputs']>128 or info['outputs']>128 or info['size']+4*sum(table_sizes)>256*1024*1024:
        raise ValueError('@faust exceeds the 64 audio channels / 128 ports / 256 MiB DSP bounds '
                         '(inputs=%s, outputs=%s, inferred audio inputs=%s, scalar exports=%s, bytes=%s)' %
                         (info['inputs'], info['outputs'], len(signals), len(exports), info['size']+4*sum(table_sizes)))
    zones=[]
    def walk(nodes,path=""):
        for node in nodes:
            if 'items' in node:walk(node['items'],path+"/"+node.get("label",""))
            elif 'index' in node:
                label=node['label'];external=label[len('__za_in_'):] if label.startswith('__za_in_') else None
                zones.append({'offset':int(node['index']),'default':float(node.get('init',0)),'variable':external,'type':node['type'],'label':label,'path':path+'/'+label,'minimum':float(node.get('min',0)),'maximum':float(node.get('max',1)),'step':float(node.get('step',1 if node['type'] in ('button','checkbox') else 0))})
    walk(info['ui'])
    if table_sizes:
        missing_controls=set(controls)-{z['variable'] for z in zones if z['variable']}
        if missing_controls:raise ValueError('@faust: JSFX controls cannot initialize generated tables: '+', '.join(sorted(missing_controls)))
    for z in zones:
        if z['offset']<0 or z['offset']+8>info['size']:raise ValueError('Invalid Faust LLVM control layout')
    return {'kind':'faust','index':stage['index'],'line':stage['line'],'name':name,'inputs':int(info['inputs']),'outputs':int(info['outputs']),'audio_outputs':audio_outputs,'size':int(info['size']),'table_sizes':table_sizes,'zones':zones,'imports':imported,'signals':signals,'exports':exports,'condition':stage.get('condition'),'capture_stage':stage.get('capture_stage',-1),'block_mode':stage.get('block_mode',False),'backend':'llvm','faust_version':info.get('version'),'source_sha256':hashlib.sha256(source.encode()).hexdigest()},module

def stage_nodes(c,pipeline,plan):
    found={s['index']:[] for s in plan['stages'] if s['kind']!='faust'}
    for kind in ('block','sample'):
        current=None
        for node in pipeline['programs'][kind]:
            if isinstance(node,c.Call) and node.fn.startswith(MARKER):current=int(node.fn[len(MARKER):]);continue
            if current is None:raise ValueError('Mixed DSP section lost its stage marker')
            found[current].append(node)
        pipeline['programs'][kind]=[n for values in [found[s['index']] for s in plan['stages'] if s['kind']==kind] for n in values]
    return found

def binding(name,known,aliases):
    name=name.lower()
    if name in aliases:return (1,aliases[name],known.get(name,-1))
    if re.fullmatch(r'slider\d+',name):return (1,int(name[6:])-1)
    if re.fullmatch(r'spl\d+',name):return (2,int(name[3:]))
    if name=='srate':return (3,0)
    if name=='samplesblock':return (4,0)
    return (0,known[name])

def attach(c,emitter,plan,nodes,known,aliases,fn_defs):
    descriptors=[];modules=[];sample_rw={}
    for stage in plan['stages']:
        if stage['kind']=='faust':
            if stage.get('block_mode'):
                preceding=[p for p in plan['stages'] if p['kind']=='sample' and p['index']<stage['index']]
                if preceding:
                    owner=preceding[-1]['index'];stage['capture_stage']=owner
                    stage['stream_vars']={v.lower() for v in sample_rw[owner]['writes']}
            desc,module=compile_stage(stage,known,aliases,include_paths=plan.get('include_paths',()));descriptors.append(desc);modules.append(module)
        else:
            name='jsfx_mixed_stage_'+str(stage['index']);emitter.emit_section_fn(name,nodes[stage['index']])
            descriptors.append({'kind':stage['kind'],'index':stage['index'],'line':stage['line'],'name':name})
            if stage['kind']=='sample':
                rw=c._summarize_section_rw(nodes[stage['index']],set(fn_defs))
                # Include every reachable helper, conservatively. Aliased heap and
                # shared scalar dependencies force a sample-accurate island.
                def helpers(items,seen):
                    if isinstance(items,c.Call) and items.fn in fn_defs and items.fn not in seen:
                        seen.add(items.fn);helpers(fn_defs[items.fn].body,seen)
                    if isinstance(items,c.Node):
                        for value in vars(items).values():helpers(value,seen)
                    elif isinstance(items,(list,tuple)):
                        for value in items:helpers(value,seen)
                seen=set();helpers(nodes[stage['index']],seen)
                writes=set(rw.written_vars);reads=set(rw.read_vars)
                for fn in seen:
                    extra=c._summarize_section_rw([fn_defs[fn].body],set(fn_defs));writes|=extra.written_vars;reads|=extra.read_vars
                sample_rw[stage['index']]={'writes':writes,'reads':reads,'uncertain':rw.writes_unknown_state or rw.writes_dynamic_slider}
    islands=[];current=[]
    def finish():
        if not current:return
        samples=[x for x in current if x['kind']=='sample'];faust=[x for x in current if x['kind']=='faust']
        writes=set().union(*(sample_rw[x['index']]['writes'] for x in samples)) if samples else set()
        reads=set().union(*(sample_rw[x['index']]['reads'] for x in samples)) if samples else set()
        imports=set().union(*(set(x['imports'])-set(x['signals']) for x in faust)) if faust else set()
        exports=set().union(*(set(x['exports']) for x in faust)) if faust else set()
        # Two EEL sections can alias arbitrary heap state. Never reorder them.
        keys=lambda names:{tuple(binding(n,known,aliases)[:2]) for n in names if n.lower() in known or n.lower() in aliases or re.fullmatch(r'slider\d+',n.lower()) or n.lower() in ('srate','samplesblock')}
        # Audio ports already carry whole sample streams. Only scalar/control
        # dependencies and potentially aliased EEL state require interleaving.
        fused=len(samples)>1 or bool(keys(writes) & keys(imports)) or bool(keys(reads) & keys(exports)) or bool(keys(exports) & keys(imports))
        if imports and any(sample_rw[x['index']]['uncertain'] for x in samples):fused=True
        if fused and any(x.get('block_mode') for x in faust):
            raise ValueError('@faust block has unresolved sample interleaving; separate its stages with an explicit @block boundary')
        island=len(islands)
        for x in current:x['island']=island;x['fused']=fused
        islands.append({'fused':fused,'stages':[x['index'] for x in current]});current.clear()
    for x in descriptors:
        if x['kind']=='block':finish();x['island']=-1;x['fused']=False
        else:current.append(x)
    finish()
    # Bindings are numerical state indices, not shared external globals.
    for x in descriptors:
        if x['kind']=='faust':
            for z in x['zones']:z['binding']=binding(z['variable'],known,aliases) if z['variable'] else None
            x['signal_bindings']=[binding(n,known,aliases) for n in x['signals']]
            condition=x.get('condition')
            if condition:
                low=condition.lower()
                slider=re.fullmatch(r'slider([1-9]\d*)',low)
                if not (low in known or low in aliases or low in ('srate','samplesblock') or (slider and int(slider[1])<=256)):
                    raise ValueError('@faust block when requires a known scalar control: '+condition)
            x['condition_binding']=binding(condition,known,aliases) if condition else (-1,0)
            if x.get('condition') and any(tuple(binding(v,known,aliases)[:2])==tuple(x['condition_binding'][:2]) for rw in sample_rw.values() for v in rw['writes'] if v.lower() in known or v.lower() in aliases):
                raise ValueError('@faust block when condition must not be written by a sample stage')
            x['export_bindings']=[binding(n,known,aliases) for n in x['exports']]
    stream=0
    for consumer in descriptors:
        for source in consumer.get('signal_bindings',[]):
            if source[0]==2:continue
            owner=next((s for s in descriptors if s['index']==consumer.get('capture_stage')),None)
            if owner is not None:owner.setdefault('captures',[]).append({'stream':stream,'binding':source})
            stream+=1
    for stage in descriptors:
        if stage['kind']=='sample':emit_bulk_sample(c,emitter,stage)
    return descriptors,modules,islands


def emit_bulk_sample(c,e,stage):
    """Keep the EEL frame loop in LLVM, as in the ordinary native backend."""
    sample=e.module.globals[stage['name']];sample.attributes.add('alwaysinline')
    stage['bulk_name']=stage['name']+'_bulk'
    doublep=e.double.as_pointer()
    fn=e._function(e.module,c.ir.FunctionType(c.ir.VoidType(),[
        e.state_ptr,doublep,e.i32,e.i32,e.i32,e.i32,doublep.as_pointer()]),name=stage['bulk_name'])
    b=c.ir.IRBuilder(fn.append_basic_block('entry'))
    state,audio,capacity,first,count,channels,streams=fn.args
    zero=c.ir.Constant(e.i32,0);one=c.ir.Constant(e.i32,1)
    captured=[(item,b.load(b.gep(streams,[c.ir.Constant(e.i32,item['stream'])]))) for item in stage.get('captures',[])]
    def loop(start,end,body,label):
        pre=b.block;cond=fn.append_basic_block(label+'_cond');work=fn.append_basic_block(label+'_body');done=fn.append_basic_block(label+'_end')
        b.branch(cond);b.position_at_end(cond)
        index=b.phi(e.i32);index.add_incoming(start,pre)
        b.cbranch(b.icmp_signed('<',index,end),work,done);b.position_at_end(work)
        body(index)
        nxt=b.add(index,one);index.add_incoming(nxt,b.block);b.branch(cond);b.position_at_end(done)
    def frame(index):
        def input_channel(channel):
            value=b.load(b.gep(audio,[b.add(b.mul(channel,capacity),index)]))
            e._store_cell(b,value,b.gep(state,[zero,zero,channel],inbounds=True))
        loop(zero,channels,input_channel,'input')
        b.call(sample,[state])
        for item,target in captured:
            with b.if_then(b.icmp_unsigned('!=',target,c.ir.Constant(doublep,None))):
                kind,slot,*_=item['binding']
                field={0:2,1:1,3:5,4:6}[kind]
                indices=[zero,c.ir.Constant(e.i32,field)]
                pointer=b.gep(state,indices,inbounds=True)
                if kind==0 and e.native_gfx_legacy:
                    pointer=b.gep(b.load(pointer),[c.ir.Constant(e.i32,slot)],inbounds=True)
                elif kind in (0,1):
                    pointer=b.gep(state,indices+[c.ir.Constant(e.i32,slot)],inbounds=True)
                value=e._load_cell(b,pointer)
                b.store(value,b.gep(target,[index]))
        def output_channel(channel):
            value=e._load_cell(b,b.gep(state,[zero,zero,channel],inbounds=True))
            b.store(value,b.gep(audio,[b.add(b.mul(channel,capacity),index)]))
        loop(zero,channels,output_channel,'output')
    loop(first,b.add(first,count),frame,'frame');b.ret_void()

def emit_process(c,e):
    f32pp=c.ir.FloatType().as_pointer().as_pointer()
    ty=c.ir.FunctionType(c.ir.VoidType(),[e.state_ptr,f32pp,f32pp,e.i32,e.i32])
    fn=e._function(e.module,ty,name='jsfx_process_block');external=e._function(e.module,ty,name='jsfx_faust_process')
    b=c.ir.IRBuilder(fn.append_basic_block('entry'));b.call(external,list(fn.args));b.ret_void()

def merge(module,extras):
    result=llvm.parse_assembly(str(module))
    for extra in extras:result.link_in(extra)
    result.verify();return result

def header(meta):
    if not meta.get('has_faust'):return ''
    stages=meta['faust_stages'];out=['#ifdef __cplusplus','#include "JsfxFaustPlan.h"','namespace jsfx_faust {','using Binding=za::jsfx::Binding; using Zone=za::jsfx::Zone; using Stage=za::jsfx::Stage;','inline constexpr int quantum='+str(meta.get('faust_quantum',0))+';']
    out.append('inline constexpr Stage eelStage(int kind,int island,bool fused,void (*eel)(DSPJSFX_State*),void (*bulk)(DSPJSFX_State*,double*,int,int,int,int,double**)) { Stage s{};s.kind=kind;s.island=island;s.fused=fused;s.eel=eel;s.bulk=bulk;return s; }')
    def arr(name,values,typ):
        out.append('inline constexpr '+typ+' '+name+'[] = {'+','.join(values or ['{}'])+'};');return name
    for s in stages:
        name=s['name']
        if s['kind']=='faust':
            out+=['extern "C" void '+prefix+name+args+';' for prefix,args in [('classInit','(int)'),('instanceConstants','(void*,int)'),('instanceClear','(void*)'),('allocate','(void*)'),('destroy','(void*)'),('compute','(void*,int,double**,double**)'),('bindTables','(void**)'),('seedTables','(void**)')]]
            arr('tables_'+name,[str(n) for n in s['table_sizes']],'int')
            arr('zones_'+name,['{'+str(z['offset'])+','+repr(z['default'])+',{'+','.join(map(str,z['binding'] or (-1,0)))+'}}' for z in s['zones']],'Zone')
            arr('signals_'+name,['{'+','.join(map(str,b))+'}' for b in s['signal_bindings']],'Binding')
            arr('exports_'+name,['{'+','.join(map(str,b))+'}' for b in s['export_bindings']],'Binding')
        else:
            out.append('extern "C" void '+name+'(DSPJSFX_State*);')
            if s.get('bulk_name'):out.append('extern "C" void '+s['bulk_name']+'(DSPJSFX_State*,double*,int,int,int,int,double**);')
    rows=[]
    for s in stages:
        if s['kind']!='faust':rows.append('eelStage('+','.join([str(0 if s['kind']=='block' else 1),str(s['island']),str(s['fused']).lower(),'&'+s['name'],'&'+s['bulk_name'] if s.get('bulk_name') else 'nullptr'])+')')
        else:
            n=s['name'];values=['2',str(s['island']),str(s['fused']).lower(),'nullptr',str(s['size']),str(s['inputs']),str(s['outputs']),str(s['audio_outputs'])]+['&'+p+n for p in ['classInit','instanceConstants','instanceClear','allocate','destroy','compute']]+['zones_'+n,str(len(s['zones'])),'signals_'+n,str(len(s['signals'])),'exports_'+n,str(len(s['exports'])),'tables_'+n,str(len(s['table_sizes'])),'&bindTables'+n,'&seedTables'+n,str(s.get('capture_stage',-1)),'{'+','.join(map(str,s.get('condition_binding',(-1,0))))+'}']
            rows.append('{'+','.join(values)+'}')
    out.append('inline constexpr Stage stages[] = {'+','.join(rows)+'};');out+=['}','#endif']
    return '\n'.join(out)+'\n'
