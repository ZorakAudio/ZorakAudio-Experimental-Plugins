"""Structured task syntax, safety validation and LLVM emission helpers."""
from dataclasses import fields, is_dataclass

DEFERRED = {'defer': 0, 'defer_after': 1, 'defer_for': 2, 'defer_reduce': 3, 'defer_arena': 4}
APIS = {'task_status': (0,1), 'task_finished': (1,1), 'task_result': (2,1),
        'task_cancel': (3,1), 'task_release': (4,1), 'task_value': (5,2),
        'defer_all': (6,None), 'task_buffer_create': (10,1),
        'task_buffer_set': (11,3), 'task_buffer_read': (12,2),
        'task_buffer_seal': (13,1), 'task_buffer_release': (14,1),
        'task_arena_create': (20,3), 'task_arena_status': (21,1),
        'task_arena_copy_in': (22,3), 'task_arena_seal': (23,1),
        'task_arena_copy_out': (24,3), 'task_arena_get': (25,2),
        'task_arena_release': (26,1), 'task_report': (27,4),
        'task_arena_progress': (28,2), 'task_arena_commit': (29,None), 'task_arena_clone': (30,3),
        'task_arena_preserve': (31,3), 'task_arena_adopt': (32,None)}
REDUCTIONS = {'sum':2, 'product':3, 'min':4, 'max':5, 'any':6, 'all':7}
CONSTANTS = {'task_invalid':-1, 'task_busy':0, 'task_pending':1, 'task_running':2,
             'task_succeeded':3, 'task_cancelled':4, 'task_failed':5}

def walk(value):
    yield value
    if isinstance(value,(list,tuple)):
        for item in value: yield from walk(item)
    elif is_dataclass(value):
        for f in fields(value):
            if f.name not in ('id','span'): yield from walk(getattr(value,f.name))

def parse(c,p,fn,span):
    args=[]
    prefix={'defer':0,'defer_after':1,'defer_for':2,'defer_reduce':4,'defer_arena':2}[fn]
    for i in range(prefix):
        p._skip_seps();arg=p.parse_expr(0)
        if fn in ('defer_for','defer_reduce') and i==0 and not isinstance(arg,c.Var):
            raise SyntaxError('Deferred iteration requires an index variable')
        if fn in ('defer_for','defer_reduce') and i==0 and (
                arg.name in c.BUILTIN_NAMES or arg.name in CONSTANTS or
                arg.name.startswith(('#','gfx_','mouse_')) or
                c._is_slider_var_name(arg.name) or c._is_spl_var_name(arg.name)):
            raise SyntaxError('Deferred iteration requires an ordinary private index variable')
        if fn=='defer_reduce' and i==2:
            if not isinstance(arg,c.Var) or arg.name not in REDUCTIONS:
                raise SyntaxError('Reduction must be SUM, PRODUCT, MIN, MAX, ANY or ALL')
            arg=c.Num(p._new_id(),arg.span,float(REDUCTIONS[arg.name]))
        args.append(arg);p._skip_seps();p._eat('punc',',')
    p._skip_seps();body=[]
    while not(p.cur.kind=='punc' and p.cur.text==')'):
        body.append(p.parse_stmt_or_expr_for_seq());p._skip_seps()
    p._adv()
    args.append(c.Seq(p._new_id(),span,body) if body else c.Num(p._new_id(),span,0.))
    return c.Call(p._new_id(),span,fn,args)

def validate(c,programs,functions,slider_aliases=()):
    tasks=[v for v in walk([programs.get(k,[]) for k in programs]+[f.body for f in functions.values()])
           if isinstance(v,c.Call) and v.fn in DEFERRED]
    pure=set(c._PURE_HOIST_CALLS)|{'abs'}
    pure.discard('__memtop')
    allowed=pure|set(DEFERRED)|{'defer_all','task_status','task_finished','task_result','task_value','task_buffer_read'}
    def check(body,seen,arena=False):
        for v in walk(body):
            if isinstance(v,c.StrLit) or isinstance(v,c.Index) and not arena:
                raise ValueError('Deferred bodies cannot access live indexed memory or strings; use sealed task buffers')
            if isinstance(v,c.Var) and (v.name.startswith(('#','gfx_','mouse_')) or (v.name in slider_aliases and not arena) or v.name in {'gmem','mem','midi_bus','ext_midi_bus'} or (c._is_slider_var_name(v.name) and not arena) or c._is_spl_var_name(v.name)):
                raise ValueError('Deferred body cannot access host state: '+v.name)
            if isinstance(v,c.Assign) and isinstance(v.target,c.Var) and (v.target.name in c.BUILTIN_NAMES or c._is_slider_var_name(v.target.name) or v.target.name in slider_aliases):
                raise ValueError('Deferred body cannot write host state: '+v.target.name)
            if isinstance(v,c.Call):
                if v.fn in functions:
                    if v.fn in seen: raise ValueError('Recursive deferred helpers are unsupported')
                    check(functions[v.fn].body,seen|{v.fn},arena)
                elif v.fn not in allowed and not (arena and v.fn in {'memset','memcpy','fft','ifft','fft_permute','fft_ipermute','fft_real','ifft_real','__memtop','sample_read2','sample_get','sample_len','sample_srate','sample_channels','sample_peak','task_report'}):
                    raise ValueError('Unsafe or unsupported deferred builtin: '+v.fn)
    for task in tasks:check(task.args[-1],set(),task.fn=='defer_arena')
    enabled=bool(tasks or any(isinstance(v,c.Call) and v.fn in APIS for v in walk(list(programs.values())+[f.body for f in functions.values()])))
    if enabled:
        def serialization_uses_tasks(body, seen):
            for value in walk(body):
                if isinstance(value, c.Call):
                    if value.fn in DEFERRED or value.fn in APIS:
                        return True
                    if value.fn in functions and value.fn not in seen and serialization_uses_tasks(functions[value.fn].body, seen | {value.fn}):
                        return True
            return False
        if serialization_uses_tasks(programs.get('serialize', []), set()):
            raise ValueError('Tasks are not supported in @serialize; task handles are transient')
        for value in walk(list(programs.values())+[f.body for f in functions.values()]):
            if isinstance(value,c.Assign) and isinstance(value.target,c.Var) and value.target.name in CONSTANTS:
                raise ValueError('Task status constants are read-only')
    return enabled

def emit_api(c,e,b,st,name,values):
    ir=c.ir
    callee=e._buildins.get('jsfx_task_api')
    if callee is None:
        callee=e._function(e.module,ir.FunctionType(e.double,[e.state_ptr,e.i32,e.double.as_pointer(),e.i32]),name='jsfx_task_api')
        e._buildins['jsfx_task_api']=callee
    slot=e._task_entry_alloca(b,ir.ArrayType(e.double,max(1,len(values))))
    for i,v in enumerate(values):b.store(v,b.gep(slot,[e._const_i32(0),e._const_i32(i)]))
    return b.call(callee,[st,e._const_i32(APIS[name][0] if name in APIS else 50),b.gep(slot,[e._const_i32(0),e._const_i32(0)]),e._const_i32(len(values))])

def emit(c,e,b,st,node):
    ir=c.ir
    if node.fn in APIS:
        _,arity=APIS[node.fn]
        if arity is not None and len(node.args)!=arity or arity is None and not 1<=len(node.args)<=(4097 if node.fn in ('task_arena_commit','task_arena_adopt') else 32):
            raise ValueError('Invalid task API argument count: '+node.fn)
        if node.fn in ('task_arena_commit','task_arena_adopt'):
            if not all(isinstance(a,c.Var) and e.sym.resolve(a.name).kind=='var' for a in node.args[1:]):
                raise ValueError('task_arena_commit requires global variable names')
            return emit_api(c,e,b,st,node.fn,[e.emit_expr(b,st,node.args[0])]+[e._const_f64(e.sym.resolve(a.name).index) for a in node.args[1:]])
        if node.fn=='task_arena_get':
            if not isinstance(node.args[1],c.Var) or e.sym.resolve(node.args[1].name).kind!='var':
                raise ValueError('task_arena_get requires a global variable name')
            return emit_api(c,e,b,st,node.fn,[e.emit_expr(b,st,node.args[0]),e._const_f64(e.sym.resolve(node.args[1].name).index)])
        return emit_api(c,e,b,st,node.fn,[e.emit_expr(b,st,a) for a in node.args])
    # Capture surrounding function locals explicitly. Globals live in a private
    # runtime snapshot, including globals used indirectly by helper functions.
    locals_map={}
    for scope in e._local_slots_stack:locals_map.update(scope)
    if len(locals_map)>64:raise ValueError('Deferred capture limit is 64 function locals')
    captured=[(name,e._load_cell(b,ptr)) for name,ptr in locals_map.items()]
    arena=e.emit_expr(b,st,node.args[0]) if node.fn=='defer_arena' else None
    dependency=e.emit_expr(b,st,node.args[1] if arena is not None else node.args[0]) if node.fn in ('defer_after','defer_arena') else e._const_f64(0)
    count=e.emit_expr(b,st,node.args[1]) if node.fn in ('defer_for','defer_reduce') else e._const_f64(1)
    mode=int(node.args[2].value) if node.fn=='defer_reduce' else 1 if node.fn=='defer_for' else 0
    identity=e.emit_expr(b,st,node.args[3]) if node.fn=='defer_reduce' else e._const_f64(0)
    callback_type=ir.FunctionType(e.double,[e.state_ptr,e.double,e.double.as_pointer()])
    callback=e._function(e.module,callback_type,name='jsfx_deferred_'+str(len(getattr(e,'_task_callbacks',[]))))
    if not hasattr(e,'_task_callbacks'):e._task_callbacks=[]
    e._task_callbacks.append(callback)
    cb=ir.IRBuilder(callback.append_basic_block('entry'))
    slots={}
    for i,(name,_) in enumerate(captured):
        ptr=cb.alloca(e.double);cb.store(cb.load(cb.gep(callback.args[2],[e._const_i32(i)])),ptr);slots[name]=ptr
    if node.fn in ('defer_for','defer_reduce'):
        name=node.args[0].name;ptr=cb.alloca(e.double);cb.store(callback.args[1],ptr);slots[name]=ptr
    saved_gfx=e._emitting_gfx;saved_stack=e._local_slots_stack;saved_hoists=e._hoisted_value_stack
    e._emitting_gfx=False;e._local_slots_stack=[slots];e._hoisted_value_stack=[]
    try:
        checkpoint(c,e,cb,callback.args[0])
        result=e.emit_expr(cb,callback.args[0],node.args[-1]);cb.ret(result)
    finally:e._emitting_gfx=saved_gfx;e._local_slots_stack=saved_stack;e._hoisted_value_stack=saved_hoists
    # Do not put an alloca inside an audio sample loop: allocate in entry once.
    slot=e._task_entry_alloca(b,ir.ArrayType(e.double,max(1,len(captured))))
    for i,(_,v) in enumerate(captured):b.store(v,b.gep(slot,[e._const_i32(0),e._const_i32(i)]))
    if arena is not None:
        callee=e._buildins.get('jsfx_task_submit_arena')
        if callee is None:
            callee=e._function(e.module,ir.FunctionType(e.double,[e.state_ptr,callback_type.as_pointer(),e.double.as_pointer(),e.i32,e.double,e.double]),name='jsfx_task_submit_arena')
            e._buildins['jsfx_task_submit_arena']=callee
        return b.call(callee,[st,callback,b.gep(slot,[e._const_i32(0),e._const_i32(0)]),e._const_i32(len(captured)),arena,dependency])
    callee=e._buildins.get('jsfx_task_submit')
    if callee is None:
        callee=e._function(e.module,ir.FunctionType(e.double,[e.state_ptr,callback_type.as_pointer(),e.double.as_pointer(),e.i32,e.double,e.double,e.i32,e.double]),name='jsfx_task_submit')
        e._buildins['jsfx_task_submit']=callee
    return b.call(callee,[st,callback,b.gep(slot,[e._const_i32(0),e._const_i32(0)]),e._const_i32(len(captured)),dependency,count,e._const_i32(mode),identity])

def checkpoint(c,e,b,st):
    cancelled=emit_api(c,e,b,st,'checkpoint',[])
    with b.if_then(b.fcmp_ordered('!=',cancelled,e._const_f64(0))):
        if isinstance(b.function.function_type.return_type,c.ir.VoidType):b.ret_void()
        else:b.ret(e._const_f64(0))
