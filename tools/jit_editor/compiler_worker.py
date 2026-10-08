"""Private, short-lived compiler helper for the standalone Windows JIT Editor.

Only this helper imports Python/llvmlite. The DAW links the resulting native IR.
Uses the production compiler pipeline. This is not a sandbox for untrusted code.
"""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import re
import sys
import time
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import dsp_jsfx_aot as compiler
from llvmlite import binding as llvm

NATIVE = set(re.findall(r"JSFX_RUNTIME_EXPORT\((jsfx_[A-Za-z_0-9]+)\)", (ROOT / "JsfxRuntimeExports.inc").read_text(encoding="utf-8")))

MATH = set("sin cos tan asin acos atan atan2 sinh cosh tanh exp exp2 log log2 log10 pow sqrt floor ceil round trunc fmod remainder fabs copysign fmin fmax memcpy memmove memset malloc calloc free".split())


def walk(value):
    if isinstance(value, compiler.Node):
        yield value
        for child in vars(value).values():
            yield from walk(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from walk(child)


def compile_request(request, directory):
    started = checkpoint = time.perf_counter()
    phases = {}
    def phase(name):
        nonlocal checkpoint
        now=time.perf_counter();phases[name]=now-checkpoint;checkpoint=now
    source = request["source"]
    frontend_backend = request.get("frontend", "python-reference")
    if frontend_backend not in ("python-reference", "cpp-frontend"):
        raise ValueError("Unknown compiler frontend: " + str(frontend_backend))
    native_result = None
    if not isinstance(source, str):
        raise ValueError("Source must be text")
    if request.get("mode", "jsfx") == "faust":
        source = "@faust block\n" + source
    if request.get('mode','jsfx') != 'faust':
        from scripts.jsfx_source import resolve_source
        if (ROOT / "jsfx_eel_pp.exe").exists():os.environ["JSFX_EEL_PP"]=str(ROOT / "jsfx_eel_pp.exe")
        origin=Path(request.get('sourcePath') or directory / 'editor.jsfx').resolve()
        try:
            if frontend_backend == "cpp-frontend":
                from native_frontend import run_native
                native_result = run_native(ROOT, directory, source, origin)
                source = native_result["resolution"]["text"]
            else:
                source=resolve_source(origin,text=source).text
        except Exception as error:
            if "missing import" in str(error).lower():
                raise ValueError("Missing dependency: use Open source... or Source folder... with the original dependency files.\n" + str(error)) from error
            raise
    phase("sourceResolution")
    parse_program = None
    if native_result is not None:
        from native_frontend import run_native, parser_from_result
        plan = compiler.faust_compiler.split(source)
        if plan:
            # The existing Faust planner inserts EEL stage markers. Parse that
            # generated EEL stream natively too; Faust code remains with Faust.
            native_result = run_native(ROOT, directory, plan["text"])
        parse_program = parser_from_result(compiler, native_result)
    pipeline = compiler.prepare_jsfx_pipeline(source, native_gfx_legacy=True,include_serialize=True,
                                              parse_program=parse_program)
    phase("frontend")
    if pipeline.get("faust_plan"):
        pipeline["faust_plan"]["include_paths"] = [str(ROOT.parent / "faust" / "share" / "faust")]
        os.environ["JSFX_FAUST_COMPILER"] = str(ROOT.parent / "faust" / "bin" / "faust.exe")
    module, meta = compiler.compile_jsfx_to_ir(source, pipeline=pipeline, native_gfx_legacy=True, state_var_capacity=0)
    # Native code uses the production GFX services, but its editor adapter must
    # preserve AOT's separate scalar view rather than alias DSP scratch cells.
    ownership=str(compiler.parse_jsfx_options(source).get('ownership','legacy')).lower()
    if ownership in ('auto','hybrid'):
        gfx_flags=compiler.analyze_gfx_var_sync(source,meta['vars'])['flags']
    elif ownership=='ui_only':gfx_flags={name:0 for name in meta['vars']}
    else:gfx_flags={name:3 for name in meta['vars']}
    host_gfx={'gfx_r','gfx_g','gfx_b','gfx_a','gfx_a2','gfx_x','gfx_y','gfx_w','gfx_h','gfx_dest','gfx_mode','gfx_texth','gfx_clear','gfx_ext_retina','gfx_ext_flags','gfx_frame','mouse_x','mouse_y','mouse_cap','mouse_wheel','mouse_hwheel'}
    for name in gfx_flags:
        if name.startswith('__fnlocal__') or name in host_gfx:gfx_flags[name]=0
    if not meta['sections_present'].get('gfx'):gfx_flags={name:0 for name in gfx_flags}
    meta['editor_gfx_var_flags']=gfx_flags
    phase("irEmissionAndFaust")
    editor_sliders=[]
    used={int(n)-1 for n in re.findall(r"(?mi)^\s*slider(\d+)\s*:",source)}
    for stage in meta['faust_stages']:
        for zone in stage.get('zones',[]):
            if zone.get('binding') is not None or zone['type'] not in ('hslider','vslider','nentry','button','checkbox'):continue
            slot=next((n for n in range(256) if n not in used),None)
            if slot is None:raise ValueError('More than 256 combined JSFX/Faust controls')
            used.add(slot);zone['binding']=[1,slot]
            editor_sliders.append(dict(slot=slot,id='faust_'+str(stage['index'])+'_'+hashlib.sha1(zone.get('path',zone['label']).encode()).hexdigest()[:16],label=zone['label'],type=zone['type'],minimum=zone['minimum'],maximum=zone['maximum'],step=zone['step'],default=zone['default']))
    meta['editor_sliders']=editor_sliders
    sliders = [] # Parsed by the shared production C++ slider declaration parser.
    exports = ["jsfx_init", "jsfx_slider", "jsfx_gfx_aot", "jsfx_serialize", "jsfx_process_block"]
    if not meta["has_faust"]:
        exports += ["jsfx_block", "jsfx_sample"]
    for stage in meta["faust_stages"]:
        if stage['kind']!='faust':
            exports.append(stage['name'])
            if stage.get('bulk_name'):exports.append(stage['bulk_name'])
            continue
        name = stage["name"]
        exports += [prefix + name for prefix in ("classInit", "instanceConstants", "instanceClear", "allocate", "destroy", "compute", "bindTables", "seedTables")]
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    # llvmlite's jitdefault code model uses ELF even on Windows. The plugin's
    # default ORC target uses the native Windows COFF layout; match it explicitly.
    target = llvm.Target.from_default_triple().create_target_machine(codemodel="default", jit=False)
    # Unlike AOT plugins, every JIT Program has its own ORC module; its Faust
    # engine is prepared before publication and executes on one audio thread.
    # Module-private bindings therefore cannot be shared across instances.
    # Windows ORC has no emulated-TLS allocator; retain the ownership contract
    # rather than introducing process-global tables or a fake TLS service.
    ir_text = re.sub(r"(?m)^(@__za_tls_[A-Za-z_0-9]+ = internal) thread_local", r"\1", str(module))
    module = llvm.parse_assembly(ir_text)
    module.triple = llvm.get_default_triple()
    module.data_layout = str(target.target_data)
    # Pure Faust can erase every JSFX GEP, leaving only Faust's packed structs.
    # Ask the compiler for its bridge type instead of guessing from struct order.
    emitter = compiler.LLVMModuleEmitter(compiler.SymTable(meta["vars"]), native_gfx_legacy=True, has_tasks=meta["has_tasks"], has_faust=meta["has_faust"], state_var_capacity=0)
    layout = llvm.parse_assembly(str(emitter.module) + "\n@bridge = external global " + str(emitter.state_ty) + "\n")
    state = layout.get_global_variable("bridge").global_value_type
    offsets = {name: target.target_data.get_element_offset(state, index) for index, name in
               ((0, "spl"), (1, "sliders"), (2, "vars"), (5, "srate"), (6, "samplesblock"), (14, "currentBlockSize"), (15, "currentSampleRate"))}
    state_size = target.target_data.get_abi_size(state)
    for function in module.functions:
        if not function.is_declaration and function.name not in exports:
            function.linkage = "internal"
    module.verify()
    phase("irPreparation")
    tuning=llvm.create_pipeline_tuning_options(speed_level=2)
    if len(str(module))>8*1024*1024:tuning.inlining_threshold=32
    with llvm.create_pass_builder(target,tuning) as builder:
        with builder.getModulePassManager() as passes:
            passes.run(module, builder)
    module.verify()
    phase("llvmOptimization")
    # LLVM prints unused external declarations too; only reject ones referenced by
    # a remaining definition/global. This avoids manufacturing host-function stubs.
    bodies = re.sub(r"(?m)^declare [^\n]*\n", "", str(module))
    # Scan once, rather than running one full-IR regex for every declaration.
    referenced = set(re.findall(r'@([A-Za-z_][A-Za-z_0-9.]*)', bodies))
    imports = []
    for function in module.functions:
        if function.is_declaration and function.name in referenced:
            if function.name.startswith("llvm."):
                continue
            if function.name not in MATH and function.name not in NATIVE:
                raise ValueError("Runtime binding missing: " + function.name)
            imports.append(function.name)
    phase("importValidation")
    actual_frontend = "faust" if request.get("mode", "jsfx") == "faust" else frontend_backend
    result = dict(frontend=actual_frontend, phaseSeconds=phases, version=1, stateAbi=17, stateSize=state_size, offsets=offsets, exports=exports, imports=imports,
                  sliders=sliders, expandedSource=source, metadata=meta, compileSeconds=time.perf_counter() - started)
    (directory / "program.ll").write_text(str(module), encoding="utf-8")
    (directory / "program.json").write_text(json.dumps(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    job = Path(sys.argv[1]).resolve()
    try:
        request = json.loads((job / "request.json").read_text(encoding="utf-8-sig"))
        result = compile_request(request, job)
        print(json.dumps({"ok": True, "seconds": result["compileSeconds"]}))
    except Exception as error:
        (job / "error.txt").write_text(str(error), encoding="utf-8")
        print(str(error), file=sys.stderr)
        sys.exit(1)
