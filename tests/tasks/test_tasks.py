"""Compiler contracts and linked AOT/runtime integration, without JUCE."""
import sys
import unittest
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if sys.platform=='win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)
sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as c
from llvmlite import binding as llvm

class Tasks(unittest.TestCase):
    def compile(self,text,**kw):
        module,meta=c.compile_jsfx_to_ir(text,**kw)
        llvm.parse_assembly(str(module)).verify()
        return module,meta
    def test_no_task_abi_change_for_existing_scripts(self):
        _,meta=self.compile('@sample\nspl0*=0.5;')
        self.assertFalse(meta['has_tasks']);self.assertNotIn('void* taskContext',c._emit_header(meta))
    def test_nested_loops_and_function_captures(self):
        self.compile('@init\nfunction f(x) local(y) (y=x;defer(loop(10,y+=1;);while(y<20)(y+=1;);defer(y;);););t=f(2);')
    def test_worker_capability(self):
        self.assertFalse(self.compile('@block\nx=task_finished(t);')[1]['has_task_workers'])
        self.assertFalse(self.compile('@init\nb=task_buffer_create(8);task_buffer_seal(b);')[1]['has_task_workers'])
        self.assertFalse(self.compile('@init\nfunction unused() (t=defer(3;););')[1]['has_task_workers'])
        self.assertTrue(self.compile('@init\nfunction later() (t=defer(3;););\n@block\ntrigger ? later();')[1]['has_task_workers'])
        self.assertTrue(self.compile('@init\nt=defer_all(a,b);')[1]['has_task_workers'])
        self.assertTrue(self.compile('@init\na=task_arena_create(64,0,0);')[1]['has_task_workers'])
    def test_unsafe_access_rejected_through_helpers(self):
        for body in ['0[0];','spl0;','slider1;','gfx_rect(0,0,1,1);','rand();','task_buffer_set(b,0,1);']:
            with self.subTest(body=body),self.assertRaisesRegex(ValueError,'Deferred|deferred'):
                self.compile('@init\nfunction unsafe() ('+body+');t=defer(unsafe(););')
    def test_arena_memory_and_slider_snapshot(self):
        self.compile('@init\nfunction f() (0[0]=slider1;memset(4,2,8);fft(16,8););a=task_arena_create(64,0,0);t=defer_arena(a,0,f(););task_arena_commit(a,x);task_arena_preserve(a,0,16);task_arena_adopt(a,x);')
        for body in ['spl0;','slider1=2;','gmem[0];','rand();','sample_pool_loaded(1);']:
            with self.subTest(body=body),self.assertRaises(ValueError):
                self.compile('@init\nt=defer_arena(a,0,'+body+');')
        with self.assertRaises(ValueError):
            self.compile('@init\nt=defer(0[0];);')
    def test_arena_sample_metadata_and_adoption_destinations(self):
        self.compile('@init\nt=defer_arena(a,0,sample_get(p,0);sample_len(p,1);sample_srate(p,1);sample_channels(p,1);sample_peak(p,1););')
        for expr in ['task_arena_adopt(a,x+1);','task_arena_adopt(a,slider1);','task_arena_adopt(a,0[0]);']:
            with self.subTest(expr=expr),self.assertRaises(ValueError):
                self.compile('@init\n'+expr)
    def test_bad_reducer_and_index(self):
        for expr in ['defer_reduce(i,10,BOGUS,0,i;)','defer_for(2,10,3;)']:
            with self.assertRaises(SyntaxError):self.compile('@init\nt='+expr+';')
    def test_gfx_requires_native_execution(self):
        with self.assertRaisesRegex(ValueError,'require'):self.compile('@gfx\nt=defer(3;);')
        self.compile('@gfx\nt=defer(3;);',native_gfx_legacy=True)
        self.compile('@gfx\nt=defer(3;);',native_gfx_prototype=True)
        self.compile('@gfx\n// defer(3;)\ngfx_drawstr("task_status(1)");')
        with self.assertRaisesRegex(ValueError,'require'):
            self.compile('@gfx\nt=defer_all(a,b);')
    def test_private_gfx_writes_are_not_live_dsp_writes(self):
        self.compile('@init\nx=1;\n@sample\nspl0=x;\n@gfx\nt=defer(x+=1;x;);',native_gfx_prototype=True)
    def test_slider_alias_is_host_state(self):
        with self.assertRaisesRegex(ValueError,'host state'):
            self.compile('slider1:gain=1<0,2,0.1>Gain\n@init\nt=defer(gain;);',native_gfx_legacy=True)
    def test_serialization_cannot_silently_discard_tasks(self):
        with self.assertRaisesRegex(ValueError,'@serialize'):
            self.compile('@serialize\nt=defer(3;);')
    def test_linked_runtime(self):
        llvm.initialize_native_target();llvm.initialize_native_asmprinter()
        source=(ROOT/'tests/tasks/tasks.jsfx').read_text()
        for legacy in (False,True):
            with self.subTest(legacy=legacy):
                module,meta=self.compile(source,native_gfx_legacy=legacy)
                target=llvm.Target.from_default_triple().create_target_machine()
                module.triple=llvm.get_default_triple();module.data_layout=str(target.target_data)
                compiled=llvm.parse_assembly(str(module));compiled.verify()
                out=ROOT/'build/tasks'/('legacy' if legacy else 'default');out.mkdir(parents=True,exist_ok=True)
                obj=out/'tasks.obj';ll=out/'tasks.ll';ll.write_text(str(compiled))
                subprocess.run(['clang++','-c',str(ll),'-o',str(obj)],check=True,cwd=ROOT)
                (out/'JSFXDSP.h').write_text(c._emit_header(meta))
                exe=out/'tasks.exe'
                command=['clang++','-std=c++20','-DJSFX_TASKS_TESTING=1','-O1','-UNDEBUG','-I'+str(out),'-I'+str(ROOT/'src'),str(ROOT/'tests/tasks/task_runtime.cpp'),str(obj),'-o',str(exe)]
                subprocess.run(command,check=True,cwd=ROOT)
                subprocess.run([str(exe)],check=True,timeout=30,cwd=ROOT)

if __name__=='__main__':unittest.main()
