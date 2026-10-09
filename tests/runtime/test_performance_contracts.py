"""Backend flags and shared host/IPC contracts; serial, no JUCE build needed."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as compiler

class PerformanceContracts(unittest.TestCase):
    def runtime_output(self):
        out=ROOT/'build/runtime-performance';out.mkdir(parents=True,exist_ok=True)
        _,meta=compiler.compile_jsfx_to_ir('@sample\nspl0*=.5;')
        (out/'JSFXDSP.h').write_text(compiler._emit_header(meta),encoding='utf-8')
        return out
    def test_final_backend_uses_requested_level_without_reoptimizing_ir(self):
        for triple in ('x86_64-pc-windows-msvc','arm64-apple-darwin'):
            for level in range(4):
                with self.subTest(triple=triple,level=level):
                    module,_=compiler.compile_jsfx_to_ir('@sample\nspl0*=.5;')
                    with patch.object(compiler.shutil,'which',return_value='clang'),patch.object(compiler.subprocess,'check_call') as call:
                        compiler._aot_opt_and_emit(module,opt_level=level,target_triple=triple,emit_obj='unused.obj',emit_asm=None)
                        command=call.call_args.args[0]
                    self.assertIn('-O'+str(level),command)
                    self.assertIn('--target='+triple,command)
                    self.assertIn('-disable-llvm-passes',command)
                    self.assertNotIn('-ffast-math',command)
    def test_host_and_ipc(self):
        out=self.runtime_output()
        clang=shutil.which('clang++')
        self.assertIsNotNone(clang,'clang++ is needed for runtime checks')
        for name,sources in [('host_bindings_check',[]),
                             ('shared_memory_check',[ROOT/'src/DspJsfxSharedMemory.cpp']),
                             ('ipc_idle_check',[ROOT/'src/DspJsfxSharedMemory.cpp'])]:
            exe=out/(name+('.exe' if os.name=='nt' else ''))
            command=[clang,'-std=c++20','-O2','-UNDEBUG','-I'+str(ROOT/'src'),'-I'+str(out),str(Path(__file__).with_name(name+'.cpp')),*map(str,sources),'-o',str(exe)]
            if os.name!='nt':command+=['-pthread']
            subprocess.run(command,check=True,cwd=ROOT)
            subprocess.run([str(exe)],check=True,cwd=ROOT,timeout=30)

    @unittest.skipUnless(sys.platform=='linux','Darwin syscall contract shim runs on Linux; macOS runs native checks')
    def test_darwin_shared_memory_contract(self):
        out=self.runtime_output()
        clang=shutil.which('clang++')
        self.assertIsNotNone(clang,'clang++ is needed for runtime checks')
        for name in ('shared_memory_check','ipc_idle_check'):
            exe=out/(name+'-darwin-contract')
            command=[clang,'-std=c++20','-O2','-UNDEBUG','-DZA_JSFX_TEST_DARWIN_SHM=1',
                     '-I'+str(ROOT/'src'),'-I'+str(out),str(Path(__file__).with_name(name+'.cpp')),
                     str(ROOT/'src/DspJsfxSharedMemory.cpp'),str(Path(__file__).with_name('darwin_shm_contract.cpp')),
                     '-Wl,--wrap=shm_open,--wrap=flock,--wrap=close,--wrap=ftruncate','-pthread','-o',str(exe)]
            subprocess.run(command,check=True,cwd=ROOT)
            subprocess.run([str(exe)],check=True,cwd=ROOT,timeout=30)

if __name__=='__main__':unittest.main()
