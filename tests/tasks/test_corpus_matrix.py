"""Exercise the actual Corpus matrix state machine at its region/task limits."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as compiler
from llvmlite import binding as llvm

source=(ROOT/'plugins/Spectral/Corpus/src/Corpus.jsfx').read_text()
functions=source[source.index('function xc_coarse_similarity'):source.index('function xc_recur_finish_row')]
fixture='''@init
function cp_01(x) (min(1,max(0,x)););
'''+functions+'''
xc_coarse=0;xc_matrix=24576;xc_phase=4;xc_cursor=0;
@block
xc_tasks_retire();
test_restart ? (xc_tasks_cancel();xc_phase=4;xc_cursor=0;test_restart=0;);
xc_phase==4 ? xc_matrix_step();
'''
out=ROOT/'build/tasks/corpus-matrix';out.mkdir(parents=True,exist_ok=True)
module,meta=compiler.compile_jsfx_to_ir(fixture)
llvm.initialize_native_target();llvm.initialize_native_asmprinter()
target=llvm.Target.from_default_triple().create_target_machine()
module.triple=llvm.get_default_triple();module.data_layout=str(target.target_data)
llvm.parse_assembly(str(module)).verify()
(out/'matrix.ll').write_text(str(module));(out/'JSFXDSP.h').write_text(compiler._emit_header(meta))
obj=out/'matrix.obj';exe=out/'matrix.exe'
subprocess.run(['clang++','-c',str(out/'matrix.ll'),'-o',str(obj)],check=True)
subprocess.run(['clang++','-std=c++17','-O1','-UNDEBUG','-I'+str(out),'-I'+str(ROOT/'src'),
                str(ROOT/'tests/tasks/corpus_matrix.cpp'),str(obj),'-o',str(exe)],check=True)
subprocess.run([str(exe)],check=True,timeout=180)
