"""Generate the common state ABI and native GFX opcode declarations, using AOT."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dsp_jsfx_aot as compiler
_, metadata = compiler.compile_jsfx_to_ir('@gfx 640 400\ngfx_rect(0,0,gfx_w,gfx_h);', native_gfx_legacy=True, state_var_capacity=0)
# Layout/opcodes are static. Actual names, aliases and strings are instance data.
metadata['has_tasks'] = True
metadata['vars'] = {}
metadata['string_literals'] = []
metadata['named_strings'] = {}
Path(sys.argv[1]).write_text(compiler._emit_header(metadata), encoding='utf-8')
