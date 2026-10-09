"""Compiler and source expansion contracts for embedded Faust."""
from pathlib import Path
import tempfile,sys,unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import dsp_jsfx_aot as c
from scripts.jsfx_source import resolve_source
class Contracts(unittest.TestCase):
 def test_conditional_block_rejects_sample_written_gate(self):
  with self.assertRaisesRegex(ValueError,'condition must not be written'):
   c.compile_jsfx_to_ir('@init\ngate=0;\n@sample\ngate=1;\n@block\nmarker=0;\n@faust block when gate\nprocess=_,_;')
 def test_block_rejects_sample_feedback(self):
  with self.assertRaisesRegex(ValueError,'unresolved sample interleaving'):
   c.compile_jsfx_to_ir('@init\nmeter=0;\n@faust block\nmeter=abs(spl0);process=spl0,spl1;\n@sample\nspl0=meter;')
 def test_block_streams_are_not_host_audio_ports(self):
  _,m=c.compile_jsfx_to_ir('@init\ngain=0.5;\n@sample\ngain+=0.01;\n@block\ngain=100;\n@faust block\nprocess=spl0*gain,spl1*gain;')
  stage=m['faust_stages'][-1]
  self.assertIn('gain',stage['signals']);self.assertFalse(stage['fused'])
  self.assertEqual(stage['capture_stage'],0)
  self.assertEqual(m['io_channels']['inputs'],2)
 def test_compile_timeout_names_section_and_preserves_limit(self):
  from unittest.mock import patch
  import subprocess
  from scripts.jsfx_faust_compiler import compile_stage
  with patch('scripts.jsfx_faust_compiler.subprocess.run',side_effect=subprocess.TimeoutExpired('faust',120)) as command:
   with self.assertRaisesRegex(ValueError,r'@faust at line 42 exceeded the 120-second build limit'):
    compile_stage({'source':'process=_,_;','index':0,'line':42},{},{},faust_cmd='faust')
   self.assertEqual(command.call_args.kwargs['timeout'],120)
 def test_unknown(self):
  with self.assertRaisesRegex(ValueError,'undefined Faust/JSFX variable typo'):
   c.compile_jsfx_to_ir('@faust\nprocess=_,_:*(typo),_;')
 def test_local_precedence(self):
  _,m=c.compile_jsfx_to_ir('@init\ngain=0.5;\n@faust\ngain=0.25;process=_,_:*(gain),_;')
  self.assertEqual(m['faust_stages'][0]['imports'],[])
  self.assertEqual(m['faust_stages'][0]['exports'],['gain'])
 def test_combined_task_abi(self):
  _,m=c.compile_jsfx_to_ir('@init\ngain=0.5;t=defer(2+3;);\n@faust\nprocess=_,_:*(gain),_;')
  header=c._emit_header(m);self.assertTrue(m['has_tasks'] and m['has_faust'])
  self.assertIn('void* taskContext',header);self.assertIn('void* faustContext',header)
 def test_ordinary_abi(self):
  _,m=c.compile_jsfx_to_ir('@sample\nspl0*=0.5;')
  self.assertFalse(m['has_faust']);self.assertNotIn('void* faustContext',c._emit_header(m))
 def test_hoists(self):
  with self.assertRaisesRegex(ValueError,'hoisting'):
   c.compile_jsfx_to_ir('@faust\nprocess=_,_;',enable_loop_hoists=True)
 def test_ui_ownership(self):
  with self.assertRaisesRegex(ValueError,'UI writes'):
   c.compile_jsfx_to_ir('@init\ngain=0.5;\n@faust\nprocess=_,_:*(gain),_;\n@gfx\ngain=0.4;',native_gfx_prototype=True)
 def test_table_controls(self):
  with self.assertRaisesRegex(ValueError,'@faust'):
   c.compile_jsfx_to_ir('@init\ngain=0.5;\n@faust\nprocess=rdtable(16,gain,int(_)%16),_;')
 def test_resolver_and_relative_library(self):
  with tempfile.TemporaryDirectory(prefix='jsfx-relative-library Ω-') as d:
   d=Path(d);(d/'helper.jsfx-inc').write_text('@init\nfunction twice(x)(x*2);\n')
   (d/'custom.lib').write_text('half=0.5;')
   p=d/'mixed.jsfx';p.write_text('import helper.jsfx-inc\n@init\ngain=0.5;\n@block\ngain=twice(gain);\n@faust\nimport("custom.lib");process=_,_:*(half),_;\n@block\ngain=0.25;\n@faust\nprocess=_,_;\n')
   source=resolve_source(p).text
   self.assertEqual(source.count('@faust'),2);self.assertEqual(source.count('@block'),2);self.assertIn('import("custom.lib")',source)
   pipeline=c.prepare_jsfx_pipeline(source);pipeline['faust_plan']['include_paths']=[d]
   _,m=c.compile_jsfx_to_ir(source,pipeline=pipeline)
   self.assertEqual([s['kind'] for s in m['faust_stages']],['block','faust','block','faust'])
if __name__=='__main__':unittest.main()

