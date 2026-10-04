"""Compiler gates for live shared guest state; runtime tests use production helpers."""
import re
import sys
import unittest
import shutil
import subprocess
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dsp_jsfx_aot as c

class LegacyCompilerTests(unittest.TestCase):
    def compile(self, text):
        return c.compile_jsfx_to_ir(text, native_gfx_legacy=True)

    def test_large_math_shares_code_and_keeps_receiver_cells(self):
        from llvmlite import binding as llvm
        source = '@init\nfunction coeff(x) local(n) instance(a,b)(n+=1;a=x+n;b=a*2;'+16*'b+=a/100;'+'b;);\n'
        source += '\n'.join(f'voice{i}.coeff({i});' for i in range(200))
        module, meta = self.compile(source+'\n@gfx\nvoice0.coeff(3);voice1.coeff(4);')
        helpers = [f for f in module.functions if f.name.startswith('jsfx_fn_')]
        self.assertEqual(len(helpers), 2)  # One shared body in each caller section.
        self.assertIn('voice0.a', meta['vars'])
        self.assertIn('voice199.b', meta['vars'])
        self.assertFalse(any(n.startswith('__receiver_cell_') for n in meta['vars']))
        llvm.parse_assembly(str(module)).verify()

    def test_namespace_members_keep_receiver_specialization(self):
        source = '@init\nfunction band(x) instance(child,gain)(gain=x;child.left=gain;'+16*'child.left+=gain/100;'+'child.left;);left.band(2);right.band(4);'
        module, meta = self.compile(source)
        self.assertEqual(len([f for f in module.functions if f.name.startswith('jsfx_fn_')]), 2)
        self.assertIn('left.child.left', meta['vars'])
        self.assertIn('right.child.left', meta['vars'])
        self.assertFalse(any(n.startswith('__receiver_cell_') for n in meta['vars']))

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required for block ABI check')
    def test_empty_sample_preserves_separate_and_in_place_audio(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                module, meta = c.compile_jsfx_to_ir('@block\nticks+=1;\n@sample\n', native_gfx_legacy=legacy)
                (folder/'case.h').write_text(c._emit_header(meta))
                c._aot_opt_and_emit(str(module), 2, str(folder/'case.o'), None,
                                   position_independent=True, native_gfx_legacy=legacy)
                (folder/'check.cpp').write_text('''#include "case.h"
#include <cassert>
int main() {
 DSPJSFX_State state{};
 float left[]{1, 2, 3}, right[]{4, 5, 6}, outLeft[3]{}, outRight[3]{};
 const float* input[]{left, right}; float* output[]{outLeft, outRight};
 jsfx_process_block(&state, input, output, 2, 3);
 for (int i=0; i<3; ++i) { assert(left[i]==outLeft[i]); assert(right[i]==outRight[i]); }
 float* inplace[]{left, right};
 jsfx_process_block(&state, input, inplace, 2, 3);
 assert(left[0]==1 && right[2]==6);
 jsfx_process_block(&state, nullptr, nullptr, 0, 3);
 jsfx_process_block(&state, nullptr, nullptr, 2, 0);
}
''')
                built = subprocess.run(['c++', '-std=c++20', '-I'+str(Path(c.__file__).parent/'src'), str(folder/'check.cpp'), str(folder/'case.o'),
                                        '-o', str(folder/'check')], capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                subprocess.run([str(folder/'check')], check=True, capture_output=True)

    def test_legacy_block_labels_remain_short_in_dense_code(self):
        from llvmlite import binding as llvm
        module, _ = self.compile('@init\n'+'\n'.join(f'mem[{i}]=1;' for i in range(200))+'\n@gfx\nmem[0]=2;')
        self.assertTrue(all(len(b.name)<16 for f in module.functions for b in f.blocks))
        llvm.parse_assembly(str(module)).verify()

    def test_streamed_cli_ir_matches_inspectable_api_ir(self):
        source = '@init\nfunction paint(flag) (flag ? (loop(4,gfx_getpixel(r,g,b));));\n@gfx\npaint(1);'
        normal, normal_meta = self.compile(source)
        streamed, streamed_meta = c.compile_jsfx_to_ir(source, native_gfx_legacy=True, stream_functions=True)
        self.assertEqual(normal_meta, streamed_meta)
        self.assertEqual(str(normal), str(streamed))

    def test_addresses_and_callback_storage_dominate_conditional_loops(self):
        from llvmlite import binding as llvm
        module, _ = self.compile('@init\nfunction paint(flag) (flag ? (loop(4,gfx_getpixel(r,g,b));));\n@gfx\npaint(1);')
        parsed = llvm.parse_assembly(str(module))
        parsed.verify()
        for function in parsed.functions:
            for block in function.blocks:
                for instruction in block.instructions:
                    if instruction.opcode == 'alloca':
                        self.assertEqual(block.name, 'entry')

    def test_arbitrary_guest_writes_need_no_contract(self):
        m, meta = self.compile('@sample\nspl0 *= gain;\n@gfx\ngain=mouse_x;0[4]+=1;')
        self.assertEqual(meta['gfx_var_sync_mode'], 'native-legacy-shared')
        self.assertTrue(meta['gfx_mem_shared'])
        self.assertNotIn('jsfx_native_read_mem', str(m))
        self.assertIn('store atomic double', str(m))

    def test_compound_assignments_are_not_atomic_rmw(self):
        m, _ = self.compile('@sample\ncounter+=1;\n@gfx\ncounter+=1;')
        self.assertNotIn('atomicrmw', str(m))
        self.assertIn('load atomic double', str(m))

    def test_dynamic_slider_and_sample_output_writes(self):
        m, _ = self.compile('@sample\nspl0*=gain;\n@gfx\nslider(1)=0.5;spl(0)=2;')
        self.assertIn('store atomic double', str(m))

    def test_invalid_maxmem_and_idle_options_are_rejected(self):
        for option in ('maxmem=0', 'maxmem=268435457', 'maxmem=bad', 'gfx_idle', 'gfx_idle_only'):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.compile('options:'+option+'\n@gfx\nx=1;')

    def test_no_nonatomic_double_access_to_guest_storage(self):
        m, _ = self.compile(Path(__file__).with_name('legacy_shared_probe.jsfx').read_text())
        text = str(m)
        # Explicit atomic rvalue operands have private alloca temporaries.
        self.assertFalse(re.search(r'\bload double,', text))
        temporaries = set(re.findall(r'(%"[^"]+") = alloca double', text))
        stores = [x for x in text.splitlines() if re.search(r'\bstore double\b', x)
                  and x.rsplit('double* ', 1)[-1].split(',')[0] not in temporaries]
        self.assertEqual(len(stores), 1)  # private currentSampleRate in DSP context

    def test_custom_hoists_are_rejected(self):
        for flag in ('enable_section_hoists', 'enable_loop_hoists'):
            with self.assertRaisesRegex(ValueError, 'scalar hoisting'):
                c.compile_jsfx_to_ir('@gfx\nx=1;', native_gfx_legacy=True, **{flag: True})

    def test_maxmem_and_abi_are_explicit(self):
        _, meta = self.compile('options:maxmem=262144\n@gfx\n0[262143]=1;')
        h = c._emit_header(meta)
        self.assertIn('DSPJSFX_MAX_MEM_CELLS 262144LL', h)
        self.assertIn('DSPJSFX_RUNTIME_STATE_ABI 5', h)
        self.assertIn('DSPJSFX_NATIVE_GFX_LEGACY 1', h)

    def test_unsupported_apis_still_fail_at_compile_time(self):
        for expr in ('gfx_unimplemented();', 'gfx_blit(0,1);', 'file_open();', 'gfx_unimplemented(1);'):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                self.compile('@gfx\n'+expr)

    def test_extended_drawing_uses_reference_dispatch(self):
        m, _ = self.compile('@gfx\ngfx_set(1,.5,0,1,1,2,.7);gfx_setimgdim(2,32,32);'
            'gfx_getimgdim(2,0[3],slider1);gfx_blit(2,1,0);gfx_getpixel(x,x,x);'
            'gfx_printf("%.2f",slider1);gfx_drawstr("label",1,200,40);')
        self.assertIn('jsfx_native_gfx_dispatch', str(m))

    def test_plugins_without_custom_gfx_can_use_legacy_state(self):
        _, meta = self.compile('@sample\nspl0*=slider1;')
        self.assertTrue(meta['native_gfx_legacy'])

    def test_multiline_strings_and_packed_font_flags(self):
        _, meta = self.compile('@gfx\ngfx_setfont(1,"Arial",12,\'bi\');gfx_drawstr("a\nb");')
        self.assertIn('a\nb', [item['text'] for item in meta['string_literals']])

    def test_named_slider_is_one_cell_and_uses_direct_notifications(self):
        m, meta = self.compile('slider1:gain=1<0,2>Gain\n@sample\nspl0*=gain;\n@gfx\ngain=0.5;sliderchange(gain);')
        self.assertEqual(meta['slider_aliases'], {'gain': 0})
        self.assertIn('jsfx_native_slider_event', str(m))

    def test_named_slider_aliases_are_case_insensitive(self):
        m, meta = self.compile('slider1:Gain=1<0,2>Gain\n@sample\nspl0*=gAiN;\n@gfx\nGAIN=0.5;sliderchange(Gain);')
        self.assertEqual(meta['slider_aliases'], {'gain': 0})
        header = c._emit_header(meta)
        aliases = header.split('DSPJSFX_LEGACY_SLIDER_ALIASES', 1)[1].split(';', 1)[0]
        self.assertRegex(aliases, r'\{0(?:,|\})')
        self.assertIn('jsfx_native_slider_event', str(m))
        self.assertIn('DSPJSFX_LEGACY_SLIDER_ALIASES', c._emit_header(meta))

    def test_invalid_named_slider_cannot_escape_the_state_array(self):
        for index in (0, 257):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.compile(f'slider{index}:gain=1<0,2>Gain\n@gfx\ngain=2;')

if __name__ == '__main__': unittest.main()
