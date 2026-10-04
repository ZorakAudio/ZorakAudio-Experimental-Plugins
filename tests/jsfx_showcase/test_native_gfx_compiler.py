"""Regression gates for the opt-in display-only native graphics compiler."""
import sys
import json
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dsp_jsfx_aot as compiler

class NativeGraphicsCompilerTests(unittest.TestCase):
    def compile(self, source, native=True):
        return compiler.compile_jsfx_to_ir(source, native_gfx_prototype=native)

    def rejects(self, source, message):
        with self.assertRaisesRegex(ValueError, message):
            self.compile(source)

    def test_default_path_stays_opt_out(self):
        module, meta = self.compile('@gfx 20 20\ngfx_blit(0,1,0);', False)
        self.assertFalse(meta['native_gfx_prototype'])
        self.assertNotIn('jsfx_gfx_aot', str(module))

    def test_graphics_helpers_are_emitted_with_native_calls(self):
        source = '@gfx 20 20\nfunction label(s) (gfx_drawstr(s);); label("hello");'
        module, _ = self.compile(source)
        self.assertIn('call double @"jsfx_native_gfx_drawstr"', str(module))

    def test_unsupported_call_inside_helper_fails(self):
        self.rejects('@gfx\nfunction inner() (gfx_blit(0,1,0);); function outer() (inner();); outer();', 'gfx_blit')

    def test_measurestr_uses_scalar_output_pointers_in_helpers(self):
        source = '@gfx\nfunction label(s) local(w h) (gfx_measurestr(s,w,h);gfx_line(0,h,w,h););label("hello");'
        module, _ = self.compile(source)
        self.assertIn('call double @"jsfx_native_gfx_measurestr"', str(module))
        self.assertIn('call double @"jsfx_native_gfx_line"', str(module))

    def test_measurestr_outputs_count_as_ui_writes(self):
        self.rejects('@init\nlevel=1;\n@sample\nspl0*=level;\n@gfx\ngfx_measurestr("hello",level,h);', 'UI writes to DSP')

    def test_measurestr_rejects_non_scalar_outputs(self):
        for target in ['1+2', '0[0]', 'slider1', 'srate', '#s']:
            with self.subTest(target=target):
                self.rejects('@gfx\ngfx_measurestr("hello",'+target+',h);', 'outputs require')

    def test_heap_access_inside_helper_fails(self):
        self.rejects('@gfx\nfunction read_heap() (0[0];); read_heap();', 'indexed memory')

    def test_ui_writes_to_dsp_state_fail(self):
        self.rejects('@init\ncontrol=1;\n@sample\nspl0*=control;\n@gfx\ncontrol=0.5;', 'UI writes to DSP')

    def test_graphics_initialization_inside_audio_helper_fails(self):
        self.rejects('@init\nfunction setup() (gfx_set(1);); setup();\n@gfx\ngfx_rect(0,0,10,10);', 'audio sections')

    def test_mouse_input_fails_in_display_subset(self):
        self.rejects('@gfx\ngfx_rect(mouse_x,0,10,10);', 'mouse_x')

    def test_unsafe_format_and_argument_mismatch_fail(self):
        for expression in ['sprintf(#s,"%n",1);', 'sprintf(#s,"%f");', 'sprintf(#s,"%9999f",1);']:
            with self.assertRaises(ValueError):
                self.compile('@gfx\n'+expression)

    def test_graphics_local_state_is_not_published_from_audio(self):
        _, meta = self.compile('@init\nmeter=0.5;\n@gfx\ncounter+=1;gfx_rect(0,0,meter*20,counter);')
        self.assertEqual(meta['gfx_var_flags']['meter'], compiler.GFX_VAR_FLAG_TO_GFX)
        self.assertEqual(meta['gfx_var_flags']['counter'], 0)
        self.assertFalse(meta['gfx_mem_shared'])

    def contract(self, value, source):
        return '// za_native_gfx: ' + json.dumps(value) + '\n' + source

    def test_interactive_controls_use_native_host_hooks(self):
        module, meta = self.compile(self.contract({'interactive': True},
            '@gfx\nslider1=mouse_x;sliderchange(slider1);slider_automate(slider1,1);gfx_getchar();strcpy(#s,"hello");strcat(#s,"!");strncpy(#t,#s,3);gfx_drawstr(#t);strlen(#t);time_precise();gfx_showmenu(#s);'))
        self.assertEqual(meta['gfx_var_sync_mode'], 'native-interactive')
        self.assertIn('jsfx_native_slider_event', str(module))
        self.assertNotIn('call i32 @"jsfx_sliderchange"', str(module))

    def test_explicit_preview_and_command_ownership(self):
        source = self.contract({'interactive': True, 'locals': ['preview'], 'commands': ['solo']},
            '@init\npreview=1;solo=0;\n@sample\nspl0*=preview+solo;\n@gfx\npreview=slider1;solo=mouse_cap&1;')
        _, meta = self.compile(source)
        self.assertEqual(meta['gfx_var_flags']['preview'], 0)
        self.assertEqual(meta['gfx_var_flags']['solo'], compiler.GFX_VAR_FLAG_FROM_GFX)
        self.assertFalse(meta['gfx_mem_shared'])

    def test_bounded_publication_without_heap_allocation(self):
        source = self.contract({'views': [{'base': 'base', 'count': 'count', 'stride': 4, 'fields': [1], 'max': 8}]},
            '@init\nbase=100;count=8;base[1]=2;\n@gfx\nmeter=base[1];gfx_rect(0,0,meter,2);')
        module, meta = self.compile(source)
        self.assertEqual(meta['gfx_var_flags']['base'], compiler.GFX_VAR_FLAG_TO_GFX)
        self.assertEqual(meta['gfx_var_flags']['count'], compiler.GFX_VAR_FLAG_TO_GFX)
        self.assertIn('jsfx_native_read_mem', str(module))
        self.assertFalse(meta['gfx_mem_shared'])

    def test_publication_does_not_authorize_heap_writes_or_gmem(self):
        for expression in ['base[1]=3;', 'x=gmem[1];']:
            source = self.contract({'views': [{'base': 'base', 'count': 8, 'max': 8}]},
                '@init\nbase=100;\n@gfx\n'+expression)
            with self.assertRaises(ValueError): self.compile(source)

    def test_publication_limits_and_layout_ownership_are_checked(self):
        for view in [{'base': 'base', 'count': 8, 'max': 1000000},
                     {'base': 'base', 'count': 8, 'max': 8, 'fields': [4], 'stride': 4},
                     {'base': 'missing', 'count': 8, 'max': 8}]:
            with self.subTest(view=view):
                with self.assertRaises(ValueError):
                    self.compile(self.contract({'views': [view]}, '@init\nbase=100;\n@gfx\ngfx_rect(0,0,2,2);'))
        self.rejects(self.contract({'views': [{'base': 'base', 'count': 8, 'max': 8}]},
            '@init\nbase=100;\n@gfx\nbase=200;x=base[0];'), 'DSP variables|audio owned')

    def test_commands_cannot_restore_armed_state(self):
        self.rejects(self.contract({'commands': ['solo'], 'persist': ['solo']},
            '@gfx\nsolo=1;gfx_rect(0,0,2,2);'), 'cannot be persisted')

    def test_native_literals_preserve_utf8(self):
        _, meta = self.compile('@gfx\ngfx_drawstr("Grüße 日本語");')
        header = compiler._emit_header(meta)
        self.assertIn('0xc3, 0xbc', header)
        self.assertIn('0xe6, 0x97, 0xa5', header)

if __name__ == '__main__':
    unittest.main()
