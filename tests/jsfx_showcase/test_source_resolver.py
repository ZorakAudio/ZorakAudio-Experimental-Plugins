from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.jsfx_source import SourceResolver, SourceError, resolve_source, apply_host_options
from scripts.verify_jsfx_snapshot import verify, executable_tokens
from scripts import build
import dsp_jsfx_aot as aot

class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def write(self, path, text):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding='utf-8')
        return file
    def expand(self, text):
        return resolve_source(self.write('main.jsfx', text))
    def test_no_import_unchanged(self):
        text='desc:Bob\'s effect\nprovides: deps/*\n@init\nx=1;\n@gfx 100 80\n'
        self.assertEqual(self.expand(text).text, text)
    def test_bare_subdirectory_dependency(self):
        p=self.write('deps/lib.jsfx-inc','@init\nfunction f()(7);\n')
        r=self.expand('desc:x\nprovides:deps/*\nimport lib.jsfx-inc\n@sample\nspl0=f();\n')
        self.assertIn(p, r.dependencies); self.assertIn('function f()',r.text)
    def test_nested_relative_and_windows_separator(self):
        self.write('deps/lib.jsfx-inc','import "child\\leaf.jsfx-inc";\n@init\nx=1;\n')
        self.write('deps/child/leaf.jsfx-inc','@init\nx=2;\n')
        r=self.expand('import deps/lib.jsfx-inc\n@init\nx=3;\n')
        self.assertLess(r.text.index('x=2'),r.text.index('x=1'));self.assertLess(r.text.index('x=1'),r.text.index('x=3'))
    def test_main_processing_overrides_import(self):
        self.write('lib','@init\nx=1;\n@sample\nspl0=12;\n@gfx 80 30\ngfx_rect(0,0,2,2);\n')
        r=self.expand('import lib\n@sample\nspl0=3;\n')
        self.assertNotIn('spl0=12',r.text);self.assertIn('@gfx 80 30',r.text)
    def test_empty_main_section_still_overrides(self):
        self.write('lib','@sample\nspl0=12;\n')
        self.assertNotIn('spl0=12',self.expand('import lib\n@sample\n').text)
    def test_first_postorder_fallback_not_concatenation(self):
        self.write('a','import leaf\n@sample\nspl0=2;\n')
        self.write('leaf','@sample\nspl0=1;\n')
        self.write('b','@sample\nspl0=3;\n')
        r=self.expand('import a\nimport b\n')
        self.assertIn('spl0=1',r.text);self.assertNotIn('spl0=2',r.text);self.assertNotIn('spl0=3',r.text)
    def test_diamond_import_executes_init_once(self):
        self.write('a','import c\n@init\na+=1;\n');self.write('b','import c\n@init\nb+=1;\n');self.write('c','@init\nc+=1;\n')
        r=self.expand('import a\nimport b\n')
        self.assertEqual(r.text.count('c+=1'),1);self.assertEqual(len(r.dependencies),4)
    def test_library_metadata_not_injected(self):
        self.write('lib','desc:wrong\nslider1:7<0,9,1>wrong\n@init\nx=1;\n')
        r=self.expand('desc:right\nimport lib\n')
        self.assertNotIn('wrong',r.text);self.assertTrue(r.text.startswith('desc:right'))
    def test_cycles_diagnostic(self):
        self.write('a','import b\n@init\n');self.write('b','import a\n@init\n')
        with self.assertRaisesRegex(SourceError,'Cyclic.*a.*b.*a'):self.expand('import a\n')
    def test_ambiguous_names_rejected(self):
        self.write('a/lib','@init\n');self.write('b/lib','@init\n')
        with self.assertRaisesRegex(SourceError,'ambiguous'):self.expand('import lib\n')
    def test_direct_sibling_beats_recursive_duplicate(self):
        self.write('lib','@init\nx=1;\n');self.write('deps/lib','@init\nx=2;\n')
        self.assertNotIn('x=2',self.expand('import lib\n').text)
    def test_case_fallback_is_deterministic(self):
        self.write('deps/Mixed.JSFX-INC','@init\nx=1;\n')
        self.assertIn('x=1',self.expand('import mixed.jsfx-inc\n').text)
    def test_missing_lists_attempts(self):
        with self.assertRaisesRegex(SourceError,'missing import.*nothing'):self.expand('import nothing\n')
    def test_comments_strings_and_quotes(self):
        self.write('a b','@init\nx=1;\n')
        r=self.expand('/*\nimport nope\n@fake\n*/\nimport "a b"; // comment\n@gfx\n#s="hello\nimport not_a_directive\n@also_not_a_section\n";\n')
        self.assertIn('x=1',r.text);self.assertEqual(len(r.dependencies),2)
        self.assertEqual(set(r.section_sources),{'init','gfx'})
    def test_supplied_buffer_is_not_reread(self):
        file=self.write('main.jsfx','@init\nx=1;\n')
        self.assertEqual(resolve_source(file,text='@init\nx=2;\n').text,'@init\nx=2;\n')
    def test_path_escape_rejected(self):
        with self.assertRaises(SourceError):self.expand('import ../escape.jsfx\n')
    def test_no_section_library_rejected(self):
        self.write('lib','function f()(4);')
        with self.assertRaisesRegex(SourceError,'sectionless'):self.expand('import lib\n')
    def test_build_and_compiler_use_same_expander(self):
        self.write('deps/lib','@init\nfunction f()(7);\n')
        p=self.write('main.jsfx','import lib\n@sample\nspl0=f();\n')
        self.assertEqual(build.preprocess_jsfx_imports_from_path(p),aot.preprocess_jsfx_imports(p.read_text(),p))
    def test_host_options_leave_source_file_unchanged(self):
        p=self.write('main.jsfx','desc:x\n@init\nx=1;\n');before=p.read_bytes()
        r=apply_host_options(resolve_source(p),{'eel2Stores':True,'gfxMemory':'explicit'})
        self.assertEqual(p.read_bytes(),before);self.assertTrue(r.text.startswith('desc:x'))
        self.assertIn('options:za_eel2_stores=1',r.text);self.assertIn('gfx_sync_policy EXPLICIT',r.text)
    def test_host_options_fail_on_typos(self):
        for invalid in ({'foo':True},{'eel2Stores':'true'},{'gfxMemory':'never'},'no'):
            with self.assertRaises(SourceError):apply_host_options(self.expand('@init\n'),invalid)
    def test_manifest_has_all_hashes(self):
        self.write('deps/lib','@init\nx=1;\n')
        r=self.expand('import lib\n');m=r.manifest(self.root)
        self.assertEqual(len(m['files']),2);self.assertTrue(all(len(x['sha256'])==64 for x in m['files']))
    def test_shipped_snapshot_and_discovery(self):
        package=ROOT/'plugins/JoepVanlier/SaikeAbyss'
        self.assertEqual(len(verify(package)),4)
        from scripts.pluginlib import discover_plugins
        s=next(s for s in discover_plugins(ROOT) if s.slug=='SaikeAbyss')
        self.assertEqual(len(resolve_source(s.entry_path).dependencies),4)
    def test_upstream_notice_is_staged(self):
        from scripts.pluginlib import discover_plugins
        plugin=next(p for p in discover_plugins(ROOT) if p.slug=='SaikeAbyss')
        build.copy_upstream_notice(plugin,self.root/'staged')
        self.assertEqual((self.root/'staged/SaikeAbyss.LICENSE.upstream.txt').read_bytes(),
                         (plugin.root_dir/'LICENSE.upstream').read_bytes())
    def test_snapshot_tokens_preserve_strings_and_numbers(self):
        self.assertEqual(executable_tokens('@init\nx = 1; // hi\n'), executable_tokens('@init\nx=1;'))
        self.assertNotEqual(executable_tokens('@init\n#s="a b";'),executable_tokens('@init\n#s="ab";'))
        self.assertNotEqual(executable_tokens('@init\nx=1 2;'),executable_tokens('@init\nx=12;'))

if __name__=='__main__':unittest.main()
