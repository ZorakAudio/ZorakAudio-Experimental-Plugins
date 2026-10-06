"""Policy gates: automatic Legacy selection must reach both compiler and host."""
from pathlib import Path
import contextlib, io, json, sys, tempfile, unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import build
from pluginlib import discover_plugins
SPECS=discover_plugins(ROOT)
class BuildModes(unittest.TestCase):
    def test_all_joep_always_legacy(self):
        specs=[s for s in SPECS if s.category=='JoepVanlier'];self.assertEqual(len(specs),50)
        for s in specs:
            for p,l in [(False,False),(True,False),(False,True)]:self.assertEqual(build.native_gfx_modes_for_plugin(s,prototype=p,legacy=l),(False,True),s.slug)
    def test_other_jsfx_preserve_options(self):
        specs=[s for s in SPECS if s.category!='JoepVanlier' and s.plugin_type=='jsfx' and s.raw.get('nativeGfx')!='legacy'];self.assertEqual(len(specs),27)
        for s in specs:
            for p,l in [(False,False),(True,False),(False,True)]:self.assertEqual(build.native_gfx_modes_for_plugin(s,prototype=p,legacy=l),(p,l),s.slug)
    def test_corpus_always_legacy(self):
        spec=next(s for s in SPECS if s.slug=='Corpus')
        for p,l in [(False,False),(True,False),(False,True)]:
            self.assertEqual(build.native_gfx_modes_for_plugin(spec,prototype=p,legacy=l),(False,True))
    def test_faust_never_legacy(self):
        specs=[s for s in SPECS if s.plugin_type=='faust'];self.assertEqual(len(specs),5)
        for s in specs:self.assertEqual(build.native_gfx_modes_for_plugin(s,legacy=True),(False,False))
    def test_main_passes_identical_compiler_host_modes(self):
        class Configured(Exception):pass
        joep=next(s for s in SPECS if s.category=='JoepVanlier');other=next(s for s in SPECS if s.slug=='Sample');faust=next(s for s in SPECS if s.plugin_type=='faust')
        for spec,opt,legacy in [(joep,[],True),(joep,['--native-gfx-prototype'],True),(other,[],False),(other,['--native-gfx-legacy'],True),(faust,['--native-gfx-legacy'],False)]:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);(root/'scripts').mkdir();meta=root/'meta.json';meta.write_text('{}');calls=[]
                def configured(cmd,*a,**k):calls.append(cmd);raise Configured
                with patch.object(build,'__file__',str(root/'scripts/build.py')),patch.object(build,'discover_plugins',return_value=[spec]),patch.object(build,'host_os',return_value='linux'),patch.object(build,'write_plugin_readme_header'),patch.object(build,'build_jsfx_aot',return_value=(root/'d.o',root/'d.h',meta,root/'d.ll')) as compile_,patch.object(build,'run',side_effect=configured),patch.object(sys,'argv',['build.py',*opt]),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(Configured):build.main()
                    self.assertIn('-DZA_NATIVE_GFX_LEGACY='+('ON' if legacy else 'OFF'),calls[0])
                    if spec.plugin_type=='jsfx':self.assertEqual(compile_.call_args.kwargs['native_gfx_legacy'],legacy);self.assertFalse(compile_.call_args.kwargs['native_gfx_prototype'])
    def test_direct_manifest_helper_overrides_prototype(self):
        class Compiling(Exception):pass
        spec=next(s for s in SPECS if s.slug=='Corpus');commands=[]
        for prototype in [False,True]:
            with tempfile.TemporaryDirectory() as t:
                def stop(cmd,*a,**k):commands.append(cmd);raise Compiling
                with patch.object(build,'run',side_effect=stop):
                    with self.assertRaises(Compiling):build.build_jsfx_aot(ROOT,Path(t),spec.slug,spec.entry_path,native_gfx_prototype=prototype)
                self.assertIn('--native-gfx-legacy',commands[-1]);self.assertNotIn('--native-gfx-prototype',commands[-1])
    def test_shadow_monitor_rejected_before_mutations(self):
        spec=next(s for s in SPECS if s.category=='JoepVanlier')
        with patch.object(build,'discover_plugins',return_value=[spec]),patch.object(build,'host_os') as host,patch.object(sys,'argv',['build.py','--correctness-check']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:build.main()
            self.assertEqual(e.exception.code,2);host.assert_not_called()
if __name__=='__main__':unittest.main()
