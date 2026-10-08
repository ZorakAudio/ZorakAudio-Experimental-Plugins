"""Exercise fresh, reused, conflicting and concurrent dependency builds.

Use the actual pinned upstream wrapper files in isolated Git fixtures. Never
revert or mutate the developer's real submodules.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/jit_editor/apply_wrapper_patches.py"
FILES = (
    ("libs/JUCE", "modules/juce_audio_plugin_client/juce_audio_plugin_client_VST3.cpp"),
    ("libs/clap-juce-extensions", "src/wrapper/clap-juce-wrapper.cpp"),
)


def run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kwargs)


class WrapperPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="za wrapper checks ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = {}
        for relative, source in FILES:
            folder = self.root / relative
            folder.mkdir(parents=True)
            subprocess.run(["git", "init", "--quiet", str(folder)], check=True)
            (folder / "CMakeLists.txt").write_text("# fixture\n", encoding="utf-8")
            original = subprocess.check_output(["git", "-C", str(ROOT / relative), "show", f"HEAD:{source}"])
            self.base[relative] = original.replace(b"\r\n", b"\n")
            target = folder / source
            target.parent.mkdir(parents=True)
            target.write_bytes(self.base[relative])

    def invoke(self, *args):
        return run([sys.executable, str(SCRIPT), "--root", str(self.root), *args])

    def wrappers(self):
        return {(self.root / relative / source): (self.root / relative / source).read_bytes()
                for relative, source in FILES}

    def test_fresh_lf_checkout_and_repeat_does_not_rewrite(self):
        self.assertNotEqual(self.invoke("--check").returncode, 0)
        self.assertEqual(self.invoke().returncode, 0)
        before = self.wrappers()
        times = {p: p.stat().st_mtime_ns for p in before}
        self.assertEqual(self.invoke().returncode, 0)
        self.assertEqual(self.invoke("--check").returncode, 0)
        self.assertEqual(before, self.wrappers())
        self.assertEqual(times, {p: p.stat().st_mtime_ns for p in before})

    def test_crlf_checkout(self):
        for relative, source in FILES:
            (self.root / relative / source).write_bytes(self.base[relative].replace(b"\n", b"\r\n"))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.invoke("--check").returncode, 0)

    def test_unrelated_edits_survive(self):
        for relative, source in FILES:
            with (self.root / relative / source).open("ab") as target:
                target.write(b"\n// unrelated local change\n")
        self.assertEqual(self.invoke().returncode, 0)
        self.assertTrue(all(b"// unrelated local change" in data for data in self.wrappers().values()))

    def test_conflict_is_preflighted_without_partial_application(self):
        target = self.root / FILES[1][0] / FILES[1][1]
        target.write_bytes(target.read_bytes().replace(b"const bool forceLegacyParamIDs = false;",
                                                       b"const bool forceLegacyParamIDs = local_override;"))
        before = self.wrappers()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conflicts", result.stderr)
        self.assertEqual(before, self.wrappers())

    def test_missing_dependency_has_actionable_error(self):
        (self.root / FILES[1][0] / "CMakeLists.txt").unlink()
        before = self.wrappers()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("git submodule update --init --recursive", result.stderr)
        self.assertEqual(before, self.wrappers())
        self.assertEqual(self.invoke("--only", "juce").returncode, 0)

    def test_concurrent_builds_apply_once(self):
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _: self.invoke(), range(4)))
        for result in results:
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.invoke("--check").returncode, 0)

    def test_reused_cmake_build_repairs_reverted_wrapper(self):
        tools = self.root / "tools/jit_editor"
        tools.mkdir(parents=True)
        shutil.copy2(SCRIPT, tools)
        shutil.copytree(SCRIPT.with_name("wrapper-patches"), tools / "wrapper-patches")
        cmake = self.root / "cmake"
        cmake.mkdir()
        shutil.copy2(ROOT / "cmake/ApplyWrapperPatches.cmake", cmake)
        harness = self.root / "harness"
        harness.mkdir()
        (harness / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.21)\nproject(PatchHarness LANGUAGES NONE)\n'
            'set(ZA_ENABLE_CLAP ON)\ninclude("${ZA_ROOT}/cmake/ApplyWrapperPatches.cmake")\n',
            encoding="utf-8",
        )
        build = self.root / "harness-build"
        configured = run(["cmake", "-G", "Ninja", "-S", str(harness), "-B", str(build),
                          f"-DZA_ROOT={self.root}", f"-DPython3_EXECUTABLE={sys.executable}"])
        self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
        for relative, source in FILES:
            (self.root / relative / source).write_bytes(self.base[relative])
        self.assertNotEqual(self.invoke("--check").returncode, 0)
        rebuilt = run(["cmake", "--build", str(build), "--target", "za_wrapper_patches"])
        self.assertEqual(rebuilt.returncode, 0, rebuilt.stdout + rebuilt.stderr)
        self.assertEqual(self.invoke("--check").returncode, 0)


class ArtifactLayoutTests(unittest.TestCase):
    def test_outer_mac_bundles_and_windows_linux_binaries(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import build
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outer_clap = root / "CLAP/Effect.clap"
            (outer_clap / "Contents/MacOS").mkdir(parents=True)
            (outer_clap / "Contents/MacOS/Effect.clap").write_bytes(b"payload")
            outer_vst = root / "VST3/Effect.vst3"
            (outer_vst / "Contents/x86_64-win").mkdir(parents=True)
            (outer_vst / "Contents/x86_64-win/Effect.vst3").write_bytes(b"payload")
            binary = root / "CLAP/Other.clap"
            binary.write_bytes(b"binary")
            self.assertEqual(set(build.collect_stageable_clap_artifacts(root)), {outer_clap, binary})
            self.assertEqual(build.collect_stageable_vst3_artifacts(root), [outer_vst])


if __name__ == "__main__":
    unittest.main()
