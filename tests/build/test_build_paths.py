"""Keep Windows linker paths short without changing plugin/package identities."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path, PureWindowsPath
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build
from pluginlib import discover_plugins
from release_collections import build_catalog

SPECS = build_catalog(discover_plugins(ROOT), ROOT)


class BuildPathTests(unittest.TestCase):
    def test_reported_ci_path_is_rejected_with_remedy(self):
        root = PureWindowsPath(
            "D:/a/ZorakAudio-Experimental-Plugins/ZorakAudio-Experimental-Plugins/build/windows"
        )
        spec = next(s for s in SPECS if s.slug == "joep_saikemultispectralanalyzer_old")
        with self.assertRaisesRegex(ValueError, r"279 characters.*--build-root"):
            build.validate_windows_build_paths(root, [spec], "Release")

    def test_short_ci_root_fits_entire_release_catalog(self):
        root = PureWindowsPath("D:/a/_temp/za-catalog/windows")
        self.assertTrue(SPECS, "Release catalog must not be empty")
        for config in ("Release", "RelWithDebInfo", "Debug"):
            build.validate_windows_build_paths(root, SPECS, config)

    def test_default_and_relative_absolute_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for platform in ("windows", "linux", "macos"):
                self.assertEqual(build.platform_build_root(root, platform), root / "build" / platform)
                self.assertEqual(build.platform_build_root(root, platform, "short"), root / "short" / platform)
                absolute = root / "outside-checkout" / "short"
                self.assertEqual(build.platform_build_root(root / "checkout", platform, absolute), absolute / platform)

    def test_platform_cannot_escape_selected_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            for platform in ("../", "", "windows/../.."):
                with self.assertRaises(ValueError):
                    build.platform_build_root(Path(temporary), platform)

    def test_clean_only_preserves_other_build_trees(self):
        spec = SPECS[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            default = root / "build/windows/keep.txt"
            selected = root / "short/windows/remove.txt"
            other = root / "short/linux/keep.txt"
            for path in (default, selected, other):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            with patch.object(build, "__file__", str(root / "scripts/build.py")), \
                 patch.object(build, "discover_plugins", return_value=[spec]), \
                 patch.object(build, "build_catalog", return_value=[spec]), \
                 patch.object(build, "host_os", return_value="windows"), \
                 patch.object(build, "run") as run, \
                 patch.object(sys, "argv", ["build.py", "--build-root", "short", "--clean-only"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                build.main()
            self.assertTrue(default.is_file())
            self.assertTrue(other.is_file())
            self.assertFalse(selected.parent.exists())
            run.assert_not_called()

    def test_override_reaches_compiler_cmake_and_packaging(self):
        spec = next(s for s in SPECS if s.slug == "joep_bandjoiner")
        with tempfile.TemporaryDirectory(prefix="za-") as temporary:
            root = Path(temporary).resolve()
            base = root / "short"
            expected_build = base / "windows" / spec.slug
            commands = []
            compiler_dirs = []

            def compile_fixture(repo_root, cmake_build, *args, **kwargs):
                compiler_dirs.append(cmake_build)
                obj = cmake_build / "JSFXDSP.obj"
                obj.write_bytes(b"fixture")
                meta = cmake_build / "JSFXDSP_meta.json"
                meta.write_text("{}", encoding="utf-8")
                return obj, cmake_build / "JSFXDSP.h", meta, cmake_build / "JSFXDSP.ll"

            def run_fixture(command, *args, **kwargs):
                commands.append(command)
                if "--build" in command:
                    directory = Path(command[command.index("--build") + 1])
                    artifacts = directory / f"{spec.slug}_artefacts/Release"
                    payload = artifacts / f"VST3/{spec.slug}.vst3/Contents/x86_64-win/{spec.slug}.vst3"
                    payload.parent.mkdir(parents=True)
                    payload.write_bytes(b"vst3 fixture")
                    clap = artifacts / f"CLAP/{spec.slug}.clap"
                    clap.parent.mkdir(parents=True)
                    clap.write_bytes(b"clap fixture")

            with patch.object(build, "__file__", str(root / "scripts/build.py")), \
                 patch.object(build, "discover_plugins", return_value=[spec]), \
                 patch.object(build, "build_catalog", return_value=[spec]), \
                 patch.object(build, "host_os", return_value="windows"), \
                 patch.object(build, "is_macos", return_value=False), \
                 patch.object(build, "build_jsfx_aot", side_effect=compile_fixture), \
                 patch.object(build, "run", side_effect=run_fixture), \
                 patch.dict(build.os.environ, {"ZA_CMAKE_GENERATOR": "Visual Studio 17 2022"}), \
                 patch.object(sys, "argv", ["build.py", "--only", spec.slug, "--tag", "probe",
                                            "--build-root", str(base)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                build.main()

            self.assertEqual(compiler_dirs, [expected_build])
            configure = next(cmd for cmd in commands if "-S" in cmd)
            self.assertEqual(configure[configure.index("-B") + 1], str(expected_build))
            self.assertIn(f"-DPLUGIN_SLUG={spec.slug}", configure)
            self.assertIn(f"-DBUNDLE_ID={spec.bundle_id}", configure)
            self.assertIn(f"-DCLAP_ID={spec.clap_id}", configure)
            archive = root / "dist/probe/windows/ZorakAudio-Experimental-Plugins-probe-windows.zip"
            with zipfile.ZipFile(archive) as package:
                manifest = json.loads(next(package.read(name) for name in package.namelist()
                                           if name.endswith("/manifest.json")))
                entry = manifest["plugins"][0]
                self.assertEqual(entry["slug"], spec.slug)
                self.assertEqual(entry["bundleId"], spec.bundle_id)
                self.assertEqual(entry["clapId"], spec.clap_id)
                self.assertEqual(entry["installPath"], spec.install_display)
                self.assertTrue(any(name.endswith(
                    f"VST3/{spec.install_rel_dir.as_posix()}/{spec.slug}.vst3/Contents/x86_64-win/{spec.slug}.vst3"
                ) for name in package.namelist()))
                self.assertTrue(any(name.endswith(
                    f"CLAP/{spec.install_rel_dir.as_posix()}/{spec.slug}.clap"
                ) for name in package.namelist()))


if __name__ == "__main__":
    unittest.main()
