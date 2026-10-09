"""Stage a relocatable Linux compiler payload from explicit local toolchains.

The supported baseline is Ubuntu 24.04 x86-64. glibc and its loader remain OS
dependencies; other ELF dependencies are bundled with relative search paths.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[2]


def stage_linux(destination, python, faust):
    if not shutil.which('patchelf'):
        raise ValueError('Linux compiler packaging requires patchelf')
    executable = python / 'bin/python3'
    probe = '''import sys,sysconfig,json,llvmlite,importlib.metadata
d=importlib.metadata.distribution('llvmlite')
print(json.dumps(dict(binary=sys._base_executable,version=sys.version,prefix=sys.base_prefix,minor='%d.%d'%sys.version_info[:2],stdlib=sysconfig.get_path('stdlib'),package=llvmlite.__file__,llvmlite=llvmlite.__version__,licenses=[str(d.locate_file(f)) for f in d.files if 'licenses/' in str(f)])))'''
    info = json.loads(subprocess.check_output([str(executable), '-I', '-c', probe], text=True))
    destination.mkdir(parents=True, exist_ok=True)
    py = destination / 'python'
    (py / 'bin').mkdir(parents=True)
    shutil.copy2(Path(info['binary']).resolve(), py / 'bin/python3')
    stdlib = Path(info['stdlib'])
    lib = py / 'lib' / ('python' + info['minor'])
    lib.mkdir(parents=True)
    excluded = {'site-packages', 'dist-packages', '__pycache__', 'test', 'tests',
                'idlelib', 'tkinter', 'turtledemo', 'ensurepip'}
    with zipfile.ZipFile(py / 'lib' / ('python' + info['minor'].replace('.', '') + '.zip'),
                         'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sorted(stdlib.rglob('*.py')):
            relative = source.relative_to(stdlib)
            if not excluded.intersection(relative.parts) and str(relative) not in ('sitecustomize.py', 'usercustomize.py'):
                archive.write(source, str(relative))
    # CPython's executable-relative prefix detection requires this landmark.
    shutil.copy2(stdlib / 'os.py', lib / 'os.py')
    shutil.copytree(stdlib / 'lib-dynload', lib / 'lib-dynload',
                    ignore=shutil.ignore_patterns('_tkinter*'))
    package = Path(info['package']).parent
    shutil.copytree(package, py / 'site-packages/llvmlite',
                    ignore=shutil.ignore_patterns('__pycache__', 'tests'))
    (lib / 'sitecustomize.py').write_text(
        'import sys\nfrom pathlib import Path\n'
        'root=Path(__file__).resolve().parents[2]\n'
        'sys.path.insert(0,str(root/"site-packages"))\n', encoding='utf-8')
    compiler = destination / 'compiler'
    (compiler / 'scripts').mkdir(parents=True)
    for source in (REPO / 'dsp_jsfx_aot.py', REPO / 'src/JsfxRuntimeExports.inc',
                   Path(__file__).with_name('compiler_worker.py'),
                   Path(__file__).with_name('native_frontend.py')):
        shutil.copy2(source, compiler)
    for name in ('jsfx_faust_compiler.py', 'jsfx_tasks_compiler.py', 'jsfx_source.py', 'jsfx_preprocessor.py'):
        shutil.copy2(REPO / 'scripts' / name, compiler / 'scripts')
    (compiler / 'scripts/__init__.py').write_text('', encoding='utf-8')
    for source in (REPO / 'build/native-compiler/jsfx_frontend',
                   REPO / 'build/tools/jsfx_eel_pp/bin/jsfx_eel_pp',
                   REPO / 'build/jit-editor/za_compiler_launcher'):
        if not source.is_file():
            raise ValueError('Build the compiler helper before packaging: ' + str(source))
        shutil.copy2(source, compiler)
    (destination / 'faust/bin').mkdir(parents=True)
    shutil.copy2(faust / 'bin/faust', destination / 'faust/bin')
    for source in (faust / 'share/faust').rglob('*.lib'):
        target = destination / 'faust/share/faust' / source.relative_to(faust / 'share/faust')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    licenses = destination / 'licenses'
    licenses.mkdir()
    python_notice = next((p for p in (Path(info['prefix']) / 'LICENSE.txt', stdlib / 'LICENSE.txt',
        Path('/usr/share/doc/python' + info['minor'] + '/copyright')) if p.is_file()), None)
    if python_notice is None: raise ValueError('Python license notice missing from explicit compiler installation')
    for source, name in ((REPO / 'LICENSE', 'Repository-LICENSE'),
                         (REPO / 'libs/JUCE/LICENSE.md', 'JUCE-LICENSE.md'),
                         (REPO / 'src/WDL/LICENSE.txt', 'WDL-LICENSE.txt'),
                         (python_notice, 'Python-copyright')):
        if not source.is_file():
            raise ValueError('Required compiler notice missing: ' + str(source))
        shutil.copy2(source, licenses / name)
    for source in (REPO / 'libs/clap-juce-extensions').glob('LICENSE*'):
        if source.is_file(): shutil.copy2(source, licenses / ('CLAP-JUCE-' + source.name))
    for source in map(Path, info['licenses']):
        if source.is_file(): shutil.copy2(source, licenses / ('llvmlite-' + source.name))
    for source in (faust / 'share/faust').glob('*COPYING*'):
        shutil.copy2(source, licenses / ('Faust-' + source.name))
    native_libraries = destination / 'lib'
    native_libraries.mkdir()
    # glibc is tied to the target OS/loader. Bundle the remaining dependencies.
    platform_libs = {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2',
                     'librt.so.1', 'libutil.so.1', 'libresolv.so.2', 'ld-linux-x86-64.so.2'}
    elf_files = []
    for source in destination.rglob('*'):
        if source.is_file():
            with source.open('rb') as stream:
                if stream.read(4) == b'\x7fELF': elf_files.append(source)
    dependencies = {}
    for source in elf_files:
        output = subprocess.check_output(['ldd', str(source)], text=True)
        if 'not found' in output: raise ValueError('Missing ELF dependency:\n' + output)
        for soname, path in re.findall(r'^\s*(\S+) => (/\S+) ', output, re.MULTILINE):
            if soname not in platform_libs: dependencies[soname] = Path(path)
    for soname, source in dependencies.items():
        target = native_libraries / soname
        shutil.copy2(source.resolve(), target)
        elf_files.append(target)
        owned = subprocess.run(['dpkg-query', '-S', str(source)], capture_output=True, text=True)
        if owned.returncode:
            owned = subprocess.run(['dpkg-query', '-S', str(source.resolve())], capture_output=True, text=True)
        for line in owned.stdout.splitlines():
            package_name = line.split(': ', 1)[0].split(':')[0]
            notice = Path('/usr/share/doc') / package_name / 'copyright'
            if notice.is_file(): shutil.copy2(notice, licenses / (package_name + '-copyright'))
    for source in elf_files:
        relative = os.path.relpath(native_libraries, source.parent)
        subprocess.run(['patchelf', '--set-rpath', '$ORIGIN/' + relative, str(source)], check=True)
    manifest = dict(platform='Linux x86-64 (Ubuntu 24.04 baseline)', python=info['version'],
                    llvmlite=info['llvmlite'], compilerBackend='LLVM ORC; shared production compiler/runtime',
                    bundledLibraries=sorted(dependencies), files={})
    for category in ('python', 'compiler', 'faust', 'lib', 'licenses'):
        manifest['files'][category] = sum(p.stat().st_size for p in (destination / category).rglob('*') if p.is_file())
    (destination / 'runtime-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
