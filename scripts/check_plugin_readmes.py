"""Check every buildable help page and compile its embedded UTF-8 text.

The compiler check verifies the same generated header used by both native
plugin formats. It does not require compiling the entire DSP catalog.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from pluginlib import discover_plugins, read_plugin_readme
from build import write_plugin_readme_header


def check(root: Path, out: Path, compiler: str):
    specs = discover_plugins(root)
    rows, expected, includes, checks = [], bytearray(), [], []
    for index, spec in enumerate(specs):
        text = read_plugin_readme(spec.readme_path)
        body = text.split('\n', 1)[-1].strip()
        if len(body.split()) < 50:
            raise ValueError(f'{spec.readme_path}: insufficient user help after the title')
        if '\ufffd' in text:
            raise ValueError(f'{spec.readme_path}: contains a Unicode replacement character')
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            local = target.split('#', 1)[0]
            if local and not (spec.readme_path.parent / local).exists():
                raise ValueError(f'{spec.readme_path}: broken local link {target}')
        folder = out / spec.slug
        folder.mkdir(parents=True, exist_ok=True)
        header = write_plugin_readme_header(folder, spec.readme_path)
        includes.append(f'namespace plugin_{index} {{\n#include "{header.as_posix()}"\n}}')
        checks.append(f'std::cout.write(plugin_{index}::kPluginReadmeMarkdownText, sizeof(plugin_{index}::kPluginReadmeMarkdownText) - 1);')
        expected.extend(text.encode('utf-8'))
        rows.append(dict(plugin=spec.slug, path=spec.readme_path.relative_to(root).as_posix(), words=len(text.split()), bytes=len(text.encode('utf-8'))))
    source = out / 'embedded_help_check.cpp'
    source.write_text('#include <iostream>\n#ifdef _WIN32\n#include <fcntl.h>\n#include <io.h>\n#endif\n' + '\n'.join(includes) + '\nint main(){\n#ifdef _WIN32\n_setmode(_fileno(stdout), _O_BINARY);\n#endif\n' + '\n'.join(checks) + '\n}\n', encoding='utf-8')
    exe = out / ('embedded_help_check.exe' if __import__('os').name == 'nt' else 'embedded_help_check')
    subprocess.run([compiler, '-std=c++17', str(source), '-o', str(exe)], check=True)
    actual = subprocess.check_output([str(exe)])
    if actual != bytes(expected):
        raise ValueError('Compiled embedded help differs from README UTF-8 bytes')
    report = dict(plugins=len(rows), joep_plugins=sum(s.category == 'JoepVanlier' for s in specs), compiled_embedding='exact UTF-8 byte match', pages=rows)
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"PASS: {len(rows)} plugin help pages; {report['joep_plugins']} JoepVanlier; exact compiled UTF-8 embedding.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--compiler', default=shutil.which('clang++') or shutil.which('g++'))
    args = parser.parse_args()
    if not args.compiler:
        parser.error('A C++ compiler is required to verify embedded help bytes.')
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        check(args.root.resolve(), args.out.resolve(), args.compiler)
    else:
        with tempfile.TemporaryDirectory(prefix='za-plugin-help-') as directory:
            check(args.root.resolve(), Path(directory), args.compiler)


if __name__ == '__main__':
    main()
