"""Compare the native frontend with the current Python production implementation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
import zipfile
from Reference import ROOT, frontend


def first_difference(a, b, path="$"):
    if type(a) is not type(b): return f"{path}: types {type(a).__name__}/{type(b).__name__}"
    if isinstance(a, dict):
        if a.keys() != b.keys(): return f"{path}: fields {a.keys()}/{b.keys()}"
        for key in a:
            diff = first_difference(a[key], b[key], f"{path}.{key}")
            if diff: return diff
    elif isinstance(a, list):
        if len(a) != len(b): return f"{path}: lengths {len(a)}/{len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = first_difference(x, y, f"{path}[{i}]")
            if diff: return diff
    elif a != b: return f"{path}: {str(a)[:200]!r}/{str(b)[:200]!r}"


def fixtures(root):
    snippets = {
        "precedence": "a+b+=2;x=a+b*c/d%3^2; a&&b||c; a|b&c~d; a-b+c; a/b*c; a===b;a!==b;",
        "continuation": "a\n + b\n || c; -x\n * y; x\n ? y\n : z;",
        "literals": r'''x=$~53; y=$xFF; z=0Xabcdef0123456789abcdef; a=$'Z'; b='é'; c='ABCD'; #str="A\0B\x00C\x80\n\"\\\q";''',
        "unicode strings": 'x="é🌈";\ny="ζ";',
        "function": "function Ns.Foo(a,b c) instance(X, Y) global(Z) local(P q) (p=a;b?x+=q;); ns.foo(1,2,3);",
        "while forms": "while(i+=1;i<30);while(x)\n(x-=1); y=while(k+=1;k<5);",
        "while newline": "i=0;while\n// name/group whitespace\n\n(i+=1;i<3);while /*comment*/\n(i<5)\n(i+=1);x=while\r\n(i+=1;i<7);",
        "while separator": "while;\n(i+=1;i<3);",
        "sequences": "();(a;b;);loop(3,x+=1;y+=2);loop(4,);a[];spl(3)=slider(8);",
        "ternary": "a?:b;c?d;e?(x;y):(z);a?b?c:d:e;",
        "comments": "/*multi\nline*/ X=1;//stuff\n#A=\"hi\";#A+=\"there\";",
        "defer": "t=defer(x=1;defer(y=x+1));t=defer_after(t,z=2);t=defer_arena(a,t,0[2]=3);",
        "reduce": "t=defer_for(i,16,i*i);r=defer_reduce(j,16,SUM,0,j);",
        "empty deferred": "defer();defer_for(i,3,);defer_reduce(j,2,PRODUCT,1,);",
        "numeric rounding": "x=1e309;y=1e-999;z=-0.0;w=9007199254740993;",
        "bad character": "x=`;",
        "unterminated comment": "x=1;/*oops",
        "unterminated string": 'x="oops',
        "unterminated escape": 'x="oops\\',
        "long character": "x='ABCDE';",
        "invalid lvalue": "(a+b)=1;",
        "bad argument": "sin(,1);",
        "bad while": "while();",
        "missing delimiter": "x[2;",
        "invalid reduce": "defer_reduce(i,2,FOO,0,i);",
        "invalid task index": "defer_for(slider1,2,1);",
        "index expression": "defer_for(i+1,2,1);",
    }
    for name, source in snippets.items():
        yield name, dict(protocol=1, stage="frontend", source=source, snippet=True, baseLine=7)
    for key,value in [("resolve","yes"),("snippet",1),("baseLine",0),("baseLine",True),("sourcePath",""),("searchRoots","."),("searchRoots",[""]),("optimization",2)]:
        yield "protocol:"+key+repr(value), dict(protocol=1,stage="frontend",source="x=1;",**{key:value})
    # Seeded generated expressions check nested precedence, not just fixed examples.
    rng = random.Random(1515)
    def expression(depth):
        if depth == 0: return rng.choice(["a", "b", "3", ".25", "mem[2]", "sin(x)"])
        return "("+expression(depth-1)+rng.choice(["+","-","*","/","^","<","===","&&","||","~"])+expression(depth-1)+")"
    for i in range(100):
        yield f"generated-{i}", dict(protocol=1, stage="frontend", source="x="+expression(4)+";", snippet=True)
    package = root / "package"
    package.mkdir()
    (package / "leaf.jsfx-inc").write_text("@init\nfunction f(x)(x*.5)\n@sample\nspl0=123;\n", encoding="utf-8")
    (package / "left.jsfx-inc").write_text("import leaf.jsfx-inc\n@init\nx=1;\n", encoding="utf-8")
    (package / "right.jsfx-inc").write_text("import leaf.jsfx-inc\n@init\ny=2;\n", encoding="utf-8")
    (package / "comment-only").write_text("// library\n", encoding="utf-8")
    (package / "sectionless").write_text("x=1;\n", encoding="utf-8")
    (package / "cycle").write_text("import cycle\n@init\nx=1;", encoding="utf-8")
    for folder in ["one","two"]:
        (package / folder).mkdir()
        (package / folder / "ambiguous").write_text("@init\nx=1;", encoding="utf-8")
    (root / "outside").write_text("@init\nx=1;", encoding="utf-8")
    for name, source in {
        "diamond and override": "import left.jsfx-inc\nimport right.jsfx-inc\n@sample\nspl0=f(spl0);",
        "empty root overrides fallback": "import leaf.jsfx-inc\n@sample\n",
        "fallback": "import leaf.jsfx-inc\n@block\nx=1;",
        "metadata wildcard": "provides:\n dependencies/*\nabout:\n someone's tool\nimport leaf.jsfx-inc\n@sample\nspl0=2;",
        "comments masking": '/* import missing */\nimport leaf.jsfx-inc\n@sample\nx="import missing";/*\nimport missing\n*/',
        "repeated init": "import leaf.jsfx-inc\n@init\nx=1;\n@init\ny=2;",
        "mixed section order": "import leaf.jsfx-inc\n@faust block\nprocess=_,_;\n@sample\nx=1;\n@faust sample\nprocess=_,_;\n@sample\nx=2;",
        "config preprocess": 'config: count "Count" 3 2 3\n@init\n<? printf("x=%d;", count); ?>\n',
        "config hex": 'config: count "Count" -$xA\n@init\n<? printf("x=%d;", count); ?>',
        "config mask": 'config: count "Count" $~3\n@init\n<? printf("x=%d;", count); ?>',
        "config wide hex": 'config: count "Count" $x123456789abcdef0123456789\n@init\n<? printf("x=%.17g;", count); ?>',
        "config invalid hex float": 'config: count "Count" 0x1.2\n@init\n<? printf("x=%d;", count); ?>',
        "config non-finite": 'config: count "Count" inf\n@init\n<? printf("x=%d;", count); ?>',
        "comment-only import": "import comment-only\n@init\nx=1;",
        "cycle": "import cycle\n@init\nx=1;",
        "missing": "import absent\n@init\nx=1;",
        "outside root": "import ../outside\n@init\nx=1;",
        "ambiguous": "import ambiguous\n@init\nx=1;",
        "sectionless": "import sectionless\n@init\nx=1;",
        "duplicate section": "@sample\nx=1;\n@sample\nx=2;",
        "invalid import": "import\n@init\nx=1;",
        "trailing import": "import leaf.jsfx-inc bad\n@init\nx=1;",
        "duplicate config": 'config: x "X" 1\nconfig: X "X" 1\n@init\nx=1;',
        "bad config": 'config: x "X" $~54\n@init\nx=1;',
        "preprocessor failure": "@init\n<? invalid( ?>",
    }.items():
        yield name, dict(protocol=1, stage="frontend", source=source, resolve=True,
                         sourcePath=str(package / "main.jsfx"), packageRoot=str(package))


def comparable(result):
    if result["ok"]:
        return dict(ok=True, resolution=result["resolution"], sections=result["sections"])
    error = result["diagnostics"][0]
    if error["phase"] == "source":
        # Resolver wording includes implementation-specific search details.
        categories = ["Cyclic", "nesting exceeds", "outside", "ambiguous", "missing import", "sectionless", "duplicate @", "invalid import", "duplicate config", "invalid config"]
        category = next((s for s in categories if s in error["message"]), "preprocessing")
        return dict(ok=False, phase="source", category=category)
    return dict(ok=False, phase=error["phase"], message=error["message"], line=error.get("line",0), column=error.get("column",0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--all-plugins", action="store_true")
    parser.add_argument("--freeze-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    dependency_hashes = {}
    archive = zipfile.ZipFile(args.freeze_reference, "w", zipfile.ZIP_DEFLATED) if args.freeze_reference else None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-frontend-", dir=args.output.parent.resolve()) as temp:
        work = Path(temp)
        cases = list(fixtures(work))
        if archive:
            for file in work.rglob("*"):
                if file.is_file(): archive.writestr("fixtures/"+file.relative_to(work).as_posix(),file.read_bytes())
        if args.catalog:
            for row in json.loads(args.catalog.read_text(encoding="utf-8")):
                path = Path(row["source"])
                cases.append(("catalog:"+row["slug"],dict(protocol=1, stage="frontend", source=path.read_text(encoding="utf-8-sig"), resolve=True, sourcePath=str(path))))
        if args.all_plugins:
            for config in sorted((ROOT / "plugins").rglob("plugin.json")):
                settings = json.loads(config.read_text(encoding="utf-8-sig"))
                if settings.get("pluginType", "jsfx") != "jsfx": continue
                path = (config.parent / settings["entry"]).resolve()
                cases.append(("catalog:"+settings["slug"],dict(protocol=1, stage="frontend", source=path.read_text(encoding="utf-8-sig"), resolve=True, sourcePath=str(path))))
        for name, request in cases:
            started = time.perf_counter()
            reference = frontend(request)
            for file in reference.get("resolution",{}).get("dependencies",[]):
                path = Path(file)
                if path.is_file() and path.is_relative_to(ROOT):
                    dependency_hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            reference_seconds = time.perf_counter()-started
            if archive:
                # Relocate paths while preserving everything semantic, including spans.
                frozen = json.dumps(dict(name=name, request=request, expected=comparable(reference)), sort_keys=True, ensure_ascii=False)
                for origin, replacement in [(str(work.resolve()), "$FIXTURES"), (str(ROOT), "$REPO")]:
                    frozen = frozen.replace(json.dumps(origin)[1:-1], replacement)
                    frozen = frozen.replace(origin, replacement)
                archive.writestr(f"{len(rows):04d}.json", frozen)
            req, out = work / "request.json", work / "result.json"
            req.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            started = time.perf_counter()
            env = dict(os.environ, PATH="")
            process = subprocess.run([str(args.helper.resolve()), str(req), str(out)], capture_output=True, text=True, timeout=120, env=env)
            native_seconds = time.perf_counter()-started
            native = json.loads(out.read_text(encoding="utf-8"))
            diff = first_difference(comparable(reference),comparable(native))
            if name=="sectionless" and native.get("diagnostics",[{}])[0].get("file")!=str(work/"package"/"sectionless"):
                diff="Imported-file diagnostic lost its source filename"
            if process.returncode != (0 if native["ok"] else 1): diff = "Unexpected helper exit: "+str(process.returncode)
            row = dict(name=name, passed=diff is None, accepted=reference["ok"], referenceSeconds=reference_seconds,
                       nativeProcessSeconds=native_seconds, nativePhaseSeconds=native.get("phaseSeconds"), difference=diff,
                       sourceSha256=hashlib.sha256(request["source"].encode()).hexdigest())
            rows.append(row)
            if diff:
                fail = args.output.parent / "failures"
                fail.mkdir(exist_ok=True,parents=True)
                stem = str(len(rows))
                (fail / (stem+"-request.json")).write_text(json.dumps(request),encoding="utf-8")
                (fail / (stem+"-reference.json")).write_text(json.dumps(reference),encoding="utf-8")
                (fail / (stem+"-native.json")).write_text(json.dumps(native),encoding="utf-8")
            print(json.dumps(dict(name=name, passed=diff is None, difference=diff)), flush=True)
    report = dict(protocol=1, passed=all(r["passed"] for r in rows), cases=rows, dependencyHashes=dependency_hashes,
                  referenceFiles={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [ROOT/"dsp_jsfx_aot.py", ROOT/"scripts/jsfx_source.py", ROOT/"scripts/jsfx_tasks_compiler.py"]})
    if archive:
        archive.writestr("manifest.json", json.dumps({k:v for k,v in report.items() if k != "cases"}, indent=2))
        archive.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
