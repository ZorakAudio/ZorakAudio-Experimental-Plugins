"""Consume the native frontend's AST through the existing production pipeline.

No Python parse fallback: a native failure is reported and the plugin keeps its
previous program. Lowering, symbol analysis, Faust planning and LLVM emission are
the same production passes used by the standard compiler.
"""
from dataclasses import fields
import json
import os
from pathlib import Path
import struct
import subprocess


def run_native(root, job, source, origin=None):
    helper = root / ("jsfx_frontend.exe" if os.name == "nt" else "jsfx_frontend")
    if not helper.is_file():
        raise ValueError("C++ frontend is missing from the compiler bundle; reinstall the complete new runtime")
    request = dict(protocol=1, stage="frontend", source=source, resolve=origin is not None)
    if origin is not None:
        request["sourcePath"] = str(origin)
    input_file, output_file = job / "native-request.json", job / "native-result.json"
    input_file.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    output_file.unlink(missing_ok=True)
    process = subprocess.run([str(helper), str(input_file), str(output_file)],
                             capture_output=True, timeout=110,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if not output_file.is_file():
        raise ValueError("C++ frontend exited without a result (exit %d)" % process.returncode)
    result = json.loads(output_file.read_text(encoding="utf-8"))
    if result.get("protocol") != 1 or result.get("backend") != "cpp-experimental":
        raise ValueError("Invalid C++ frontend response")
    if process.returncode or not result.get("ok"):
        diagnostic = (result.get("diagnostics") or [{}])[0]
        location = diagnostic.get("file") or "Expanded source"
        if diagnostic.get("line"):
            location += ":%d:%d" % (diagnostic["line"], diagnostic.get("column", 0))
        raise ValueError("%s: %s" % (location, diagnostic.get("message", "Native frontend failed")))
    for section in result.get("sections", []):
        section.pop("tokens", None)  # Comparison data isn't needed by lowering.
    return result


def parser_from_result(compiler, result):
    sections = {(s["source"], s["line"]): s["ast"]
                for s in result["sections"] if "ast" in s}
    types = {name: getattr(compiler, name) for name in
             ("Num", "StrLit", "Var", "Index", "Unary", "Binary", "Assign", "Call",
              "Loop", "Ternary", "Seq", "If", "While", "FunctionDef")}

    def parse(source, line):
        if (source, line) not in sections:
            raise ValueError("C++ frontend did not return the requested section")
        next_id = 0

        def decode(value):
            nonlocal next_id
            if isinstance(value, list):
                return [decode(v) for v in value]
            if not isinstance(value, dict):
                return value
            name = value.get("type")
            if name not in types:
                raise ValueError("Unknown native AST node: " + str(name))
            cls = types[name]
            expected = {f.name for f in fields(cls)} - {"id"}
            if value.keys() - {"type"} != expected:
                raise ValueError("Native AST fields do not match " + name)
            span = value["span"]
            args = {k: decode(v) for k, v in value.items() if k not in ("type", "span", "value")}
            if "value" in value:
                v = value["value"]
                args["value"] = (struct.unpack(">d", bytes.fromhex(v))[0] if name == "Num"
                                 else bytes.fromhex(v).decode("utf-8") if name == "StrLit" else decode(v))
            # IDs label compiler blocks/analysis caches; they are not guest state.
            # Assign unique postorder IDs per section, as the reference parser does.
            next_id += 1
            return cls(id=next_id, span=compiler.Span(span["line"], span["col"]), **args)

        return decode(sections[(source, line)])

    return parse
