"""Differential oracle: invokes the production resolver, lexer and parser unchanged."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from pathlib import Path
import re
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dsp_jsfx_aot as compiler
from scripts.jsfx_source import SourceResolver


def normalize(value):
    if isinstance(value, compiler.Node):
        return dict(type=type(value).__name__, **{
            f.name: normalize(getattr(value, f.name)) for f in fields(value) if f.name != "id"
        })
    if is_dataclass(value):
        return {f.name: normalize(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, float):
        return struct.pack(">d", value).hex()
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [normalize(x) for x in value]
    return value


def frontend(request):
    result = dict(protocol=1, backend="python-reference", stage="frontend", ok=False)
    phase = "protocol"
    try:
        if type(request.get("protocol")) is not int or request["protocol"] != 1:
            raise ValueError("Expected protocol 1")
        if request.get("stage") != "frontend":
            raise ValueError("Only frontend stage is implemented")
        if not isinstance(request.get("source"), str):
            raise ValueError("source must be a string")
        allowed={"protocol","stage","source","resolve","sourcePath","packageRoot","searchRoots","snippet","baseLine"}
        unknown=sorted(request.keys()-allowed)
        if unknown: raise ValueError("Unknown request field: "+unknown[0])
        for name in ["resolve","snippet"]:
            if name in request and type(request[name]) is not bool: raise ValueError(name+" must be boolean")
        for name in ["sourcePath","packageRoot"]:
            if name in request and (not isinstance(request[name],str) or not request[name]): raise ValueError(name+" must be a nonempty string")
        if "baseLine" in request and (type(request["baseLine"]) is not int or not 1<=request["baseLine"]<=2147483647): raise ValueError("baseLine must be a positive 32-bit integer")
        if "searchRoots" in request:
            if not isinstance(request["searchRoots"],list): raise ValueError("searchRoots must be an array")
            if any(not isinstance(p,str) or not p for p in request["searchRoots"]): raise ValueError("searchRoots entries must be nonempty strings")
        source = request["source"]
        phase = "limits"
        if len(source.encode("utf-8")) > 4194304:
            raise ValueError("Source exceeds 4 MiB")
        phase = "source"
        if request.get("resolve"):
            path = Path(request["sourcePath"]).resolve()
            resolver = SourceResolver(Path(request.get("packageRoot") or path.parent),
                                      search_roots=tuple(map(Path, request.get("searchRoots", []))))
            resolved = resolver.expand(path, text=source)
            resolution = dict(text=resolved.text, dependencies=list(map(str, resolved.dependencies)),
                              sectionSources={k: list(map(str, v)) for k, v in resolved.section_sources.items()})
            source = resolved.text
        else:
            resolution = dict(text=source, dependencies=[], sectionSources={})
        result["resolution"] = resolution
        phase = "limits"
        if len(source.encode("utf-8")) > 4194304:
            raise ValueError("Expanded source exceeds 4 MiB")
        sections = ([dict(name="snippet", source=source, line=request.get("baseLine", 1))]
                    if request.get("snippet") else
                    [dict(name=k, source=s, line=l) for k, (s, l) in compiler.extract_sections(source).items()])
        for section in sections:
            if section["name"] == "faust" or section["name"].startswith("faust_"):
                section["opaque"] = True
                continue
            phase = "lexer"
            lexer = compiler.Lexer(section["source"], base_line=section["line"])
            tokens = []
            while True:
                token = lexer.next()
                t = normalize(token)
                if token.kind == "str":
                    t["text"] = token.text.encode("utf-8").hex()
                tokens.append(t)
                if token.kind == "eof":
                    break
            section["tokens"] = tokens
            phase = "parser"
            section["ast"] = normalize(compiler.Parser(section["source"], base_line=section["line"]).parse_program())
            def encode_literals(value):
                if isinstance(value, dict):
                    if value.get("type") == "StrLit":
                        value["value"] = value["value"].encode("utf-8").hex()
                    else:
                        for v in value.values(): encode_literals(v)
                elif isinstance(value, list):
                    for v in value: encode_literals(v)
            encode_literals(section["ast"])
        result.update(ok=True, sections=sections)
    except Exception as error:
        message = str(error)
        match = re.search(r" at (\d+):(\d+)", message)
        result["diagnostics"] = [dict(severity="error", phase=phase,
                                     message=message.split(" at ")[0] if match else message,
                                     file=request.get("sourcePath"),
                                     line=int(match[1]) if match else 0,
                                     column=int(match[2]) if match else 0)]
    return result
