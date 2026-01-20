#!/usr/bin/env python3
"""
dsp_jsfx_to_llvm.py

DSP-JSFX -> LLVM IR (llvmlite) compiler front-end.

Contract (DSP-JSFX):
- Sections: @init, @slider, @block, @sample (any may be missing)
- DSP-only: no @gfx, no strings, no MIDI, no preprocessor.
- Type: everything is double.
- Variables: spl0..spl63, slider1..slider64, user vars (persistent), builtins:
    - mem  (numeric base pointer index = 0.0)
    - srate (read/write state field)
    - samplesblock (read/write state field; host should set before @block)
- Memory:
    - mem[...] is heap-backed (double*).
    - Pointer-style indexing is allowed: a[b] == mem[(int)a + (int)b].
      (Pointer values are numeric indices into mem; mem itself is index 0.)
    - Indices convert via truncation (fptosi) and clamp to >= 0.
    - If idx >= memN, IR emits a call to external:
        void jsfx_ensure_mem(State* st, i64 needed);
      which must grow (and update st->mem, st->memN).

Language subset:
- Statements: if/else, while, expression statements.
- Expressions: numbers, identifiers, unary + - !, binary + - * /,
  comparisons, short-circuit && ||, ternary ?:, assignments (=, +=, -=, *=, /=),
  parentheses, sequence blocks: ( a; b; c; ) returning last expr value.
- loop(count, body) expression: repeats body count times, returns last value (or 0).

Output:
- LLVM IR module with:
    void jsfx_init(State* st)
    void jsfx_slider(State* st)
    void jsfx_block(State* st)
    void jsfx_sample(State* st)

Usage:
  python dsp_jsfx_to_llvm.py input.jsfx > out.ll
  python dsp_jsfx_to_llvm.py input.jsfx --out out.ll
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import shutil
import subprocess
import tempfile


from llvmlite import ir
from llvmlite import binding as llvm


# -----------------------------
# Lexer
# -----------------------------

@dataclass(frozen=True)
class Span:
    line: int
    col: int

@dataclass(frozen=True)
class Tok:
    kind: str   # 'eof','eol','num','ident','kw','op','punc','semi'
    text: str
    span: Span

_MULTI_OPS = [
    "==","!=","<=",">=",
    "+=","-=","*=","/=",
    "&&","||",
    "<<", ">>",
]

_SINGLE = set("()[]{},;:+-*/=<>&|!?:^%~\n")


class Lexer:
    def __init__(self, src: str, base_line: int = 1):
        self.src = src
        self.i = 0
        self.line = base_line
        self.col = 1

    def _peek(self, n: int = 0) -> str:
        j = self.i + n
        return self.src[j] if j < len(self.src) else "\0"

    def _adv(self, n: int = 1) -> None:
        for _ in range(n):
            if self.i >= len(self.src):
                return
            c = self.src[self.i]
            self.i += 1
            if c == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1

    def _span(self) -> Span:
        return Span(self.line, self.col)

    def next(self) -> Tok:
        while True:
            if self.i >= len(self.src):
                return Tok("eof", "", self._span())

            c = self._peek()

            # whitespace (but keep newlines)
            if c in " \t\r":
                self._adv()
                continue

            # newline
            if c == "\n":
                sp = self._span()
                self._adv()
                return Tok("eol", "\n", sp)

            # line comment //
            if c == "/" and self._peek(1) == "/":
                while self._peek() not in ("\n", "\0"):
                    self._adv()
                continue

            # block comment /* ... */
            if c == "/" and self._peek(1) == "*":
                self._adv(2)
                while True:
                    if self._peek() == "\0":
                        raise SyntaxError("Unterminated /* comment */")
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._adv(2)
                        break
                    self._adv()
                continue

            sp = self._span()

            # multi-char operators
            two = c + self._peek(1)
            if two in _MULTI_OPS:
                self._adv(2)
                return Tok("op", two, sp)

            # number
            if c.isdigit() or (c == "." and self._peek(1).isdigit()):
                m = re.match(r"[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?|\.[0-9]+([eE][+-]?[0-9]+)?", self.src[self.i:])
                assert m
                txt = m.group(0)
                self._adv(len(txt))
                return Tok("num", txt, sp)

            # identifier / keyword
            if c.isalpha() or c == "_" or c == "$":
                m = re.match(r"[$A-Za-z_][$A-Za-z0-9_]*", self.src[self.i:])
                assert m
                txt = m.group(0)
                self._adv(len(txt))
                kind = "kw" if txt in ("if", "else", "while") else "ident"
                return Tok(kind, txt, sp)


            # single char tokens/operators
            if c in _SINGLE:
                self._adv()
                if c == ";":
                    return Tok("semi", c, sp)
                if c in "();,[]{}":
                    return Tok("punc", c, sp)
                if c in "+-*/=<>&|!?:%~" or c == "^":
                    return Tok("op", c, sp)


                raise SyntaxError(f"Lexer internal: unexpected single token {c!r}")

            raise SyntaxError(f"Unexpected character {c!r} at {sp.line}:{sp.col}")


# -----------------------------
# AST
# -----------------------------

class Node:
    id: int
    span: Span

@dataclass
class Num(Node):
    id: int
    span: Span
    value: float

@dataclass
class Var(Node):
    id: int
    span: Span
    name: str

@dataclass
class Index(Node):
    id: int
    span: Span
    base: Node
    index: Node

@dataclass
class Unary(Node):
    id: int
    span: Span
    op: str
    a: Node

@dataclass
class Binary(Node):
    id: int
    span: Span
    op: str
    l: Node
    r: Node

@dataclass
class Assign(Node):
    id: int
    span: Span
    op: str      # =, +=, ...
    target: Node # Var or Index
    value: Node

@dataclass
class Call(Node):
    id: int
    span: Span
    fn: str
    args: List[Node]

@dataclass
class Loop(Node):
    id: int
    span: Span
    count: Node
    body: Node

@dataclass
class Ternary(Node):
    id: int
    span: Span
    cond: Node
    then: Node
    els: Node

@dataclass
class Seq(Node):
    id: int
    span: Span
    items: List[Node]

@dataclass
class If(Node):
    id: int
    span: Span
    cond: Node
    then: Node
    els: Optional[Node]

@dataclass
class While(Node):
    id: int
    span: Span
    cond: Node
    body: Node

@dataclass
class FunctionDef(Node):
    id: int
    span: Span
    name: str
    params: List[str]
    locals: List[str]
    body: Node



# -----------------------------
# Pratt parser
# -----------------------------

# Higher number = tighter binding.
_PRECEDENCE: Dict[str, int] = {
    "=": 1, "+=": 1, "-=": 1, "*=": 1, "/=": 1,
    "?": 2,  # handled specially, but used as threshold
    "||": 3, "|": 3,
    "&&": 4,
    "==": 5, "!=": 5,
    "<": 6, "<=": 6, ">": 6, ">=": 6,
    "+": 7, "-": 7,
    "*": 8, "/": 8,
    "^": 9,
}
_PRECEDENCE.update({
    "|": 3,
    "&": 5,
    "<<": 6, ">>": 6,
    "%": 8,
})

_TERNARY_PREC = 2
_RIGHT_ASSOC = {"=", "+=", "-=", "*=", "/="}



class Parser:
    def __init__(self, src: str, base_line: int = 1):
        self.src_text = src
        self.base_line = base_line
        self.lex = Lexer(src, base_line=base_line)
        self.cur = self.lex.next()
        self.nxt = self.lex.next()
        self._next_id = 1

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _adv(self) -> None:
        self.cur = self.nxt
        self.nxt = self.lex.next()

    def _eat(self, kind: str, text: Optional[str] = None) -> Tok:
        if self.cur.kind != kind:
            raise SyntaxError(self._fmt_err(f"Expected {kind}, got {self.cur.kind} {self.cur.text!r}"))
        if text is not None and self.cur.text != text:
            raise SyntaxError(self._fmt_err(f"Expected {text!r}, got {self.cur.text!r}"))
        t = self.cur
        self._adv()
        return t

    def _skip_seps(self) -> None:
        while self.cur.kind in ("eol", "semi"):
            self._adv()

    def _fmt_err(self, msg: str) -> str:
        line = getattr(self.cur.span, "line", 0) or 0
        col  = getattr(self.cur.span, "col", 0) or 0

        # Show the exact line from this section snippet, but report file-absolute line:col.
        src_line = ""
        try:
            lines = self.src_text.splitlines()
            rel = line - self.base_line + 1  # 1-based within snippet
            if 1 <= rel <= len(lines):
                src_line = lines[rel - 1]
        except Exception:
            src_line = ""

        caret = ""
        if src_line:
            c = max(1, min(col, len(src_line) + 1))
            caret = " " * (c - 1) + "^"

        loc = f"{line}:{col}" if line and col else "?:?"
        if src_line:
            return f"{msg} at {loc}\n{src_line}\n{caret}"
        return f"{msg} at {loc}"
    def parse_program(self) -> List[Node]:
        out: List[Node] = []
        self._skip_seps()
        while self.cur.kind != "eof":
            out.append(self.parse_stmt())
            self._skip_seps()
        return out

    def parse_stmt(self) -> Node:
        if self.cur.kind == "kw" and self.cur.text == "if":
            return self.parse_if()
        if self.cur.kind == "kw" and self.cur.text == "while":
            return self.parse_while()
        if self.cur.kind == "ident" and self.cur.text == "function":
            return self.parse_function_def()
        return self.parse_expr(0)


    def parse_if(self) -> Node:
        kw = self._eat("kw", "if")
        self._eat("punc", "(")
        cond = self.parse_expr(0)
        self._eat("punc", ")")

        self._skip_seps()
        then = self.parse_expr(0)
        self._skip_seps()

        els = None
        if self.cur.kind == "kw" and self.cur.text == "else":
            self._adv()
            self._skip_seps()
            els = self.parse_expr(0)
            self._skip_seps()
        return If(self._new_id(), kw.span, cond, then, els)

    def parse_while(self) -> Node:
        kw = self._eat("kw", "while")
        self._eat("punc", "(")
        cond = self.parse_expr(0)
        self._eat("punc", ")")
        self._skip_seps()
        body = self.parse_expr(0)
        return While(self._new_id(), kw.span, cond, body)
    
    def parse_function_def(self) -> Node:
        # function name(a,b,c) local(x,y) ( body );
        t_fun = self._eat("ident", "function")

        if self.cur.kind != "ident":
            raise SyntaxError(self._fmt_err("Expected function name after 'function'"))
        t_name = self._eat("ident")
        fn_name = t_name.text

        # params
        self._eat("punc", "(")
        params: List[str] = []
        if not (self.cur.kind == "punc" and self.cur.text == ")"):
            while True:
                if self.cur.kind != "ident":
                    raise SyntaxError(self._fmt_err("Expected parameter name"))
                params.append(self._eat("ident").text)
                if self.cur.kind == "punc" and self.cur.text == ",":
                    self._adv()
                    continue
                break
        self._eat("punc", ")")

        # optional local(...)
        locals_: List[str] = []
        self._skip_seps()

        if self.cur.kind == "ident" and self.cur.text == "local":
            self._adv()  # consume 'local'
            self._eat("punc", "(")
            self._skip_seps()

            if not (self.cur.kind == "punc" and self.cur.text == ")"):
                while True:
                    self._skip_seps()

                    # allow trailing comma/newlines before ')'
                    if self.cur.kind == "punc" and self.cur.text == ")":
                        break

                    if self.cur.kind != "ident":
                        raise SyntaxError(self._fmt_err("Expected local variable name"))
                    locals_.append(self._eat("ident").text)

                    self._skip_seps()
                    if self.cur.kind == "punc" and self.cur.text == ",":
                        self._adv()
                        continue
                    break

            self._skip_seps()
            self._eat("punc", ")")

        self._skip_seps()



        self._skip_seps()

        # body must be a parenthesized expression/sequence
        if not (self.cur.kind == "punc" and self.cur.text == "("):
            raise SyntaxError(self._fmt_err("Expected '(' to start function body"))
        body = self.parse_primary()  # parses (...) as expr or Seq

        self._skip_seps()
        # optional trailing semicolon
        if self.cur.kind == "semi":
            self._adv()

        return FunctionDef(self._new_id(), t_fun.span, fn_name, params, locals_, body)


    def parse_expr(self, min_prec: int) -> Node:
        lhs = self.parse_prefix()

        while True:
            # assignment / binary ops
            if self.cur.kind != "op":
                break
            op = self.cur.text
            if op == "?" or op == ":":
                break
            prec = _PRECEDENCE.get(op)
            if prec is None or prec < min_prec:
                break

            assoc_right = (op in _RIGHT_ASSOC)
            self._adv()
            rhs = self.parse_expr(prec + (0 if assoc_right else 1))

            if op in _RIGHT_ASSOC:
                if not isinstance(lhs, (Var, Index)):
                    raise SyntaxError(self._fmt_err("Assignment target must be a variable or index"))
                lhs = Assign(self._new_id(), lhs.span, op, lhs, rhs)
            else:
                lhs = Binary(self._new_id(), lhs.span, op, lhs, rhs)

        # ternary (JSFX allows "cond ? then" with implicit else 0)
        if self.cur.kind == "op" and self.cur.text == "?" and _TERNARY_PREC >= min_prec:
            q = self.cur
            self._adv()  # consume '?'
            self._skip_seps()

            then = self.parse_expr(0)
            self._skip_seps()

            if self.cur.kind == "op" and self.cur.text == ":":
                self._adv()
                self._skip_seps()
                els = self.parse_expr(_TERNARY_PREC)
            else:
                # no ':' => else is 0.0
                els = Num(self._new_id(), q.span, 0.0)

            lhs = Ternary(self._new_id(), q.span, lhs, then, els)


        return lhs

    def parse_prefix(self) -> Node:
        if self.cur.kind == "op" and self.cur.text in ("+", "-", "!"):
            t = self.cur
            self._adv()
            a = self.parse_prefix()
            return Unary(self._new_id(), t.span, t.text, a)
        return self.parse_postfix()

    def parse_postfix(self) -> Node:
        node = self.parse_primary()
        while True:
            # call
            if self.cur.kind == "punc" and self.cur.text == "(":
                sp = self.cur.span
                self._adv()  # consume '('

                if not isinstance(node, Var):
                    raise SyntaxError(self._fmt_err("Can only call a named function"))
                fn = node.name

                # ---- SPECIAL: loop(count, body) where body may be un-comma'd multiline sequence ----
                if fn == "loop":
                    self._skip_seps()

                    # count expr
                    count = self.parse_expr(0)
                    self._skip_seps()

                    # optional comma after count
                    if self.cur.kind == "punc" and self.cur.text == ",":
                        self._adv()
                    self._skip_seps()

                    # body: parse as sequence until ')'
                    # Accept either a single expr or multiple separated by ;/newline.
                    body_first = None
                    items: List[Node] = []

                    # empty body => 0
                    if self.cur.kind == "punc" and self.cur.text == ")":
                        self._adv()
                        node = Loop(self._new_id(), sp, count, Num(self._new_id(), sp, 0.0))
                        continue

                    body_first = self.parse_stmt_or_expr_for_seq()
                    items.append(body_first)

                    while True:
                        self._skip_seps()
                        if self.cur.kind == "punc" and self.cur.text == ")":
                            self._adv()
                            break
                        items.append(self.parse_stmt_or_expr_for_seq())

                    body_node: Node = items[0] if len(items) == 1 else Seq(self._new_id(), sp, items)
                    node = Loop(self._new_id(), sp, count, body_node)
                    continue
                # ---- end loop special ----

                # generic call (your existing improved separator-skipping version)
                args: List[Node] = []
                self._skip_seps()
                if not (self.cur.kind == "punc" and self.cur.text == ")"):
                    while True:
                        self._skip_seps()
                        args.append(self.parse_expr(0))
                        self._skip_seps()
                        if self.cur.kind == "punc" and self.cur.text == ",":
                            self._adv()
                            continue
                        break
                self._skip_seps()
                self._eat("punc", ")")
                node = Call(self._new_id(), sp, fn, args)
                continue


            # indexing
            if self.cur.kind == "punc" and self.cur.text == "[":
                sp = self.cur.span
                self._adv()
                self._skip_seps()
                idx = self.parse_expr(0)
                self._skip_seps()
                self._eat("punc", "]")
                node = Index(self._new_id(), sp, node, idx)
                continue

            break
        return node

    def parse_primary(self) -> Node:
        if self.cur.kind == "num":
            t = self._eat("num")
            return Num(self._new_id(), t.span, float(t.text))

        if self.cur.kind == "ident":
            t = self._eat("ident")
            return Var(self._new_id(), t.span, t.text)

        if self.cur.kind == "punc" and self.cur.text == "(":
            sp = self.cur.span
            self._adv()
            # allow leading newlines/semicolons inside paren-sequences
            self._skip_seps()

            # allow empty paren group: () or ( \n )
            if self.cur.kind == "punc" and self.cur.text == ")":
                self._adv()
                return Seq(self._new_id(), sp, [])

            first = self.parse_stmt_or_expr_for_seq()
            if self.cur.kind == "punc" and self.cur.text == ")":
                self._adv()
                return first
            items = [first]
            while True:
                # consume any number of separators between items
                self._skip_seps()

                # end of sequence
                if self.cur.kind == "punc" and self.cur.text == ")":
                    self._adv()
                    break

                # if we didn't hit ')', we must be at the start of another stmt/expr
                items.append(self.parse_stmt_or_expr_for_seq())


            return Seq(self._new_id(), sp, items)

        raise SyntaxError(self._fmt_err("Expected number, identifier, or '('"))

    def parse_stmt_or_expr_for_seq(self) -> Node:
        if self.cur.kind == "kw" and self.cur.text == "if":
            return self.parse_if()
        if self.cur.kind == "kw" and self.cur.text == "while":
            return self.parse_while()
        return self.parse_expr(0)


# -----------------------------
# JSFX section extraction
# -----------------------------

_SECTION_RE = re.compile(r"^\s*@([A-Za-z_][A-Za-z0-9_]*)\s*$")

def extract_sections(jsfx_text: str) -> Dict[str, Tuple[str, int]]:
    """
    Returns {section_name: (section_text, start_line)} where start_line is the
    first line number of section_text in the original file (1-based).
    """
    lines = jsfx_text.splitlines(True)  # keep newlines
    sections: Dict[str, List[str]] = {}
    starts: Dict[str, int] = {}

    current: Optional[str] = None
    for i, ln in enumerate(lines):
        m = _SECTION_RE.match(ln)
        if m:
            current = m.group(1)
            sections.setdefault(current, [])
            starts.setdefault(current, i + 2)  # first line after marker
            continue
        if current is not None:
            sections[current].append(ln)

    out: Dict[str, Tuple[str, int]] = {}
    for k, v in sections.items():
        out[k] = ("".join(v), starts.get(k, 1))
    return out


# -----------------------------
# Symbol table (stable var indices)
# -----------------------------

BUILTIN_NAMES = {"mem", "srate", "samplesblock"}

@dataclass(frozen=True)
class SymRef:
    kind: str    # spl, slider, var, builtin
    index: int   # spl/slider/var index; builtin field id

class SymTable:
    def __init__(self, user_vars: Dict[str, int]):
        self.vars = dict(user_vars)  # stable mapping

    def resolve(self, name: str) -> SymRef:
        if name.startswith("spl"):
            suf = name[3:]
            if suf.isdigit():
                idx = int(suf)
                if 0 <= idx < 64:
                    return SymRef("spl", idx)
                raise ValueError(f"Invalid spl index: {name}")
            # NOT spl<number> => it's a normal variable like "splitSamp"


        if name.startswith("slider"):
            suf = name[6:]
            if suf.isdigit():
                n = int(suf)
                idx = n - 1
                if 0 <= idx < 64:
                    return SymRef("slider", idx)
                raise ValueError(f"Invalid slider index: {name}")
            # NOT slider<number> => normal var like "sliderGainThing"


        if name == "mem":
            # numeric base index of heap is always 0.0
            return SymRef("builtin", 0)

        if name == "srate":
            return SymRef("builtin", 1)

        if name == "samplesblock":
            return SymRef("builtin", 2)

        if name not in self.vars:
            raise ValueError(f"Unknown variable {name!r} (not declared by analysis)")
        return SymRef("var", self.vars[name])


def collect_user_vars(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef]) -> Dict[str, int]:
    names: Set[str] = set()

    def rec(n: Node, locals: Set[str]) -> None:
        if isinstance(n, Var):
            if n.name in locals:
                return
            if n.name in BUILTIN_NAMES:
                return
            # skip only real spl registers spl0..spl63
            if n.name.startswith("spl") and n.name[3:].isdigit():
                return

            # skip only real slider registers slider1..slider64
            if n.name.startswith("slider") and n.name[6:].isdigit():
                return

            if n.name.startswith("$"):   # treat $... as special/const, not state vars
                return
            names.add(n.name)
            return

        if isinstance(n, Num):
            return
        if isinstance(n, Index):
            rec(n.base, locals); rec(n.index, locals); return
        if isinstance(n, Unary):
            rec(n.a, locals); return
        if isinstance(n, Binary):
            rec(n.l, locals); rec(n.r, locals); return
        if isinstance(n, Assign):
            rec(n.target, locals); rec(n.value, locals); return
        if isinstance(n, Call):
            for a in n.args: rec(a, locals)
            return
        if isinstance(n, Loop):
            rec(n.count, locals); rec(n.body, locals); return
        if isinstance(n, Ternary):
            rec(n.cond, locals); rec(n.then, locals); rec(n.els, locals); return
        if isinstance(n, Seq):
            for it in n.items: rec(it, locals)
            return
        if isinstance(n, If):
            rec(n.cond, locals); rec(n.then, locals)
            if n.els: rec(n.els, locals)
            return
        if isinstance(n, While):
            rec(n.cond, locals); rec(n.body, locals); return

        raise TypeError(type(n))

    # sections
    for prog in programs.values():
        for st in prog:
            rec(st, set())

    # function bodies (exclude params+locals)
    for f in fn_defs.values():
        localset = set(f.params) | set(f.locals)
        rec(f.body, localset)

    return {name: i for i, name in enumerate(sorted(names))}


def extract_function_defs(programs: Dict[str, List[Node]]) -> Tuple[Dict[str, FunctionDef], Dict[str, List[Node]]]:
    fns: Dict[str, FunctionDef] = {}
    out: Dict[str, List[Node]] = {}

    for sec, prog in programs.items():
        new_prog: List[Node] = []
        for n in prog:
            if isinstance(n, FunctionDef):
                # last one wins (matches JSFX “redefine” behavior loosely; good enough for now)
                fns[n.name] = n
            else:
                new_prog.append(n)
        out[sec] = new_prog

    return fns, out


# -----------------------------
# LLVM IR emission (llvmlite)
# -----------------------------

class LLVMModuleEmitter:
    def __init__(self, sym: SymTable):
        self.sym = sym

        self.double = ir.DoubleType()
        self.i1 = ir.IntType(1)
        self.i32 = ir.IntType(32)
        self.i64 = ir.IntType(64)

        # State layout:
        # 0: double spl[64]
        # 1: double sliders[64]
        # 2: double vars[NUM]
        # 3: double* mem
        # 4: i64 memN
        # 5: double srate
        # 6: double samplesblock
        self.var_cap = max(1, (max(sym.vars.values()) + 1) if sym.vars else 1)

        self.state_ty = ir.LiteralStructType([
            ir.ArrayType(self.double, 64),
            ir.ArrayType(self.double, 64),
            ir.ArrayType(self.double, self.var_cap),
            self.double.as_pointer(),
            self.i64,
            self.double,
            self.double,
        ])
        self.state_ptr = self.state_ty.as_pointer()

        self.module = ir.Module(name="dsp_jsfx_module")

        # extern ensure
        self.fn_ensure = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [self.state_ptr, self.i64]),
            name="jsfx_ensure_mem"
        )

        self._intrinsics: Dict[str, ir.Function] = {}
        self.user_fn_defs: Dict[str, FunctionDef] = {}
        self.user_fn_ir: Dict[str, ir.Function] = {}
        self._local_slots_stack: List[Dict[str, ir.Value]] = []


        self._buildins: Dict[str, ir.Function] = {}

    def _const_f64(self, v: float) -> ir.Constant:
        return ir.Constant(self.double, float(v))

    def _const_i64(self, v: int) -> ir.Constant:
        return ir.Constant(self.i64, int(v))

    def _truthy(self, x: ir.Value, builder: ir.IRBuilder) -> ir.Value:
        return builder.fcmp_ordered("!=", x, self._const_f64(0.0))

    def _declare_math(self, fn: str) -> ir.Function:
        if fn in self._intrinsics:
            return self._intrinsics[fn]

        if fn in ("sin", "cos", "sqrt", "fabs", "floor", "ceil"):
            name = {
                "sin": "llvm.sin.f64",
                "cos": "llvm.cos.f64",
                "sqrt": "llvm.sqrt.f64",
                "fabs": "llvm.fabs.f64",
                "floor": "llvm.floor.f64",
                "ceil": "llvm.ceil.f64",
            }[fn]

            f = ir.Function(self.module, ir.FunctionType(self.double, [self.double]), name=name)
            self._intrinsics[fn] = f
            return f

        if fn in ("pow", "exp", "log"):
            f = ir.Function(self.module, ir.FunctionType(self.double, [self.double, self.double]) if fn == "pow" else ir.FunctionType(self.double, [self.double]), name=fn)
            self._intrinsics[fn] = f
            return f

        raise ValueError(f"Unknown builtin {fn}")

    def _get_slot_ptr(self, builder: ir.IRBuilder, st: ir.Value, name: str) -> ir.Value:
        # Local variables (function params/locals) shadow globals
        if self._local_slots_stack:
            loc = self._local_slots_stack[-1].get(name)
            if loc is not None:
                return loc

        ref = self.sym.resolve(name)
        zero = ir.Constant(self.i32, 0)

        if ref.kind == "builtin":
            if ref.index == 0:  # mem constant
                raise ValueError("mem has no address")
            # srate field 5, samplesblock field 6
            field = 5 if ref.index == 1 else 6
            fld = ir.Constant(self.i32, field)
            return builder.gep(st, [zero, fld], inbounds=True)

        field = {"spl": 0, "slider": 1, "var": 2}[ref.kind]
        fld = ir.Constant(self.i32, field)
        idx = ir.Constant(self.i32, ref.index)
        return builder.gep(st, [zero, fld, idx], inbounds=True)

    def _get_mem_ptr(self, builder: ir.IRBuilder, st: ir.Value) -> ir.Value:
        zero = ir.Constant(self.i32, 0)
        fld = ir.Constant(self.i32, 3)
        pptr = builder.gep(st, [zero, fld], inbounds=True)
        return builder.load(pptr)

    def _get_memN(self, builder: ir.IRBuilder, st: ir.Value) -> ir.Value:
        zero = ir.Constant(self.i32, 0)
        fld = ir.Constant(self.i32, 4)
        nptr = builder.gep(st, [zero, fld], inbounds=True)
        return builder.load(nptr)

    def _mem_elem_ptr(self, builder, st_ptr, base_expr, idx_expr):
        """
        EEL2 bracket indexing semantics:

            addr = trunc((base + idx) + 0.00001)

        NOT trunc(base) + trunc(idx)
        """

        # Evaluate base and index as f64
        base_v = self.emit_expr(builder, st_ptr, base_expr)  # f64
        idx_v  = self.emit_expr(builder, st_ptr, idx_expr)   # f64

        # EEL2 legacy rounding: add 1e-5 before trunc
        summed = builder.fadd(base_v, idx_v)
        summed = builder.fadd(summed, self._const_f64(1.0e-5))
        

        # Memory indexing: truncate ONCE to i64 (do NOT wrap to 32-bit here)
        addr_i64 = builder.fptosi(summed, self.i64)


        # Clamp negative to 0 (JSFX behavior for negative mem indexes is effectively 0-safe)
        zero_i64 = ir.Constant(self.i64, 0)
        isneg = builder.icmp_signed("<", addr_i64, zero_i64)
        addr_i64 = builder.select(isneg, zero_i64, addr_i64)

        # If addr >= memN, grow/ensure memory so mem[addr] is valid (JSFX semantics)
        memN = self._get_memN(builder, st_ptr)  # i64
        need_grow = builder.icmp_signed(">=", addr_i64, memN)
        with builder.if_then(need_grow):
            one_i64 = ir.Constant(self.i64, 1)
            needN = builder.add(addr_i64, one_i64)
            # Your runtime helper should resize/ensure at least needN doubles
            builder.call(self.fn_ensure, [st_ptr, needN])


        # Base pointer to mem (double*)
        mem_base = self._get_mem_ptr(builder, st_ptr)

        # Return &mem[addr]
        return builder.gep(mem_base, [addr_i64], inbounds=False)


    
    def _to_i32(self, builder, x):
        # JSFX-style: truncate toward 0 to *some* integer, then wrap to 32-bit
        xi64 = builder.fptosi(x, self.i64)        # safe for your magnitudes
        return builder.trunc(xi64, self.i32)      # wraps mod 2^32


    def _to_f64(self, builder, x_i32):
        return builder.sitofp(x_i32, self.double)

    
    def declare_user_functions(self, fn_defs: Dict[str, FunctionDef]) -> None:
        self.user_fn_defs = dict(fn_defs)
        # Signature: double fn(DSPJSFX_State* st, double a0, double a1, ...)
        for name, fdef in self.user_fn_defs.items():
            arg_types = [self.state_ptr] + [self.double] * len(fdef.params)
            fnty = ir.FunctionType(self.double, arg_types)
            self.user_fn_ir[name] = ir.Function(self.module, fnty, name=f"jsfx_fn_{name}")

    def emit_user_functions(self) -> None:
        for name, fdef in self.user_fn_defs.items():
            fn = self.user_fn_ir[name]
            entry = fn.append_basic_block("entry")
            b = ir.IRBuilder(entry)

            st = fn.args[0]
            st.name = "st"

            # Create local slots for params+locals (alloca double)
            locals_map: Dict[str, ir.Value] = {}

            # params
            for i, p in enumerate(fdef.params):
                slot = b.alloca(self.double, name=f"p_{p}")
                b.store(fn.args[i + 1], slot)
                locals_map[p] = slot

            # locals
            for l in fdef.locals:
                if l in locals_map:
                    continue
                slot = b.alloca(self.double, name=f"l_{l}")
                b.store(self._const_f64(0.0), slot)
                locals_map[l] = slot

            self._local_slots_stack.append(locals_map)
            retv = self.emit_expr(b, st, fdef.body)
            self._local_slots_stack.pop()

            b.ret(retv)


    def emit_section_fn(self, name: str, prog: List[Node]) -> ir.Function:
        fn = ir.Function(self.module, ir.FunctionType(ir.VoidType(), [self.state_ptr]), name=name)
        entry = fn.append_basic_block("entry")
        builder = ir.IRBuilder(entry)
        st = fn.args[0]

        for st_node in prog:
            self.emit_stmt(builder, st, st_node)

        if not builder.block.is_terminated:
            builder.ret_void()
        return fn

    def emit_stmt(self, builder: ir.IRBuilder, st: ir.Value, n: Node) -> None:
        if isinstance(n, If):
            self.emit_if(builder, st, n); return
        if isinstance(n, While):
            self.emit_while(builder, st, n); return
        _ = self.emit_expr(builder, st, n)

    def emit_if(self, builder: ir.IRBuilder, st: ir.Value, n: If) -> None:
        condv = self.emit_expr(builder, st, n.cond)
        cond = self._truthy(condv, builder)

        fn = builder.function
        then_bb = fn.append_basic_block(f"then_{n.id}")
        else_bb = fn.append_basic_block(f"else_{n.id}") if n.els is not None else None
        merge_bb = fn.append_basic_block(f"merge_{n.id}")

        if else_bb is None:
            builder.cbranch(cond, then_bb, merge_bb)
        else:
            builder.cbranch(cond, then_bb, else_bb)

        # then
        builder.position_at_end(then_bb)
        self.emit_stmt(builder, st, n.then)
        if not builder.block.is_terminated:
            builder.branch(merge_bb)

        # else
        if else_bb is not None:
            builder.position_at_end(else_bb)
            self.emit_stmt(builder, st, n.els)  # type: ignore[arg-type]
            if not builder.block.is_terminated:
                builder.branch(merge_bb)

        builder.position_at_end(merge_bb)

    def emit_while(self, builder: ir.IRBuilder, st: ir.Value, n: While) -> None:
        fn = builder.function
        pre_bb = builder.block
        cond_bb = fn.append_basic_block(f"while_cond_{n.id}")
        body_bb = fn.append_basic_block(f"while_body_{n.id}")
        after_bb = fn.append_basic_block(f"while_after_{n.id}")

        builder.branch(cond_bb)

        builder.position_at_end(cond_bb)
        condv = self.emit_expr(builder, st, n.cond)
        cond = self._truthy(condv, builder)
        builder.cbranch(cond, body_bb, after_bb)

        builder.position_at_end(body_bb)
        self.emit_stmt(builder, st, n.body)
        if not builder.block.is_terminated:
            builder.branch(cond_bb)

        builder.position_at_end(after_bb)

    def emit_expr(self, builder: ir.IRBuilder, st: ir.Value, n: Node) -> ir.Value:
        # literals
        if isinstance(n, Num):
            return self._const_f64(n.value)

        # variable
        if isinstance(n, Var):
            if n.name == "mem":
                return self._const_f64(0.0)

            # common JSFX constants (expand if needed)
            if n.name == "$pi":
                return self._const_f64(math.pi)
            if n.name == "$e":
                return self._const_f64(math.e)

            ptr = self._get_slot_ptr(builder, st, n.name)  # handles locals + globals + srate/samplesblock
            return builder.load(ptr)


        # indexing
        if isinstance(n, Index):
            # a[b] == mem[(int)a + (int)b]; mem itself is base 0.
            ptr = self._mem_elem_ptr(builder, st, n.base, n.index)
            return builder.load(ptr)

        # unary
        if isinstance(n, Unary):
            a = self.emit_expr(builder, st, n.a)
            if n.op == "+":
                return a
            if n.op == "-":
                return builder.fsub(self._const_f64(0.0), a)
            if n.op == "!":
                isz = builder.fcmp_ordered("==", a, self._const_f64(0.0))
                return builder.select(isz, self._const_f64(1.0), self._const_f64(0.0))
            raise ValueError(f"Unsupported unary op {n.op}")

        # ternary
        if isinstance(n, Ternary):
            return self.emit_ternary(builder, st, n)

        # loop expression
        if isinstance(n, Loop):
            return self.emit_loop_expr(builder, st, n)

        # binary
        if isinstance(n, Binary):
            if n.op in ("&&", "||"):
                return self.emit_logical(builder, st, n.op, n.l, n.r)

            l = self.emit_expr(builder, st, n.l)
            r = self.emit_expr(builder, st, n.r)

            if n.op == "+":
                return builder.fadd(l, r)
            if n.op == "-":
                return builder.fsub(l, r)
            if n.op == "*":
                return builder.fmul(l, r)
            if n.op == "/":
                return builder.fdiv(l, r)
            if n.op == "^":
                fdecl = self._declare_math("pow")
                return builder.call(fdecl, [l, r])


            if n.op in ("<", "<=", ">", ">=", "==", "!="):
                opmap = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "one"}
                c = builder.fcmp_ordered(opmap[n.op], l, r)
                return builder.select(c, self._const_f64(1.0), self._const_f64(0.0))

            # bitwise / shifts (JSFX-style: int ops on truncated values, return double)
            if n.op in ("|", "&", "<<", ">>"):
                li = self._to_i32(builder, l)
                ri = self._to_i32(builder, r)

                # IMPORTANT:
                # Only mask the RHS for SHIFT operations (shift count).
                # Do NOT mask the RHS for plain AND/OR, or you destroy bitmasks like 16383, 2147483647, etc.
                if n.op in ("<<", ">>"):
                    ri = builder.and_(ri, ir.Constant(self.i32, 31))

                if n.op == "|":
                    oi = builder.or_(li, ri)
                elif n.op == "&":
                    oi = builder.and_(li, ri)
                elif n.op == "<<":
                    oi = builder.shl(li, ri)
                else:
                    oi = builder.ashr(li, ri)  # arithmetic shift right (likely matches JSFX)

                return self._to_f64(builder, oi)


            if n.op == "%":
                li = self._to_i32(builder, l)
                ri = self._to_i32(builder, r)
                oi = builder.srem(li, ri)
                return self._to_f64(builder, oi)

            raise ValueError(f"Unsupported binary op {n.op}")

        # assignment
        if isinstance(n, Assign):
            rhs = self.emit_expr(builder, st, n.value)

            # resolve target pointer
            if isinstance(n.target, Var):
                if n.target.name == "mem":
                    raise ValueError("Cannot assign to mem")
                ptr = self._get_slot_ptr(builder, st, n.target.name)  # works for locals too

            elif isinstance(n.target, Index):
                ptr = self._mem_elem_ptr(builder, st, n.target.base, n.target.index)
            else:
                raise ValueError("Invalid assignment target")

            if n.op == "=":
                builder.store(rhs, ptr)
                return rhs

            cur = builder.load(ptr)
            if n.op == "+=":
                out = builder.fadd(cur, rhs)
            elif n.op == "-=":
                out = builder.fsub(cur, rhs)
            elif n.op == "*=":
                out = builder.fmul(cur, rhs)
            elif n.op == "/=":
                out = builder.fdiv(cur, rhs)
            else:
                raise ValueError(f"Unsupported assign op {n.op}")

            builder.store(out, ptr)
            return out

        # call
        if isinstance(n, Call):
            fn = n.fn
            if fn == "abs":
                fn = "fabs"

            # User-defined function call
            if n.fn in self.user_fn_ir:
                callee = self.user_fn_ir[n.fn]
                argv = [st] + [self.emit_expr(builder, st, a) for a in n.args]
                return builder.call(callee, argv)


            if fn in ("min", "max"):
                if len(n.args) != 2:
                    raise ValueError(f"{fn} expects 2 args")
                a = self.emit_expr(builder, st, n.args[0])
                b = self.emit_expr(builder, st, n.args[1])
                if fn == "min":
                    c = builder.fcmp_ordered("olt", a, b)
                    return builder.select(c, a, b)
                c = builder.fcmp_ordered("ogt", a, b)
                return builder.select(c, a, b)

            if fn in ("sin", "cos", "sqrt", "fabs", "floor", "ceil"):
                if len(n.args) != 1:
                    raise ValueError(f"{fn} expects 1 arg")
                fdecl = self._declare_math(fn)
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(fdecl, [a0])

            if fn == "pow":
                if len(n.args) != 2:
                    raise ValueError("pow expects 2 args")
                fdecl = self._declare_math("pow")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                return builder.call(fdecl, [a0, a1])

            if fn in ("exp", "log"):
                if len(n.args) != 1:
                    raise ValueError(f"{fn} expects 1 arg")
                fdecl = self._declare_math(fn)
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(fdecl, [a0])

            raise ValueError(f"Unknown function call {n.fn}")

        # sequence
        if isinstance(n, Seq):
            last = self._const_f64(0.0)
            for item in n.items:
                if isinstance(item, If):
                    self.emit_if(builder, st, item)
                    last = self._const_f64(0.0)
                elif isinstance(item, While):
                    self.emit_while(builder, st, item)
                    last = self._const_f64(0.0)
                else:
                    last = self.emit_expr(builder, st, item)
            return last

        # allow if/while as expression (returns 0)
        if isinstance(n, If):
            self.emit_if(builder, st, n)
            return self._const_f64(0.0)
        if isinstance(n, While):
            self.emit_while(builder, st, n)
            return self._const_f64(0.0)

        raise ValueError(f"Unhandled node type: {type(n).__name__}")

    def emit_logical(self, builder: ir.IRBuilder, st: ir.Value, op: str, lnode: Node, rnode: Node) -> ir.Value:
        """
        Short-circuit && and ||. Returns 1.0/0.0.
        """
        fn = builder.function
        lval = self.emit_expr(builder, st, lnode)
        lbool = self._truthy(lval, builder)

        rhs_bb = fn.append_basic_block(f"log_rhs_{op}_{lnode.id}")
        merge_bb = fn.append_basic_block(f"log_merge_{op}_{lnode.id}")

        if op == "&&":
            builder.cbranch(lbool, rhs_bb, merge_bb)
            # false path => result false
            false_from = builder.block
        else:  # ||
            builder.cbranch(lbool, merge_bb, rhs_bb)
            # true path => result true
            true_from = builder.block

        # rhs block
        builder.position_at_end(rhs_bb)
        rval = self.emit_expr(builder, st, rnode)
        rbool = self._truthy(rval, builder)
        rhs_end = builder.block
        if not builder.block.is_terminated:
            builder.branch(merge_bb)

        # merge
        builder.position_at_end(merge_bb)
        phi = builder.phi(self.i1)

        if op == "&&":
            # incoming from lfalse edge and rhs
            phi.add_incoming(ir.Constant(self.i1, 0), false_from)
            phi.add_incoming(rbool, rhs_end)
        else:
            phi.add_incoming(ir.Constant(self.i1, 1), true_from)
            phi.add_incoming(rbool, rhs_end)

        return builder.select(phi, self._const_f64(1.0), self._const_f64(0.0))

    def emit_ternary(self, builder: ir.IRBuilder, st: ir.Value, n: Ternary) -> ir.Value:
        fn = builder.function
        condv = self.emit_expr(builder, st, n.cond)
        cond = self._truthy(condv, builder)

        then_bb = fn.append_basic_block(f"tern_then_{n.id}")
        else_bb = fn.append_basic_block(f"tern_else_{n.id}")
        merge_bb = fn.append_basic_block(f"tern_merge_{n.id}")

        builder.cbranch(cond, then_bb, else_bb)

        builder.position_at_end(then_bb)
        tval = self.emit_expr(builder, st, n.then)
        then_end = builder.block
        if not builder.block.is_terminated:
            builder.branch(merge_bb)

        builder.position_at_end(else_bb)
        eval_ = self.emit_expr(builder, st, n.els)
        else_end = builder.block
        if not builder.block.is_terminated:
            builder.branch(merge_bb)

        builder.position_at_end(merge_bb)
        phi = builder.phi(self.double)
        phi.add_incoming(tval, then_end)
        phi.add_incoming(eval_, else_end)
        return phi

    def emit_loop_expr(self, builder: ir.IRBuilder, st: ir.Value, n: Loop) -> ir.Value:
        """
        loop(count, body): repeats body count times, returns last body's value or 0.
        """
        fn = builder.function
        pre_bb = builder.block

        count_v = self.emit_expr(builder, st, n.count)
        count_i = builder.fptosi(count_v, self.i64)
        # clamp >=0
        is_neg = builder.icmp_signed("<", count_i, self._const_i64(0))
        n_i = builder.select(is_neg, self._const_i64(0), count_i)

        cond_bb = fn.append_basic_block(f"loop_cond_{n.id}")
        body_bb = fn.append_basic_block(f"loop_body_{n.id}")
        after_bb = fn.append_basic_block(f"loop_after_{n.id}")

        builder.branch(cond_bb)

        builder.position_at_end(cond_bb)
        phi_i = builder.phi(self.i64)
        phi_last = builder.phi(self.double)
        phi_i.add_incoming(self._const_i64(0), pre_bb)
        phi_last.add_incoming(self._const_f64(0.0), pre_bb)

        keep_going = builder.icmp_signed("<", phi_i, n_i)
        builder.cbranch(keep_going, body_bb, after_bb)

        builder.position_at_end(body_bb)
        v = self.emit_expr(builder, st, n.body)
        i_next = builder.add(phi_i, self._const_i64(1))
        latch_bb = builder.block
        if not builder.block.is_terminated:
            builder.branch(cond_bb)

        # Add incoming from latch
        phi_i.add_incoming(i_next, latch_bb)
        phi_last.add_incoming(v, latch_bb)

        builder.position_at_end(after_bb)
        # cond_bb dominates after_bb, so phi_last is valid here
        return phi_last



def emit_process_block_fn(self, fn_init: ir.Function, fn_slider: ir.Function, fn_block: ir.Function, fn_sample: ir.Function) -> ir.Function:
    """
    Emits:
        void jsfx_process_block(State* st,
                                const float* const* inputs,
                                float* const* outputs,
                                i32 numChannels,
                                i32 numSamples);

    Semantics:
    - st->samplesblock = (double)numSamples
    - call jsfx_slider(st)
    - call jsfx_block(st)
    - for each sample:
        load inputs[ch][i] into st.spl[ch] (as double)
        call jsfx_sample(st)
        store st.spl[ch] to outputs[ch][i] (as float)
    """
    f32 = ir.FloatType()
    f32p = f32.as_pointer()
    f32pp = f32p.as_pointer()

    fn = ir.Function(
        self.module,
        ir.FunctionType(ir.VoidType(), [self.state_ptr, f32pp, f32pp, self.i32, self.i32]),
        name="jsfx_process_block"
    )

    # Encourage inlining of section fns into the loop when optimized
    for f in (fn_init, fn_slider, fn_block, fn_sample):
        try:
            f.attributes.add("alwaysinline")
        except Exception:
            pass

    entry = fn.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    st = fn.args[0]
    inputs = fn.args[1]
    outputs = fn.args[2]
    nCh = fn.args[3]
    nSamp = fn.args[4]

    # clamp channels to [0, 64]
    zero_i32 = ir.Constant(self.i32, 0)
    max_i32 = ir.Constant(self.i32, 64)
    neg = builder.icmp_signed("<", nCh, zero_i32)
    nCh0 = builder.select(neg, zero_i32, nCh)
    gt = builder.icmp_signed(">", nCh0, max_i32)
    chLim = builder.select(gt, max_i32, nCh0)

    # st->samplesblock = (double)nSamp
    z = ir.Constant(self.i32, 0)
    fld_samplesblock = ir.Constant(self.i32, 6)
    sb_ptr = builder.gep(st, [z, fld_samplesblock], inbounds=True)
    nSamp64 = builder.sext(nSamp, self.i64)
    sb_val = builder.sitofp(nSamp64, self.double)
    builder.store(sb_val, sb_ptr)

    # Call block 
    builder.call(fn_block, [st])

    # Outer sample loop blocks
    samp_cond = fn.append_basic_block("samp_cond")
    samp_body = fn.append_basic_block("samp_body")
    samp_end  = fn.append_basic_block("samp_end")

    builder.branch(samp_cond)

    # samp_cond
    builder.position_at_end(samp_cond)
    i_phi = builder.phi(self.i32, name="i")
    i_phi.add_incoming(zero_i32, entry)
    in_range = builder.icmp_signed("<", i_phi, nSamp)
    builder.cbranch(in_range, samp_body, samp_end)

    # samp_body
    builder.position_at_end(samp_body)

    # ---- channel input loop ----
    ch_in_cond = fn.append_basic_block("ch_in_cond")
    ch_in_body = fn.append_basic_block("ch_in_body")
    ch_in_end  = fn.append_basic_block("ch_in_end")
    builder.branch(ch_in_cond)

    builder.position_at_end(ch_in_cond)
    ch_phi = builder.phi(self.i32, name="ch_in")
    ch_phi.add_incoming(zero_i32, samp_body)
    ch_ok = builder.icmp_signed("<", ch_phi, chLim)
    builder.cbranch(ch_ok, ch_in_body, ch_in_end)

    builder.position_at_end(ch_in_body)
    # inputs[ch]
    in_pp = builder.gep(inputs, [ch_phi])
    in_p  = builder.load(in_pp)
    in_sp = builder.gep(in_p, [i_phi])
    in_f  = builder.load(in_sp)
    in_d  = builder.fpext(in_f, self.double)

    # store to st.spl[ch]
    fld_spl = ir.Constant(self.i32, 0)
    spl_ptr = builder.gep(st, [z, fld_spl, ch_phi], inbounds=True)
    builder.store(in_d, spl_ptr)

    ch_next = builder.add(ch_phi, ir.Constant(self.i32, 1))
    ch_phi.add_incoming(ch_next, builder.block)
    builder.branch(ch_in_cond)

    # end inputs
    builder.position_at_end(ch_in_end)

    # call per-sample function
    builder.call(fn_sample, [st])

    # ---- channel output loop ----
    ch_out_cond = fn.append_basic_block("ch_out_cond")
    ch_out_body = fn.append_basic_block("ch_out_body")
    ch_out_end  = fn.append_basic_block("ch_out_end")
    builder.branch(ch_out_cond)

    builder.position_at_end(ch_out_cond)
    ch2_phi = builder.phi(self.i32, name="ch_out")
    ch2_phi.add_incoming(zero_i32, ch_in_end)
    ch2_ok = builder.icmp_signed("<", ch2_phi, chLim)
    builder.cbranch(ch2_ok, ch_out_body, ch_out_end)

    builder.position_at_end(ch_out_body)
    # load st.spl[ch2]
    spl2_ptr = builder.gep(st, [z, fld_spl, ch2_phi], inbounds=True)
    out_d = builder.load(spl2_ptr)
    out_f = builder.fptrunc(out_d, f32)

    # outputs[ch2][i] = out_f
    out_pp = builder.gep(outputs, [ch2_phi])
    out_p  = builder.load(out_pp)
    out_sp = builder.gep(out_p, [i_phi])
    builder.store(out_f, out_sp)

    ch2_next = builder.add(ch2_phi, ir.Constant(self.i32, 1))
    ch2_phi.add_incoming(ch2_next, builder.block)
    builder.branch(ch_out_cond)

    builder.position_at_end(ch_out_end)

    # increment sample index
    i_next = builder.add(i_phi, ir.Constant(self.i32, 1))
    i_phi.add_incoming(i_next, builder.block)
    builder.branch(samp_cond)

    # end
    builder.position_at_end(samp_end)
    builder.ret_void()

    return fn


def compile_jsfx_to_ir(jsfx_text: str) -> Tuple[ir.Module, Dict[str, Any]]:
    sections = extract_sections(jsfx_text)

    programs: Dict[str, List[Node]] = {}
    for sec in ("init", "slider", "block", "sample"):
        if sec in sections:
            code, start_line = sections[sec]
            parser = Parser(code, base_line=start_line)
            programs[sec] = parser.parse_program()
        else:
            programs[sec] = []

    fn_defs, programs = extract_function_defs(programs)
    user_vars = collect_user_vars(programs, fn_defs)


    sym = SymTable(user_vars)

    emitter = LLVMModuleEmitter(sym)
    emitter.declare_user_functions(fn_defs)

    fn_init = emitter.emit_section_fn("jsfx_init", programs["init"])
    fn_slider = emitter.emit_section_fn("jsfx_slider", programs["slider"])
    fn_block = emitter.emit_section_fn("jsfx_block", programs["block"])
    fn_sample = emitter.emit_section_fn("jsfx_sample", programs["sample"])

    emitter.emit_user_functions()

    emit_process_block_fn(emitter, fn_init, fn_slider, fn_block, fn_sample)

    meta = {
        "vars": user_vars,
        "var_cap": emitter.var_cap,
        "sections_present": {k: bool(v) for k, v in programs.items()},
    }
    return emitter.module, meta



def _emit_header(meta: Dict[str, Any]) -> str:
    var_cap = int(meta.get("var_cap", 1))
    lines = []
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")
    lines.append("typedef struct DSPJSFX_State {")
    lines.append("    double spl[64];")
    lines.append("    double sliders[64];")
    lines.append(f"    double vars[{var_cap}];")
    lines.append("    double* mem;")
    lines.append("    int64_t memN;")
    lines.append("    double srate;")
    lines.append("    double samplesblock;")
    lines.append("} DSPJSFX_State;")
    lines.append("")
    lines.append("/* Sections */")
    lines.append("void jsfx_init(DSPJSFX_State* st);")
    lines.append("void jsfx_slider(DSPJSFX_State* st);")
    lines.append("void jsfx_block(DSPJSFX_State* st);")
    lines.append("void jsfx_sample(DSPJSFX_State* st);")
    lines.append("")
    lines.append("/* Entry point intended to be called from JUCE processBlock().")
    lines.append("   inputs/outputs are arrays of channel pointers (non-interleaved). */")
    lines.append("void jsfx_process_block(DSPJSFX_State* st,")
    lines.append("                        const float* const* inputs,")
    lines.append("                        float* const* outputs,")
    lines.append("                        int32_t numChannels,")
    lines.append("                        int32_t numSamples);")
    lines.append("")
    lines.append("/* Runtime hook required by mem[] growth checks. You must provide this when linking.")
    lines.append("   Even if you never exceed memN, the symbol must exist. */")
    lines.append("void jsfx_ensure_mem(DSPJSFX_State* st, int64_t needed);")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    return "\n".join(lines)

def _aot_opt_and_emit(mod_ir: ir.Module,
                     opt_level: int,
                     emit_obj: Optional[str],
                     emit_asm: Optional[str],
                     target_triple: Optional[str] = None) -> str:
    """Return optimized LLVM IR text. Optionally emits object and/or assembly."""
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    triple = target_triple or llvm.get_default_triple()
    target = llvm.Target.from_triple(triple)
    tm = target.create_target_machine(opt=opt_level)

    # Configure IR module for this target
    mod_ir.triple = triple
    mod_ir.data_layout = str(tm.target_data)

    llvm_mod = llvm.parse_assembly(str(mod_ir))
    llvm_mod.verify()

    if opt_level > 0:
        # New pass manager default pipeline (roughly -O{opt_level})
        pto = llvm.PipelineTuningOptions(speed_level=int(opt_level), size_level=0)
        pb = llvm.create_pass_builder(tm, pto)
        pm = pb.getModulePassManager()
        pm.run(llvm_mod, pb)

    # IMPORTANT:
    # llvmlite's tm.emit_object() is not reliably MSVC-linkable on Windows.
    # For Windows targets, we shell out to clang to produce a proper COFF .obj.
    if emit_obj:
        if target_triple and (("windows" in target_triple.lower()) or ("apple" in target_triple.lower())):
            clang = shutil.which("clang")
            if not clang:
                raise RuntimeError(
                    "clang not found on PATH. Install LLVM for Windows and ensure clang.exe is on PATH."
                )

            with tempfile.TemporaryDirectory() as td:
                ll_path = Path(td) / "aot.ll"
                ll_path.write_text(str(llvm_mod), encoding="utf-8")

                cmd = [
                    clang,
                    f"--target={target_triple}",
                    "-c",
                    str(ll_path),
                    "-o",
                    str(Path(emit_obj)),
                ]
                subprocess.check_call(cmd)
        else:
            Path(emit_obj).write_bytes(tm.emit_object(llvm_mod))

    if emit_asm:
        Path(emit_asm).write_text(tm.emit_assembly(llvm_mod), encoding="utf-8")

    return str(llvm_mod)



def main() -> int:
    ap = argparse.ArgumentParser(description="DSP-JSFX -> LLVM IR + AOT object + JUCE-callable entry point")
    ap.add_argument("input", help="Path to .jsfx file")
    ap.add_argument("--out-ll", default="", help="Write LLVM IR (.ll) to this path (default: stdout)")
    ap.add_argument("--out-obj", default="", help="Emit AOT object file (.o/.obj) to this path")
    ap.add_argument("--out-asm", default="", help="Emit AOT assembly (.s) to this path")
    ap.add_argument("--out-h", default="", help="Emit C/C++ header (.h) with State + prototypes")
    ap.add_argument("--meta", default="", help="Optional JSON metadata output")
    ap.add_argument("--opt", type=int, default=2, help="Optimization level for AOT (0-3). Default 2.")
    ap.add_argument("--target", default="", help="LLVM target triple for AOT object/asm (e.g. x86_64-pc-windows-msvc)")
    args = ap.parse_args()

    txt = Path(args.input).read_text(encoding="utf-8", errors="replace")
    mod, meta = compile_jsfx_to_ir(txt)

    want_aot = bool(args.out_obj or args.out_asm)
    if want_aot:
        ir_text = _aot_opt_and_emit(
            mod_ir=mod,
            opt_level=max(0, min(3, int(args.opt))),
            emit_obj=args.out_obj or None,
            emit_asm=args.out_asm or None,
            target_triple=(args.target or None),
        )

    else:
        ir_text = str(mod)

    if args.out_ll:
        Path(args.out_ll).write_text(ir_text, encoding="utf-8")
    else:
        print(ir_text)

    if args.out_h:
        Path(args.out_h).write_text(_emit_header(meta), encoding="utf-8")

    if args.meta:
        Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
