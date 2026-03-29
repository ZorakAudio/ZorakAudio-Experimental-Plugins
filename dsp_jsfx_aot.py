#!/usr/bin/env python3
"""
dsp_jsfx_to_llvm.py

DSP-JSFX -> LLVM IR (llvmlite) compiler front-end.

Contract (DSP-JSFX):
- Sections: @init, @slider, @block, @sample (any may be missing)
- DSP-first AOT subset: no @gfx compilation, no preprocessor.
- Strings are accepted as opaque handles for compatibility helpers.
- MIDI builtins midirecv()/midisend() are supported in @block and @sample.
- Type: everything is double.
- Variables: spl0..spl63, slider1..slider64, user vars (persistent), builtins:
    - mem  (numeric base pointer index = 0.0)
    - srate (read/write state field)
    - samplesblock (read/write state field; host should set before @block)
    - MIDI calls: midirecv(offset,msg1,msg2,msg3), midisend(offset,msg1,msg2,msg3)
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
from collections import Counter
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
            if c.isalpha() or c in "_$#":
                # Allow dotted identifiers (e.g. u.next_bank) as a single symbol.
                # Also allow JSFX string variables like #menu_item.
                m = re.match(r"[#$A-Za-z_][#$A-Za-z0-9_]*(?:\.[#$A-Za-z_][#$A-Za-z0-9_]*)*", self.src[self.i:])
                assert m
                txt = m.group(0)
                self._adv(len(txt))
                kind = "kw" if txt in ("if", "else", "while") else "ident"
                return Tok(kind, txt, sp)

            # quoted literal
            # Accept both double-quoted strings and single-quoted char/string-ish
            # literals. Many JSFX UI helpers use forms like:
            #   gfx_setfont(2, "Arial", 10, 'b');
            # We keep the runtime semantics intentionally lightweight here and
            # represent either form as an opaque string handle token.
            if c in ('"', "'"):
                quote = c
                self._adv()  # consume opening quote
                out = []
                while True:
                    ch = self._peek()
                    if ch == "\0":
                        raise SyntaxError(self._fmt_err("Unterminated string literal"))
                    if ch in ("\n", "\r"):
                        raise SyntaxError(self._fmt_err("Newline in string literal"))
                    if ch == quote:
                        self._adv()  # closing quote
                        break
                    if ch == "\\":
                        self._adv()  # consume backslash
                        esc = self._peek()
                        if esc == "\0":
                            raise SyntaxError(self._fmt_err("Unterminated string escape"))
                        self._adv()
                        if esc == "n":
                            out.append("\n")
                        elif esc == "r":
                            out.append("\r")
                        elif esc == "t":
                            out.append("\t")
                        elif esc == quote:
                            out.append(quote)
                        elif esc == "\\":
                            out.append("\\")
                        else:
                            # Unknown escapes: keep the escaped character verbatim.
                            out.append(esc)
                        continue

                    out.append(ch)
                    self._adv()

                return Tok("str", "".join(out), sp)


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
class StrLit(Node):
    id: int
    span: Span
    value: str

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
    instances: List[str]
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

    def _skip_eol(self) -> None:
        # Like _skip_seps(), but ONLY consumes newlines, not semicolons.
        # This enables multi-line expressions without accidentally joining
        # semicolon-separated statements.
        while self.cur.kind == "eol":
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
        # JSFX/EEL2 user functions support optional qualifier lists between
        # the parameter list and the body, e.g.:
        #   function foo(x y) local(a b) instance(s1, s2) ( ... );
        # The qualifier order is flexible; global() is accepted and ignored
        # because unqualified variables are already global in our IR model.
        t_fun = self._eat("ident", "function")

        if self.cur.kind != "ident":
            raise SyntaxError(self._fmt_err("Expected function name after 'function'"))
        t_name = self._eat("ident")
        fn_name = t_name.text

        def parse_name_list(kind_label: str, *, allow_whitespace: bool = True) -> List[str]:
            out: List[str] = []
            self._eat("punc", "(")
            self._skip_seps()

            if not (self.cur.kind == "punc" and self.cur.text == ")"):
                while True:
                    self._skip_seps()

                    # allow trailing comma/newlines before ')'
                    if self.cur.kind == "punc" and self.cur.text == ")":
                        break

                    if self.cur.kind != "ident":
                        raise SyntaxError(self._fmt_err(f"Expected {kind_label} name"))
                    out.append(self._eat("ident").text)

                    self._skip_seps()
                    # JSFX allows names in these lists to be separated by
                    # commas OR whitespace/newlines.
                    if self.cur.kind == "punc" and self.cur.text == ",":
                        self._adv()
                        continue
                    if allow_whitespace and self.cur.kind == "ident":
                        continue
                    break

            self._skip_seps()
            self._eat("punc", ")")
            return out

        # params: JSFX accepts comma- or whitespace-separated names here.
        params = parse_name_list("parameter", allow_whitespace=True)

        locals_: List[str] = []
        instances: List[str] = []
        self._skip_seps()

        while self.cur.kind == "ident" and self.cur.text in ("local", "instance", "global"):
            qual = self.cur.text
            self._adv()
            names = parse_name_list(f"{qual} variable", allow_whitespace=True)

            if qual == "local":
                locals_.extend(names)
            elif qual == "instance":
                instances.extend(names)
            else:
                # global() is documentation / explicit declaration in JSFX;
                # no special lowering is required here.
                pass

            self._skip_seps()

        # body must be a parenthesized expression/sequence
        if not (self.cur.kind == "punc" and self.cur.text == "("):
            raise SyntaxError(self._fmt_err("Expected '(' to start function body"))
        body = self.parse_primary()  # parses (...) as expr or Seq

        self._skip_seps()
        # optional trailing semicolon
        if self.cur.kind == "semi":
            self._adv()

        return FunctionDef(self._new_id(), t_fun.span, fn_name, params, locals_, instances, body)



    def _is_assign_target(self, n: Node) -> bool:
        # Valid assignment targets ("lvalues") in our JSFX subset:
        #   - variables (x, slider1, spl0, ...)
        #   - memory indexing (mem[...] or ptr[...] => Index nodes)
        #   - dynamic slider/spl access: slider(i), spl(i)
        if isinstance(n, (Var, Index)):
            return True
        if isinstance(n, Call) and n.fn in ("slider", "spl") and len(n.args) == 1:
            return True
        return False

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
                if not self._is_assign_target(lhs):
                    raise SyntaxError(self._fmt_err("Assignment target must be a variable, index, or slider()/spl() reference"))
                lhs = Assign(self._new_id(), lhs.span, op, lhs, rhs)
            else:
                lhs = Binary(self._new_id(), lhs.span, op, lhs, rhs)

        # Allow multiline ternary where '?' starts on the next line.
        # We ONLY skip newlines in this specific situation to avoid
        # accidentally merging separate statements.
        while self.cur.kind == "eol" and (self.nxt.kind == "eol" or (self.nxt.kind == "op" and self.nxt.text == "?")):
            self._adv()

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
                els = self.parse_expr(0)
            else:
                # no ':' => else is 0.0
                els = Num(self._new_id(), q.span, 0.0)

            lhs = Ternary(self._new_id(), q.span, lhs, then, els)


        return lhs

    def parse_prefix(self) -> Node:
        # Allow line breaks inside expressions (JSFX is newline-tolerant).
        self._skip_eol()

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

        if self.cur.kind == "str":
            t = self._eat("str")
            return StrLit(self._new_id(), t.span, t.text)

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

_SECTION_RE = re.compile(r"^\s*@([A-Za-z_][A-Za-z0-9_]*)\b.*$")

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

        if isinstance(n, (Num, StrLit)):
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



# -----------------------------
# spl[] I/O inference (for JUCE bus layout)
# -----------------------------

_SPL_RE = re.compile(r"^spl([0-9]+)$")
_PIN_RE = re.compile(r"^\s*(in_pin|out_pin)\s*:\s*(.*?)\s*$", re.IGNORECASE)

def parse_pin_hints(jsfx_text: str) -> Dict[str, Optional[int]]:
    """Parse JSFX pin declarations.

    Returns {'inputs': int|None, 'outputs': int|None}. None means "not explicitly declared".
    JSFX allows repeated in_pin:/out_pin: lines; the special token 'none' declares zero pins.
    """
    saw: Dict[str, bool] = {'inputs': False, 'outputs': False}
    counts: Dict[str, int] = {'inputs': 0, 'outputs': 0}

    for raw_line in jsfx_text.splitlines():
        line = raw_line
        if '//' in line:
            line = line.split('//', 1)[0]
        if ';' in line:
            line = line.split(';', 1)[0]
        m = _PIN_RE.match(line)
        if not m:
            continue
        kind = 'inputs' if m.group(1).lower() == 'in_pin' else 'outputs'
        value = m.group(2).strip()
        saw[kind] = True
        if value.lower() == 'none':
            counts[kind] = 0
            continue
        counts[kind] += 1

    return {k: (counts[k] if saw[k] else None) for k in ('inputs', 'outputs')}

def detect_midi_usage(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef]) -> Dict[str, bool]:
    uses_midirecv = False
    uses_midisend = False

    def rec(n: Node) -> None:
        nonlocal uses_midirecv, uses_midisend
        if isinstance(n, (Num, StrLit, Var)):
            return
        if isinstance(n, Index):
            rec(n.base); rec(n.index); return
        if isinstance(n, Unary):
            rec(n.a); return
        if isinstance(n, Binary):
            rec(n.l); rec(n.r); return
        if isinstance(n, Assign):
            rec(n.target); rec(n.value); return
        if isinstance(n, Call):
            if n.fn == 'midirecv':
                uses_midirecv = True
            elif n.fn == 'midisend':
                uses_midisend = True
            for a in n.args:
                rec(a)
            return
        if isinstance(n, Loop):
            rec(n.count); rec(n.body); return
        if isinstance(n, Ternary):
            rec(n.cond); rec(n.then); rec(n.els); return
        if isinstance(n, Seq):
            for it in n.items: rec(it)
            return
        if isinstance(n, If):
            rec(n.cond); rec(n.then)
            if n.els: rec(n.els)
            return
        if isinstance(n, While):
            rec(n.cond); rec(n.body); return
        if isinstance(n, FunctionDef):
            rec(n.body); return
        raise TypeError(type(n))

    for prog in programs.values():
        for st in prog:
            rec(st)

    for f in fn_defs.values():
        rec(f.body)

    return {
        'uses_midi': uses_midirecv or uses_midisend,
        'accepts_midi_input': uses_midirecv,
        'produces_midi_output': uses_midisend,
    }

def infer_spl_io(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef], pin_hints: Optional[Dict[str, Optional[int]]] = None) -> Dict[str, int]:
    """Infer minimum input/output channel counts from splN usage.

    Heuristic:
      - Any read of splN implies at least (N+1) input channels.
      - Any write to splN (assignment target) implies at least (N+1) output channels.

    Returned counts are clamped to 1..64.

    NOTE:
      In REAPER JSFX can see higher track channels even when a plugin only exposes fewer pins.
      For CLAP/VST we must declare enough channels for those spl registers to exist.
    """
    reads: Set[int] = set()
    writes: Set[int] = set()

    def _record(name: str, is_write: bool) -> None:
        m = _SPL_RE.match(name)
        if not m:
            return
        try:
            idx = int(m.group(1))
        except ValueError:
            return
        if idx < 0 or idx >= 64:
            return
        (writes if is_write else reads).add(idx)

    def rec(n: Node, locals: Set[str], write_ctx: bool = False) -> None:
        if isinstance(n, Var):
            if n.name in locals:
                return
            _record(n.name, write_ctx)
            return

        if isinstance(n, (Num, StrLit)):
            return
        if isinstance(n, Index):
            rec(n.base, locals, False)
            rec(n.index, locals, False)
            return
        if isinstance(n, Unary):
            rec(n.a, locals, False)
            return
        if isinstance(n, Binary):
            rec(n.l, locals, False)
            rec(n.r, locals, False)
            return
        if isinstance(n, Assign):
            # Target is written
            rec(n.target, locals, True)

            # Compound assignment (+=, *=, ...) also reads the old target value
            if n.op != "=":
                rec(n.target, locals, False)

            rec(n.value, locals, False)
            return
        if isinstance(n, Call):
            for a in n.args:
                rec(a, locals, False)
            return
        if isinstance(n, Loop):
            rec(n.count, locals, False)
            rec(n.body, locals, False)
            return
        if isinstance(n, Ternary):
            rec(n.cond, locals, False)
            rec(n.then, locals, False)
            rec(n.els, locals, False)
            return
        if isinstance(n, Seq):
            for it in n.items:
                rec(it, locals, False)
            return
        if isinstance(n, If):
            rec(n.cond, locals, False)
            rec(n.then, locals, False)
            if n.els:
                rec(n.els, locals, False)
            return
        if isinstance(n, While):
            rec(n.cond, locals, False)
            rec(n.body, locals, False)
            return

        raise TypeError(type(n))

    # sections
    for prog in programs.values():
        for st in prog:
            rec(st, set(), False)

    # function bodies (exclude params+locals)
    for f in fn_defs.values():
        localset = set(f.params) | set(f.locals)
        rec(f.body, localset, False)

    max_read = max(reads) if reads else -1
    max_write = max(writes) if writes else -1

    inferred_in_ch = (max_read + 1) if max_read >= 0 else 0
    inferred_out_ch = (max_write + 1) if max_write >= 0 else 0

    pin_hints = pin_hints or {}
    declared_in = pin_hints.get('inputs')
    declared_out = pin_hints.get('outputs')

    in_ch = int(inferred_in_ch)
    out_ch = int(inferred_out_ch)

    if declared_in is not None:
        in_ch = int(declared_in)
    if declared_out is not None:
        out_ch = int(declared_out)

    # If a script has no explicit I/O usage and no explicit pin declarations,
    # keep a conservative stereo fallback for backward compatibility.
    if declared_in is None and declared_out is None and in_ch == 0 and out_ch == 0:
        in_ch = 2
        out_ch = 2

    # Mirror unspecified side to the specified side only when there was no explicit
    # declaration forcing silence on that edge.
    if declared_in is None and in_ch == 0 and out_ch > 0:
        in_ch = out_ch
    if declared_out is None and out_ch == 0 and in_ch > 0:
        out_ch = in_ch

    in_ch = max(0, min(64, int(in_ch)))
    out_ch = max(0, min(64, int(out_ch)))
    process_ch = max(in_ch, out_ch)

    return {
        "inputs": in_ch,
        "outputs": out_ch,
        "process": process_ch,
        "max_read": max_read,
        "max_write": max_write,
    }


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


def _mangle_jsfx_component(text: str) -> str:
    out: List[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append(f"_x{ord(ch):02X}_")
    if not out:
        return "_"
    if out[0][0].isdigit():
        out.insert(0, "_")
    return "".join(out)


def _make_specialized_fn_name(section: str, fn_name: str, namespace: Optional[str]) -> str:
    sec_m = _mangle_jsfx_component(section)
    fn_m = _mangle_jsfx_component(fn_name)
    if namespace:
        ns_m = _mangle_jsfx_component(namespace)
        return f"__fn__{sec_m}__{fn_m}__ns__{ns_m}"
    return f"__fn__{sec_m}__{fn_m}"


def _make_persistent_local_name(section: str, fn_name: str, local_name: str) -> str:
    sec_m = _mangle_jsfx_component(section)
    fn_m = _mangle_jsfx_component(fn_name)
    loc_m = _mangle_jsfx_component(local_name)
    return f"__fnlocal__{sec_m}__{fn_m}__{loc_m}"


def _make_instance_var_name(namespace: str, var_name: str) -> str:
    # Use the actual JSFX pseudo-object spelling so instance(foo) and
    # explicit accesses like this.foo / whatever.foo resolve to the same
    # backing state slot.
    return f"{namespace}.{var_name}"


def _node_uses_relative_namespace(n: Node) -> bool:
    if isinstance(n, Var):
        return n.name == "this" or n.name.startswith("this.")
    if isinstance(n, (Num, StrLit)):
        return False
    if isinstance(n, Index):
        return _node_uses_relative_namespace(n.base) or _node_uses_relative_namespace(n.index)
    if isinstance(n, Unary):
        return _node_uses_relative_namespace(n.a)
    if isinstance(n, Binary):
        return _node_uses_relative_namespace(n.l) or _node_uses_relative_namespace(n.r)
    if isinstance(n, Assign):
        return _node_uses_relative_namespace(n.target) or _node_uses_relative_namespace(n.value)
    if isinstance(n, Call):
        if n.fn == "this" or n.fn.startswith("this."):
            return True
        return any(_node_uses_relative_namespace(a) for a in n.args)
    if isinstance(n, Loop):
        return _node_uses_relative_namespace(n.count) or _node_uses_relative_namespace(n.body)
    if isinstance(n, Ternary):
        return (_node_uses_relative_namespace(n.cond) or
                _node_uses_relative_namespace(n.then) or
                _node_uses_relative_namespace(n.els))
    if isinstance(n, Seq):
        return any(_node_uses_relative_namespace(it) for it in n.items)
    if isinstance(n, If):
        return (_node_uses_relative_namespace(n.cond) or
                _node_uses_relative_namespace(n.then) or
                (n.els is not None and _node_uses_relative_namespace(n.els)))
    if isinstance(n, While):
        return _node_uses_relative_namespace(n.cond) or _node_uses_relative_namespace(n.body)
    if isinstance(n, FunctionDef):
        return _node_uses_relative_namespace(n.body)
    raise TypeError(type(n))


def _resolve_relative_namespace(prefix: str, current_namespace: Optional[str]) -> Optional[str]:
    if prefix == "this":
        return current_namespace
    if prefix.startswith("this."):
        suffix = prefix[5:]
        if current_namespace:
            return f"{current_namespace}.{suffix}" if suffix else current_namespace
        return suffix or current_namespace
    return prefix


def lower_user_functions(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef]) -> Tuple[Dict[str, List[Node]], Dict[str, FunctionDef]]:
    """Lower JSFX user-function local()/instance() namespace semantics.

    Strategy:
      - local() variables become persistent state vars with per-caller-section
        mangled names, so repeated calls do not reset them to zero.
      - instance() variables become namespaced persistent state vars keyed by
        the call-site namespace (e.g. monLP.onepole2_lp -> monLP.s1 / monLP.s2).
      - user functions are specialized per caller section, and additionally per
        namespace when needed, so relative namespace references and local()
        storage follow JSFX/EEL2 behavior closely enough for DSP code.

    This intentionally does not attempt to implement full dynamic/relative
    namespace features such as this.. hierarchy walking.
    """
    if not fn_defs:
        return programs, fn_defs

    fn_needs_namespace: Dict[str, bool] = {
        name: bool(fdef.instances) or _node_uses_relative_namespace(fdef.body)
        for name, fdef in fn_defs.items()
    }

    specialized_defs: Dict[str, FunctionDef] = {}
    specialized_name_cache: Dict[Tuple[str, str, Optional[str]], str] = {}
    in_progress: Set[Tuple[str, str, Optional[str]]] = set()

    def rewrite_var_name(name: str, params: Set[str], local_map: Dict[str, str], instance_map: Dict[str, str], current_namespace: Optional[str]) -> str:
        if name in params:
            return name
        if name in local_map:
            return local_map[name]
        if name in instance_map:
            return instance_map[name]
        if name == "this":
            return current_namespace or name
        if name.startswith("this."):
            suffix = name[5:]
            if current_namespace:
                return f"{current_namespace}.{suffix}" if suffix else current_namespace
            return suffix or name
        return name

    def resolve_user_call(fn_name: str, current_namespace: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if fn_name in fn_defs:
            return fn_name, None

        parts = fn_name.split(".")
        if len(parts) >= 2 and parts[-1] in fn_defs:
            base = parts[-1]
            prefix = ".".join(parts[:-1])
            return base, _resolve_relative_namespace(prefix, current_namespace)

        return None, None

    def ensure_specialized(section: str, base_fn: str, call_namespace: Optional[str]) -> str:
        if base_fn not in fn_defs:
            return base_fn

        orig = fn_defs[base_fn]
        namespace_key = call_namespace if fn_needs_namespace.get(base_fn, False) else None
        if fn_needs_namespace.get(base_fn, False) and not namespace_key:
            namespace_key = base_fn

        key = (section, base_fn, namespace_key)
        cached = specialized_name_cache.get(key)
        if cached is not None:
            return cached
        if key in in_progress:
            raise ValueError(f"Recursive or cyclic user-function specialization detected for {base_fn}")

        spec_name = _make_specialized_fn_name(section, base_fn, namespace_key)
        specialized_name_cache[key] = spec_name
        in_progress.add(key)

        local_map = {name: _make_persistent_local_name(section, base_fn, name) for name in orig.locals}
        instance_map = {name: _make_instance_var_name(namespace_key, name) for name in orig.instances} if namespace_key else {}
        params = set(orig.params)

        body = rewrite_node(orig.body, section, namespace_key, params, local_map, instance_map)

        specialized_defs[spec_name] = FunctionDef(
            orig.id,
            orig.span,
            spec_name,
            list(orig.params),
            [],
            [],
            body,
        )

        in_progress.remove(key)
        return spec_name

    def rewrite_call_name(fn_name: str, section: str, current_namespace: Optional[str]) -> str:
        base_fn, call_namespace = resolve_user_call(fn_name, current_namespace)
        if base_fn is None:
            return fn_name
        return ensure_specialized(section, base_fn, call_namespace)

    def rewrite_node(n: Node, section: str, current_namespace: Optional[str], params: Set[str], local_map: Dict[str, str], instance_map: Dict[str, str]) -> Node:
        if isinstance(n, (Num, StrLit)):
            return n
        if isinstance(n, Var):
            new_name = rewrite_var_name(n.name, params, local_map, instance_map, current_namespace)
            if new_name == n.name:
                return n
            return Var(n.id, n.span, new_name)
        if isinstance(n, Index):
            return Index(n.id, n.span, rewrite_node(n.base, section, current_namespace, params, local_map, instance_map), rewrite_node(n.index, section, current_namespace, params, local_map, instance_map))
        if isinstance(n, Unary):
            return Unary(n.id, n.span, n.op, rewrite_node(n.a, section, current_namespace, params, local_map, instance_map))
        if isinstance(n, Binary):
            return Binary(n.id, n.span, n.op, rewrite_node(n.l, section, current_namespace, params, local_map, instance_map), rewrite_node(n.r, section, current_namespace, params, local_map, instance_map))
        if isinstance(n, Assign):
            return Assign(n.id, n.span, n.op, rewrite_node(n.target, section, current_namespace, params, local_map, instance_map), rewrite_node(n.value, section, current_namespace, params, local_map, instance_map))
        if isinstance(n, Call):
            new_fn = rewrite_call_name(n.fn, section, current_namespace)
            new_args = [rewrite_node(a, section, current_namespace, params, local_map, instance_map) for a in n.args]
            if new_fn == n.fn and len(new_args) == len(n.args) and all(a1 is a2 for a1, a2 in zip(new_args, n.args)):
                return n
            return Call(n.id, n.span, new_fn, new_args)
        if isinstance(n, Loop):
            return Loop(n.id, n.span, rewrite_node(n.count, section, current_namespace, params, local_map, instance_map), rewrite_node(n.body, section, current_namespace, params, local_map, instance_map))
        if isinstance(n, Ternary):
            return Ternary(n.id, n.span, rewrite_node(n.cond, section, current_namespace, params, local_map, instance_map), rewrite_node(n.then, section, current_namespace, params, local_map, instance_map), rewrite_node(n.els, section, current_namespace, params, local_map, instance_map))
        if isinstance(n, Seq):
            return Seq(n.id, n.span, [rewrite_node(it, section, current_namespace, params, local_map, instance_map) for it in n.items])
        if isinstance(n, If):
            return If(n.id, n.span, rewrite_node(n.cond, section, current_namespace, params, local_map, instance_map), rewrite_node(n.then, section, current_namespace, params, local_map, instance_map), rewrite_node(n.els, section, current_namespace, params, local_map, instance_map) if n.els is not None else None)
        if isinstance(n, While):
            return While(n.id, n.span, rewrite_node(n.cond, section, current_namespace, params, local_map, instance_map), rewrite_node(n.body, section, current_namespace, params, local_map, instance_map))
        if isinstance(n, FunctionDef):
            raise TypeError("Unexpected nested FunctionDef during lowering")
        raise TypeError(type(n))

    lowered_programs: Dict[str, List[Node]] = {}
    for section, prog in programs.items():
        lowered_programs[section] = [
            rewrite_node(node, section, None, set(), {}, {})
            for node in prog
        ]

    return lowered_programs, specialized_defs



# -----------------------------
# Loop-invariant scalar hoisting
# -----------------------------

_PURE_HOIST_CALLS: Set[str] = {
    "min", "max",
    "sin", "cos", "sqrt", "fabs", "floor", "ceil",
    "pow", "exp", "log", "tan", "log10",
}

_SLIDER_VAR_RE = re.compile(r"^slider([1-9][0-9]?)$")
_JSFX_HEX_CONST_RE = re.compile(r"^\$x[0-9A-Fa-f]+$")


class _LoopMutationSummary:
    __slots__ = (
        "written_vars",
        "writes_mem",
        "writes_dynamic_slider",
        "writes_dynamic_spl",
        "writes_unknown_state",
    )

    def __init__(self) -> None:
        self.written_vars: Set[str] = set()
        self.writes_mem = False
        self.writes_dynamic_slider = False
        self.writes_dynamic_spl = False
        self.writes_unknown_state = False

    def merge(self, other: '_LoopMutationSummary') -> '_LoopMutationSummary':
        self.written_vars.update(other.written_vars)
        self.writes_mem = self.writes_mem or other.writes_mem
        self.writes_dynamic_slider = self.writes_dynamic_slider or other.writes_dynamic_slider
        self.writes_dynamic_spl = self.writes_dynamic_spl or other.writes_dynamic_spl
        self.writes_unknown_state = self.writes_unknown_state or other.writes_unknown_state
        return self


def _is_slider_var_name(name: str) -> bool:
    m = _SLIDER_VAR_RE.fullmatch(name)
    return m is not None and 1 <= int(m.group(1)) <= 64


def _is_spl_var_name(name: str) -> bool:
    m = _SPL_RE.fullmatch(name)
    return m is not None and 0 <= int(m.group(1)) < 64


def _is_compile_time_constant_var(name: str) -> bool:
    if name == "mem":
        return True
    if name in ("$pi", "$e"):
        return True
    return _JSFX_HEX_CONST_RE.fullmatch(name) is not None


def _note_mutated_lvalue(node: Node, out: _LoopMutationSummary) -> None:
    if isinstance(node, Var):
        if node.name != "mem":
            out.written_vars.add(node.name)
        return
    if isinstance(node, Index):
        out.writes_mem = True
        return
    if isinstance(node, Call) and len(node.args) == 1 and node.fn in ("slider", "spl"):
        if node.fn == "slider":
            out.writes_dynamic_slider = True
        else:
            out.writes_dynamic_spl = True
        return
    out.writes_unknown_state = True


def _summarize_loop_mutations(node: Node, user_fn_names: Set[str]) -> _LoopMutationSummary:
    out = _LoopMutationSummary()

    def rec(n: Node) -> None:
        if isinstance(n, (Num, StrLit, Var)):
            return
        if isinstance(n, Index):
            rec(n.base)
            rec(n.index)
            return
        if isinstance(n, Unary):
            rec(n.a)
            return
        if isinstance(n, Binary):
            rec(n.l)
            rec(n.r)
            return
        if isinstance(n, Assign):
            _note_mutated_lvalue(n.target, out)
            rec(n.target)
            rec(n.value)
            return
        if isinstance(n, Call):
            for a in n.args:
                rec(a)

            fn = n.fn
            if fn in user_fn_names:
                out.writes_unknown_state = True
                return

            if fn == "midirecv":
                for a in n.args[:4]:
                    _note_mutated_lvalue(a, out)
                return

            if fn == "file_var" and len(n.args) >= 2:
                _note_mutated_lvalue(n.args[1], out)
                return

            if fn == "file_riff" and len(n.args) >= 3:
                _note_mutated_lvalue(n.args[1], out)
                _note_mutated_lvalue(n.args[2], out)
                return

            if fn in ("memset", "fft", "ifft", "fft_permute", "fft_ipermute", "file_mem"):
                out.writes_mem = True
                return

            if fn == "abs":
                fn = "fabs"

            if fn in _PURE_HOIST_CALLS:
                return

            if fn.startswith("gfx_") or fn in (
                "slider", "spl",
                "midisend", "rand",
                "freembuf", "sliderchange", "slider_automate",
                "file_open", "file_close", "file_rewind", "file_seek", "file_avail", "file_text",
                "sprintf", "printf",
                "strcpy", "strcat", "strcmp", "strlen",
                "str_getchar", "str_setchar",
                "str_insert", "str_delete", "str_mid", "strncpy",
                "file_read", "file_write", "file_string",
            ):
                return

            out.writes_unknown_state = True
            return
        if isinstance(n, Loop):
            rec(n.count)
            rec(n.body)
            return
        if isinstance(n, Ternary):
            rec(n.cond)
            rec(n.then)
            rec(n.els)
            return
        if isinstance(n, Seq):
            for it in n.items:
                rec(it)
            return
        if isinstance(n, If):
            rec(n.cond)
            rec(n.then)
            if n.els is not None:
                rec(n.els)
            return
        if isinstance(n, While):
            rec(n.cond)
            rec(n.body)
            return
        if isinstance(n, FunctionDef):
            rec(n.body)
            return
        raise TypeError(type(n))

    rec(node)
    return out


def _is_loop_invariant_expr(node: Node, summary: _LoopMutationSummary, user_fn_names: Set[str]) -> bool:
    if isinstance(node, (Num, StrLit)):
        return True

    if isinstance(node, Var):
        if _is_compile_time_constant_var(node.name):
            return True
        if node.name in summary.written_vars:
            return False
        if summary.writes_unknown_state:
            return False
        if _is_slider_var_name(node.name) and summary.writes_dynamic_slider:
            return False
        if _is_spl_var_name(node.name) and summary.writes_dynamic_spl:
            return False
        return True

    if isinstance(node, Index):
        return False

    if isinstance(node, Unary):
        return _is_loop_invariant_expr(node.a, summary, user_fn_names)

    if isinstance(node, Binary):
        return (_is_loop_invariant_expr(node.l, summary, user_fn_names) and
                _is_loop_invariant_expr(node.r, summary, user_fn_names))

    if isinstance(node, Ternary):
        return (_is_loop_invariant_expr(node.cond, summary, user_fn_names) and
                _is_loop_invariant_expr(node.then, summary, user_fn_names) and
                _is_loop_invariant_expr(node.els, summary, user_fn_names))

    if isinstance(node, Call):
        if node.fn in user_fn_names:
            return False
        fn = "fabs" if node.fn == "abs" else node.fn
        if fn not in _PURE_HOIST_CALLS:
            return False
        return all(_is_loop_invariant_expr(a, summary, user_fn_names) for a in node.args)

    if isinstance(node, (Assign, Loop, Seq, If, While, FunctionDef)):
        return False

    raise TypeError(type(node))


def _collect_loop_invariant_candidates(node: Node,
                                       summary: _LoopMutationSummary,
                                       user_fn_names: Set[str],
                                       out: List[Node]) -> None:
    if isinstance(node, (Loop, While)):
        return

    if isinstance(node, (Unary, Binary, Ternary, Call)) and _is_loop_invariant_expr(node, summary, user_fn_names):
        out.append(node)
        return

    if isinstance(node, (Num, StrLit, Var)):
        return
    if isinstance(node, Index):
        _collect_loop_invariant_candidates(node.base, summary, user_fn_names, out)
        _collect_loop_invariant_candidates(node.index, summary, user_fn_names, out)
        return
    if isinstance(node, Unary):
        _collect_loop_invariant_candidates(node.a, summary, user_fn_names, out)
        return
    if isinstance(node, Binary):
        _collect_loop_invariant_candidates(node.l, summary, user_fn_names, out)
        _collect_loop_invariant_candidates(node.r, summary, user_fn_names, out)
        return
    if isinstance(node, Assign):
        _collect_loop_invariant_candidates(node.target, summary, user_fn_names, out)
        _collect_loop_invariant_candidates(node.value, summary, user_fn_names, out)
        return
    if isinstance(node, Call):
        for a in node.args:
            _collect_loop_invariant_candidates(a, summary, user_fn_names, out)
        return
    if isinstance(node, Ternary):
        _collect_loop_invariant_candidates(node.cond, summary, user_fn_names, out)
        _collect_loop_invariant_candidates(node.then, summary, user_fn_names, out)
        _collect_loop_invariant_candidates(node.els, summary, user_fn_names, out)
        return
    if isinstance(node, Seq):
        for it in node.items:
            _collect_loop_invariant_candidates(it, summary, user_fn_names, out)
        return
    if isinstance(node, If):
        _collect_loop_invariant_candidates(node.cond, summary, user_fn_names, out)
        _collect_loop_invariant_candidates(node.then, summary, user_fn_names, out)
        if node.els is not None:
            _collect_loop_invariant_candidates(node.els, summary, user_fn_names, out)
        return
    if isinstance(node, FunctionDef):
        _collect_loop_invariant_candidates(node.body, summary, user_fn_names, out)
        return
    raise TypeError(type(node))


def plan_loop_invariant_hoists(programs: Dict[str, List[Node]],
                               fn_defs: Dict[str, FunctionDef]) -> Dict[int, List[Node]]:
    user_fn_names = set(fn_defs.keys())
    out: Dict[int, List[Node]] = {}

    def visit(n: Node) -> None:
        if isinstance(n, Loop):
            summary = _summarize_loop_mutations(n.body, user_fn_names)
            cands: List[Node] = []
            _collect_loop_invariant_candidates(n.body, summary, user_fn_names, cands)
            if cands:
                out[n.id] = cands
            visit(n.count)
            visit(n.body)
            return

        if isinstance(n, While):
            summary = _summarize_loop_mutations(n.cond, user_fn_names)
            summary.merge(_summarize_loop_mutations(n.body, user_fn_names))
            cands: List[Node] = []
            _collect_loop_invariant_candidates(n.cond, summary, user_fn_names, cands)
            _collect_loop_invariant_candidates(n.body, summary, user_fn_names, cands)
            if cands:
                out[n.id] = cands
            visit(n.cond)
            visit(n.body)
            return

        if isinstance(n, (Num, StrLit, Var)):
            return
        if isinstance(n, Index):
            visit(n.base)
            visit(n.index)
            return
        if isinstance(n, Unary):
            visit(n.a)
            return
        if isinstance(n, Binary):
            visit(n.l)
            visit(n.r)
            return
        if isinstance(n, Assign):
            visit(n.target)
            visit(n.value)
            return
        if isinstance(n, Call):
            for a in n.args:
                visit(a)
            return
        if isinstance(n, Ternary):
            visit(n.cond)
            visit(n.then)
            visit(n.els)
            return
        if isinstance(n, Seq):
            for it in n.items:
                visit(it)
            return
        if isinstance(n, If):
            visit(n.cond)
            visit(n.then)
            if n.els is not None:
                visit(n.els)
            return
        if isinstance(n, FunctionDef):
            visit(n.body)
            return
        raise TypeError(type(n))

    for prog in programs.values():
        for st in prog:
            visit(st)

    for f in fn_defs.values():
        visit(f.body)

    return out


class _SectionRWInfo:
    __slots__ = (
        "read_vars",
        "write_counts",
        "writes_mem",
        "writes_dynamic_slider",
        "writes_dynamic_spl",
        "writes_unknown_state",
    )

    def __init__(self) -> None:
        self.read_vars: Set[str] = set()
        self.write_counts: Counter[str] = Counter()
        self.writes_mem = False
        self.writes_dynamic_slider = False
        self.writes_dynamic_spl = False
        self.writes_unknown_state = False

    @property
    def written_vars(self) -> Set[str]:
        return set(self.write_counts.keys())

    def merge(self, other: '_SectionRWInfo') -> '_SectionRWInfo':
        self.read_vars.update(other.read_vars)
        self.write_counts.update(other.write_counts)
        self.writes_mem = self.writes_mem or other.writes_mem
        self.writes_dynamic_slider = self.writes_dynamic_slider or other.writes_dynamic_slider
        self.writes_dynamic_spl = self.writes_dynamic_spl or other.writes_dynamic_spl
        self.writes_unknown_state = self.writes_unknown_state or other.writes_unknown_state
        return self


def _note_section_write_var(out: _SectionRWInfo, name: str) -> None:
    if name != "mem":
        out.write_counts[name] += 1


def _collect_section_lvalue_rw(node: Node, user_fn_names: Set[str], out: _SectionRWInfo) -> None:
    if isinstance(node, Var):
        _note_section_write_var(out, node.name)
        return
    if isinstance(node, Index):
        out.writes_mem = True
        _collect_section_rw(node.base, user_fn_names, out)
        _collect_section_rw(node.index, user_fn_names, out)
        return
    if isinstance(node, Call) and len(node.args) == 1 and node.fn in ("slider", "spl"):
        _collect_section_rw(node.args[0], user_fn_names, out)
        if node.fn == "slider":
            out.writes_dynamic_slider = True
        else:
            out.writes_dynamic_spl = True
        return
    out.writes_unknown_state = True
    _collect_section_rw(node, user_fn_names, out)


def _collect_section_rw(node: Node, user_fn_names: Set[str], out: _SectionRWInfo) -> None:
    if isinstance(node, (Num, StrLit)):
        return
    if isinstance(node, Var):
        out.read_vars.add(node.name)
        return
    if isinstance(node, Index):
        _collect_section_rw(node.base, user_fn_names, out)
        _collect_section_rw(node.index, user_fn_names, out)
        return
    if isinstance(node, Unary):
        _collect_section_rw(node.a, user_fn_names, out)
        return
    if isinstance(node, Binary):
        _collect_section_rw(node.l, user_fn_names, out)
        _collect_section_rw(node.r, user_fn_names, out)
        return
    if isinstance(node, Assign):
        _collect_section_lvalue_rw(node.target, user_fn_names, out)
        _collect_section_rw(node.value, user_fn_names, out)
        return
    if isinstance(node, Call):
        fn = node.fn
        if fn == "midirecv":
            for a in node.args[:4]:
                _collect_section_lvalue_rw(a, user_fn_names, out)
            return
        if fn == "file_var":
            if len(node.args) >= 1:
                _collect_section_rw(node.args[0], user_fn_names, out)
            if len(node.args) >= 2:
                _collect_section_lvalue_rw(node.args[1], user_fn_names, out)
            for a in node.args[2:]:
                _collect_section_rw(a, user_fn_names, out)
            return
        if fn == "file_riff":
            if len(node.args) >= 1:
                _collect_section_rw(node.args[0], user_fn_names, out)
            if len(node.args) >= 2:
                _collect_section_lvalue_rw(node.args[1], user_fn_names, out)
            if len(node.args) >= 3:
                _collect_section_lvalue_rw(node.args[2], user_fn_names, out)
            for a in node.args[3:]:
                _collect_section_rw(a, user_fn_names, out)
            return

        for a in node.args:
            _collect_section_rw(a, user_fn_names, out)

        if fn in user_fn_names:
            out.writes_unknown_state = True
            return

        if fn in ("memset", "fft", "ifft", "fft_permute", "fft_ipermute", "file_mem"):
            out.writes_mem = True
            return

        return
    if isinstance(node, Loop):
        _collect_section_rw(node.count, user_fn_names, out)
        _collect_section_rw(node.body, user_fn_names, out)
        return
    if isinstance(node, Ternary):
        _collect_section_rw(node.cond, user_fn_names, out)
        _collect_section_rw(node.then, user_fn_names, out)
        _collect_section_rw(node.els, user_fn_names, out)
        return
    if isinstance(node, Seq):
        for it in node.items:
            _collect_section_rw(it, user_fn_names, out)
        return
    if isinstance(node, If):
        _collect_section_rw(node.cond, user_fn_names, out)
        _collect_section_rw(node.then, user_fn_names, out)
        if node.els is not None:
            _collect_section_rw(node.els, user_fn_names, out)
        return
    if isinstance(node, While):
        _collect_section_rw(node.cond, user_fn_names, out)
        _collect_section_rw(node.body, user_fn_names, out)
        return
    if isinstance(node, FunctionDef):
        _collect_section_rw(node.body, user_fn_names, out)
        return
    raise TypeError(type(node))


def _summarize_section_rw(nodes: List[Node], user_fn_names: Set[str]) -> _SectionRWInfo:
    out = _SectionRWInfo()
    for node in nodes:
        _collect_section_rw(node, user_fn_names, out)
    return out


class _StageInvariantSummary:
    __slots__ = (
        "mutable_vars",
        "writes_dynamic_slider",
        "writes_unknown_state",
        "allow_slider_vars",
        "allow_samplesblock",
        "allow_srate",
    )

    def __init__(self,
                 mutable_vars: Set[str],
                 *,
                 writes_dynamic_slider: bool,
                 writes_unknown_state: bool,
                 allow_slider_vars: bool,
                 allow_samplesblock: bool,
                 allow_srate: bool) -> None:
        self.mutable_vars = set(mutable_vars)
        self.writes_dynamic_slider = writes_dynamic_slider
        self.writes_unknown_state = writes_unknown_state
        self.allow_slider_vars = allow_slider_vars
        self.allow_samplesblock = allow_samplesblock
        self.allow_srate = allow_srate


def _is_stage_invariant_expr(node: Node,
                             summary: _StageInvariantSummary,
                             user_fn_names: Set[str]) -> bool:
    if isinstance(node, (Num, StrLit)):
        return True

    if isinstance(node, Var):
        if _is_compile_time_constant_var(node.name):
            return True
        if summary.writes_unknown_state:
            return False
        if node.name == "srate":
            return summary.allow_srate and node.name not in summary.mutable_vars
        if node.name == "samplesblock":
            return summary.allow_samplesblock and node.name not in summary.mutable_vars
        if _is_spl_var_name(node.name):
            return False
        if _is_slider_var_name(node.name):
            return (summary.allow_slider_vars and
                    not summary.writes_dynamic_slider and
                    node.name not in summary.mutable_vars)
        return node.name not in summary.mutable_vars

    if isinstance(node, Index):
        return False

    if isinstance(node, Unary):
        return _is_stage_invariant_expr(node.a, summary, user_fn_names)

    if isinstance(node, Binary):
        return (_is_stage_invariant_expr(node.l, summary, user_fn_names) and
                _is_stage_invariant_expr(node.r, summary, user_fn_names))

    if isinstance(node, Ternary):
        return (_is_stage_invariant_expr(node.cond, summary, user_fn_names) and
                _is_stage_invariant_expr(node.then, summary, user_fn_names) and
                _is_stage_invariant_expr(node.els, summary, user_fn_names))

    if isinstance(node, Call):
        if node.fn in user_fn_names:
            return False
        fn = "fabs" if node.fn == "abs" else node.fn
        if fn not in _PURE_HOIST_CALLS:
            return False
        return all(_is_stage_invariant_expr(a, summary, user_fn_names) for a in node.args)

    if isinstance(node, (Assign, Loop, Seq, If, While, FunctionDef)):
        return False

    raise TypeError(type(node))


def _adjusted_write_counts(base: Counter[str], name: str) -> Counter[str]:
    out = Counter(base)
    if out.get(name, 0) > 0:
        out[name] -= 1
        if out[name] <= 0:
            del out[name]
    return out


def _make_user_var_assignment_candidate(node: Node) -> Optional[Tuple[str, Node]]:
    if not isinstance(node, Assign):
        return None
    if node.op != "=":
        return None
    if not isinstance(node.target, Var):
        return None
    name = node.target.name
    if name in BUILTIN_NAMES:
        return None
    if _is_slider_var_name(name) or _is_spl_var_name(name):
        return None
    if name.startswith("$"):
        return None
    return name, node.value


def _plan_sample_to_block_hoists(programs: Dict[str, List[Node]], user_fn_names: Set[str]) -> None:
    slider_info = _summarize_section_rw(programs["slider"], user_fn_names)
    active_sample_info = _summarize_section_rw(programs["sample"], user_fn_names)

    if slider_info.writes_unknown_state or active_sample_info.writes_unknown_state:
        return

    new_sample: List[Node] = []
    moved: List[Node] = []
    prefix_reads: Set[str] = set()
    prefix_writes: Set[str] = set()

    for node in programs["sample"]:
        info = _summarize_section_rw([node], user_fn_names)
        cand = _make_user_var_assignment_candidate(node)
        movable = False

        if cand is not None:
            target_name, rhs = cand
            rhs_info = _summarize_section_rw([rhs], user_fn_names)
            adjusted_counts = _adjusted_write_counts(active_sample_info.write_counts, target_name)
            mutable_vars = set(adjusted_counts.keys()) | slider_info.written_vars
            stage_summary = _StageInvariantSummary(
                mutable_vars,
                writes_dynamic_slider=(active_sample_info.writes_dynamic_slider or slider_info.writes_dynamic_slider),
                writes_unknown_state=False,
                allow_slider_vars=True,
                allow_samplesblock=True,
                allow_srate=True,
            )

            movable = (
                target_name not in prefix_reads and
                target_name not in prefix_writes and
                target_name not in slider_info.read_vars and
                target_name not in slider_info.written_vars and
                target_name not in adjusted_counts and
                target_name not in rhs_info.read_vars and
                _is_stage_invariant_expr(rhs, stage_summary, user_fn_names)
            )

        if movable:
            moved.append(node)
            active_sample_info.write_counts = _adjusted_write_counts(active_sample_info.write_counts, target_name)
            continue

        new_sample.append(node)
        prefix_reads.update(info.read_vars)
        prefix_writes.update(info.written_vars)

    if moved:
        programs["sample"] = new_sample
        programs["block"] = list(programs["block"]) + moved


def _plan_block_to_init_hoists(programs: Dict[str, List[Node]], user_fn_names: Set[str]) -> None:
    slider_info = _summarize_section_rw(programs["slider"], user_fn_names)
    sample_info = _summarize_section_rw(programs["sample"], user_fn_names)
    active_block_info = _summarize_section_rw(programs["block"], user_fn_names)

    if slider_info.writes_unknown_state or sample_info.writes_unknown_state or active_block_info.writes_unknown_state:
        return

    new_block: List[Node] = []
    moved: List[Node] = []
    prefix_reads: Set[str] = set()
    prefix_writes: Set[str] = set()

    for node in programs["block"]:
        info = _summarize_section_rw([node], user_fn_names)
        cand = _make_user_var_assignment_candidate(node)
        movable = False

        if cand is not None:
            target_name, rhs = cand
            rhs_info = _summarize_section_rw([rhs], user_fn_names)
            adjusted_counts = _adjusted_write_counts(active_block_info.write_counts, target_name)
            mutable_vars = set(adjusted_counts.keys()) | slider_info.written_vars | sample_info.written_vars
            stage_summary = _StageInvariantSummary(
                mutable_vars,
                writes_dynamic_slider=(active_block_info.writes_dynamic_slider or slider_info.writes_dynamic_slider or sample_info.writes_dynamic_slider),
                writes_unknown_state=False,
                allow_slider_vars=False,
                allow_samplesblock=False,
                allow_srate=True,
            )

            movable = (
                target_name not in prefix_reads and
                target_name not in prefix_writes and
                target_name not in slider_info.read_vars and
                target_name not in slider_info.written_vars and
                target_name not in adjusted_counts and
                target_name not in sample_info.written_vars and
                target_name not in rhs_info.read_vars and
                _is_stage_invariant_expr(rhs, stage_summary, user_fn_names)
            )

        if movable:
            moved.append(node)
            active_block_info.write_counts = _adjusted_write_counts(active_block_info.write_counts, target_name)
            continue

        new_block.append(node)
        prefix_reads.update(info.read_vars)
        prefix_writes.update(info.written_vars)

    if moved:
        programs["block"] = new_block
        programs["init"] = list(programs["init"]) + moved


def hoist_section_invariants(programs: Dict[str, List[Node]],
                             fn_defs: Dict[str, FunctionDef]) -> Dict[str, List[Node]]:
    out = {sec: list(nodes) for sec, nodes in programs.items()}
    user_fn_names = set(fn_defs.keys())

    _plan_sample_to_block_hoists(out, user_fn_names)
    _plan_block_to_init_hoists(out, user_fn_names)
    return out


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
        # 7: DSPJSFX_MidiEvent* midiIn
        # 8: i32 midiInCount
        # 9: i32 midiInReadIndex
        # 10: i32 midiInCapacity
        # 11: DSPJSFX_MidiEvent* midiOut
        # 12: i32 midiOutCount
        # 13: i32 midiOutCapacity
        # 14: i32 currentBlockSize
        # 15: double currentSampleRate
        # 16: i32 pendingNoteCleanup
        # 17..22: optional diagnostics counters
        # 23: i64 pendingSliderChangeMask
        # 24: i64 pendingSliderAutomateMask
        # 25: i64 pendingSliderAutomateEndMask
        # 26: uint32_t randMT[624]
        # 27: uint32_t randIndex (0 means uninitialized; mirrors EEL2 __idx)
        # 28: int64_t sliderVisibleMask (bitmask, 1=visible)
        # 29: int32_t sliderVisibilityInit
        self.var_cap = max(1, (max(sym.vars.values()) + 1) if sym.vars else 1)
        self.midi_event_ty = ir.LiteralStructType([self.i32, self.i32, self.i32, self.i32])

        self.state_ty = ir.LiteralStructType([
            ir.ArrayType(self.double, 64),
            ir.ArrayType(self.double, 64),
            ir.ArrayType(self.double, self.var_cap),
            self.double.as_pointer(),
            self.i64,
            self.double,
            self.double,
            self.midi_event_ty.as_pointer(),
            self.i32,
            self.i32,
            self.i32,
            self.midi_event_ty.as_pointer(),
            self.i32,
            self.i32,
            self.i32,
            self.double,
            self.i32,
            self.i32,
            self.i32,
            self.i32,
            self.i32,
            self.i32,
            self.i32,
            self.i64,
            self.i64,
            self.i64,
            ir.ArrayType(self.i32, 624),
            self.i32,
            self.i64,
            self.i32,
        ])
        self.state_ptr = self.state_ty.as_pointer()

        self.module = ir.Module(name="dsp_jsfx_module")

        # extern ensure
        self.fn_ensure = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [self.state_ptr, self.i64]),
            name="jsfx_ensure_mem"
        )

        self.fn_midirecv = ir.Function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer()]),
            name="jsfx_midirecv"
        )
        self.fn_midisend = ir.Function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double, self.double]),
            name="jsfx_midisend"
        )
        self.fn_sliderchange = ir.Function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double]),
            name="jsfx_sliderchange"
        )
        self.fn_slider_automate = ir.Function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double]),
            name="jsfx_slider_automate"
        )

        self._intrinsics: Dict[str, ir.Function] = {}
        self._rand_gen32_fn: Optional[ir.Function] = None
        self.user_fn_defs: Dict[str, FunctionDef] = {}
        self.user_fn_ir: Dict[str, ir.Function] = {}
        self.loop_hoists: Dict[int, List[Node]] = {}
        self._local_slots_stack: List[Dict[str, ir.Value]] = []
        self._hoisted_value_stack: List[Dict[int, ir.Value]] = []


        self._buildins: Dict[str, ir.Function] = {}

        # String literal pool (DSP-only):
        # We represent string literals as opaque numeric handles (doubles).
        # This keeps parsing/compilation working for scripts that include string
        # helpers (often used by @gfx) inside @init/@slider.
        # Full EEL2 string semantics are NOT implemented in this DSP compiler.
        self._str_lits: Dict[str, int] = {}
        self._next_str_id: int = 0
        self._str_handle_base: int = (1 << 40)

    def _intern_string(self, s: str) -> int:
        # Return a stable nonzero handle for this literal.
        # We offset into a high range to reduce accidental collisions with typical
        # mem() indices or small numeric constants.
        if s in self._str_lits:
            return self._str_lits[s]
        hid = self._str_handle_base + self._next_str_id
        self._next_str_id += 1
        self._str_lits[s] = hid
        return hid

    def _const_f64(self, v: float) -> ir.Constant:
        return ir.Constant(self.double, float(v))

    def _const_i64(self, v: int) -> ir.Constant:
        return ir.Constant(self.i64, int(v))

    def _const_i32(self, v: int) -> ir.Constant:
        return ir.Constant(self.i32, int(v))

    def _lookup_hoisted_value(self, node: Node) -> Optional[ir.Value]:
        for env in reversed(self._hoisted_value_stack):
            cached = env.get(node.id)
            if cached is not None:
                return cached
        return None

    def _compute_loop_hoisted_values(self, builder: ir.IRBuilder, st: ir.Value, loop_id: int) -> Dict[int, ir.Value]:
        out: Dict[int, ir.Value] = {}
        for node in self.loop_hoists.get(loop_id, []):
            out[node.id] = self.emit_expr(builder, st, node)
        return out

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

        if fn in ("pow", "exp", "log", "tan", "log10"):
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


    def _dyn_state_array_ptr(self, builder: ir.IRBuilder, st: ir.Value, which: str, idx_expr: Node) -> Tuple[ir.Value, ir.Value]:
        """Return (ptr, in_range) for dynamic access to st->spl[] or st->sliders[].

        JSFX/EEL2 idiom:
          - spl(i) is 0-based (spl(0) == spl0)
          - slider(i) is 1-based (slider(1) == slider1)

        Out-of-range reads return 0, and out-of-range writes are ignored.
        """
        # Evaluate index expression as f64, then convert to integer with the same
        # tiny bias used elsewhere (matches EEL2's common truncation behavior).
        idx_v = self.emit_expr(builder, st, idx_expr)
        idx_v = builder.fadd(idx_v, self._const_f64(1.0e-5))
        idx_i64 = builder.fptosi(idx_v, self.i64)

        if which == "slider":
            # slider(1) -> slider1 -> index 0 in our array
            idx0_i64 = builder.sub(idx_i64, self._const_i64(1))
            field = 1  # sliders[64]
        elif which == "spl":
            # spl(0) -> spl0 -> index 0
            idx0_i64 = idx_i64
            field = 0  # spl[64]
        else:
            raise ValueError(f"Unknown dynamic state array {which!r}")

        zero_i64 = self._const_i64(0)
        sixty4_i64 = self._const_i64(64)

        ge0 = builder.icmp_signed(">=", idx0_i64, zero_i64)
        lt64 = builder.icmp_signed("<", idx0_i64, sixty4_i64)
        in_range = builder.and_(ge0, lt64)

        # Clamp to [0, 63] so GEP is always in-bounds even when out-of-range;
        # we still use 'in_range' to return 0 / ignore writes.
        idx_clamped = idx0_i64
        isneg = builder.icmp_signed("<", idx_clamped, zero_i64)
        idx_clamped = builder.select(isneg, zero_i64, idx_clamped)

        max63_i64 = self._const_i64(63)
        isgt = builder.icmp_signed(">", idx_clamped, max63_i64)
        idx_clamped = builder.select(isgt, max63_i64, idx_clamped)

        idx_i32 = builder.trunc(idx_clamped, self.i32)

        z = ir.Constant(self.i32, 0)
        fld = ir.Constant(self.i32, field)
        ptr = builder.gep(st, [z, fld, idx_i32], inbounds=True)
        return ptr, in_range

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

    def _get_rand_mt_ptr(self, builder: ir.IRBuilder, st: ir.Value) -> ir.Value:
        zero = ir.Constant(self.i32, 0)
        fld = ir.Constant(self.i32, 26)
        return builder.gep(st, [zero, fld], inbounds=True)

    def _get_rand_idx_ptr(self, builder: ir.IRBuilder, st: ir.Value) -> ir.Value:
        zero = ir.Constant(self.i32, 0)
        fld = ir.Constant(self.i32, 27)
        return builder.gep(st, [zero, fld], inbounds=True)

    def _get_slider_visible_mask_ptr(self, builder: ir.IRBuilder, st: ir.Value) -> ir.Value:
        zero = ir.Constant(self.i32, 0)
        fld = ir.Constant(self.i32, 28)
        return builder.gep(st, [zero, fld], inbounds=True)

    def _get_slider_visibility_init_ptr(self, builder: ir.IRBuilder, st: ir.Value) -> ir.Value:
        zero = ir.Constant(self.i32, 0)
        fld = ir.Constant(self.i32, 29)
        return builder.gep(st, [zero, fld], inbounds=True)

    def _ensure_slider_visibility_initialized(self, builder: ir.IRBuilder, st: ir.Value) -> None:
        init_ptr = self._get_slider_visibility_init_ptr(builder, st)
        init_v = builder.load(init_ptr)
        needs_init = builder.icmp_signed("==", init_v, self._const_i32(0))
        with builder.if_then(needs_init):
            vis_ptr = self._get_slider_visible_mask_ptr(builder, st)
            builder.store(ir.Constant(self.i64, -1), vis_ptr)
            builder.store(self._const_i32(1), init_ptr)

    def _ensure_rand_gen32_fn(self) -> ir.Function:
        if self._rand_gen32_fn is not None:
            return self._rand_gen32_fn

        N = 624
        M = 397
        MATRIX_A = 0x9908B0DF
        UPPER_MASK = 0x80000000
        LOWER_MASK = 0x7FFFFFFF

        fn = ir.Function(self.module, ir.FunctionType(self.i32, [self.state_ptr]), name="jsfx_rand_genrand_int32")
        fn.linkage = "internal"
        self._rand_gen32_fn = fn

        entry = fn.append_basic_block("entry")
        init_seed = fn.append_basic_block("init_seed")
        init_cond = fn.append_basic_block("init_cond")
        init_body = fn.append_basic_block("init_body")
        init_done = fn.append_basic_block("init_done")
        after_init = fn.append_basic_block("after_init")
        twist_entry = fn.append_basic_block("twist_entry")
        twist1_cond = fn.append_basic_block("twist1_cond")
        twist1_body = fn.append_basic_block("twist1_body")
        twist1_done = fn.append_basic_block("twist1_done")
        twist2_cond = fn.append_basic_block("twist2_cond")
        twist2_body = fn.append_basic_block("twist2_body")
        twist2_done = fn.append_basic_block("twist2_done")
        twist_last = fn.append_basic_block("twist_last")
        after_twist = fn.append_basic_block("after_twist")
        no_twist = fn.append_basic_block("no_twist")
        temper = fn.append_basic_block("temper")

        st = fn.args[0]
        st.name = "st"

        c0 = self._const_i32(0)
        c1 = self._const_i32(1)
        c7 = self._const_i32(7)
        c11 = self._const_i32(11)
        c15 = self._const_i32(15)
        c18 = self._const_i32(18)
        c30 = self._const_i32(30)
        cN = self._const_i32(N)
        cM = self._const_i32(M)
        cN_minus_M = self._const_i32(N - M)
        cN_minus_1 = self._const_i32(N - 1)
        cMatrixA = self._const_i32(MATRIX_A)
        cUpperMask = self._const_i32(UPPER_MASK)
        cLowerMask = self._const_i32(LOWER_MASK)
        cTemperB = self._const_i32(0x9D2C5680)
        cTemperC = self._const_i32(0xEFC60000)
        cSeed = self._const_i32(0x4141F00D)
        cInitMul = self._const_i32(1812433253)

        def _as_i32(idx: Any) -> ir.Value:
            if isinstance(idx, ir.Value):
                return idx
            return self._const_i32(int(idx))

        def _mt_elem_ptr(builder: ir.IRBuilder, idx: Any) -> ir.Value:
            mt_ptr = self._get_rand_mt_ptr(builder, st)
            return builder.gep(mt_ptr, [c0, _as_i32(idx)], inbounds=True)

        def _load_mt(builder: ir.IRBuilder, idx: Any) -> ir.Value:
            return builder.load(_mt_elem_ptr(builder, idx))

        def _store_mt(builder: ir.IRBuilder, idx: Any, value: ir.Value) -> None:
            builder.store(value, _mt_elem_ptr(builder, idx))

        def _twist_value(builder: ir.IRBuilder, idx_a: Any, idx_b: Any, idx_src: Any) -> ir.Value:
            y_hi = builder.and_(_load_mt(builder, idx_a), cUpperMask)
            y_lo = builder.and_(_load_mt(builder, idx_b), cLowerMask)
            y = builder.or_(y_hi, y_lo)
            y_lsb = builder.and_(y, c1)
            use_matrix = builder.icmp_unsigned("!=", y_lsb, c0)
            mag = builder.select(use_matrix, cMatrixA, c0)
            mixed = builder.xor(builder.lshr(y, c1), mag)
            return builder.xor(_load_mt(builder, idx_src), mixed)

        b = ir.IRBuilder(entry)
        idx_ptr = self._get_rand_idx_ptr(b, st)
        idx0 = b.load(idx_ptr, name="rand_idx")
        idx_is_zero = b.icmp_unsigned("==", idx0, c0)
        b.cbranch(idx_is_zero, init_seed, after_init)

        b = ir.IRBuilder(init_seed)
        _store_mt(b, 0, cSeed)
        b.branch(init_cond)

        b = ir.IRBuilder(init_cond)
        mti_init = b.phi(self.i32, name="mti_init")
        mti_init.add_incoming(c1, init_seed)
        init_more = b.icmp_unsigned("<", mti_init, cN)
        b.cbranch(init_more, init_body, init_done)

        b = ir.IRBuilder(init_body)
        prev_idx = b.sub(mti_init, c1)
        prev = _load_mt(b, prev_idx)
        prev_mix = b.xor(prev, b.lshr(prev, c30))
        seeded = b.add(b.mul(cInitMul, prev_mix), mti_init)
        _store_mt(b, mti_init, seeded)
        next_mti = b.add(mti_init, c1)
        b.branch(init_cond)
        mti_init.add_incoming(next_mti, init_body)

        b = ir.IRBuilder(init_done)
        b.store(cN, idx_ptr)
        b.branch(after_init)

        b = ir.IRBuilder(after_init)
        mti_cur = b.phi(self.i32, name="mti")
        mti_cur.add_incoming(idx0, entry)
        mti_cur.add_incoming(cN, init_done)
        need_twist = b.icmp_unsigned(">=", mti_cur, cN)
        b.cbranch(need_twist, twist_entry, no_twist)

        b = ir.IRBuilder(twist_entry)
        b.store(c1, idx_ptr)
        b.branch(twist1_cond)

        b = ir.IRBuilder(twist1_cond)
        kk1 = b.phi(self.i32, name="kk1")
        kk1.add_incoming(c0, twist_entry)
        twist1_more = b.icmp_unsigned("<", kk1, cN_minus_M)
        b.cbranch(twist1_more, twist1_body, twist1_done)

        b = ir.IRBuilder(twist1_body)
        kk1_p1 = b.add(kk1, c1)
        kk1_pM = b.add(kk1, cM)
        twist1_val = _twist_value(b, kk1, kk1_p1, kk1_pM)
        _store_mt(b, kk1, twist1_val)
        kk1_next = b.add(kk1, c1)
        b.branch(twist1_cond)
        kk1.add_incoming(kk1_next, twist1_body)

        b = ir.IRBuilder(twist1_done)
        b.branch(twist2_cond)

        b = ir.IRBuilder(twist2_cond)
        kk2 = b.phi(self.i32, name="kk2")
        kk2.add_incoming(cN_minus_M, twist1_done)
        twist2_more = b.icmp_unsigned("<", kk2, cN_minus_1)
        b.cbranch(twist2_more, twist2_body, twist2_done)

        b = ir.IRBuilder(twist2_body)
        kk2_p1 = b.add(kk2, c1)
        kk2_src = b.sub(kk2, cN_minus_M)
        twist2_val = _twist_value(b, kk2, kk2_p1, kk2_src)
        _store_mt(b, kk2, twist2_val)
        kk2_next = b.add(kk2, c1)
        b.branch(twist2_cond)
        kk2.add_incoming(kk2_next, twist2_body)

        b = ir.IRBuilder(twist2_done)
        b.branch(twist_last)

        b = ir.IRBuilder(twist_last)
        twist_last_val = _twist_value(b, N - 1, 0, M - 1)
        _store_mt(b, N - 1, twist_last_val)
        b.branch(after_twist)

        b = ir.IRBuilder(after_twist)
        b.branch(temper)

        b = ir.IRBuilder(no_twist)
        next_idx = b.add(mti_cur, c1)
        b.store(next_idx, idx_ptr)
        b.branch(temper)

        b = ir.IRBuilder(temper)
        out_idx = b.phi(self.i32, name="out_idx")
        out_idx.add_incoming(mti_cur, no_twist)
        out_idx.add_incoming(c0, after_twist)
        y = _load_mt(b, out_idx)
        y = b.xor(y, b.lshr(y, c11))
        y = b.xor(y, b.and_(b.shl(y, c7), cTemperB))
        y = b.xor(y, b.and_(b.shl(y, c15), cTemperC))
        y = b.xor(y, b.lshr(y, c18))
        b.ret(y)

        return fn

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

    def _get_midirecv_lvalue_ptr(self, builder: ir.IRBuilder, st: ir.Value, node: Node) -> ir.Value:
        if isinstance(node, Var):
            if node.name == "mem":
                raise ValueError("midirecv output arguments must be assignable variables or mem[] slots")
            return self._get_slot_ptr(builder, st, node.name)
        if isinstance(node, Index):
            return self._mem_elem_ptr(builder, st, node.base, node.index)
        raise ValueError("midirecv output arguments must be assignable variables or mem[] slots")

    def _emit_slider_mask_arg(self, builder: ir.IRBuilder, st: ir.Value, arg: Node) -> ir.Value:
        # JSFX sliderchange()/slider_automate() accept either a direct slider
        # variable reference (slider1, slider2, ...) or an integer bitmask.
        # In AOT we can resolve direct slider variables at compile time and pass
        # numeric bitmask expressions through verbatim.
        if isinstance(arg, Var):
            m = re.fullmatch(r"slider([1-9][0-9]?)", arg.name)
            if m is not None:
                idx1 = int(m.group(1))
                if 1 <= idx1 <= 64:
                    return self._const_f64(float(1 << (idx1 - 1)))
        return self.emit_expr(builder, st, arg)

    
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
        hoisted = self._compute_loop_hoisted_values(builder, st, n.id)
        self._hoisted_value_stack.append(hoisted)
        try:
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
        finally:
            self._hoisted_value_stack.pop()

    def emit_expr(self, builder: ir.IRBuilder, st: ir.Value, n: Node) -> ir.Value:
        cached = self._lookup_hoisted_value(n)
        if cached is not None:
            return cached

        # literals
        if isinstance(n, Num):
            return self._const_f64(n.value)

        if isinstance(n, StrLit):
            return self._const_f64(float(self._intern_string(n.value)))

        # variable
        if isinstance(n, Var):
            if n.name == "mem":
                return self._const_f64(0.0)

            # common JSFX constants (expand if needed)
            if n.name == "$pi":
                return self._const_f64(math.pi)
            if n.name == "$e":
                return self._const_f64(math.e)
            if n.name.startswith("$x") and len(n.name) > 2:
                try:
                    return self._const_f64(float(int(n.name[2:], 16)))
                except ValueError:
                    pass

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

            guarded = False

            # resolve target pointer (and, for dynamic refs, an in-range mask)
            if isinstance(n.target, Var):
                if n.target.name == "mem":
                    raise ValueError("Cannot assign to mem")
                ptr = self._get_slot_ptr(builder, st, n.target.name)  # works for locals too
                in_range = ir.Constant(self.i1, 1)

            elif isinstance(n.target, Index):
                ptr = self._mem_elem_ptr(builder, st, n.target.base, n.target.index)
                in_range = ir.Constant(self.i1, 1)

            elif isinstance(n.target, Call) and n.target.fn in ("slider", "spl") and len(n.target.args) == 1:
                # Dynamic access: slider(i) / spl(i)
                ptr, in_range = self._dyn_state_array_ptr(builder, st, n.target.fn, n.target.args[0])
                guarded = True

            else:
                raise ValueError("Invalid assignment target")

            if n.op == "=":
                if guarded:
                    with builder.if_then(in_range):
                        builder.store(rhs, ptr)
                else:
                    builder.store(rhs, ptr)
                return rhs

            cur = builder.load(ptr)
            if guarded:
                # JSFX behavior: out-of-range reads as 0
                cur = builder.select(in_range, cur, self._const_f64(0.0))

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

            if guarded:
                # JSFX behavior: out-of-range writes are ignored
                with builder.if_then(in_range):
                    builder.store(out, ptr)
            else:
                builder.store(out, ptr)

            return out

# call
        if isinstance(n, Call):
            fn = n.fn

            # Dynamic slider/spl access (JSFX idiom)
            if fn in ("slider", "spl"):
                if len(n.args) != 1:
                    raise ValueError(f"{fn} expects 1 arg")
                ptr, in_range = self._dyn_state_array_ptr(builder, st, fn, n.args[0])
                val = builder.load(ptr)
                return builder.select(in_range, val, self._const_f64(0.0))


            if fn == "midirecv":
                if len(n.args) != 4:
                    raise ValueError("midirecv expects 4 args")
                out_offset = self._get_midirecv_lvalue_ptr(builder, st, n.args[0])
                out_msg1 = self._get_midirecv_lvalue_ptr(builder, st, n.args[1])
                out_msg2 = self._get_midirecv_lvalue_ptr(builder, st, n.args[2])
                out_msg3 = self._get_midirecv_lvalue_ptr(builder, st, n.args[3])
                ret = builder.call(self.fn_midirecv, [st, out_offset, out_msg1, out_msg2, out_msg3])
                return self._to_f64(builder, ret)

            if fn == "midisend":
                if len(n.args) != 4:
                    raise ValueError("midisend expects 4 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                a3 = self.emit_expr(builder, st, n.args[3])
                ret = builder.call(self.fn_midisend, [st, a0, a1, a2, a3])
                return self._to_f64(builder, ret)

            # ------------------------------------------------------------
            # File I/O (DSP-JSFX runtime)
            #
            # REAPER JSFX provides file_*() APIs. In DSP-JSFX we keep the API
            # surface, but route it to host-provided non-blocking runtime
            # helpers (disk I/O happens off the audio thread).
            # ------------------------------------------------------------
            if fn == "file_open":
                if len(n.args) < 1 or len(n.args) > 2:
                    raise ValueError("file_open expects 1 or 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1]) if len(n.args) == 2 else self._const_f64(0.0)

                rt_name = "jsfx_file_open"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0, a1])

            if fn == "file_close":
                if len(n.args) != 1:
                    raise ValueError("file_close expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                rt_name = "jsfx_file_close"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0])

            if fn == "file_rewind":
                if len(n.args) != 1:
                    raise ValueError("file_rewind expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                rt_name = "jsfx_file_rewind"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0])

            if fn == "file_seek":
                if len(n.args) != 2:
                    raise ValueError("file_seek expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                rt_name = "jsfx_file_seek"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0, a1])

            if fn == "file_avail":
                if len(n.args) != 1:
                    raise ValueError("file_avail expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                rt_name = "jsfx_file_avail"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0])

            if fn == "file_text":
                if len(n.args) != 1:
                    raise ValueError("file_text expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                rt_name = "jsfx_file_text"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0])

            if fn == "file_mem":
                if len(n.args) != 3:
                    raise ValueError("file_mem expects 3 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                rt_name = "jsfx_file_mem"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double, self.double])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0, a1, a2])

            if fn == "file_var":
                if len(n.args) != 2:
                    raise ValueError("file_var expects 2 args")

                h = self.emit_expr(builder, st, n.args[0])

                # Second arg should be an lvalue. If it is not, we still
                # evaluate it for side-effects and pass NULL.
                dst_ptr = None
                if isinstance(n.args[1], Var) and n.args[1].name != "mem":
                    dst_ptr = self._get_slot_ptr(builder, st, n.args[1].name)
                elif isinstance(n.args[1], Index):
                    dst_ptr = self._mem_elem_ptr(builder, st, n.args[1].base, n.args[1].index)
                else:
                    _ = self.emit_expr(builder, st, n.args[1])

                if dst_ptr is None:
                    dst_ptr = ir.Constant(self.double.as_pointer(), None)

                rt_name = "jsfx_file_var"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double.as_pointer()])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl

                return builder.call(fdecl, [st, h, dst_ptr])

            if fn == "file_riff":
                if len(n.args) != 3:
                    raise ValueError("file_riff expects 3 args")

                h = self.emit_expr(builder, st, n.args[0])

                out1 = None
                if isinstance(n.args[1], Var) and n.args[1].name != "mem":
                    out1 = self._get_slot_ptr(builder, st, n.args[1].name)
                elif isinstance(n.args[1], Index):
                    out1 = self._mem_elem_ptr(builder, st, n.args[1].base, n.args[1].index)
                else:
                    _ = self.emit_expr(builder, st, n.args[1])

                out2 = None
                if isinstance(n.args[2], Var) and n.args[2].name != "mem":
                    out2 = self._get_slot_ptr(builder, st, n.args[2].name)
                elif isinstance(n.args[2], Index):
                    out2 = self._mem_elem_ptr(builder, st, n.args[2].base, n.args[2].index)
                else:
                    _ = self.emit_expr(builder, st, n.args[2])

                if out1 is None:
                    out1 = ir.Constant(self.double.as_pointer(), None)
                if out2 is None:
                    out2 = ir.Constant(self.double.as_pointer(), None)

                rt_name = "jsfx_file_riff"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double.as_pointer(), self.double.as_pointer()])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl

                return builder.call(fdecl, [st, h, out1, out2])


            # ------------------------------------------------------------
            # DSP-only compatibility stubs
            #
            # Many JSFX scripts define helpers inside @init/@slider that are
            # only used by @gfx (formatting, drawing, UI glue). Those helpers
            # often call gfx_* and/or string/file functions.
            #
            # This AOT compiler intentionally does not implement @gfx nor full
            # string/file I/O semantics, but we still want such scripts to
            # compile (and run DSP).
            #
            # Strategy: treat these calls as no-ops that evaluate their args
            # (preserving side effects) and return 0.
            # ------------------------------------------------------------
            if fn.startswith("gfx_") or fn in (
                # formatting / strings
                "sprintf", "printf",
                "strcpy", "strcat", "strcmp", "strlen",
                "str_getchar", "str_setchar",
                "str_insert", "str_delete", "str_mid", "strncpy",

                # file I/O helpers that are still NOT implemented in the DSP runtime
                "file_read", "file_write", "file_string",
            ):
                for a in n.args:
                    _ = self.emit_expr(builder, st, a)
                return self._const_f64(0.0)
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

            if fn in ("exp", "log", "tan", "log10"):
                if len(n.args) != 1:
                    raise ValueError(f"{fn} expects 1 arg")
                fdecl = self._declare_math(fn)
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(fdecl, [a0])

            if fn == "rand":
                # JSFX builtin: rand([max])
                #
                # Mirrors EEL2's nseel_int_rand():
                #   x = floor(arg);
                #   if (x < 1.0) x = 1.0;
                #   return genrand_int32() * (1.0 / 0xFFFFFFFF) * x;
                #
                # We intentionally keep the RNG state per DSPJSFX_State instance
                # rather than process-global static storage. The algorithm, seed,
                # index progression, floor/clamp behavior, and output scaling match
                # EEL2; only the cross-instance global coupling is omitted.
                if len(n.args) > 1:
                    raise ValueError("rand expects 0 or 1 args")

                if len(n.args) == 1:
                    floor_fn = self._declare_math("floor")
                    max_v = self.emit_expr(builder, st, n.args[0])
                    max_v = builder.call(floor_fn, [max_v])
                else:
                    max_v = self._const_f64(1.0)

                use_one = builder.fcmp_ordered("<", max_v, self._const_f64(1.0))
                max_v = builder.select(use_one, self._const_f64(1.0), max_v)

                gen_fn = self._ensure_rand_gen32_fn()
                rand_u32 = builder.call(gen_fn, [st])
                rand_f64 = builder.uitofp(rand_u32, self.double)
                scale = self._const_f64(1.0 / 4294967295.0)
                return builder.fmul(builder.fmul(rand_f64, scale), max_v)

            if fn == "freembuf":
                # JSFX builtin: freembuf(top)
                #
                # This is a *hint* to the host memory manager that indices >= top
                # may be freed. REAPER does not guarantee it will actually free or
                # clear memory.
                #
                # Our AOT runtime heap is grow-only (jsfx_ensure_mem). Treat freembuf
                # as a no-op so scripts that call it still compile and run correctly.
                if len(n.args) != 1:
                    raise ValueError("freembuf expects 1 arg")
                _ = self.emit_expr(builder, st, n.args[0])
                return self._const_f64(0.0)

            if fn == "sliderchange":
                # JSFX builtin: sliderchange(slider_mask_or_var)
                #
                # Record internally-driven slider changes so the host bridge can
                # re-run @slider and mirror the new value back to the host param.
                if len(n.args) != 1:
                    raise ValueError("sliderchange expects 1 arg")
                mask = self._emit_slider_mask_arg(builder, st, n.args[0])
                ret = builder.call(self.fn_sliderchange, [st, mask])
                return self._to_f64(builder, ret)

            if fn == "slider_automate":
                # JSFX builtin: slider_automate(slider_mask_or_var, [is_end_gesture])
                #
                # Record host automation gesture hints for internally-driven slider
                # changes so the runtime can begin/end touch gestures correctly.
                if len(n.args) not in (1, 2):
                    raise ValueError("slider_automate expects 1 or 2 args")
                mask = self._emit_slider_mask_arg(builder, st, n.args[0])
                end_touch = self.emit_expr(builder, st, n.args[1]) if len(n.args) == 2 else self._const_f64(0.0)
                ret = builder.call(self.fn_slider_automate, [st, mask, end_touch])
                return self._to_f64(builder, ret)

            if fn == "slider_show":
                # JSFX builtin: slider_show(mask_or_sliderX[, value])
                #
                # REAPER semantics:
                #   - with no value: query visibility and return requested visible bits
                #   - value == -1: toggle requested bits
                #   - value == 0 : hide requested bits
                #   - otherwise  : show requested bits
                #
                # In AOT/JUCE there is no native REAPER slider panel, so we track
                # visibility state internally for scripts that query it, but this has
                # no direct host-UI side effect.
                if len(n.args) not in (1, 2):
                    raise ValueError("slider_show expects 1 or 2 args")

                self._ensure_slider_visibility_initialized(builder, st)

                mask_f = self._emit_slider_mask_arg(builder, st, n.args[0])
                zero_f = self._const_f64(0.0)
                mask_nonneg = builder.select(
                    builder.fcmp_ordered("<", mask_f, zero_f),
                    zero_f,
                    mask_f,
                )
                mask_i64 = builder.fptoui(mask_nonneg, self.i64)

                vis_ptr = self._get_slider_visible_mask_ptr(builder, st)
                vis_i64 = builder.load(vis_ptr)

                if len(n.args) == 2:
                    mode_v = self.emit_expr(builder, st, n.args[1])
                    is_toggle = builder.fcmp_ordered("==", mode_v, self._const_f64(-1.0))
                    is_hide = builder.fcmp_ordered("==", mode_v, self._const_f64(0.0))

                    all_ones = ir.Constant(self.i64, -1)
                    hidden = builder.and_(vis_i64, builder.xor(mask_i64, all_ones))
                    toggled = builder.xor(vis_i64, mask_i64)
                    shown = builder.or_(vis_i64, mask_i64)

                    vis_after_toggle_or_show = builder.select(is_toggle, toggled, shown)
                    vis_i64 = builder.select(is_hide, hidden, vis_after_toggle_or_show)
                    builder.store(vis_i64, vis_ptr)

                requested_visible = builder.and_(vis_i64, mask_i64)
                return builder.uitofp(requested_visible, self.double)

            if fn == "memset":
                # JSFX builtin: memset(dest, value, length)
                # Sets mem[dest .. dest+length-1] = value. Returns dest (double).
                if len(n.args) != 3:
                    raise ValueError("memset expects 3 args")

                dest_v  = self.emit_expr(builder, st, n.args[0])
                value_v = self.emit_expr(builder, st, n.args[1])
                len_v   = self.emit_expr(builder, st, n.args[2])

                # JSFX-style address rounding: trunc(x + 1e-5)
                dest_sum = builder.fadd(dest_v, self._const_f64(1.0e-5))
                dest_i64 = builder.fptosi(dest_sum, self.i64)

                zero_i64 = self._const_i64(0)
                isneg = builder.icmp_signed("<", dest_i64, zero_i64)
                dest_i64 = builder.select(isneg, zero_i64, dest_i64)

                # length = max(0, trunc(length))
                len_i64 = builder.fptosi(len_v, self.i64)
                isnegL = builder.icmp_signed("<", len_i64, zero_i64)
                len_i64 = builder.select(isnegL, zero_i64, len_i64)

                # Ensure memory for dest + len
                end_i64 = builder.add(dest_i64, len_i64)
                memN = self._get_memN(builder, st)
                need_grow = builder.icmp_signed(">", end_i64, memN)
                with builder.if_then(need_grow):
                    builder.call(self.fn_ensure, [st, end_i64])

                # for (i=0; i<len; ++i) mem[dest+i] = value
                fnc = builder.function
                pre_bb = builder.block
                cond_bb = fnc.append_basic_block(f"memset_cond_{n.id}")
                body_bb = fnc.append_basic_block(f"memset_body_{n.id}")
                after_bb = fnc.append_basic_block(f"memset_after_{n.id}")

                builder.branch(cond_bb)

                builder.position_at_end(cond_bb)
                i_phi = builder.phi(self.i64, name=f"memset_i_{n.id}")
                i_phi.add_incoming(zero_i64, pre_bb)
                cond = builder.icmp_signed("<", i_phi, len_i64)
                builder.cbranch(cond, body_bb, after_bb)

                builder.position_at_end(body_bb)
                mem_base = self._get_mem_ptr(builder, st)
                idx_i64 = builder.add(dest_i64, i_phi)
                ptr = builder.gep(mem_base, [idx_i64], inbounds=False)
                builder.store(value_v, ptr)
                i_next = builder.add(i_phi, self._const_i64(1))
                body_end = builder.block
                builder.branch(cond_bb)
                i_phi.add_incoming(i_next, body_end)

                builder.position_at_end(after_bb)
                return dest_v


            if fn in ("fft", "ifft", "fft_permute", "fft_ipermute"):
                # JSFX FFT helpers (used by some scripts for spectral/topology work).
                # We route these to small C++ runtime helpers that operate on st->mem
                # as an interleaved complex buffer: base[2*i]=re, base[2*i+1]=im.
                if len(n.args) != 2:
                    raise ValueError(f"{fn} expects 2 args")

                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])

                rt_name = {
                    "fft": "jsfx_fft",
                    "ifft": "jsfx_ifft",
                    "fft_permute": "jsfx_fft_permute",
                    "fft_ipermute": "jsfx_fft_ipermute",
                }[fn]

                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double])
                    fdecl = ir.Function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl

                return builder.call(fdecl, [st, a0, a1])

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

        count_v = self.emit_expr(builder, st, n.count)
        count_i = builder.fptosi(count_v, self.i64)
        # clamp >=0
        is_neg = builder.icmp_signed("<", count_i, self._const_i64(0))
        n_i = builder.select(is_neg, self._const_i64(0), count_i)

        hoisted = self._compute_loop_hoisted_values(builder, st, n.id)
        self._hoisted_value_stack.append(hoisted)
        try:
            pre_bb = builder.block
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
        finally:
            self._hoisted_value_stack.pop()



def emit_process_block_fn(self, fn_init: ir.Function, fn_slider: ir.Function, fn_block: ir.Function, fn_sample: ir.Function, has_sample_work: bool) -> ir.Function:
    """
    Emits:
        void jsfx_process_block(State* st,
                                const float* const* inputs,
                                float* const* outputs,
                                i32 numChannels,
                                i32 numSamples);

    Semantics:
    - st->samplesblock = (double)numSamples
    - st->currentBlockSize = numSamples
    - st->currentSampleRate = st->srate
    - call jsfx_block(st)
    - for each sample (if numChannels > 0):
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

    # mirror block metadata for runtime helpers
    fld_blocksize = ir.Constant(self.i32, 14)
    blocksize_ptr = builder.gep(st, [z, fld_blocksize], inbounds=True)
    builder.store(nSamp, blocksize_ptr)

    fld_currate = ir.Constant(self.i32, 15)
    currate_ptr = builder.gep(st, [z, fld_currate], inbounds=True)
    fld_srate = ir.Constant(self.i32, 5)
    srate_ptr = builder.gep(st, [z, fld_srate], inbounds=True)
    builder.store(builder.load(srate_ptr), currate_ptr)

    # Call block 
    builder.call(fn_block, [st])

    # If @block changed sliders internally (via sliderchange/slider_automate),
    # run @slider once before @sample so derived state tracks the new values
    # within the same host block.
    fld_sliderchg = ir.Constant(self.i32, 23)
    fld_sliderauto = ir.Constant(self.i32, 24)
    fld_sliderautoend = ir.Constant(self.i32, 25)
    sliderchg_ptr = builder.gep(st, [z, fld_sliderchg], inbounds=True)
    sliderauto_ptr = builder.gep(st, [z, fld_sliderauto], inbounds=True)
    sliderautoend_ptr = builder.gep(st, [z, fld_sliderautoend], inbounds=True)
    sliderchg_mask = builder.load(sliderchg_ptr)
    sliderauto_mask = builder.load(sliderauto_ptr)
    sliderautoend_mask = builder.load(sliderautoend_ptr)
    any_sliderchg = builder.or_(sliderchg_mask, sliderauto_mask)
    any_sliderchg = builder.or_(any_sliderchg, sliderautoend_mask)
    have_sliderchg = builder.icmp_signed("!=", any_sliderchg, ir.Constant(self.i64, 0))
    with builder.if_then(have_sliderchg):
        builder.call(fn_slider, [st])

    # Fast path: if the script has no top-level @sample work, stop after @block.
    # This matters a lot for MIDI-only/controller JSFX where the generic runtime
    # would otherwise pay a per-sample loop cost for no useful work.
    if not has_sample_work:
        builder.ret_void()
        return fn

    # Outer sample loop blocks
    samp_cond = fn.append_basic_block("samp_cond")
    samp_body = fn.append_basic_block("samp_body")
    samp_end  = fn.append_basic_block("samp_end")

    pre_loop_bb = builder.block
    builder.branch(samp_cond)

    # samp_cond
    builder.position_at_end(samp_cond)
    i_phi = builder.phi(self.i32, name="i")
    i_phi.add_incoming(zero_i32, pre_loop_bb)
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
    programs, fn_defs = lower_user_functions(programs, fn_defs)
    programs = hoist_section_invariants(programs, fn_defs)
    loop_hoists = plan_loop_invariant_hoists(programs, fn_defs)
    user_vars = collect_user_vars(programs, fn_defs)

    pin_hints = parse_pin_hints(jsfx_text)
    io_channels = infer_spl_io(programs, fn_defs, pin_hints=pin_hints)
    midi_caps = detect_midi_usage(programs, fn_defs)

    sym = SymTable(user_vars)

    emitter = LLVMModuleEmitter(sym)
    emitter.loop_hoists = loop_hoists
    emitter.declare_user_functions(fn_defs)

    fn_init = emitter.emit_section_fn("jsfx_init", programs["init"])
    fn_slider = emitter.emit_section_fn("jsfx_slider", programs["slider"])
    fn_block = emitter.emit_section_fn("jsfx_block", programs["block"])
    fn_sample = emitter.emit_section_fn("jsfx_sample", programs["sample"])

    emitter.emit_user_functions()

    has_sample_work = len(programs["sample"]) > 0
    emit_process_block_fn(emitter, fn_init, fn_slider, fn_block, fn_sample, has_sample_work)

    plugin_kind = "audio_effect"
    if midi_caps["uses_midi"]:
        if io_channels["inputs"] == 0 and io_channels["outputs"] == 0:
            plugin_kind = "midi_effect"
        elif io_channels["inputs"] == 0 and io_channels["outputs"] > 0 and midi_caps["accepts_midi_input"]:
            plugin_kind = "instrument"
        elif io_channels["inputs"] > 0 or io_channels["outputs"] > 0:
            plugin_kind = "hybrid"
        else:
            plugin_kind = "midi_effect"

    meta = {
        "vars": user_vars,
        "var_cap": emitter.var_cap,
        "sections_present": {k: bool(v) for k, v in programs.items()},
        "io_channels": io_channels,
        "pin_hints": pin_hints,
        "midi": midi_caps,
        "plugin_kind": plugin_kind,
        "has_sample_section": has_sample_work,
    }
    return emitter.module, meta



def _emit_header(meta: Dict[str, Any]) -> str:
    var_cap = int(meta.get("var_cap", 1))
    user_vars: Dict[str, int] = dict(meta.get("vars", {}) or {})

    io_meta: Dict[str, Any] = dict(meta.get("io_channels", {}) or {})
    in_ch = int(io_meta.get("inputs", 0) or 0)
    out_ch = int(io_meta.get("outputs", 0) or 0)
    proc_ch = int(io_meta.get("process", max(in_ch, out_ch)) or max(in_ch, out_ch))
    midi_meta: Dict[str, Any] = dict(meta.get("midi", {}) or {})
    uses_midi = 1 if midi_meta.get("uses_midi") else 0
    accepts_midi_input = 1 if midi_meta.get("accepts_midi_input") else 0
    produces_midi_output = 1 if midi_meta.get("produces_midi_output") else 0
    plugin_kind = str(meta.get("plugin_kind", "audio_effect") or "audio_effect")

    in_ch = max(0, min(64, in_ch))
    out_ch = max(0, min(64, out_ch))
    proc_ch = max(0, min(64, proc_ch))


    def _c_escape(s: str) -> str:
        return s.replace('\\', r'\\').replace('"', r'\\"')

    lines = []
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")

    lines.append("")
    lines.append("/* Inferred minimum I/O channel counts (from splN usage / pin declarations) */")
    lines.append(f"#define DSPJSFX_INPUT_CHANNELS {in_ch}")
    lines.append(f"#define DSPJSFX_OUTPUT_CHANNELS {out_ch}")
    lines.append(f"#define DSPJSFX_PROCESS_CHANNELS {proc_ch}")
    lines.append(f"#define DSPJSFX_USES_MIDI {uses_midi}")
    lines.append(f"#define DSPJSFX_ACCEPTS_MIDI_INPUT {accepts_midi_input}")
    lines.append(f"#define DSPJSFX_PRODUCES_MIDI_OUTPUT {produces_midi_output}")
    lines.append(f"#define DSPJSFX_HAS_SAMPLE_SECTION {1 if meta.get('has_sample_section', False) else 0}")
    lines.append(f'#define DSPJSFX_PLUGIN_KIND "{_c_escape(plugin_kind)}"')
    lines.append("")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")
    lines.append("typedef struct DSPJSFX_MidiEvent {")
    lines.append("    int32_t sampleOffset;")
    lines.append("    int32_t msg1;")
    lines.append("    int32_t msg2;")
    lines.append("    int32_t msg3;")
    lines.append("} DSPJSFX_MidiEvent;")
    lines.append("")
    lines.append("typedef struct DSPJSFX_State {")
    lines.append("    double spl[64];")
    lines.append("    double sliders[64];")
    lines.append(f"    double vars[{var_cap}];")
    lines.append("    double* mem;")
    lines.append("    int64_t memN;")
    lines.append("    double srate;")
    lines.append("    double samplesblock;")
    lines.append("    DSPJSFX_MidiEvent* midiIn;")
    lines.append("    int32_t midiInCount;")
    lines.append("    int32_t midiInReadIndex;")
    lines.append("    int32_t midiInCapacity;")
    lines.append("    DSPJSFX_MidiEvent* midiOut;")
    lines.append("    int32_t midiOutCount;")
    lines.append("    int32_t midiOutCapacity;")
    lines.append("    int32_t currentBlockSize;")
    lines.append("    double currentSampleRate;")
    lines.append("    int32_t pendingNoteCleanup;")
    lines.append("    int32_t midiInDropped;")
    lines.append("    int32_t midiOutDropped;")
    lines.append("    int32_t midiInCountLastBlock;")
    lines.append("    int32_t midiOutCountLastBlock;")
    lines.append("    int32_t midiInPeak;")
    lines.append("    int32_t midiOutPeak;")
    lines.append("    int64_t pendingSliderChangeMask;")
    lines.append("    int64_t pendingSliderAutomateMask;")
    lines.append("    int64_t pendingSliderAutomateEndMask;")
    lines.append("    uint32_t randMT[624];")
    lines.append("    uint32_t randIndex;")
    lines.append("    int64_t sliderVisibleMask;")
    lines.append("    int32_t sliderVisibilityInit;")
    lines.append("} DSPJSFX_State;")
    lines.append("")

    # Export the AOT compiler's user-var symbol mapping so host code can bind EEL/JSFX
    # variables by *name* to the correct vars[] index.
    #
    # NOTE: MSVC does not allow zero-sized arrays, so we emit a 1-element dummy when empty.
    items_by_index = sorted(user_vars.items(), key=lambda kv: int(kv[1]))
    count = len(items_by_index)
    lines.append("/* User vars (name -> vars[] index) */")
    lines.append("typedef struct DSPJSFX_VarDesc { const char* name; int32_t index; } DSPJSFX_VarDesc;")
    lines.append(f"#define DSPJSFX_VARS_COUNT {count}")
    arr_size = count if count > 0 else 1
    lines.append(f"static const DSPJSFX_VarDesc DSPJSFX_VARS[{arr_size}] = {{")
    if count == 0:
        lines.append('    {"", -1},')
    else:
        for name, idx in items_by_index:
            lines.append(f'    {{"{_c_escape(str(name))}", {int(idx)}}},')
    lines.append("};")
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
    lines.append("int jsfx_midirecv(DSPJSFX_State* st, double* offset, double* msg1, double* msg2, double* msg3);")
    lines.append("int jsfx_midisend(DSPJSFX_State* st, double offset, double msg1, double msg2, double msg3);")
    lines.append("int jsfx_sliderchange(DSPJSFX_State* st, double sliderMask);")
    lines.append("int jsfx_slider_automate(DSPJSFX_State* st, double sliderMask, double endTouch);")
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
    triple = target_triple or llvm.get_default_triple()
    default_triple = llvm.get_default_triple()

    if target_triple and target_triple != default_triple:
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
    else:
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