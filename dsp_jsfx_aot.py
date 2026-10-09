#!/usr/bin/env python3
"""
dsp_jsfx_to_llvm.py

DSP-JSFX -> LLVM IR (llvmlite) compiler front-end.

Contract (DSP-JSFX):
- Sections: @init, @slider, @block, @sample (any may be missing)
- DSP-first AOT subset: no @gfx compilation. Basic textual import preprocessing is supported.
- Strings are accepted as lightweight handles with literal/dynamic runtime support for MIDI helpers.
- MIDI builtins midirecv()/midisend() plus buf/str/sysEx variants are supported in @block and @sample.
- Type: everything is double.
- Variables: spl0..spl63, slider1..slider256, user vars (persistent), builtins:
    - mem  (numeric base pointer index = 0.0)
    - srate (read/write state field)
    - samplesblock (read/write state field; host should set before @block)
    - MIDI calls: midirecv()/midisend() short-message forms, plus midirecv_buf()/midisend_buf(), midirecv_str()/midisend_str(), and midisyx().
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
  comparisons, short-circuit && ||, ternary ?:, assignments (=, +=, -=, *=, /=, %=, ^=, |=, &=, ~=),
  parentheses, sequence blocks: ( a; b; c; ) returning last expr value.
- loop(count, body) expression: repeats up to 1048576 times and returns 1.

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

import sys
from scripts import jsfx_tasks_compiler as tasks_compiler
from scripts import jsfx_faust_compiler as faust_compiler
import argparse
import contextlib
from collections import Counter
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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
    "===","!==",
    "==","!=","<=",">=",
    "+=","-=","*=","/=","%=","^=","|=","&=","~=",
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

    def _fmt_err(self, message: str) -> str:
        return f"{message} at {self.line}:{self.col}"

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

            # EEL's bit mask and historical character spellings are numeric.
            mask = re.match(r"\$~([0-9]+)", self.src[self.i:])
            if mask:
                self._adv(len(mask.group(0)))
                return Tok("num", str((1 << min(53, int(mask.group(1)))) - 1), sp)
            if c == "$" and self._peek(1) == "'" and self._peek(3) == "'":
                value = ord(self._peek(2))
                self._adv(4)
                return Tok("num", str(value), sp)
            hexadecimal = re.match(r"(?:0[xX]|\$[xX])([0-9a-fA-F]+)", self.src[self.i:])
            if hexadecimal:
                self._adv(len(hexadecimal.group(0)))
                return Tok("num", str(int(hexadecimal.group(1), 16)), sp)

            # multi-char operators. Match longest-first so JSFX exact
            # comparisons (=== / !==) are not split into == + = / != + =.
            matched_op = next((op for op in _MULTI_OPS if self.src.startswith(op, self.i)), None)
            if matched_op is not None:
                self._adv(len(matched_op))
                return Tok("op", matched_op, sp)

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

                # WDL/EEL2 resolves identifiers case-insensitively (its symbol
                # and function tables use stricmp/strnicmp). Canonicalise at
                # the lexer boundary so variables, user functions, builtins,
                # local()/instance() names, sliderN/splN and dotted namespace
                # components all share exactly the same semantics downstream.
                # String *contents* are tokenised separately and remain case
                # sensitive.
                txt = txt.lower()
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
                        elif esc in ("x", "X"):
                            hex1 = self._peek()
                            hex2 = self._peek(1)
                            if (re.fullmatch(r"[0-9A-Fa-f]", hex1) is not None and
                                    re.fullmatch(r"[0-9A-Fa-f]", hex2) is not None):
                                out.append(chr(int(hex1 + hex2, 16)))
                                self._adv(2)
                            else:
                                out.append(esc)
                        elif esc == "0":
                            out.append("\0")
                        else:
                            # Unknown escapes: keep the escaped character verbatim.
                            out.append(esc)
                        continue

                    out.append(ch)
                    self._adv()

                text = "".join(out)
                if quote == "'":
                    data = text.encode("utf-8")
                    if len(data) > 4:
                        raise SyntaxError(self._fmt_err("Packed character literal exceeds four bytes"))
                    return Tok("num", str(int.from_bytes(data, "big")), sp)
                return Tok("str", text, sp)


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
    cell_args: List[Var] = field(default_factory=list)

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
    cell_params: List[str] = field(default_factory=list)



# -----------------------------
# Pratt parser
# -----------------------------

# Higher number = tighter binding.
# Match WDL/eel2/eel2.y. EEL binds an assignment's *left* side tightly,
# but its right side consumes a complete conditional expression. Thus
# a+b+=2 means a+(b+=2), while x=a+b still means x=(a+b).
_PRECEDENCE: Dict[str, int] = {
    "=": 11, "+=": 11, "-=": 11, "*=": 11, "/=": 11, "%=": 11,
    "^=": 11, "|=": 11, "&=": 11, "~=": 11,
    "?": 1, "||": 2, "&&": 2,
    "==": 3, "!=": 3, "===": 3, "!==": 3,
    "<": 3, "<=": 3, ">": 3, ">=": 3,
    "|": 4, "&": 4, "~": 4,
    "+": 5, "-": 6, "*": 7, "/": 8,
    "%": 9, "<<": 9, ">>": 9, "^": 10,
}

_TERNARY_PREC = 1
_RIGHT_ASSOC = {"=", "+=", "-=", "*=", "/=", "%=", "^=", "|=", "&=", "~="}



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
        """Parse WDL/EEL2's two special while() forms.

        EEL2 does not treat while as a conventional language keyword in its
        grammar.  It is a special one-argument function whose argument may be
        a semicolon sequence:

            while(
              i += 1;
              table[i] != value && i < 30
            );

        The complete argument is evaluated once per iteration and its final
        value decides whether another iteration runs.  EEL2 also accepts the
        familiar postfix-body spelling:

            while(condition) (
              body;
            );

        WDL rewrites the latter internally to a one-argument while expression.
        Keep our existing While(cond, body) IR, but parse the first form as a
        condition expression with a no-op body.
        """
        kw = self._eat("kw", "while")
        # EEL whitespace may separate the function name and opening group.
        # Consume only newlines: a semicolon still terminates the statement.
        self._skip_eol()
        self._eat("punc", "(")
        self._skip_seps()

        if self.cur.kind == "punc" and self.cur.text == ")":
            raise SyntaxError(self._fmt_err("while() requires an expression"))

        # Unlike an ordinary condition, the single argument to EEL2 while()
        # may contain a semicolon-separated expression sequence.  The value of
        # the final expression controls repetition, while all preceding
        # expressions execute on every test.
        items: List[Node] = [self.parse_stmt_or_expr_for_seq()]
        while True:
            if self.cur.kind == "punc" and self.cur.text == ")":
                self._adv()
                break
            if self.cur.kind not in ("eol", "semi"):
                raise SyntaxError(self._fmt_err("Expected separator or ')' in while() expression"))
            self._skip_seps()
            if self.cur.kind == "punc" and self.cur.text == ")":
                self._adv()
                break
            items.append(self.parse_stmt_or_expr_for_seq())

        cond: Node = items[0] if len(items) == 1 else Seq(self._new_id(), kw.span, items)

        # WDL/EEL2's second form is IDENTIFIER '(' expression ')'
        # '(' expression ')'.  Newlines are whitespace to EEL, so permit them
        # between the two groups, but a semicolon terminates the one-argument
        # form and must not be consumed here.
        while self.cur.kind == "eol":
            self._adv()

        if self.cur.kind == "punc" and self.cur.text == "(":
            body = self.parse_primary()
            return While(self._new_id(), kw.span, cond, body)

        # Direct while(expr); form: evaluating cond performs the loop work; a
        # zero-valued body simply takes the back-edge when cond was truthy.
        body = Num(self._new_id(), kw.span, 0.0)
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

                    if self.cur.kind == "punc" and self.cur.text == ",":
                        self._adv()
                        continue
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

        while self.cur.kind == "ident" and self.cur.text in ("local", "instance", "global", "globals"):
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

    def _is_line_continuation_op(self, tok: Tok, min_prec: int) -> bool:
        """True when a newline followed by tok must continue the current expr.

        JSFX/EEL2 permits expressions such as:

            wrapped
                || something
                || something_else

        After an operand, newline-leading '+' and '-' continue the expression,
        just as they do on the same line in WDL/EEL2. A semicolon terminates it;
        a following sign is then parsed as a unary prefix by parse_prefix().
        Preserve this parser's newline-separated statements when the next token
        is not an infix/ternary operator. '!' is unary-only, so it cannot join.
        """
        if tok.kind != "op":
            return False
        if tok.text == "?":
            return _TERNARY_PREC >= min_prec
        if tok.text == ":":
            return False
        if tok.text == "!":
            return False
        prec = _PRECEDENCE.get(tok.text)
        return prec is not None and prec >= min_prec

    def _skip_expr_continuation_eol(self, min_prec: int) -> None:
        # Skip blank lines while looking for an explicit continuation operator,
        # but do not swallow newlines before ordinary statement starts.
        while self.cur.kind == "eol" and (
            self.nxt.kind == "eol" or self._is_line_continuation_op(self.nxt, min_prec)
        ):
            self._adv()

    def parse_expr(self, min_prec: int) -> Node:
        lhs = self.parse_prefix()
        while True:
            self._skip_expr_continuation_eol(min_prec)

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
            rhs = self.parse_expr(0 if assoc_right else prec + 1)

            if op in _RIGHT_ASSOC:
                if not self._is_assign_target(lhs):
                    raise SyntaxError(self._fmt_err("Assignment target must be a variable, index, or slider()/spl() reference"))
                if isinstance(lhs, Var) and lhs.name.startswith("#") and op in ("=", "+="):
                    lhs = Call(self._new_id(), lhs.span, "strcpy" if op == "=" else "strcat", [lhs, rhs])
                else:
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

            then = (Num(self._new_id(), q.span, 0.0) if self.cur.kind == "op" and self.cur.text == ":"
                    else self.parse_expr(0))
            # A semicolon terminates an implicit-else conditional. Consuming
            # it here loses a statement separator inside while(expr; ...).
            while self.cur.kind == "eol" and (self.nxt.kind == "eol" or
                    self.nxt.kind == "op" and self.nxt.text == ":"):
                self._adv()

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
            a = self.parse_expr(11)
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

                if fn in tasks_compiler.DEFERRED:
                    node = tasks_compiler.parse(sys.modules[__name__], self, fn, sp)
                    continue

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
                if self.cur.kind == "punc" and self.cur.text == "]":
                    idx = Num(self._new_id(), sp, 0.0)
                else:
                    idx = self.parse_expr(0)
                    self._skip_seps()
                self._eat("punc", "]")
                node = Index(self._new_id(), sp, node, idx)
                continue

            break
        return node

    def parse_primary(self) -> Node:
        if self.cur.kind == "kw" and self.cur.text == "while":
            return self.parse_while()
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



# Both DSP and GFX consume the same section-aware source resolver.
def preprocess_jsfx_imports(jsfx_text: str, source_path: Optional[Path]) -> str:
    if source_path is None:
        return jsfx_text
    from scripts.jsfx_source import resolve_source
    return resolve_source(source_path, text=jsfx_text).text

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
            # Section names are identifiers too; REAPER/WDL accepts their
            # spelling case-insensitively. Keep one canonical namespace.
            current = m.group(1).lower()
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

MAX_JSFX_SLIDERS = 256
SLIDER_MASK_WORDS = MAX_JSFX_SLIDERS // 64

BUILTIN_NAMES = {"mem", "gmem", "srate", "samplesblock", "midi_bus", "ext_midi_bus"}

@dataclass(frozen=True)
class SymRef:
    kind: str    # spl, slider, var, builtin
    index: int   # spl/slider/var index; builtin field id

class SymTable:
    def __init__(self, user_vars: Dict[str, int]):
        self.vars = dict(user_vars)  # stable mapping
        self.slider_aliases: Dict[str, int] = {}

    def resolve(self, name: str) -> SymRef:
        if name in self.slider_aliases:
            index = self.slider_aliases[name]
            if not 0 <= index < MAX_JSFX_SLIDERS:
                raise ValueError(f"Invalid slider alias index: {name}")
            return SymRef("slider", index)
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
                if 0 <= idx < MAX_JSFX_SLIDERS:
                    return SymRef("slider", idx)
                raise ValueError(f"Invalid slider index: {name}")
            # NOT slider<number> => normal var like "sliderGainThing"


        if name == "mem" and name not in self.vars:
            # numeric base index of heap is always 0.0
            return SymRef("builtin", 0)

        if name == "gmem":
            return SymRef("builtin", 5)

        if name == "srate":
            return SymRef("builtin", 1)

        if name == "samplesblock":
            return SymRef("builtin", 2)

        if name == "midi_bus":
            return SymRef("builtin", 3)

        if name == "ext_midi_bus":
            return SymRef("builtin", 4)

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

            # skip only real slider registers slider1..slider256
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
            for a in n.cell_args: rec(a, locals)
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
        localset = set(f.params) | set(f.locals) | set(f.cell_params)
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

_OPTIONS_LINE_RE = re.compile(r"^\s*options\s*:\s*(.*)$", re.IGNORECASE)

JSFX_DEFAULT_MEMTOP_SLOTS = 8 * 1024 * 1024

GFX_VAR_FLAG_TO_GFX = 1
GFX_VAR_FLAG_FROM_GFX = 2

_RUNTIME_AUDIO_OWNED_SECTIONS = ("slider", "block", "sample", "serialize")


def parse_jsfx_options(jsfx_text: str) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    for raw_line in jsfx_text.splitlines():
        m = _OPTIONS_LINE_RE.match(raw_line)
        if not m:
            continue
        payload = m.group(1).strip()
        if not payload:
            continue
        for tok in re.split(r"[\s,]+", payload):
            if not tok or "=" not in tok:
                continue
            key, value = tok.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key:
                opts[key] = value
    return opts


def resolve_jsfx_memtop_slots(options: Dict[str, str]) -> int:
    raw = str(options.get("maxmem", "") or "").strip()
    if not raw:
        return JSFX_DEFAULT_MEMTOP_SLOTS

    try:
        slots = int(float(raw))
    except Exception:
        return JSFX_DEFAULT_MEMTOP_SLOTS

    return slots if slots > 0 else JSFX_DEFAULT_MEMTOP_SLOTS


@dataclass
class _VarUsageSummary:
    reads: Set[str]
    writes: Set[str]
    reads_mem: bool = False
    writes_mem: bool = False

    def merge(self, other: '_VarUsageSummary') -> '_VarUsageSummary':
        self.reads.update(other.reads)
        self.writes.update(other.writes)
        self.reads_mem = self.reads_mem or other.reads_mem
        self.writes_mem = self.writes_mem or other.writes_mem
        return self


def _is_trackable_user_var_name(name: str, locals_: Set[str]) -> bool:
    if name in locals_:
        return False
    if name in BUILTIN_NAMES:
        return False
    if name.startswith("spl") and name[3:].isdigit():
        return False
    if name.startswith("slider") and name[6:].isdigit():
        return False
    if name.startswith("$"):
        return False
    return True


def _prepare_var_sync_analysis_units(jsfx_text: str) -> Tuple[Dict[str, List[Node]], Dict[str, FunctionDef]]:
    sections = extract_sections(jsfx_text)
    analysis_programs: Dict[str, List[Node]] = {}

    for sec in ("init", "slider", "block", "sample", "serialize", "gfx"):
        if sec in sections:
            code, start_line = sections[sec]
            parser = Parser(code, base_line=start_line)
            analysis_programs[sec] = parser.parse_program()
        else:
            analysis_programs[sec] = []

    analysis_fn_defs, analysis_programs = extract_function_defs(analysis_programs)
    analysis_programs, analysis_fn_defs = lower_user_functions(analysis_programs, analysis_fn_defs)
    return analysis_programs, analysis_fn_defs


def analyze_gfx_var_sync(jsfx_text: str, user_vars: Dict[str, int]) -> Dict[str, Any]:
    programs, fn_defs = _prepare_var_sync_analysis_units(jsfx_text)
    fn_cache: Dict[str, _VarUsageSummary] = {}
    fn_in_progress: Set[str] = set()

    def walk_node(n: Node, locals_: Set[str]) -> _VarUsageSummary:
        out = _VarUsageSummary(set(), set())

        def walk_read(node: Node) -> None:
            nonlocal out
            if isinstance(node, Var):
                if _is_trackable_user_var_name(node.name, locals_):
                    out.reads.add(node.name)
                return
            if isinstance(node, (Num, StrLit)):
                return
            if isinstance(node, Index):
                walk_read(node.base)
                walk_read(node.index)
                if isinstance(node.base, Var) and node.base.name == "mem":
                    out.reads_mem = True
                return
            if isinstance(node, Unary):
                walk_read(node.a)
                return
            if isinstance(node, Binary):
                walk_read(node.l)
                walk_read(node.r)
                return
            if isinstance(node, Assign):
                walk_assign(node)
                return
            if isinstance(node, Call):
                if node.fn in tasks_compiler.DEFERRED:
                    for arg in node.args[:-1]:
                        walk_read(arg)
                    private = walk_node(node.args[-1], locals_)
                    out.reads.update(private.reads)
                    return
                for index, a in enumerate(node.args):
                    if node.fn == "gfx_measurestr" and index in (1, 2):
                        # These are output lvalues, including helper locals.
                        if isinstance(a, Var):
                            if _is_trackable_user_var_name(a.name, locals_):
                                out.writes.add(a.name)
                        elif isinstance(a, Index):
                            walk_read(a.base)
                            walk_read(a.index)
                            out.writes_mem = True
                    else:
                        walk_read(a)
                if node.fn in fn_defs:
                    out.merge(summarize_function(node.fn))
                return
            if isinstance(node, Loop):
                walk_read(node.count)
                walk_read(node.body)
                return
            if isinstance(node, Ternary):
                walk_read(node.cond)
                walk_read(node.then)
                walk_read(node.els)
                return
            if isinstance(node, Seq):
                for it in node.items:
                    walk_read(it)
                return
            if isinstance(node, If):
                walk_read(node.cond)
                walk_read(node.then)
                if node.els:
                    walk_read(node.els)
                return
            if isinstance(node, While):
                walk_read(node.cond)
                walk_read(node.body)
                return
            raise TypeError(type(node))

        def walk_assign(node: Assign) -> None:
            nonlocal out
            walk_read(node.value)

            if isinstance(node.target, Var):
                if node.op != "=":
                    walk_read(node.target)
                if _is_trackable_user_var_name(node.target.name, locals_):
                    out.writes.add(node.target.name)
                return

            if isinstance(node.target, Index):
                walk_read(node.target.base)
                walk_read(node.target.index)
                if isinstance(node.target.base, Var) and node.target.base.name == "mem":
                    out.writes_mem = True
                    if node.op != "=":
                        out.reads_mem = True
                return

            walk_read(node.target)

        walk_read(n)
        return out

    def summarize_function(name: str) -> _VarUsageSummary:
        cached = fn_cache.get(name)
        if cached is not None:
            return cached
        if name in fn_in_progress:
            return _VarUsageSummary(set(), set())
        fn_in_progress.add(name)
        f = fn_defs[name]
        locals_ = set(f.params) | set(f.locals) | set(f.instances)
        summary = walk_node(f.body, locals_)
        fn_in_progress.remove(name)
        fn_cache[name] = summary
        return summary

    def summarize_section(section: str) -> _VarUsageSummary:
        summary = _VarUsageSummary(set(), set())
        for node in programs.get(section, []):
            summary.merge(walk_node(node, set()))
        return summary

    gfx_usage = summarize_section("gfx")
    audio_usage = _VarUsageSummary(set(), set())
    for sec in _RUNTIME_AUDIO_OWNED_SECTIONS:
        audio_usage.merge(summarize_section(sec))

    flags_by_name: Dict[str, int] = {}
    for name in user_vars.keys():
        flags = 0
        if name in audio_usage.writes and name in gfx_usage.reads:
            flags |= GFX_VAR_FLAG_TO_GFX
        if name in gfx_usage.writes and name in audio_usage.reads:
            flags |= GFX_VAR_FLAG_FROM_GFX
        flags_by_name[name] = flags

    return {
        "flags": flags_by_name,
        "gfx_reads": set(gfx_usage.reads),
        "gfx_writes": set(gfx_usage.writes),
        "audio_reads": set(audio_usage.reads),
        "audio_writes": set(audio_usage.writes),
        "mem_shared": bool(gfx_usage.reads_mem or gfx_usage.writes_mem) and bool(audio_usage.reads_mem or audio_usage.writes_mem),
    }


MIDI_RECV_FUNCTIONS: Set[str] = {"midirecv", "midirecv_buf", "midirecv_str"}
MIDI_SEND_FUNCTIONS: Set[str] = {"midisend", "midisend_buf", "midisend_str", "midisyx"}
GMEM_SETUP_FUNCTIONS: Set[str] = {"gmem_attach", "gmem_attach_size"}
GMEM_BULK_FUNCTIONS: Set[str] = {"gmem_get", "gmem_put", "gmem_fill", "gmem_zero", "gmem_copy"}
GMEM_QUERY_FUNCTIONS: Set[str] = {"gmem_size", "gmem_seq", "gmem_page"}
COMM_SETUP_FUNCTIONS: Set[str] = {"comm_join", "msg_subscribe", "msg_unsubscribe", "msg_advertise", "instance_set_name"}
COMM_BLOCK_FUNCTIONS: Set[str] = {
    "msg_send", "msg_sendto", "msg_recv",
    "msg_send_buf", "msg_sendto_buf", "msg_recv_buf",
    "msg_avail", "msg_kind", "msg_length", "msg_dropped", "msg_clear",
    "msg_peer_count", "msg_peer_id", "msg_peer_name", "msg_peer_uid", "msg_peer_caps", "msg_peer_alive",
}
COMM_MISC_FUNCTIONS: Set[str] = {"instance_id", "instance_uid", "instance_get_name"}
HOST_TRACK_FUNCTIONS: Set[str] = {
    "track_name",
    "track_name_available",
    "track_name_seq",
    "host_track_name",
    "host_track_name_available",
    "host_track_name_seq",
}

SAMPLE_POOL_SETUP_FUNCTIONS: Set[str] = {
    "sample_pool_adopt", "sample_pool_from_slot", "sample_pool_set_mode", "sample_pool_set_deferred", "sample_pool_set_budget_mb", "sample_pool_commit",
}
SAMPLE_POOL_QUERY_FUNCTIONS: Set[str] = {
    "sample_pool_state", "sample_pool_selected", "sample_pool_loaded", "sample_pool_failed",
    "sample_pool_ram_mb", "sample_pool_generation", "sample_get", "sample_len",
    "sample_channels", "sample_srate", "sample_peak", "sample_rms", "sample_preview_bins",
}
SAMPLE_POOL_READ_FUNCTIONS: Set[str] = {
    "sample_read", "sample_read_interp", "sample_read2", "sample_read2_interp",
    "sample_preview_read", "sample_name",
}
SAMPLE_POOL_EXPORT_FUNCTIONS: Set[str] = {"sample_export_mem", "sample_export_mem2"}
SAMPLE_POOL_FUNCTIONS: Set[str] = SAMPLE_POOL_SETUP_FUNCTIONS | SAMPLE_POOL_QUERY_FUNCTIONS | SAMPLE_POOL_READ_FUNCTIONS | SAMPLE_POOL_EXPORT_FUNCTIONS
LEGACY_FILE_FUNCTIONS: Set[str] = {
    "file_open", "file_open_multi", "file_close", "file_rewind", "file_seek", "file_avail",
    "file_text", "file_riff", "file_var", "file_mem", "file_multi_count", "file_multi_select",
}
COMM_IMPURE_FUNCTIONS: Set[str] = COMM_SETUP_FUNCTIONS | COMM_BLOCK_FUNCTIONS | COMM_MISC_FUNCTIONS | HOST_TRACK_FUNCTIONS | GMEM_SETUP_FUNCTIONS | GMEM_BULK_FUNCTIONS | GMEM_QUERY_FUNCTIONS | SAMPLE_POOL_SETUP_FUNCTIONS | SAMPLE_POOL_EXPORT_FUNCTIONS
COMM_SEND_FUNCTIONS: Set[str] = {"msg_send", "msg_sendto", "msg_send_buf", "msg_sendto_buf"}
COMM_RECV_FUNCTIONS: Set[str] = {"msg_recv", "msg_recv_buf"}
COMM_DISCOVERY_FUNCTIONS: Set[str] = {"msg_peer_count", "msg_peer_id", "msg_peer_name", "msg_peer_uid", "msg_peer_caps", "msg_peer_alive"}



def detect_comm_usage(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef]) -> Dict[str, Any]:
    uses_msg = False
    uses_gmem = False
    uses_msg_buffers = False
    channels_static: Set[str] = set()
    gmem_names_static: Set[str] = set()

    def note_literal_arg(target: Set[str], args: List[Node], idx: int) -> None:
        if 0 <= idx < len(args) and isinstance(args[idx], StrLit):
            target.add(args[idx].value)

    def rec(n: Node) -> None:
        nonlocal uses_msg, uses_gmem, uses_msg_buffers
        if isinstance(n, (Num, StrLit, Var)):
            return
        if isinstance(n, Index):
            if isinstance(n.base, Var) and n.base.name == "gmem":
                uses_gmem = True
            rec(n.base); rec(n.index); return
        if isinstance(n, Unary):
            rec(n.a); return
        if isinstance(n, Binary):
            rec(n.l); rec(n.r); return
        if isinstance(n, Assign):
            if isinstance(n.target, Index) and isinstance(n.target.base, Var) and n.target.base.name == "gmem":
                uses_gmem = True
            rec(n.target); rec(n.value); return
        if isinstance(n, Call):
            fn = n.fn



            if fn in COMM_SEND_FUNCTIONS or fn in COMM_RECV_FUNCTIONS or fn in COMM_DISCOVERY_FUNCTIONS or fn in {"msg_subscribe", "msg_unsubscribe", "msg_advertise", "msg_avail", "msg_kind", "msg_length", "msg_dropped", "msg_clear", "instance_id", "instance_uid", "instance_get_name", "instance_set_name", "comm_join"} or fn in HOST_TRACK_FUNCTIONS:
                uses_msg = True
            if fn in {"msg_send_buf", "msg_sendto_buf", "msg_recv_buf"}:
                uses_msg_buffers = True
            if fn in GMEM_SETUP_FUNCTIONS | GMEM_BULK_FUNCTIONS | GMEM_QUERY_FUNCTIONS:
                uses_gmem = True
            if fn in {"msg_subscribe", "msg_unsubscribe", "msg_advertise", "msg_send", "msg_send_buf", "msg_recv", "msg_recv_buf", "msg_avail", "msg_kind", "msg_dropped", "msg_clear", "msg_peer_count", "msg_peer_id"}:
                note_literal_arg(channels_static, n.args, 0)
            if fn in {"msg_sendto", "msg_sendto_buf"}:
                note_literal_arg(channels_static, n.args, 1)
            if fn in {"gmem_attach", "gmem_attach_size"}:
                note_literal_arg(gmem_names_static, n.args, 0)
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
        "uses_comm": uses_msg or uses_gmem,
        "uses_msg": uses_msg,
        "uses_gmem": uses_gmem,
        "uses_msg_buffers": uses_msg_buffers,
        "channels_static": sorted(channels_static),
        "gmem_names_static": sorted(gmem_names_static),
    }


def detect_sample_pool_usage(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef]) -> Dict[str, Any]:
    uses_sample_pool = False
    uses_export_mem = False
    uses_raw_sample_read = False
    uses_legacy_file_io = False

    def rec(n: Node) -> None:
        nonlocal uses_sample_pool, uses_export_mem, uses_raw_sample_read, uses_legacy_file_io
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
            fn = n.fn
            if fn in SAMPLE_POOL_FUNCTIONS:
                uses_sample_pool = True
            if fn in SAMPLE_POOL_EXPORT_FUNCTIONS:
                uses_export_mem = True
            if fn in {"sample_read", "sample_read_interp", "sample_read2", "sample_read2_interp"}:
                uses_raw_sample_read = True
            if fn in LEGACY_FILE_FUNCTIONS:
                uses_legacy_file_io = True
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

    for prog in programs.values():
        for st in prog:
            rec(st)
    for f in fn_defs.values():
        rec(f.body)

    return {
        "uses_sample_pool": uses_sample_pool,
        "uses_raw_sample_read": uses_raw_sample_read,
        "uses_export_mem": uses_export_mem,
        "uses_legacy_file_io": uses_legacy_file_io,
    }


def validate_builtin_sections(programs: Dict[str, List[Node]]) -> None:
    block_only = {
        "msg_send", "msg_sendto", "msg_recv",
        "msg_send_buf", "msg_sendto_buf", "msg_recv_buf",
        "msg_avail", "msg_kind", "msg_length", "msg_dropped", "msg_clear",
        "msg_peer_count", "msg_peer_id", "msg_peer_name", "msg_peer_uid", "msg_peer_caps", "msg_peer_alive",
        "gmem_get", "gmem_put", "gmem_fill", "gmem_zero", "gmem_copy",
        "sample_export_mem", "sample_export_mem2",
    }
    # Bus/string-slider setup calls are intentionally valid from @slider so a
    # per-instance text slider can rebind message/gmem endpoints without forcing
    # users to reopen the plugin. Operational message traffic remains @block-only.
    init_slider_block_setup = {"comm_join", "msg_subscribe", "msg_unsubscribe", "msg_advertise", "instance_set_name", "instance_get_name", "instance_uid", "gmem_attach", "gmem_attach_size"} | HOST_TRACK_FUNCTIONS | SAMPLE_POOL_SETUP_FUNCTIONS
    init_slider_block = {"instance_id"}
    sample_pool_runtime = SAMPLE_POOL_QUERY_FUNCTIONS | SAMPLE_POOL_READ_FUNCTIONS

    def fail(node: Node, message: str) -> None:
        raise SyntaxError(f"{message} at {node.span.line}:{node.span.col}")

    def rec(section: str, n: Node) -> None:
        if isinstance(n, (Num, StrLit, Var)):
            return
        if isinstance(n, Index):
            rec(section, n.base); rec(section, n.index); return
        if isinstance(n, Unary):
            rec(section, n.a); return
        if isinstance(n, Binary):
            rec(section, n.l); rec(section, n.r); return
        if isinstance(n, Assign):
            rec(section, n.target); rec(section, n.value); return
        if isinstance(n, Call):
            fn = n.fn
            if fn in block_only and section != "block":
                fail(n, f"{fn}() is only valid in @block")
            if fn in init_slider_block_setup and section not in ("init", "slider", "block"):
                fail(n, f"{fn}() is only valid in @init, @slider, or @block")
            if fn in init_slider_block and section not in ("init", "slider", "block"):
                fail(n, f"{fn}() is only valid in @init, @slider, or @block")
            if fn in sample_pool_runtime and section not in ("init", "slider", "block", "sample"):
                fail(n, f"{fn}() is only valid in @init, @slider, @block, or @sample")
            for a in n.args:
                rec(section, a)
            return
        if isinstance(n, Loop):
            rec(section, n.count); rec(section, n.body); return
        if isinstance(n, Ternary):
            rec(section, n.cond); rec(section, n.then); rec(section, n.els); return
        if isinstance(n, Seq):
            for it in n.items: rec(section, it)
            return
        if isinstance(n, If):
            rec(section, n.cond); rec(section, n.then)
            if n.els is not None:
                rec(section, n.els)
            return
        if isinstance(n, While):
            rec(section, n.cond); rec(section, n.body); return
        raise TypeError(type(n))

    for section, nodes in programs.items():
        for node in nodes:
            rec(section, node)


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
            if n.fn in MIDI_RECV_FUNCTIONS:
                uses_midirecv = True
            elif n.fn in MIDI_SEND_FUNCTIONS:
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

def infer_spl_io(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef], pin_hints: Optional[Dict[str, Optional[int]]] = None, *, include_fallback: bool = True) -> Dict[str, int]:
    """Infer minimum input/output channel counts from splN usage.

    Heuristic:
      - Any read of splN implies at least (N+1) input channels.
      - Any write to splN (assignment target) implies at least (N+1) output channels.

    Returned counts are clamped to 0..64. Faust pipelines disable the
    stereo/mirrored fallback because their stages provide actual port counts.

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
    if include_fallback and declared_in is None and declared_out is None and in_ch == 0 and out_ch == 0:
        in_ch = 2
        out_ch = 2

    # Mirror unspecified side to the specified side only when there was no explicit
    # declaration forcing silence on that edge.
    if include_fallback and declared_in is None and in_ch == 0 and out_ch > 0:
        in_ch = out_ch
    if include_fallback and declared_out is None and out_ch == 0 and in_ch > 0:
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
    counts = Counter(n.name for prog in programs.values() for n in prog if isinstance(n, FunctionDef))
    duplicated = {name for name, count in counts.items() if count > 1}
    latest: Dict[Tuple[str, int], str] = {}
    versions: Counter = Counter()

    def bind_name(name: str, argc: Optional[int] = None) -> str:
        if argc is None:
            return name
        if (name, argc) in latest:
            return latest[name, argc]
        prefix, dot, base = name.rpartition(".")
        return prefix + dot + latest[base, argc] if (base, argc) in latest else name

    def bind(node: Node) -> Node:
        changes = {}
        for field, value in vars(node).items():
            if isinstance(value, Node):
                changes[field] = bind(value)
            elif isinstance(value, list):
                changes[field] = [bind(v) if isinstance(v, Node) else v for v in value]
        if isinstance(node, Call):
            # EEL spells a missing single argument as an implicit zero.
            argc = max(1, len(node.args))
            changes["fn"] = bind_name(node.fn, argc)
        return replace(node, **changes) if changes else node

    for sec, prog in programs.items():
        new_prog: List[Node] = []
        for n in prog:
            if isinstance(n, FunctionDef):
                # WDL binds calls while compiling each definition. A wrapper
                # can deliberately reuse its own name and call the previous
                # definition on a child receiver (Filther's spline helpers).
                name = n.name
                body = bind(n.body)
                instances = [bind_name(v) for v in n.instances]
                if name in duplicated:
                    version = f"__eel_def_{versions[name]}_{name}"
                    versions[name] += 1
                    latest[name, max(1, len(n.params))] = version
                else:
                    version = name
                fns[version] = replace(n, name=version, body=body, instances=instances)
            else:
                new_prog.append(bind(n))
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


def lower_user_functions(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef], *, compact_receiver_math: bool = False) -> Tuple[Dict[str, List[Node]], Dict[str, FunctionDef]]:
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

    # Large synths call the same coefficient math for thousands of objects.
    # Share only scalar arithmetic leaf helpers, with receiver cells passed by
    # address. Persistent locals stay in their original per-section slots.
    # Exclude strings, heap writes, host callbacks and nested user calls;
    # Keep small helpers specialized so their existing inlining is preserved.
    receiver_cell_fields: Dict[str, List[str]] = {}
    arithmetic_calls = {"min", "max", "abs", "floor", "ceil", "sin", "cos", "tan",
                        "atan", "atan2", "exp", "log", "log10", "sqrt", "pow",
                        "sqr", "sign", "asin", "acos"}
    def scalar_leaf(node: Node) -> bool:
        if isinstance(node, Num): return True
        # A receiver cell pointer represents one scalar, not its namespace.
        # Rewriting instance(band) plus band.left to cell.left would otherwise
        # create an unrelated global variable shared by every receiver.
        if isinstance(node, Var): return "." not in node.name and not node.name.startswith("#") and node.name != "this"
        if isinstance(node, Index):
            return not (isinstance(node.base, Var) and node.base.name == "gmem") and scalar_leaf(node.base) and scalar_leaf(node.index)
        if isinstance(node, Assign) and not isinstance(node.target, Var): return False
        if isinstance(node, (Unary, Binary, Assign, Ternary, Seq)):
            return all(scalar_leaf(child) for value in vars(node).values()
                       for child in (value if isinstance(value, list) else [value]) if isinstance(child, Node))
        if isinstance(node, Call):
            return node.fn in arithmetic_calls and node.fn not in fn_defs and all(scalar_leaf(a) for a in node.args)
        return False
    def node_count(node):
        return 1 + sum(node_count(child) for value in vars(node).values()
                       for child in (value if isinstance(value, list) else [value]) if isinstance(child, Node))
    if compact_receiver_math:
        for name, definition in fn_defs.items():
            if (definition.instances and all(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", n) for n in definition.instances)
                    and scalar_leaf(definition.body) and node_count(definition.body) >= 64):
                receiver_cell_fields[name] = [n for n in definition.instances if n not in definition.params and n not in definition.locals]

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
        # instance(buffer) makes both buffer.member and buffer.method()
        # relative to the receiver, not global objects shared by every caller.
        # Longest prefix also handles dotted local/instance declarations.
        for mapping in (local_map, instance_map):
            matches = [key for key in mapping if name.startswith(key + ".")]
            if matches:
                key = max(matches, key=len)
                return mapping[key] + name[len(key):]
        if name == "this":
            return current_namespace or name
        if name.startswith("this."):
            suffix = name[5:]
            if current_namespace:
                return f"{current_namespace}.{suffix}" if suffix else current_namespace
            return suffix or name
        return name

    def resolve_user_call(fn_name: str, current_namespace: Optional[str], params: Set[str], local_map: Dict[str, str], instance_map: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
        if fn_name in fn_defs:
            declared_name = re.sub(r"^__eel_def_[0-9]+_", "", fn_name)
            prefix, dot, _ = declared_name.rpartition(".")
            # A dotted declaration fixes its receiver: scope.drawGrid()
            # shares scope.w with scope.drawSignal(), rather than owning a
            # separate scope.drawGrid.w pseudo-object.
            return fn_name, rewrite_var_name(prefix, params, local_map, instance_map, current_namespace) if dot else None

        parts = fn_name.split(".")
        if len(parts) >= 2 and parts[-1] in fn_defs:
            base = parts[-1]
            prefix = ".".join(parts[:-1])

            # EEL2/JSFX permits a qualified method name itself to appear in
            # instance(), e.g. instance(comp_l1.compressor_gain).  In that
            # form the receiver (comp_l1) is relative to the current function
            # instance just as an ordinary instance(comp_l1) declaration
            # would be.  Looking up only the receiver prefix loses that fact
            # and incorrectly specializes the call against a global comp_l1
            # namespace instead of e.g. filterbank.comp_l1.
            #
            # Preserve the declaration's fully-qualified mapping and strip the
            # method suffix back off to recover the correct receiver namespace.
            mapped_call = instance_map.get(fn_name)
            if mapped_call is not None:
                suffix = "." + base
                if mapped_call.endswith(suffix):
                    mapped_receiver = mapped_call[:-len(suffix)]
                    if mapped_receiver:
                        return base, mapped_receiver

            return base, rewrite_var_name(prefix, params, local_map, instance_map, current_namespace)

        return None, None

    def ensure_specialized(section: str, base_fn: str, call_namespace: Optional[str]) -> str:
        if base_fn not in fn_defs:
            return base_fn

        orig = fn_defs[base_fn]
        cell_fields = receiver_cell_fields.get(base_fn, [])
        namespace_key = call_namespace if fn_needs_namespace.get(base_fn, False) and not cell_fields else None
        if fn_needs_namespace.get(base_fn, False) and not namespace_key and not cell_fields:
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
        cell_params = [f"__receiver_cell_{i}" for i in range(len(cell_fields))]
        instance_map = dict(zip(cell_fields, cell_params)) if cell_fields else {name: _make_instance_var_name(namespace_key, name) for name in orig.instances} if namespace_key else {}
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
            cell_params,
        )

        in_progress.remove(key)
        return spec_name

    def rewrite_call_name(fn_name: str, section: str, current_namespace: Optional[str], params: Set[str], local_map: Dict[str, str], instance_map: Dict[str, str]) -> str:
        base_fn, call_namespace = resolve_user_call(fn_name, current_namespace, params, local_map, instance_map)
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
            new_fn = rewrite_call_name(n.fn, section, current_namespace, params, local_map, instance_map)
            new_args = [rewrite_node(a, section, current_namespace, params, local_map, instance_map) for a in n.args]
            base_fn, receiver = resolve_user_call(n.fn, current_namespace, params, local_map, instance_map)
            fields = receiver_cell_fields.get(base_fn, [])
            cells = [Var(n.id, n.span, _make_instance_var_name(receiver or base_fn, field)) for field in fields]
            if new_fn == n.fn and len(new_args) == len(n.args) and all(a1 is a2 for a1, a2 in zip(new_args, n.args)):
                return n
            return Call(n.id, n.span, new_fn, new_args, cells)
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
# Optimization debug / reporting helpers
# -----------------------------

_OPT_DEBUG_SECTION_ORDER = ("init", "slider", "block", "sample")


def _format_num_literal(v: float) -> str:
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if math.isfinite(fv):
        txt = format(fv, ".17g")
        if txt == "-0":
            txt = "0"
        return txt
    return str(fv)


def _node_to_jsfx_text(node: Node, indent: int = 0) -> str:
    pad = "  " * indent

    if isinstance(node, Num):
        return _format_num_literal(node.value)
    if isinstance(node, StrLit):
        return json.dumps(node.value)
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Index):
        return f"{_node_to_jsfx_text(node.base, indent)}[{_node_to_jsfx_text(node.index, indent)}]"
    if isinstance(node, Unary):
        return f"({node.op}{_node_to_jsfx_text(node.a, indent)})"
    if isinstance(node, Binary):
        return f"({_node_to_jsfx_text(node.l, indent)} {node.op} {_node_to_jsfx_text(node.r, indent)})"
    if isinstance(node, Assign):
        return f"{_node_to_jsfx_text(node.target, indent)} {node.op} {_node_to_jsfx_text(node.value, indent)}"
    if isinstance(node, Call):
        return f"{node.fn}({', '.join(_node_to_jsfx_text(a, indent) for a in node.args)})"
    if isinstance(node, Loop):
        return f"loop({_node_to_jsfx_text(node.count, indent)}, {_node_to_jsfx_text(node.body, indent)})"
    if isinstance(node, Ternary):
        return f"({_node_to_jsfx_text(node.cond, indent)} ? {_node_to_jsfx_text(node.then, indent)} : {_node_to_jsfx_text(node.els, indent)})"
    if isinstance(node, Seq):
        if not node.items:
            return "()"
        inner = []
        for it in node.items:
            inner.append(_stmt_to_jsfx_text(it, indent + 1, annotate=False))
        return "(\n" + "\n".join(inner) + "\n" + pad + ")"
    if isinstance(node, If):
        txt = f"if ({_node_to_jsfx_text(node.cond, indent)}) {_node_to_jsfx_text(node.then, indent)}"
        if node.els is not None:
            txt += f" else {_node_to_jsfx_text(node.els, indent)}"
        return txt
    if isinstance(node, While):
        return f"while ({_node_to_jsfx_text(node.cond, indent)}) {_node_to_jsfx_text(node.body, indent)}"
    if isinstance(node, FunctionDef):
        quals: List[str] = []
        if node.locals:
            quals.append("local(" + ", ".join(node.locals) + ")")
        if node.instances:
            quals.append("instance(" + ", ".join(node.instances) + ")")
        qual_txt = (" " + " ".join(quals)) if quals else ""
        return f"function {node.name}({', '.join(node.params)}){qual_txt} {_node_to_jsfx_text(node.body, indent)}"
    raise TypeError(type(node))


def _stmt_to_jsfx_text(node: Node, indent: int = 0, *, annotate: bool) -> str:
    pad = "  " * indent
    ann = ""
    if annotate:
        ann = f"{pad}// L{getattr(node.span, 'line', 0)} C{getattr(node.span, 'col', 0)} ID{getattr(node, 'id', 0)}\n"
    tail = "" if isinstance(node, FunctionDef) else ";"
    return ann + pad + _node_to_jsfx_text(node, indent) + tail


def _program_to_jsfx_text(nodes: List[Node], *, annotate: bool) -> str:
    if not nodes:
        return "// <empty>"
    return "\n".join(_stmt_to_jsfx_text(n, 0, annotate=annotate) for n in nodes)


def _emit_original_sections_text(sections: Dict[str, Tuple[str, int]]) -> str:
    order = list(_OPT_DEBUG_SECTION_ORDER)
    extras = [k for k in sections.keys() if k not in order]
    order.extend(sorted(extras))

    parts: List[str] = []
    for sec in order:
        if sec not in sections:
            continue
        code, start_line = sections[sec]
        parts.append(f"@{sec}    // source starts at line {start_line}")
        body = code.rstrip()
        parts.append(body if body else "// <empty>")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _emit_reconstructed_sections_text(programs: Dict[str, List[Node]],
                                      fn_defs: Dict[str, FunctionDef],
                                      *,
                                      annotate: bool) -> str:
    parts: List[str] = []
    for sec in _OPT_DEBUG_SECTION_ORDER:
        parts.append(f"@{sec}")
        parts.append(_program_to_jsfx_text(programs.get(sec, []), annotate=annotate))
        parts.append("")

    if fn_defs:
        parts.append("/* user functions */")
        for name in sorted(fn_defs.keys()):
            parts.append(_stmt_to_jsfx_text(fn_defs[name], 0, annotate=annotate))
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _make_opt_event(kind: str, node: Node, **extra: Any) -> Dict[str, Any]:
    ev: Dict[str, Any] = {
        "kind": kind,
        "node_id": int(getattr(node, "id", 0) or 0),
        "line": int(getattr(node.span, "line", 0) or 0),
        "col": int(getattr(node.span, "col", 0) or 0),
        "text": _node_to_jsfx_text(node),
    }
    ev.update(extra)
    return ev


def _record_opt_event(events: Optional[List[Dict[str, Any]]], event: Dict[str, Any]) -> None:
    if events is not None:
        events.append(event)


def _sort_opt_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        events,
        key=lambda ev: (
            str(ev.get("from_section") or ev.get("section") or ""),
            int(ev.get("loop_line") or 0),
            int(ev.get("line") or 0),
            int(ev.get("col") or 0),
            str(ev.get("kind") or ""),
            int(ev.get("node_id") or 0),
        ),
    )


def _build_opt_report(events: List[Dict[str, Any]], *, enable_section_hoists: bool, enable_loop_hoists: bool) -> Dict[str, Any]:
    ordered = _sort_opt_events(list(events))
    summary = {
        "section_hoists": sum(1 for ev in ordered if ev.get("kind") == "section_hoist"),
        "loop_hoists": sum(1 for ev in ordered if ev.get("kind") == "loop_hoist"),
    }
    return {
        "settings": {
            "section_hoists_enabled": bool(enable_section_hoists),
            "loop_hoists_enabled": bool(enable_loop_hoists),
        },
        "summary": summary,
        "events": ordered,
    }


def _emit_opt_report_text(report: Dict[str, Any]) -> str:
    settings = dict(report.get("settings") or {})
    summary = dict(report.get("summary") or {})
    events = list(report.get("events") or [])

    lines: List[str] = []
    lines.append("DSP-JSFX optimization movement report")
    lines.append("")
    lines.append(f"section hoists enabled: {1 if settings.get('section_hoists_enabled') else 0}")
    lines.append(f"loop hoists enabled: {1 if settings.get('loop_hoists_enabled') else 0}")
    lines.append(f"section hoists: {int(summary.get('section_hoists', 0) or 0)}")
    lines.append(f"loop hoists: {int(summary.get('loop_hoists', 0) or 0)}")
    lines.append("")

    section_events = [ev for ev in events if ev.get("kind") == "section_hoist"]
    loop_events = [ev for ev in events if ev.get("kind") == "loop_hoist"]

    lines.append("SECTION HOISTS")
    if not section_events:
        lines.append("  <none>")
    else:
        for ev in section_events:
            lines.append(
                f"  {ev.get('from_section')} -> {ev.get('to_section')} | L{ev.get('line')}:{ev.get('col')} | id {ev.get('node_id')} | {ev.get('target') or '?'}"
            )
            lines.append(f"    {ev.get('text')}")
    lines.append("")

    lines.append("LOOP HOISTS")
    if not loop_events:
        lines.append("  <none>")
    else:
        for ev in loop_events:
            lines.append(
                f"  {ev.get('section')} | {ev.get('loop_kind')} L{ev.get('loop_line')}:{ev.get('loop_col')} | expr L{ev.get('line')}:{ev.get('col')} | loop id {ev.get('loop_id')} | expr id {ev.get('node_id')}"
            )
            lines.append(f"    {ev.get('text')}")
    lines.append("")
    return "\n".join(lines)


def _write_opt_report(path: str, report: Dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        out_path.write_text(_emit_opt_report_text(report), encoding="utf-8")


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepare_jsfx_pipeline(jsfx_text: str,
                          *,
                          include_gfx: bool = False,
                          include_serialize: bool = False,
                          native_gfx_legacy: bool = False,
                          enable_section_hoists: bool = False,
                          enable_loop_hoists: bool = False,
                          collect_opt_report: bool = False,
                          parse_program: Optional[Callable[[str, int], List[Node]]] = None) -> Dict[str, Any]:
    if native_gfx_legacy and (enable_section_hoists or enable_loop_hoists):
        raise ValueError("Native legacy shared state is incompatible with custom scalar hoisting")
    include_gfx = include_gfx or native_gfx_legacy
    faust_plan = faust_compiler.split(jsfx_text)
    if faust_plan and (enable_section_hoists or enable_loop_hoists):
        raise ValueError("Mixed Faust sections do not support custom section/loop hoisting")
    sections = extract_sections(faust_plan["text"] if faust_plan else jsfx_text)

    programs: Dict[str, List[Node]] = {}
    for sec in _OPT_DEBUG_SECTION_ORDER + (("serialize",) if include_serialize else ()) + (("gfx",) if include_gfx else ()):
        if sec in sections:
            code, start_line = sections[sec]
            programs[sec] = (parse_program(code, start_line) if parse_program is not None
                             else Parser(code, base_line=start_line).parse_program())
        else:
            programs[sec] = []

    fn_defs, programs = extract_function_defs(programs)
    programs, fn_defs = lower_user_functions(programs, fn_defs, compact_receiver_math=native_gfx_legacy)
    validate_builtin_sections(programs)
    lowered_programs = {sec: list(nodes) for sec, nodes in programs.items()}

    opt_events: Optional[List[Dict[str, Any]]] = [] if collect_opt_report else None

    work_programs = {sec: list(nodes) for sec, nodes in lowered_programs.items()}
    if enable_section_hoists:
        work_programs = hoist_section_invariants(work_programs, fn_defs, events=opt_events)

    if enable_loop_hoists:
        loop_hoists = plan_loop_invariant_hoists(work_programs, fn_defs, events=opt_events)
    else:
        loop_hoists = {}

    return {
        "sections": sections,
        "parse_program": parse_program,
        "faust_plan": faust_plan,
        "fn_defs": fn_defs,
        "lowered_programs": lowered_programs,
        "programs": work_programs,
        "loop_hoists": loop_hoists,
        "opt_report": _build_opt_report(opt_events or [], enable_section_hoists=enable_section_hoists, enable_loop_hoists=enable_loop_hoists),
    }


def parse_native_gfx_contract(text: str, user_vars: Dict[str, int]) -> Dict[str, Any]:
    """Explicit ownership/publication contract; ordinary display builds stay strict."""
    declarations = re.findall(r"(?m)^\s*//\s*za_native_gfx:\s*(\{.*\})\s*$", text)
    if not declarations:
        return {}
    if len(declarations) != 1:
        raise ValueError("Exactly one za_native_gfx contract is allowed")
    contract = json.loads(declarations[0])
    if set(contract) - {"interactive", "locals", "commands", "views", "persist"}:
        raise ValueError("Unknown native graphics contract field")
    if not isinstance(contract.get("interactive", False), bool):
        raise ValueError("Native interactive must be boolean")
    for key in ("locals", "commands", "persist"):
        names = contract.get(key, [])
        if not isinstance(names, list) or any(not isinstance(n, str) or n not in user_vars for n in names):
            raise ValueError(f"Unknown variable in native graphics {key}")
    for key in ("locals", "commands", "persist"):
        if any(n.startswith(("#", "gfx_", "mouse_")) or n in BUILTIN_NAMES or re.fullmatch(r"(?:slider|spl)\d+", n) for n in contract.get(key, [])):
            raise ValueError("Native ownership declarations require ordinary scalar variables")
    if set(contract.get("persist", [])) & set(contract.get("commands", [])):
        raise ValueError("Native commands cannot be persisted")
    if set(contract.get("locals", [])) & set(contract.get("commands", [])):
        raise ValueError("Native commands cannot also be local previews")
    views = contract.get("views", [])
    if len(views) > 16:
        raise ValueError("Native graphics supports at most 16 publications")
    for view in views:
        if set(view) - {"base", "count", "stride", "fields", "max", "generation"}:
            raise ValueError("Unknown native publication field")
        for key in ("base", "count", "stride", "generation"):
            value = view.get(key, 1 if key == "stride" else 0)
            if not (isinstance(value, int) and 0 <= value <= 16777216 or isinstance(value, str) and value in user_vars):
                raise ValueError(f"Invalid native publication {key}")
        fields = view.get("fields", [0])
        maximum = view.get("max", 0)
        if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 65536 or not fields or len(fields) > 16 or any(not isinstance(f, int) or not 0 <= f < 65536 for f in fields) or len(set(fields)) != len(fields):
            raise ValueError("Invalid native publication bounds/fields")
        stride = view.get("stride", 1)
        if isinstance(stride, int) and (stride < 1 or max(fields) >= stride):
            raise ValueError("Native publication fields must fit the record stride")
    if sum(v["max"] * len(v.get("fields", [0])) for v in views) > 262144:
        raise ValueError("Native publication capacity exceeds 262144 cells")
    return contract


# Order is emitted into the generated header and used by the C++ dispatcher.
NATIVE_GFX_APIS = {
    "gfx_set": (1, 7), "gfx_rect": (4, 5), "gfx_rectto": (2, 2),
    "gfx_line": (4, 5), "gfx_lineto": (2, 3), "gfx_circle": (3, 5),
    "gfx_roundrect": (5, 6), "gfx_arc": (5, 6), "gfx_triangle": (6, 1024),
    "gfx_setimgdim": (3, 3), "gfx_getimgdim": (3, 3), "gfx_loadimg": (2, 2),
    "gfx_blit": (3, 13), "gfx_blitext": (3, 3), "gfx_deltablit": (9, 16),
    "gfx_transformblit": (8, 8), "gfx_setpixel": (3, 3), "gfx_getpixel": (3, 3),
    "gfx_gradrect": (8, 16), "gfx_muladdrect": (7, 12), "gfx_blurto": (2, 2),
    "gfx_drawstr": (1, 4), "gfx_printf": (1, 128), "gfx_measurestr": (1, 3),
    "gfx_drawchar": (1, 1), "gfx_drawnumber": (2, 2), "gfx_measurechar": (3, 3),
    "gfx_setfont": (1, 4), "gfx_getfont": (0, 1), "gfx_setcursor": (1, 2),
    "gfx_getdropfile": (1, 2),
}
# The vendored VM's vararg registrations check a minimum argument count;
# drawing functions ignore extra arguments after evaluating them.
NATIVE_GFX_APIS = {name: (minimum, 1024) for name, (minimum, _) in NATIVE_GFX_APIS.items()}
NATIVE_STRING_APIS = {
    "strcpy": (2,2), "strcat": (2,2), "strncpy": (3,3), "strncat": (3,3),
    "strlen": (1,1), "strcmp": (2,2), "stricmp": (2,2), "strncmp": (3,3),
    "strnicmp": (3,3), "str_getchar": (2,3), "str_setchar": (3,4),
    "str_setlen": (2,2), "strcpy_substr": (3,4), "strcpy_from": (3,3),
    "str_insert": (3,3), "str_delsub": (3,3), "sprintf": (2,1024),
}
NATIVE_FILE_APIS = {"file_open": (1,2), "file_open_multi": (1,2),
    "file_close": (1,1), "file_rewind": (1,1), "file_seek": (2,2),
    "file_avail": (1,1), "file_text": (1,1), "file_riff": (3,3),
    "file_var": (2,2), "file_mem": (3,3), "file_string": (2,2),
    "file_multi_count": (1,1), "file_multi_select": (2,2)}
# GFX file calls use the immediate worker backend. Audio keeps its existing
# host file bridge, whose I/O is prepared off the audio thread.
NATIVE_GFX_APIS.update(NATIVE_FILE_APIS)
NATIVE_GFX_APIS.update({"match": (2,1024), "matchi": (2,1024)})

def validate_native_gfx_prototype(programs: Dict[str, List[Node]], fn_defs: Dict[str, FunctionDef], contract: Optional[Dict[str, Any]] = None, *, legacy: bool = False) -> Set[str]:
    """Validate the entire reachable graphics graph, including helper bodies."""
    contract = contract or {}
    interactive = legacy or bool(contract.get("interactive"))
    supported = {"gfx_set": (1, 4), "gfx_rect": (4, 5), "gfx_circle": (3, 5),
                 "gfx_line": (4, 5), "gfx_measurestr": (1, 3),
                 "gfx_setfont": (1, 4), "gfx_drawstr": (1, 1),
                 "min": (2, 2), "max": (2, 2), "abs": (1, 1),
                 "floor": (1, 1), "ceil": (1, 1), "sin": (1, 1),
                 "cos": (1, 1), "exp": (1, 1), "log": (1, 1),
                 "sqrt": (1, 1), "pow": (2, 2)}
    supported.update({name: (arity, arity) if arity is not None else (1, 32)
                      for name, (_, arity) in tasks_compiler.APIS.items()})
    supported.update({'defer': (1,1), 'defer_after': (2,2), 'defer_for': (3,3), 'defer_reduce': (5,5)})
    if interactive:
        supported.update({"gfx_getchar": (0, 1), "gfx_showmenu": (1, 1),
                          "strcpy": (2, 2), "strcat": (2, 2), "strncpy": (3, 3),
                          "strlen": (1, 1), "time_precise": (0, 1),
                          "sliderchange": (1, 1), "slider_automate": (1, 2)})
    if legacy:
        supported.update(NATIVE_GFX_APIS)
        supported.update(NATIVE_STRING_APIS)
        supported.update({"memcpy": (3, 3), "memset": (3, 3), "freembuf": (1, 1),
                          "__memtop": (0, 0), "slider_show": (1, 2), "fft": (2, 2), "ifft": (2, 2),
                          "fft_real": (2, 2), "ifft_real": (2, 2),
                          "fft_permute": (2, 2), "fft_ipermute": (2, 2),
                          "convolve_c": (3, 3), "slider": (1, 1), "spl": (1, 1),
                          "rand": (0, 1), "atomic_get": (1, 1), "atomic_set": (2, 2),
                          "atomic_add": (2, 2), "atomic_exch": (2, 2),
                          "atomic_setifequal": (3, 3), "sqr": (1, 1),
                          "sign": (1, 1), "tan": (1, 1), "atan": (1, 1),
                          "atan2": (2, 2), "log10": (1, 1),
                          "asin": (1, 1), "acos": (1, 1)})
    if not legacy and not programs.get("gfx"):
        raise ValueError("Native GFX prototype requires a nonempty @gfx section")
    reachable: Set[str] = set()

    def visit(value: Any) -> None:
        if not legacy and isinstance(value, Var) and ((value.name.startswith("mouse_") and not interactive) or value.name.startswith("gfx_ext_") or value.name in ("gfx_clear", "gfx_mode", "gfx_dest")):
            raise ValueError(f"Native display prototype does not support {value.name}")
        if isinstance(value, Call):
            if value.fn in tasks_compiler.DEFERRED:
                # The task validator checks private bodies separately. Only
                # submission arguments execute on the graphics state.
                for arg in value.args[:-1]:
                    visit(arg)
                return
            if value.fn in fn_defs:
                if value.fn not in reachable:
                    reachable.add(value.fn)
                    visit(fn_defs[value.fn].body)
            elif value.fn == "sprintf":
                if legacy:
                    if len(value.args) < 2:
                        raise ValueError("sprintf requires destination and format")
                    for arg in value.args:
                        visit(arg)
                    return
                if (len(value.args) < 2 or not isinstance(value.args[0], Var)
                        or not value.args[0].name.startswith("#") or not isinstance(value.args[1], StrLit)):
                    raise ValueError("Native sprintf requires a named string destination and literal format")
                fmt = value.args[1].value
                fields = re.findall(r"%(?:%|[-+ 0#]*\d*(?:\.\d+)?[sfeEgGdiuxXc])", fmt)
                if re.sub(r"%(?:%|[-+ 0#]*\d*(?:\.\d+)?[sfeEgGdiuxXc])", "", fmt).find("%") >= 0:
                    raise ValueError("Unsupported native sprintf format")
                if len([f for f in fields if f != "%%"]) != len(value.args) - 2:
                    raise ValueError("Native sprintf format/argument count mismatch")
                if any(int(n) > 512 for f in fields for n in re.findall(r"\d+", f)):
                    raise ValueError("Native sprintf field width/precision exceeds 512")
            elif value.fn not in supported or not supported[value.fn][0] <= len(value.args) <= supported[value.fn][1]:
                raise ValueError(f"Native GFX prototype does not support {value.fn}({len(value.args)} args) at {value.span.line}:{value.span.col}")
            if not legacy and value.fn in ("strcpy", "strcat", "strncpy"):
                if not isinstance(value.args[0], Var) or not value.args[0].name.startswith("#"):
                    raise ValueError("Native string operations require a named string destination")
            if value.fn == "gfx_measurestr":
                for output in value.args[1:]:
                    if not legacy and (not isinstance(output, Var) or output.name.startswith("#")
                            or output.name in BUILTIN_NAMES
                            or re.fullmatch(r"(?:slider|spl)\d+", output.name)):
                        raise ValueError("Native gfx_measurestr outputs require graphics-owned scalar variables")
        if not legacy and isinstance(value, Index) and (not contract.get("views") or isinstance(value.base, Var) and value.base.name == "gmem"):
            raise ValueError(f"Native GFX prototype does not yet support indexed memory at {value.span.line}:{value.span.col}")
        if isinstance(value, Assign):
            if legacy and (isinstance(value.target, Index) or isinstance(value.target, Call)
                           and value.target.fn in ("slider", "spl")):
                pass
            elif not isinstance(value.target, Var) or value.target.name.startswith("#"):
                raise ValueError("Native GFX prototype requires scalar assignment targets")
            elif not legacy and (value.target.name in BUILTIN_NAMES or re.fullmatch(r"spl\d+", value.target.name) or re.fullmatch(r"slider\d+", value.target.name) and not interactive):
                raise ValueError(f"Native GFX prototype does not support writing {value.target.name}")
        if isinstance(value, Node):
            for child in vars(value).values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(programs["gfx"])
    checked_audio: Set[str] = set()
    def check_init(value: Any) -> None:
        if isinstance(value, Call) and value.fn in fn_defs and value.fn not in checked_audio:
            checked_audio.add(value.fn)
            check_init(fn_defs[value.fn].body)
        if isinstance(value, Call) and value.fn.startswith("gfx_") and not (legacy and section == "init" and value.fn in NATIVE_GFX_APIS):
            raise ValueError(f"Native GFX prototype does not support graphics calls from audio sections at {value.span.line}:{value.span.col}")
        if not legacy and ((isinstance(value, Var) and value.name.startswith("#")) or (isinstance(value, Call) and value.fn in (
                "sprintf", "printf", "strcpy", "strcat", "strncpy", "str_setchar",
                "str_insert", "str_delete", "str_mid", "file_string",
                "instance_uid", "instance_get_name", "track_name", "host_track_name",
                "msg_peer_name", "msg_peer_uid"))):
            raise ValueError("Native display prototype does not support mutable strings in audio sections")
        if not legacy and isinstance(value, Assign) and isinstance(value.target, Var) and value.target.name.startswith("gfx_"):
            raise ValueError("Native GFX prototype does not support graphics initialization in audio sections")
        if isinstance(value, Node):
            for child in vars(value).values():
                check_init(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_init(child)
    for section in _OPT_DEBUG_SECTION_ORDER:
        check_init(programs[section])
    return reachable


def compile_pipeline_to_ir(jsfx_text: str, pipeline: Dict[str, Any],
                           *, native_gfx_prototype: bool = False, native_gfx_legacy: bool = False, stream_functions: bool = False, state_var_capacity: int = 0) -> Tuple[ir.Module, Dict[str, Any]]:
    native_gfx_prototype = native_gfx_prototype or native_gfx_legacy
    if native_gfx_legacy and pipeline["opt_report"]["settings"].get("section_hoists_enabled"):
        raise ValueError("Native legacy shared state is incompatible with custom scalar hoisting")
    if native_gfx_legacy and pipeline["loop_hoists"]:
        raise ValueError("Native legacy shared state is incompatible with custom scalar hoisting")
    programs: Dict[str, List[Node]] = dict(pipeline["programs"])
    fn_defs: Dict[str, FunctionDef] = dict(pipeline["fn_defs"])
    loop_hoists: Dict[int, List[Node]] = dict(pipeline["loop_hoists"])

    faust_plan = pipeline.get('faust_plan')
    faust_nodes = faust_compiler.stage_nodes(sys.modules[__name__], {'programs':programs}, faust_plan) if faust_plan else {}
    faust_aliases = {name.lower():int(index)-1 for index,name in re.findall(
        r"(?m)^\s*slider([0-9]+)\s*:\s*([A-Za-z_][A-Za-z_0-9]*)\s*=",jsfx_text)}
    slider_aliases = {name.lower() for _, name in re.findall(
        r"(?m)^\s*slider([0-9]+)\s*:\s*([A-Za-z_][A-Za-z_0-9]*)\s*=", jsfx_text)}
    has_tasks = tasks_compiler.validate(sys.modules[__name__], programs, fn_defs, slider_aliases)
    gfx_source = pipeline['sections'].get('gfx', ('', 0))[0]
    gfx_code = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', '', gfx_source, flags=re.S)
    task_calls = '|'.join(sorted(set(tasks_compiler.DEFERRED) | set(tasks_compiler.APIS)))
    serialize_source = pipeline['sections'].get('serialize', ('', 0))
    serialize_code = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', '', serialize_source[0], flags=re.S)
    if serialize_source[0] and (has_tasks or re.search(r'\b(?:' + task_calls + r')\s*\(', serialize_code)):
        validation_programs = dict(programs)
        parse_program = pipeline.get('parse_program')
        validation_programs['serialize'] = (parse_program(*serialize_source) if parse_program is not None
                                            else Parser(serialize_source[0], base_line=serialize_source[1]).parse_program())
        tasks_compiler.validate(sys.modules[__name__], validation_programs, fn_defs, slider_aliases)
    if not native_gfx_prototype and re.search(r'\b(?:' + task_calls + r')\s*\(', gfx_code):
        raise ValueError('Tasks in @gfx require --native-gfx-legacy or --native-gfx-prototype')
    if has_tasks and (pipeline['loop_hoists'] or pipeline['opt_report']['settings'].get('section_hoists_enabled')):
        raise ValueError('Tasks are incompatible with custom scalar hoisting')
    user_vars = collect_user_vars(programs, fn_defs)
    if has_tasks and not native_gfx_legacy and len(user_vars) > 4096:
        raise ValueError('Task snapshot limit is 4096 scalar variables')
    if native_gfx_legacy:
        # 'mem' is an ordinary EEL variable, often an allocator cursor.
        # The older DSP dialect reserved it as a zero-valued heap base.
        user_vars.setdefault("mem", len(user_vars))
    if native_gfx_prototype:
        # Graphics globals have defaults even when only written by host calls.
        for name in ("gfx_r", "gfx_g", "gfx_b", "gfx_a", "gfx_a2", "gfx_x", "gfx_y",
                     "gfx_w", "gfx_h", "gfx_dest", "gfx_mode", "gfx_texth",
                     "gfx_clear", "gfx_ext_retina", "gfx_ext_flags", "gfx_frame"):
            if name not in user_vars:
                user_vars[name] = len(user_vars)

    options = parse_jsfx_options(jsfx_text)
    if native_gfx_legacy:
        if re.search(r"(?m)^\s*options\s*:[^\n]*\bgfx_idle(?:_only)?\b", jsfx_text):
            raise ValueError("Native legacy does not yet support gfx_idle/gfx_idle_only without an editor")
        if "maxmem" in options:
            try:
                maximum = int(float(options["maxmem"]))
            except (ValueError, OverflowError):
                raise ValueError("Native legacy maxmem must be a positive number of cells")
            if not 1 <= maximum <= 268435456:
                raise ValueError("Native legacy maxmem must be between 1 and 268435456 cells (2 GiB)")
    gfx_var_sync_mode = str(options.get("ownership", "legacy") or "legacy").strip().lower()
    if gfx_var_sync_mode in ("auto", "hybrid"):
        try:
            gfx_var_sync = analyze_gfx_var_sync(jsfx_text, user_vars)
            gfx_var_flags = gfx_var_sync["flags"]
            gfx_mem_shared = bool(gfx_var_sync.get("mem_shared"))
            gfx_var_sync_mode = "hybrid"
        except Exception:
            gfx_var_sync_mode = "legacy"
            gfx_var_flags = {name: (GFX_VAR_FLAG_TO_GFX | GFX_VAR_FLAG_FROM_GFX) for name in user_vars.keys()}
            gfx_mem_shared = True
    elif gfx_var_sync_mode == "ui_only":
        gfx_var_flags = {name: 0 for name in user_vars.keys()}
        gfx_mem_shared = False
    else:
        gfx_var_sync_mode = "legacy"
        gfx_var_flags = {name: (GFX_VAR_FLAG_TO_GFX | GFX_VAR_FLAG_FROM_GFX) for name in user_vars.keys()}
        gfx_mem_shared = True

    native_functions: Set[str] = set()
    native_contract = {"interactive": True} if native_gfx_legacy else parse_native_gfx_contract(jsfx_text, user_vars) if native_gfx_prototype else {}
    if native_gfx_legacy:
        native_functions = validate_native_gfx_prototype(programs, fn_defs, native_contract, legacy=True)
        gfx_var_flags = {name: 0 for name in user_vars}
        gfx_var_sync_mode = "native-legacy-shared"
        gfx_mem_shared = True
    elif native_gfx_prototype:
        native_functions = validate_native_gfx_prototype(programs, fn_defs, native_contract)
        usage = analyze_gfx_var_sync(jsfx_text, user_vars)
        conflicts = usage["gfx_writes"] & (usage["audio_reads"] | usage["audio_writes"])
        conflicts -= set(native_contract.get("locals", [])) | set(native_contract.get("commands", []))
        if conflicts:
            raise ValueError("Native display prototype rejects UI writes to DSP variables: " + ", ".join(sorted(conflicts)))
        gfx_var_flags = {name: GFX_VAR_FLAG_TO_GFX if name in usage["gfx_reads"]
                         and name not in usage["gfx_writes"] and not name.startswith(("gfx_", "#")) else 0
                         for name in user_vars}
        for name in native_contract.get("commands", []):
            gfx_var_flags[name] = GFX_VAR_FLAG_FROM_GFX
        for view in native_contract.get("views", []):
            for key in ("base", "count", "stride", "generation"):
                name = view.get(key)
                if isinstance(name, str):
                    if name in usage["gfx_writes"]:
                        raise ValueError("Publication layout variables must be audio owned: " + name)
                    gfx_var_flags[name] = GFX_VAR_FLAG_TO_GFX
        gfx_var_sync_mode = "native-interactive" if native_contract.get("interactive") else "native-display"
        gfx_mem_shared = False

    pin_hints = parse_pin_hints(jsfx_text)
    io_channels = infer_spl_io(programs, fn_defs, pin_hints=pin_hints, include_fallback=not bool(faust_plan))
    midi_caps = detect_midi_usage(programs, fn_defs)
    comm_caps = detect_comm_usage(programs, fn_defs)
    sample_pool_caps = detect_sample_pool_usage(programs, fn_defs)

    sym = SymTable(user_vars)
    if native_gfx_legacy or faust_plan:
        sym.slider_aliases = {name.lower(): int(index) - 1 for index, name in re.findall(
            r"(?m)^\s*slider([0-9]+)\s*:\s*([A-Za-z_][A-Za-z_0-9]*)\s*=", jsfx_text)}

    emitter = LLVMModuleEmitter(sym, native_gfx_legacy=native_gfx_legacy, stream_functions=stream_functions, has_tasks=has_tasks, has_faust=bool(faust_plan), state_var_capacity=state_var_capacity)
    emitter.native_gfx_prototype = native_gfx_prototype
    emitter.native_gfx_functions = native_functions
    emitter.jsfx_memtop_slots = resolve_jsfx_memtop_slots(options)
    # EEL2 assignments normally pass through WDL's denormal_filter_double2,
    # which maps subnormals, NaN, and +/-infinity to +0. This is observable
    # language behaviour (for example, log(0) assigned to a variable becomes
    # 0), not merely an audio-thread denormal optimisation. Keep the legacy
    # native/IEEE path available only as an explicit compatibility opt-out.
    emitter.eel2_store_filter = options.get("za_eel2_stores", "1") != "0"
    emitter.loop_hoists = loop_hoists
    emitter.declare_user_functions(fn_defs)

    fn_init = emitter.emit_section_fn("jsfx_init", programs["init"])
    fn_slider = emitter.emit_section_fn("jsfx_slider", programs["slider"])
    fn_block = emitter.emit_section_fn("jsfx_block", programs["block"])
    fn_sample = emitter.emit_section_fn("jsfx_sample", programs["sample"])
    if "serialize" in programs:
        emitter.emit_section_fn("jsfx_serialize",programs["serialize"])

    if native_gfx_prototype:
        emitter.emit_section_fn("jsfx_gfx_aot", programs["gfx"])

    faust_stages, faust_modules, faust_islands = faust_compiler.attach(sys.modules[__name__],emitter,faust_plan,faust_nodes,user_vars,faust_aliases,fn_defs) if faust_plan else ([],[],[])
    if faust_plan and native_gfx_prototype and not native_gfx_legacy:
        extra=set().union(*(set(stage.get('imports',[]))|set(stage.get('exports',[])) for stage in faust_stages))
        conflicts=usage['gfx_writes'] & {name.lower() for name in extra}
        conflicts-=set(native_contract.get('locals',[]))|set(native_contract.get('commands',[]))
        if conflicts:raise ValueError('Native display prototype rejects UI writes to Faust DSP variables: '+', '.join(sorted(conflicts)))
    emitter.emit_user_functions()

    has_sample_work = len(programs["sample"]) > 0 or bool(faust_plan)
    if faust_plan:
        faust_compiler.emit_process(sys.modules[__name__],emitter)
        for stage in faust_stages:
            if stage['kind']=='faust':
                named=max([int(n[3:])+1 for n in stage['signals'] if re.fullmatch(r'spl\d+',n.lower())]+[0])
                io_channels['inputs']=max(io_channels['inputs'],named,stage['inputs']-len(stage['signals']))
                io_channels['outputs']=max(io_channels['outputs'],stage['audio_outputs'])
        io_channels['process']=max(io_channels['inputs'],io_channels['outputs'],io_channels.get('process',0))
    else:
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

    faust_quantum = int(options.get('za_faust_quantum', '0'))
    if faust_quantum and (not faust_plan or not 2 <= faust_quantum <= 4096 or
                          not faust_stages or faust_stages[0]['kind'] != 'block' or
                          any(stage['fused'] for stage in faust_stages)):
        raise ValueError('za_faust_quantum requires an unfused Faust pipeline beginning with @block and a size from 2 to 4096')
    meta = {
        "has_tasks": has_tasks,
        "has_task_workers": tasks_compiler.needs_workers(sys.modules[__name__], programs, fn_defs),
        "has_faust": bool(faust_plan),
        "faust_stages": faust_stages,
        "faust_islands": faust_islands,
        "faust_quantum": faust_quantum,
        "vars": user_vars,
        "var_cap": emitter.var_cap,
        "sections_present": {k: bool(v) for k, v in programs.items()},
        "io_channels": io_channels,
        "pin_hints": pin_hints,
        "midi": midi_caps,
        "comm": comm_caps,
        "sample_pool": sample_pool_caps,
        "plugin_kind": plugin_kind,
        "string_literals": emitter.get_string_literals_meta(),
        "named_strings": dict(emitter._named_gfx_strings),
        "has_sample_section": has_sample_work,
        "gfx_var_sync_mode": gfx_var_sync_mode,
        "gfx_var_flags": gfx_var_flags,
        "gfx_mem_shared": gfx_mem_shared,
        "numeric_semantics": "eel2-stores" if emitter.eel2_store_filter else "native",
        "native_gfx_prototype": native_gfx_prototype,
        "native_gfx_legacy": native_gfx_legacy,
        "memtop_slots": emitter.jsfx_memtop_slots,
        "slider_aliases": sym.slider_aliases,
        "native_gfx_contract": native_contract,
        "native_gfx_size": list(map(int, re.search(r"(?m)^\s*@gfx\s+(\d+)\s+(\d+)", jsfx_text).groups())) if native_gfx_prototype and re.search(r"(?m)^\s*@gfx\s+(\d+)\s+(\d+)", jsfx_text) else [640, 400],
    }
    return faust_compiler.merge(emitter.module,faust_modules) if faust_modules else emitter.module, meta



# -----------------------------
# Loop-invariant scalar hoisting
# -----------------------------

_PURE_HOIST_CALLS: Set[str] = {
    "min", "max",
    "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2",
    "sqr", "sqrt", "pow", "exp", "log", "log10",
    "fabs", "sign", "floor", "ceil", "invsqrt",
    "__memtop",
}

_SLIDER_VAR_RE = re.compile(r"^slider([1-9][0-9]{0,2})$")
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
    return m is not None and 1 <= int(m.group(1)) <= MAX_JSFX_SLIDERS


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
                for a in n.args[:(3 if len(n.args) == 3 else 4)]:
                    _note_mutated_lvalue(a, out)
                return

            if fn == "midirecv_buf":
                if len(n.args) >= 1:
                    _note_mutated_lvalue(n.args[0], out)
                out.writes_mem = True
                return

            if fn == "midirecv_str":
                if len(n.args) >= 1:
                    _note_mutated_lvalue(n.args[0], out)
                if len(n.args) >= 2:
                    _note_mutated_lvalue(n.args[1], out)
                return

            if fn == "file_var" and len(n.args) >= 2:
                _note_mutated_lvalue(n.args[1], out)
                return

            if fn == "file_riff" and len(n.args) >= 3:
                _note_mutated_lvalue(n.args[1], out)
                _note_mutated_lvalue(n.args[2], out)
                return

            if fn == "msg_recv":
                for a in n.args[1:7]:
                    _note_mutated_lvalue(a, out)
                out.writes_unknown_state = True
                return

            if fn == "msg_recv_buf":
                for a in n.args[1:3]:
                    _note_mutated_lvalue(a, out)
                out.writes_mem = True
                out.writes_unknown_state = True
                return

            if fn in ("sample_read2", "sample_read2_interp"):
                for a in n.args[3:5]:
                    _note_mutated_lvalue(a, out)
                return

            if fn == "sample_preview_read":
                for a in n.args[3:6]:
                    _note_mutated_lvalue(a, out)
                return

            if fn == "sample_name" and len(n.args) >= 3:
                _note_mutated_lvalue(n.args[2], out)
                out.writes_unknown_state = True
                return

            if fn in SAMPLE_POOL_EXPORT_FUNCTIONS:
                out.writes_mem = True
                out.writes_unknown_state = True
                return

            if fn in ("instance_uid", "instance_get_name", "track_name", "host_track_name") and len(n.args) >= 1:
                _note_mutated_lvalue(n.args[0], out)
                out.writes_unknown_state = True
                return

            if fn in ("msg_peer_name", "msg_peer_uid") and len(n.args) >= 2:
                _note_mutated_lvalue(n.args[1], out)
                out.writes_unknown_state = True
                return

            if fn in COMM_IMPURE_FUNCTIONS:
                out.writes_unknown_state = True
                return

            if fn in ("memset", "memcpy", "fft", "ifft", "fft_real", "ifft_real", "fft_permute", "fft_ipermute", "convolve_c", "file_mem"):
                out.writes_mem = True
                return

            if fn == "abs":
                fn = "fabs"

            if fn in _PURE_HOIST_CALLS:
                return

            if fn.startswith("gfx_") or fn in (
                "slider", "spl",
                "midisend", "midisend_buf", "midisend_str", "midisyx", "rand",
                "freembuf", "sliderchange", "slider_automate",
                "file_open", "file_open_multi", "file_close", "file_rewind", "file_seek", "file_avail", "file_text",
                "file_multi_count", "file_multi_select",
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
                               fn_defs: Dict[str, FunctionDef],
                               events: Optional[List[Dict[str, Any]]] = None) -> Dict[int, List[Node]]:
    user_fn_names = set(fn_defs.keys())
    out: Dict[int, List[Node]] = {}

    def visit(n: Node, section_name: str) -> None:
        if isinstance(n, Loop):
            summary = _summarize_loop_mutations(n.body, user_fn_names)
            cands: List[Node] = []
            _collect_loop_invariant_candidates(n.body, summary, user_fn_names, cands)
            if cands:
                out[n.id] = cands
                for cand in cands:
                    _record_opt_event(events, _make_opt_event(
                        "loop_hoist",
                        cand,
                        section=section_name,
                        loop_id=n.id,
                        loop_kind="loop",
                        loop_line=int(getattr(n.span, "line", 0) or 0),
                        loop_col=int(getattr(n.span, "col", 0) or 0),
                    ))
            visit(n.count, section_name)
            visit(n.body, section_name)
            return

        if isinstance(n, While):
            summary = _summarize_loop_mutations(n.cond, user_fn_names)
            summary.merge(_summarize_loop_mutations(n.body, user_fn_names))
            cands: List[Node] = []
            _collect_loop_invariant_candidates(n.cond, summary, user_fn_names, cands)
            _collect_loop_invariant_candidates(n.body, summary, user_fn_names, cands)
            if cands:
                out[n.id] = cands
                for cand in cands:
                    _record_opt_event(events, _make_opt_event(
                        "loop_hoist",
                        cand,
                        section=section_name,
                        loop_id=n.id,
                        loop_kind="while",
                        loop_line=int(getattr(n.span, "line", 0) or 0),
                        loop_col=int(getattr(n.span, "col", 0) or 0),
                    ))
            visit(n.cond, section_name)
            visit(n.body, section_name)
            return

        if isinstance(n, (Num, StrLit, Var)):
            return
        if isinstance(n, Index):
            visit(n.base, section_name)
            visit(n.index, section_name)
            return
        if isinstance(n, Unary):
            visit(n.a, section_name)
            return
        if isinstance(n, Binary):
            visit(n.l, section_name)
            visit(n.r, section_name)
            return
        if isinstance(n, Assign):
            visit(n.target, section_name)
            visit(n.value, section_name)
            return
        if isinstance(n, Call):
            for a in n.args:
                visit(a, section_name)
            return
        if isinstance(n, Ternary):
            visit(n.cond, section_name)
            visit(n.then, section_name)
            visit(n.els, section_name)
            return
        if isinstance(n, Seq):
            for it in n.items:
                visit(it, section_name)
            return
        if isinstance(n, If):
            visit(n.cond, section_name)
            visit(n.then, section_name)
            if n.els is not None:
                visit(n.els, section_name)
            return
        if isinstance(n, FunctionDef):
            visit(n.body, section_name)
            return
        raise TypeError(type(n))

    for section_name, prog in programs.items():
        for st in prog:
            visit(st, section_name)

    for fn_name, f in fn_defs.items():
        visit(f.body, f"function:{fn_name}")

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
            for a in node.args[:(3 if len(node.args) == 3 else 4)]:
                _collect_section_lvalue_rw(a, user_fn_names, out)
            return
        if fn == "midirecv_buf":
            if len(node.args) >= 1:
                _collect_section_lvalue_rw(node.args[0], user_fn_names, out)
            if len(node.args) >= 2:
                _collect_section_rw(node.args[1], user_fn_names, out)
            if len(node.args) >= 3:
                _collect_section_rw(node.args[2], user_fn_names, out)
            out.writes_mem = True
            return
        if fn == "midirecv_str":
            if len(node.args) >= 1:
                _collect_section_lvalue_rw(node.args[0], user_fn_names, out)
            if len(node.args) >= 2:
                _collect_section_lvalue_rw(node.args[1], user_fn_names, out)
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
        if fn == "msg_recv":
            if len(node.args) >= 1:
                _collect_section_rw(node.args[0], user_fn_names, out)
            for a in node.args[1:7]:
                _collect_section_lvalue_rw(a, user_fn_names, out)
            out.writes_unknown_state = True
            return
        if fn == "msg_recv_buf":
            if len(node.args) >= 1:
                _collect_section_rw(node.args[0], user_fn_names, out)
            for a in node.args[1:3]:
                _collect_section_lvalue_rw(a, user_fn_names, out)
            for a in node.args[3:]:
                _collect_section_rw(a, user_fn_names, out)
            out.writes_mem = True
            out.writes_unknown_state = True
            return
        if fn in ("sample_read2", "sample_read2_interp"):
            for a in node.args[:3]:
                _collect_section_rw(a, user_fn_names, out)
            for a in node.args[3:5]:
                _collect_section_lvalue_rw(a, user_fn_names, out)
            return
        if fn == "sample_preview_read":
            for a in node.args[:3]:
                _collect_section_rw(a, user_fn_names, out)
            for a in node.args[3:6]:
                _collect_section_lvalue_rw(a, user_fn_names, out)
            return
        if fn == "sample_name":
            for a in node.args[:2]:
                _collect_section_rw(a, user_fn_names, out)
            if len(node.args) >= 3:
                _collect_section_lvalue_rw(node.args[2], user_fn_names, out)
            out.writes_unknown_state = True
            return
        if fn in SAMPLE_POOL_EXPORT_FUNCTIONS:
            for a in node.args:
                _collect_section_rw(a, user_fn_names, out)
            out.writes_mem = True
            out.writes_unknown_state = True
            return
        if fn in ("instance_uid", "instance_get_name", "track_name", "host_track_name"):
            for a in node.args[:1]:
                _collect_section_lvalue_rw(a, user_fn_names, out)
            out.writes_unknown_state = True
            return
        if fn in ("msg_peer_name", "msg_peer_uid"):
            if len(node.args) >= 1:
                _collect_section_rw(node.args[0], user_fn_names, out)
            if len(node.args) >= 2:
                _collect_section_lvalue_rw(node.args[1], user_fn_names, out)
            out.writes_unknown_state = True
            return

        for a in node.args:
            _collect_section_rw(a, user_fn_names, out)

        if fn in user_fn_names:
            out.writes_unknown_state = True
            return

        if fn in COMM_IMPURE_FUNCTIONS:
            out.writes_unknown_state = True
            return

        if fn in ("memset", "memcpy", "fft", "ifft", "fft_real", "ifft_real", "fft_permute", "fft_ipermute", "convolve_c", "file_mem"):
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


def _plan_sample_to_block_hoists(programs: Dict[str, List[Node]],
                                 user_fn_names: Set[str],
                                 events: Optional[List[Dict[str, Any]]] = None) -> None:
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
            _record_opt_event(events, _make_opt_event(
                "section_hoist",
                node,
                from_section="sample",
                to_section="block",
                target=target_name,
            ))
            active_sample_info.write_counts = _adjusted_write_counts(active_sample_info.write_counts, target_name)
            continue

        new_sample.append(node)
        prefix_reads.update(info.read_vars)
        prefix_writes.update(info.written_vars)

    if moved:
        programs["sample"] = new_sample
        programs["block"] = list(programs["block"]) + moved


def _plan_block_to_init_hoists(programs: Dict[str, List[Node]],
                              user_fn_names: Set[str],
                              events: Optional[List[Dict[str, Any]]] = None) -> None:
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
            _record_opt_event(events, _make_opt_event(
                "section_hoist",
                node,
                from_section="block",
                to_section="init",
                target=target_name,
            ))
            active_block_info.write_counts = _adjusted_write_counts(active_block_info.write_counts, target_name)
            continue

        new_block.append(node)
        prefix_reads.update(info.read_vars)
        prefix_writes.update(info.written_vars)

    if moved:
        programs["block"] = new_block
        programs["init"] = list(programs["init"]) + moved


def hoist_section_invariants(programs: Dict[str, List[Node]],
                             fn_defs: Dict[str, FunctionDef],
                             events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Node]]:
    out = {sec: list(nodes) for sec, nodes in programs.items()}
    user_fn_names = set(fn_defs.keys())

    _plan_sample_to_block_hoists(out, user_fn_names, events=events)
    _plan_block_to_init_hoists(out, user_fn_names, events=events)
    return out


# -----------------------------
# LLVM IR emission (llvmlite)
# -----------------------------

class _CompactFunction(ir.Function):
    """Avoid quadratic label growth from nested llvmlite if_then blocks."""
    def append_basic_block(self, name=''):
        # if_then derives its next label from the current label. Repeated
        # bounds/store guards otherwise build names thousands of bytes long.
        label = 'entry' if not self.blocks else f'b{len(self.blocks)}'
        return super().append_basic_block(label)


class _SerializedFunctionModule(ir.Module):
    """CLI-only IR container: retain emitted text instead of instruction graphs."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.serialized_functions = {}

    def _get_body_lines(self):
        lines = [it.get_declaration() for it in self.get_identified_types().values()]
        lines.extend(self.serialized_functions[v.name] if v.name in self.serialized_functions
                     else str(v) for v in self.globals.values())
        return lines


class LLVMModuleEmitter:
    def __init__(self, sym: SymTable, *, native_gfx_legacy: bool = False, stream_functions: bool = False, has_tasks: bool = False, has_faust: bool = False, state_var_capacity: int = 0):
        self.has_tasks = has_tasks
        self.native_gfx_legacy = native_gfx_legacy
        self.sym = sym

        self.double = ir.DoubleType()
        self.float = ir.FloatType()
        self.i1 = ir.IntType(1)
        self.i8 = ir.IntType(8)
        self.i32 = ir.IntType(32)
        self.i64 = ir.IntType(64)

        # State layout:
        # 0: double spl[64]
        # 1: double sliders[256]
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
        # 23: uint64_t pendingSliderChangeMask[4]
        # 24: uint64_t pendingSliderAutomateMask[4]
        # 25: uint64_t pendingSliderAutomateEndMask[4]
        # 26: uint32_t randMT[624]
        # 27: uint32_t randIndex (0 means uninitialized; mirrors EEL2 __idx)
        # 28: uint64_t sliderVisibleMask[4] (bitset, 1=visible)
        # 29: int32_t sliderVisibilityInit
        # 30: void* runtimeOpaque
        # 31: double midi_bus
        # 32: double ext_midi_bus
        # 33: void* hostOwner (instance-bound callbacks; never a global map)
        # 34: int64_t memUsed (audio-owned heap high-water mark)
        # 35: int32_t memoryFault (latched until state reset)
        self.var_cap = max(1, state_var_capacity, (max(sym.vars.values()) + 1) if sym.vars else 1)
        self.midi_event_ty = ir.LiteralStructType([self.i32, self.i32, self.i32, self.i32])

        self.state_ty = ir.LiteralStructType([
            ir.ArrayType(self.double, 64),
            ir.ArrayType(self.double, MAX_JSFX_SLIDERS),
            self.double.as_pointer() if native_gfx_legacy else ir.ArrayType(self.double, self.var_cap),
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
            ir.ArrayType(self.i64, SLIDER_MASK_WORDS),
            ir.ArrayType(self.i64, SLIDER_MASK_WORDS),
            ir.ArrayType(self.i64, SLIDER_MASK_WORDS),
            ir.ArrayType(self.i32, 624),
            self.i32,
            ir.ArrayType(self.i64, SLIDER_MASK_WORDS),
            self.i32,
            self.i8.as_pointer(),
            self.double,
            self.double,
            self.i8.as_pointer(),
            self.i64,
            self.i32,
        ])
        self.state_ptr = self.state_ty.as_pointer()
        if native_gfx_legacy:
            self.state_ty = ir.LiteralStructType(list(self.state_ty.elements) +
                                                [self.i8.as_pointer(), self.i8.as_pointer(), self.i8.as_pointer()])
            self.state_ptr = self.state_ty.as_pointer()

        if has_tasks or native_gfx_legacy:
            self.state_ty = ir.LiteralStructType(list(self.state_ty.elements) + [self.i8.as_pointer()])
            self.state_ptr = self.state_ty.as_pointer()

        if has_faust or native_gfx_legacy:
            self.state_ty = ir.LiteralStructType(list(self.state_ty.elements) + [self.i8.as_pointer()])
            self.state_ptr = self.state_ty.as_pointer()
        if native_gfx_legacy:
            self.state_ty = ir.LiteralStructType(list(self.state_ty.elements) + [self.i64])
            self.state_ptr = self.state_ty.as_pointer()

        self.module = (_SerializedFunctionModule if stream_functions else ir.Module)(name="dsp_jsfx_module", context=ir.Context())
        if native_gfx_legacy:
            # Keep the ABI identical while naming the large aggregate once.
            # Repeating its literal body on every reference inflated the IR
            # of complex instruments by hundreds of megabytes.
            fields = tuple(self.state_ty.elements)
            self.state_ty = self.module.context.get_identified_type("DSPJSFX_State")
            self.state_ty.set_body(*fields)
            self.state_ptr = self.state_ty.as_pointer()

        # extern ensure
        self.fn_ensure = self._function(
            self.module,
            ir.FunctionType(ir.VoidType(), [self.state_ptr, self.i64]),
            name="jsfx_ensure_mem"
        )

        self.fn_midirecv = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer()]),
            name="jsfx_midirecv"
        )
        self.fn_midirecv_msg23 = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer()]),
            name="jsfx_midirecv_msg23"
        )
        self.fn_midirecv_buf = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double.as_pointer(), self.double, self.double]),
            name="jsfx_midirecv_buf"
        )
        self.fn_midirecv_str = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double.as_pointer(), self.double.as_pointer()]),
            name="jsfx_midirecv_str"
        )
        self.fn_midisend = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double, self.double]),
            name="jsfx_midisend"
        )
        self.fn_midisend_msg23 = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double]),
            name="jsfx_midisend_msg23"
        )
        self.fn_midisend_buf = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double]),
            name="jsfx_midisend_buf"
        )
        self.fn_midisend_str = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double]),
            name="jsfx_midisend_str"
        )
        self.fn_midisyx = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double]),
            name="jsfx_midisyx"
        )
        self.fn_instance_id = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr]),
            name="jsfx_instance_id"
        )
        self.fn_instance_uid = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double.as_pointer()]),
            name="jsfx_instance_uid"
        )
        self.fn_instance_set_name = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double]),
            name="jsfx_instance_set_name"
        )
        self.fn_instance_get_name = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double.as_pointer()]),
            name="jsfx_instance_get_name"
        )
        self.fn_track_name = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double.as_pointer()]),
            name="jsfx_track_name"
        )
        self.fn_track_name_available = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr]),
            name="jsfx_track_name_available"
        )
        self.fn_track_name_seq = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr]),
            name="jsfx_track_name_seq"
        )
        self.fn_comm_join = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double]),
            name="jsfx_comm_join"
        )
        self.fn_gmem_attach = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double]),
            name="jsfx_gmem_attach"
        )
        self.fn_gmem_attach_size = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double]),
            name="jsfx_gmem_attach_size"
        )
        self.fn_gmem_size = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr]),
            name="jsfx_gmem_size"
        )
        self.fn_gmem_load = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double]),
            name="jsfx_gmem_load"
        )
        self.fn_gmem_store = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double, self.double]),
            name="jsfx_gmem_store"
        )
        self.fn_gmem_get = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double]),
            name="jsfx_gmem_get"
        )
        self.fn_gmem_put = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double]),
            name="jsfx_gmem_put"
        )
        self.fn_gmem_fill = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double]),
            name="jsfx_gmem_fill"
        )
        self.fn_gmem_zero = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double]),
            name="jsfx_gmem_zero"
        )
        self.fn_gmem_copy = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double]),
            name="jsfx_gmem_copy"
        )
        self.fn_gmem_seq = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double]),
            name="jsfx_gmem_seq"
        )
        self.fn_gmem_page = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double]),
            name="jsfx_gmem_page"
        )
        self.fn_msg_subscribe = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double]),
            name="jsfx_msg_subscribe"
        )
        self.fn_msg_unsubscribe = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double]),
            name="jsfx_msg_unsubscribe"
        )
        self.fn_msg_advertise = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double]),
            name="jsfx_msg_advertise"
        )
        self.fn_msg_send = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double, self.double, self.double, self.double]),
            name="jsfx_msg_send"
        )
        self.fn_msg_sendto = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double, self.double, self.double, self.double, self.double]),
            name="jsfx_msg_sendto"
        )
        self.fn_msg_avail = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double]),
            name="jsfx_msg_avail"
        )
        self.fn_msg_kind = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double]),
            name="jsfx_msg_kind"
        )
        self.fn_msg_recv = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer()]),
            name="jsfx_msg_recv"
        )
        self.fn_msg_send_buf = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double, self.double]),
            name="jsfx_msg_send_buf"
        )
        self.fn_msg_sendto_buf = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double, self.double, self.double, self.double]),
            name="jsfx_msg_sendto_buf"
        )
        self.fn_msg_recv_buf = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double.as_pointer(), self.double.as_pointer(), self.double, self.double]),
            name="jsfx_msg_recv_buf"
        )
        self.fn_msg_length = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr]),
            name="jsfx_msg_length"
        )
        self.fn_msg_dropped = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double]),
            name="jsfx_msg_dropped"
        )
        self.fn_msg_clear = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double]),
            name="jsfx_msg_clear"
        )
        self.fn_msg_peer_count = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double, self.double]),
            name="jsfx_msg_peer_count"
        )
        self.fn_msg_peer_id = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double, self.double, self.double]),
            name="jsfx_msg_peer_id"
        )
        self.fn_msg_peer_name = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double.as_pointer()]),
            name="jsfx_msg_peer_name"
        )
        self.fn_msg_peer_uid = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double.as_pointer()]),
            name="jsfx_msg_peer_uid"
        )
        self.fn_msg_peer_caps = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double]),
            name="jsfx_msg_peer_caps"
        )
        self.fn_msg_peer_alive = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double]),
            name="jsfx_msg_peer_alive"
        )
        self.fn_strlen = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double]),
            name="jsfx_strlen"
        )
        self.fn_str_getchar = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.double]),
            name="jsfx_str_getchar"
        )
        self.fn_sliderchange = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.i32]),
            name="jsfx_sliderchange"
        )
        self.fn_slider_automate = self._function(
            self.module,
            ir.FunctionType(self.i32, [self.state_ptr, self.double, self.i32, self.double]),
            name="jsfx_slider_automate"
        )
        self.fn_slider_show = self._function(
            self.module,
            ir.FunctionType(self.double, [self.state_ptr, self.double, self.i32, self.double, self.i32]),
            name="jsfx_slider_show"
        )

        self._intrinsics: Dict[str, ir.Function] = {}
        self.eel2_store_filter = False
        self._rand_gen32_fn: Optional[ir.Function] = None
        self.user_fn_defs: Dict[str, FunctionDef] = {}
        self.user_fn_ir: Dict[str, ir.Function] = {}
        self.loop_hoists: Dict[int, List[Node]] = {}
        self._local_slots_stack: List[Dict[str, ir.Value]] = []
        self._hoisted_value_stack: List[Dict[int, ir.Value]] = []


        self._buildins: Dict[str, ir.Function] = {}
        self.native_gfx_prototype = False
        self.native_gfx_functions: Set[str] = set()
        self._named_gfx_strings: Dict[str, int] = {}
        self._emitting_gfx = False
        self._state_address_cache = {}
        self._script_state_cache = {}

        # String literal pool (DSP-only):
        # We represent string literals as opaque numeric handles (doubles).
        # This keeps parsing/compilation working for scripts that include string
        # helpers (often used by @gfx) inside @init/@slider.
        # Full EEL2 string semantics are NOT implemented in this DSP compiler.
        self._str_lits: Dict[str, int] = {}
        self._next_str_id: int = 0
        self._str_handle_base: int = (1 << 40)
        self.jsfx_memtop_slots: int = JSFX_DEFAULT_MEMTOP_SLOTS

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

    def get_string_literals_meta(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for s, handle in sorted(self._str_lits.items(), key=lambda kv: kv[1]):
            items.append({"handle": int(handle), "text": s})
        return items

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

    def _function(self, module, function_type, **kwargs):
        constructor = _CompactFunction if self.native_gfx_legacy else ir.Function
        return constructor(module, function_type, **kwargs)

    def _compute_loop_hoisted_values(self, builder: ir.IRBuilder, st: ir.Value, loop_id: int) -> Dict[int, ir.Value]:
        out: Dict[int, ir.Value] = {}
        for node in self.loop_hoists.get(loop_id, []):
            out[node.id] = self.emit_expr(builder, st, node)
        return out

    def _truthy(self, x: ir.Value, builder: ir.IRBuilder) -> ir.Value:
        magnitude = builder.call(self._declare_math("fabs"), [x])
        return builder.fcmp_ordered(">=", magnitude, self._const_f64(1.0e-5))

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

            f = self._function(self.module, ir.FunctionType(self.double, [self.double]), name=name)
            self._intrinsics[fn] = f
            return f

        if fn in ("asin", "acos", "atan", "exp", "log", "tan", "log10"):
            f = self._function(self.module, ir.FunctionType(self.double, [self.double]), name=fn)
            self._intrinsics[fn] = f
            return f

        if fn in ("pow", "atan2"):
            f = self._function(self.module, ir.FunctionType(self.double, [self.double, self.double]), name=fn)
            self._intrinsics[fn] = f
            return f

        raise ValueError(f"Unknown builtin {fn}")

    def _get_slot_ptr(self, builder: ir.IRBuilder, st: ir.Value, name: str) -> ir.Value:
        # Local variables (function params/locals) shadow globals
        if self._local_slots_stack:
            loc = self._local_slots_stack[-1].get(name)
            if loc is not None:
                return loc

        if self.native_gfx_legacy:
            key = (builder.function, st, name)
            cached = self._state_address_cache.get(key)
            if cached is not None:
                return cached
            # Fixed LEGACY state addresses cannot change during a guest call.
            # Reuse addresses, never values: every read/write remains atomic.
            # This bounds frontend IR growth for thousands of receiver helpers.
            with self._at_function_entry(builder):
                pointer = self._state_slot_address(builder, st, name)
            self._state_address_cache[key] = pointer
            return pointer
        return self._state_slot_address(builder, st, name)

    def _state_slot_address(self, builder, st, name):
        st = self._script_state(builder, st)

        ref = self.sym.resolve(name)
        zero = ir.Constant(self.i32, 0)

        if ref.kind == "builtin":
            if ref.index == 0:  # mem constant
                raise ValueError("mem has no address")
            builtin_field_map = {
                1: 5,   # srate
                2: 6,   # samplesblock
                3: 31,  # midi_bus
                4: 32,  # ext_midi_bus
            }
            field = builtin_field_map.get(ref.index)
            if field is None:
                raise ValueError(f"Unsupported builtin slot index {ref.index}")
            fld = ir.Constant(self.i32, field)
            return builder.gep(st, [zero, fld], inbounds=True)

        field = {"spl": 0, "slider": 1, "var": 2}[ref.kind]
        fld = ir.Constant(self.i32, field)
        idx = ir.Constant(self.i32, ref.index)
        if ref.kind == "var" and self.native_gfx_legacy:
            cells = builder.load(builder.gep(st, [zero, fld], inbounds=True))
            return builder.gep(cells, [idx], inbounds=True)
        return builder.gep(st, [zero, fld, idx], inbounds=True)

    def _script_state(self, builder, st):
        if not self.native_gfx_legacy or not self._emitting_gfx:
            return st
        key = (builder.function, st)
        cached = self._script_state_cache.get(key)
        if cached is not None:
            return cached
        with self._at_function_entry(builder):
            raw = builder.load(builder.gep(st, [self._const_i32(0), self._const_i32(36)]))
            owner = builder.bitcast(raw, self.state_ptr)
            root = builder.select(builder.icmp_unsigned("==", raw, ir.Constant(self.i8.as_pointer(), None)), st, owner)
        self._script_state_cache[key] = root
        return root

    def _load_cell(self, builder, ptr):
        return builder.load_atomic(ptr, "monotonic", 8) if self.native_gfx_legacy and not isinstance(ptr, ir.AllocaInstr) else builder.load(ptr)

    def _store_cell(self, builder, value, ptr):
        return builder.store_atomic(value, ptr, "monotonic", 8) if self.native_gfx_legacy and not isinstance(ptr, ir.AllocaInstr) else builder.store(value, ptr)


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
            field = 1  # sliders[256]
        elif which == "spl":
            # spl(0) -> spl0 -> index 0
            idx0_i64 = idx_i64
            field = 0  # spl[64]
        else:
            raise ValueError(f"Unknown dynamic state array {which!r}")

        zero_i64 = self._const_i64(0)
        limit = MAX_JSFX_SLIDERS if which == "slider" else 64
        limit_i64 = self._const_i64(limit)

        ge0 = builder.icmp_signed(">=", idx0_i64, zero_i64)
        lt_limit = builder.icmp_signed("<", idx0_i64, limit_i64)
        in_range = builder.and_(ge0, lt_limit)

        # Clamp so GEP is always in-bounds even when out-of-range; we still use
        # 'in_range' to return 0 / ignore writes.
        idx_clamped = idx0_i64
        isneg = builder.icmp_signed("<", idx_clamped, zero_i64)
        idx_clamped = builder.select(isneg, zero_i64, idx_clamped)

        max_index_i64 = self._const_i64(limit - 1)
        isgt = builder.icmp_signed(">", idx_clamped, max_index_i64)
        idx_clamped = builder.select(isgt, max_index_i64, idx_clamped)

        idx_i32 = builder.trunc(idx_clamped, self.i32)

        z = ir.Constant(self.i32, 0)
        fld = ir.Constant(self.i32, field)
        ptr = builder.gep(self._script_state(builder, st), [z, fld, idx_i32], inbounds=True)
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

    def _ensure_rand_gen32_fn(self) -> ir.Function:
        if self._rand_gen32_fn is not None:
            return self._rand_gen32_fn

        N = 624
        M = 397
        MATRIX_A = 0x9908B0DF
        UPPER_MASK = 0x80000000
        LOWER_MASK = 0x7FFFFFFF

        fn = self._function(self.module, ir.FunctionType(self.i32, [self.state_ptr]), name="jsfx_rand_genrand_int32")
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

    def _memory_failure_return(self, builder, st):
        # Every generated entry/user function exits before an invalid access.
        # The host silences a faulted block; failed growth cannot fall through.
        fault = builder.gep(st, [self._const_i32(0), self._const_i32(35)])
        builder.store(self._const_i32(1), fault)
        ret_ty = builder.function.function_type.return_type
        if isinstance(ret_ty, ir.VoidType):
            builder.ret_void()
        else:
            builder.ret(ir.Constant(ret_ty, 0))

    def _checked_nonnegative_index(self, builder, st, value):
        # Clamp negative indices as before, but never feed NaN/Inf/2^63 to fptosi.
        valid = builder.fcmp_ordered("<", value, self._const_f64(9223372036854775808.0))
        with builder.if_then(builder.not_(valid)):
            self._memory_failure_return(builder, st)
        nonnegative = builder.select(builder.fcmp_ordered(">", value, self._const_f64(0.0)),
                                     value, self._const_f64(0.0))
        return builder.fptosi(nonnegative, self.i64)

    def _mem_address(self, builder, st, base_expr, idx_expr):
        base_v = self.emit_expr(builder, st, base_expr)
        idx_v = self.emit_expr(builder, st, idx_expr)
        summed = builder.fadd(builder.fadd(base_v, idx_v), self._const_f64(1.0e-5))
        return self._checked_nonnegative_index(builder, st, summed)

    def _ensure_mem_end(self, builder, st, end):
        with builder.if_then(builder.icmp_signed(">", end, self._get_memN(builder, st))):
            builder.call(self.fn_ensure, [st, end])
            failed = builder.or_(
                builder.icmp_signed(">", end, self._get_memN(builder, st)),
                builder.icmp_unsigned("==", self._get_mem_ptr(builder, st),
                                      ir.Constant(self.double.as_pointer(), None)))
            with builder.if_then(failed):
                self._memory_failure_return(builder, st)

    def _mem_elem_ptr(self, builder, st_ptr, base_expr, idx_expr):
        addr = self._mem_address(builder, st_ptr, base_expr, idx_expr)
        self._ensure_mem_end(builder, st_ptr, builder.add(addr, self._const_i64(1)))
        return builder.gep(self._get_mem_ptr(builder, st_ptr), [addr], inbounds=False)

    @contextlib.contextmanager
    def _at_function_entry(self, builder):
        # Nested insertion must preserve the position before an existing
        # terminator. llvmlite.goto_block restores *after* the terminator.
        old = builder.block
        entry = builder.function.entry_basic_block
        if entry.terminator is None:
            builder.position_at_end(entry)
        else:
            builder.position_before(entry.terminator)
        try:
            yield
        finally:
            if old.terminator is None:
                builder.position_at_end(old)
            else:
                builder.position_before(old.terminator)

    def _entry_alloca(self, builder, typ, **kwargs):
        # Host argument references escape to C++ callbacks. Allocate their
        # fixed storage once per invocation, not once per loop iteration.
        # Otherwise large drawing loops grow the stack until they overflow.
        with self._at_function_entry(builder):
            return builder.alloca(typ, **kwargs)

    def _call_with_outputs(self, builder, st, args, callee, outputs, api_name,
                           allow_null=False, input_refs=frozenset()):
        # Evaluate each lvalue address exactly once, left-to-right. Do not retain
        # a heap pointer while evaluating later operands (which may realloc).
        values = []
        for index, node in enumerate(args):
            if index not in outputs and index not in input_refs:
                values.append((False, self.emit_expr(builder, st, node)))
            elif isinstance(node, Var) and node.name != "gmem" and (node.name != "mem" or self.native_gfx_legacy) and not node.name.startswith(("$", "#")):
                values.append((False, self._get_slot_ptr(builder, st, node.name)))
            elif isinstance(node, Index) and not self._is_gmem_index(node):
                addr = self._mem_address(builder, st, node.base, node.index)
                self._ensure_mem_end(builder, st, builder.add(addr, self._const_i64(1)))
                values.append((True, addr))
            elif isinstance(node, Call) and node.fn in ("slider", "spl") and len(node.args) == 1:
                ptr, valid = self._dyn_state_array_ptr(builder, st, node.fn, node.args[0])
                # An invalid reference must not alias a clamped real slider.
                # It gets a private zero cell whose writes are discarded.
                dummy = self._entry_alloca(builder, self.double); dummy.align = 8
                builder.store(self._const_f64(0), dummy)
                values.append((False, builder.select(valid, ptr, dummy)))
            elif index in input_refs:
                value = self.emit_expr(builder, st, node)
                temporary = self._entry_alloca(builder, self.double); temporary.align = 8
                builder.store(value, temporary)
                values.append((False, temporary))
            elif allow_null:
                self.emit_expr(builder, st, node)
                values.append((False, ir.Constant(self.double.as_pointer(), None)))
            else:
                raise ValueError(f"{api_name} output arguments must be assignable variables or mem[] slots")
        argv = [st]
        for is_heap, value in values:
            argv.append(builder.gep(self._get_mem_ptr(builder, st), [value], inbounds=False)
                        if is_heap else value)
        return builder.call(callee, argv)

    def _to_i32(self, builder, x):
        # JSFX-style: truncate toward 0 to *some* integer, then wrap to 32-bit
        xi64 = builder.fptosi(x, self.i64)        # safe for your magnitudes
        return builder.trunc(xi64, self.i32)      # wraps mod 2^32


    def _to_f64(self, builder, x_i32):
        return builder.sitofp(x_i32, self.double)

    def _is_gmem_index(self, node: Node) -> bool:
        return isinstance(node, Index) and isinstance(node.base, Var) and node.base.name == "gmem"

    def _get_out_lvalue_ptr(self, builder: ir.IRBuilder, st: ir.Value, node: Node, api_name: str) -> ir.Value:
        if isinstance(node, Var):
            if self.native_gfx_legacy and node.name.startswith("#") and api_name in {"instance_uid", "instance_get_name", "track_name", "msg_peer_name", "msg_peer_uid", "sample_name", "midirecv_str"}:
                # Native named strings are stable handles, not scalar slots.
                # The runtime writes their existing string context in place.
                handle = self.emit_expr(builder, st, node)
                pointer = self._entry_alloca(builder, self.double, name="string_output")
                builder.store(handle, pointer)
                return pointer
            if node.name == "gmem" or node.name == "mem" and not self.native_gfx_legacy:
                raise ValueError(f"{api_name} output arguments must be assignable variables or mem[] slots")
            return self._get_slot_ptr(builder, st, node.name)
        if isinstance(node, Index):
            if self._is_gmem_index(node):
                raise ValueError(f"{api_name} output arguments must be assignable variables or mem[] slots")
            return self._mem_elem_ptr(builder, st, node.base, node.index)
        raise ValueError(f"{api_name} output arguments must be assignable variables or mem[] slots")

    def _get_midirecv_lvalue_ptr(self, builder: ir.IRBuilder, st: ir.Value, node: Node) -> ir.Value:
        return self._get_out_lvalue_ptr(builder, st, node, "midirecv")

    def _emit_slider_mask_selector(self, builder: ir.IRBuilder, st: ir.Value, arg: Node) -> Tuple[ir.Value, ir.Value]:
        """Return (legacy_mask_f64, direct_slider_index_i32).

        Numeric bitmasks remain the legacy low-64-slider form because an EEL
        double cannot carry 256 independent bits. Direct sliderN references are
        losslessly identified by index and therefore work for slider1..slider256.
        """
        target = arg
        while isinstance(target, Seq) and target.items:
            target = target.items[-1]
        value = None
        if isinstance(arg, (Seq, Assign)):
            value = self.emit_expr(builder, st, arg)
        if isinstance(target, Assign):
            target = target.target
        if isinstance(target, Var):
            if target.name in self.sym.slider_aliases:
                return self._const_f64(0.0), self._const_i32(self.sym.slider_aliases[target.name])
            m = re.fullmatch(r"slider([1-9][0-9]{0,2})", target.name)
            if m is not None:
                idx1 = int(m.group(1))
                if 1 <= idx1 <= MAX_JSFX_SLIDERS:
                    return self._const_f64(0.0), self._const_i32(idx1 - 1)
        return value if value is not None else self.emit_expr(builder, st, arg), self._const_i32(-1)

    
    def declare_user_functions(self, fn_defs: Dict[str, FunctionDef]) -> None:
        self.user_fn_defs = dict(fn_defs)
        # Signature: double fn(DSPJSFX_State* st, double a0, double a1, ...)
        for name, fdef in self.user_fn_defs.items():
            arg_types = [self.state_ptr] + [self.double] * len(fdef.params) + [self.double.as_pointer()] * len(fdef.cell_params)
            fnty = ir.FunctionType(self.double, arg_types)
            self.user_fn_ir[name] = self._function(self.module, fnty, name=f"jsfx_fn_{name}")
            if fdef.cell_params or self.native_gfx_legacy and name in self.native_gfx_functions:
                # Keep shared large math bodies shared. UI helpers retain
                # their existing no-inline policy; other audio helpers remain
                # eligible for normal inlining.
                self.user_fn_ir[name].attributes.add("noinline")

    def _task_entry_alloca(self, builder, typ):
        with builder.goto_entry_block():
            return builder.alloca(typ)

    def emit_user_functions(self) -> None:
        for name, fdef in self.user_fn_defs.items():
            self._emitting_gfx = name in self.native_gfx_functions
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
            for i, cell_name in enumerate(fdef.cell_params):
                locals_map[cell_name] = fn.args[1 + len(fdef.params) + i]

            # locals
            for l in fdef.locals:
                if l in locals_map:
                    continue
                slot = b.alloca(self.double, name=f"l_{l}")
                b.store(self._const_f64(0.0), slot)
                locals_map[l] = slot

            self._local_slots_stack.append(locals_map)
            if self.has_tasks:
                tasks_compiler.checkpoint(sys.modules[__name__], self, b, st)
            retv = self.emit_expr(b, st, fdef.body)
            self._local_slots_stack.pop()

            b.ret(retv)
            if isinstance(self.module, _SerializedFunctionModule):
                self.module.serialized_functions[fn.name] = str(fn)
                fn.blocks.clear()
                fn._clear_string_cache()
                self._state_address_cache = {key: value for key, value in self._state_address_cache.items() if key[0] is not fn}
                self._script_state_cache = {key: value for key, value in self._script_state_cache.items() if key[0] is not fn}
                # llvmlite instructions form parent/operand cycles. Collect
                # periodically instead of retaining every receiver graph.
                if len(self.module.serialized_functions) % 128 == 0:
                    import gc
                    gc.collect()
        self._emitting_gfx = False


    def emit_section_fn(self, name: str, prog: List[Node]) -> ir.Function:
        fn = self._function(self.module, ir.FunctionType(ir.VoidType(), [self.state_ptr]), name=name)
        entry = fn.append_basic_block("entry")
        builder = ir.IRBuilder(entry)
        st = fn.args[0]

        previous_gfx = self._emitting_gfx
        self._emitting_gfx = name == "jsfx_gfx_aot" and self.native_gfx_prototype
        try:
            for st_node in prog:
                self.emit_stmt(builder, st, st_node)
        finally:
            self._emitting_gfx = previous_gfx

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

    def emit_while(self, builder: ir.IRBuilder, st: ir.Value, n: While) -> ir.Value:
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
            counter = builder.phi(self.i32, name="while_iterations")
            counter.add_incoming(self._const_i32(0), pre_bb)
            if self.has_tasks:
                tasks_compiler.checkpoint(sys.modules[__name__], self, builder, st)
            condv = self.emit_expr(builder, st, n.cond)
            cond = self._truthy(condv, builder)
            cond_end = builder.block
            builder.cbranch(cond, body_bb, after_bb)
            builder.position_at_end(body_bb)
            self.emit_stmt(builder, st, n.body)
            latch = builder.block
            has_latch = not latch.is_terminated
            if has_latch:
                next_count = builder.add(counter, self._const_i32(1))
                counter.add_incoming(next_count, latch)
                # WDL's per-loop budget is independent for nested loops. The
                # final condition is not evaluated an extra time at the cap.
                builder.cbranch(builder.icmp_unsigned("<", next_count, self._const_i32(1048576)), cond_bb, after_bb)
            builder.position_at_end(after_bb)
            result = builder.phi(self.double)
            result.add_incoming(condv, cond_end)
            if has_latch:
                result.add_incoming(condv, latch)
            return result
        finally:
            self._hoisted_value_stack.pop()

    def _filter_assignment(self, builder: ir.IRBuilder, value: ir.Value) -> ir.Value:
        if not self.eel2_store_filter:
            return value
        # WDL denormal_filter_double2: exponent 0 (subnormal/zero) and
        # exponent 2047 (NaN/infinity) become +0 on a checked store.
        bits = builder.bitcast(value, self.i64)
        exponent = builder.and_(builder.lshr(bits, ir.Constant(self.i64, 52)), ir.Constant(self.i64, 2047))
        normal = builder.icmp_unsigned("<", builder.sub(exponent, ir.Constant(self.i64, 1)), ir.Constant(self.i64, 2046))
        return builder.select(normal, value, self._const_f64(0.0))

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
            if self.has_tasks and n.name in tasks_compiler.CONSTANTS:
                return self._const_f64(tasks_compiler.CONSTANTS[n.name])
            if (self._emitting_gfx or self.native_gfx_legacy) and n.name.startswith("#"):
                name = n.name if n.name != "#" else f"#anonymous_{len(self._named_gfx_strings)}"
                handle = self._named_gfx_strings.setdefault(name, (1 << 39) + len(self._named_gfx_strings))
                return self._const_f64(handle)
            if n.name == "mem" and not self.native_gfx_legacy:
                return self._const_f64(0.0)
            if n.name == "gmem":
                raise ValueError("gmem may only be used as gmem[index]")

            # common JSFX constants (expand if needed)
            if n.name == "$pi":
                return self._const_f64(math.pi)
            if n.name == "$phi":
                return self._const_f64((1.0 + math.sqrt(5.0)) * 0.5)
            if n.name == "$e":
                return self._const_f64(math.e)
            if n.name.startswith("$x") and len(n.name) > 2:
                try:
                    return self._const_f64(float(int(n.name[2:], 16)))
                except ValueError:
                    pass

            ptr = self._get_slot_ptr(builder, st, n.name)  # handles locals + globals + srate/samplesblock
            return self._load_cell(builder, ptr)


        # indexing
        if isinstance(n, Index):
            if self._emitting_gfx and not self.native_gfx_legacy:
                addr = self._mem_address(builder, st, n.base, n.index)
                symbol = "jsfx_native_read_mem"
                fdecl = self._buildins.get(symbol)
                if fdecl is None:
                    fdecl = self._function(self.module, ir.FunctionType(self.double, [self.state_ptr, self.i64]), name=symbol)
                    self._buildins[symbol] = fdecl
                return builder.call(fdecl, [st, addr])
            if self._is_gmem_index(n):
                idx = self.emit_expr(builder, st, n.index)
                return builder.call(self.fn_gmem_load, [st, idx])
            # a[b] == mem[(int)a + (int)b]; mem itself is base 0.
            ptr = self._mem_elem_ptr(builder, st, n.base, n.index)
            return self._load_cell(builder, ptr)

        # unary
        if isinstance(n, Unary):
            a = self.emit_expr(builder, st, n.a)
            if n.op == "+":
                return a
            if n.op == "-":
                return builder.fsub(self._const_f64(0.0), a)
            if n.op == "!":
                isz = builder.not_(self._truthy(a, builder))
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


            if n.op in ("<", "<=", ">", ">="):
                opmap = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge"}
                c = builder.fcmp_ordered(opmap[n.op], l, r)
                return builder.select(c, self._const_f64(1.0), self._const_f64(0.0))

            # WDL/EEL2 deliberately gives == / != fuzzy numeric semantics,
            # using NSEEL_CLOSEFACTOR (1e-5), while === / !== are exact
            # floating-point comparisons. Keep that distinction here rather
            # than treating all four operators as C-style equality.
            if n.op in ("==", "!="):
                diff = builder.fsub(l, r)
                fabs_decl = self._declare_math("fabs")
                absdiff = builder.call(fabs_decl, [diff])
                if n.op == "==":
                    c = builder.fcmp_ordered("<", absdiff, self._const_f64(1.0e-5))
                else:
                    c = builder.fcmp_ordered(">=", absdiff, self._const_f64(1.0e-5))
                return builder.select(c, self._const_f64(1.0), self._const_f64(0.0))

            if n.op in ("===", "!=="):
                if n.op == "===":
                    c = builder.fcmp_ordered("==", l, r)
                else:
                    # C/WDL exact != is true for unordered (NaN) comparisons.
                    c = builder.fcmp_unordered("!=", l, r)
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
                if n.target.name == "mem" and not self.native_gfx_legacy:
                    raise ValueError("Cannot assign to mem")
                ptr = self._get_slot_ptr(builder, st, n.target.name)  # works for locals too
                in_range = ir.Constant(self.i1, 1)

            elif isinstance(n.target, Index) and self._is_gmem_index(n.target):
                idx = self.emit_expr(builder, st, n.target.index)
                cur = builder.call(self.fn_gmem_load, [st, idx]) if n.op != "=" else None
                if n.op == "=":
                    out = rhs
                elif n.op == "+=":
                    out = builder.fadd(cur, rhs)
                elif n.op == "-=":
                    out = builder.fsub(cur, rhs)
                elif n.op == "*=":
                    out = builder.fmul(cur, rhs)
                elif n.op == "/=":
                    out = builder.fdiv(cur, rhs)
                elif n.op == "%=":
                    li = self._to_i32(builder, cur)
                    ri = self._to_i32(builder, rhs)
                    out = self._to_f64(builder, builder.srem(li, ri))
                elif n.op == "^=":
                    fdecl = self._declare_math("pow")
                    out = builder.call(fdecl, [cur, rhs])
                elif n.op in ("|=", "&=", "~="):
                    li = self._to_i32(builder, cur)
                    ri = self._to_i32(builder, rhs)
                    if n.op == "|=":
                        oi = builder.or_(li, ri)
                    elif n.op == "&=":
                        oi = builder.and_(li, ri)
                    else:
                        oi = builder.xor(li, ri)
                    out = self._to_f64(builder, oi)
                else:
                    raise ValueError(f"Unsupported assign op {n.op}")
                return builder.call(self.fn_gmem_store, [st, idx, self._filter_assignment(builder, out)])

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
                rhs = self._filter_assignment(builder, rhs)
                if guarded:
                    with builder.if_then(in_range):
                        self._store_cell(builder, rhs, ptr)
                else:
                    self._store_cell(builder, rhs, ptr)
                return rhs

            cur = self._load_cell(builder, ptr)
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
            elif n.op == "%=":
                li = self._to_i32(builder, cur)
                ri = self._to_i32(builder, rhs)
                out = self._to_f64(builder, builder.srem(li, ri))
            elif n.op == "^=":
                fdecl = self._declare_math("pow")
                out = builder.call(fdecl, [cur, rhs])
            elif n.op in ("|=", "&=", "~="):
                li = self._to_i32(builder, cur)
                ri = self._to_i32(builder, rhs)
                if n.op == "|=":
                    oi = builder.or_(li, ri)
                elif n.op == "&=":
                    oi = builder.and_(li, ri)
                else:
                    oi = builder.xor(li, ri)
                out = self._to_f64(builder, oi)
            else:
                raise ValueError(f"Unsupported assign op {n.op}")

            out = self._filter_assignment(builder, out)
            if guarded:
                # JSFX behavior: out-of-range writes are ignored
                with builder.if_then(in_range):
                    self._store_cell(builder, out, ptr)
            else:
                self._store_cell(builder, out, ptr)

            return out

# call
        if isinstance(n, Call):
            if n.fn in tasks_compiler.DEFERRED or n.fn in tasks_compiler.APIS:
                return tasks_compiler.emit(sys.modules[__name__], self, builder, st, n)
            fn = n.fn
            if self.native_gfx_legacy and fn in NATIVE_STRING_APIS:
                minimum, maximum = NATIVE_STRING_APIS[fn]
                if not minimum <= len(n.args) <= maximum:
                    raise ValueError(f"{fn} expects {minimum}..{maximum} arguments")
                count = len(n.args)
                args = self._entry_alloca(builder, self.double, size=self._const_i32(count))
                args.align = 8
                for i, arg in enumerate(n.args):
                    builder.store(self.emit_expr(builder, st, arg), builder.gep(args, [self._const_i32(i)]))
                symbol = "jsfx_native_string_dispatch"
                callee = self._buildins.get(symbol)
                if callee is None:
                    callee = self._function(self.module, ir.FunctionType(self.double,
                        [self.state_ptr, self.i32, self.double.as_pointer(), self.i32]), name=symbol)
                    self._buildins[symbol] = callee
                return builder.call(callee, [st, self._const_i32(list(NATIVE_STRING_APIS).index(fn)), args, self._const_i32(count)])

            if self.native_gfx_legacy and fn in ("atomic_get", "atomic_set", "atomic_add", "atomic_exch", "atomic_setifequal"):
                expected = {"atomic_get": 1, "atomic_set": 2, "atomic_add": 2,
                            "atomic_exch": 2, "atomic_setifequal": 3}[fn]
                if len(n.args) != expected:
                    raise ValueError(f"{fn} expects {expected} args")
                # WDL atomic helpers receive references for their operands.
                # Simple variable addends/comparands are read inside the same
                # critical section, including atomic_add(x, x).
                types = [self.state_ptr] + [self.double.as_pointer()] * expected
                symbol = "jsfx_" + fn
                callee = self._buildins.get(symbol)
                if callee is None:
                    callee = self._function(self.module, ir.FunctionType(self.double, types), name=symbol)
                    self._buildins[symbol] = callee
                return self._call_with_outputs(builder, st, n.args, callee,
                                               {0, 1} if fn == "atomic_exch" else {0}, fn,
                                               input_refs=set(range(1, expected)) if fn != "atomic_exch" else set())

            # Dynamic slider/spl access (JSFX idiom)
            if fn in ("slider", "spl"):
                if len(n.args) != 1:
                    raise ValueError(f"{fn} expects 1 arg")
                ptr, in_range = self._dyn_state_array_ptr(builder, st, fn, n.args[0])
                val = self._load_cell(builder, ptr)
                return builder.select(in_range, val, self._const_f64(0.0))

            if fn == "instance_id":
                if len(n.args) != 0:
                    raise ValueError("instance_id expects 0 args")
                return builder.call(self.fn_instance_id, [st])

            if fn == "instance_uid":
                if len(n.args) != 1:
                    raise ValueError("instance_uid expects 1 arg")
                out_str = self._get_out_lvalue_ptr(builder, st, n.args[0], "instance_uid")
                ret = builder.call(self.fn_instance_uid, [st, out_str])
                return self._to_f64(builder, ret)

            if fn == "instance_set_name":
                if len(n.args) != 1:
                    raise ValueError("instance_set_name expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                ret = builder.call(self.fn_instance_set_name, [st, a0])
                return self._to_f64(builder, ret)

            if fn == "instance_get_name":
                if len(n.args) != 1:
                    raise ValueError("instance_get_name expects 1 arg")
                out_str = self._get_out_lvalue_ptr(builder, st, n.args[0], "instance_get_name")
                ret = builder.call(self.fn_instance_get_name, [st, out_str])
                return self._to_f64(builder, ret)

            if fn in ("track_name", "host_track_name"):
                if len(n.args) != 1:
                    raise ValueError("track_name expects 1 arg")
                out_str = self._get_out_lvalue_ptr(builder, st, n.args[0], "track_name")
                ret = builder.call(self.fn_track_name, [st, out_str])
                return self._to_f64(builder, ret)

            if fn in ("track_name_available", "host_track_name_available"):
                if len(n.args) != 0:
                    raise ValueError("track_name_available expects 0 args")
                return builder.call(self.fn_track_name_available, [st])

            if fn in ("track_name_seq", "host_track_name_seq"):
                if len(n.args) != 0:
                    raise ValueError("track_name_seq expects 0 args")
                return builder.call(self.fn_track_name_seq, [st])

            if fn == "comm_join":
                if len(n.args) != 1:
                    raise ValueError("comm_join expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                ret = builder.call(self.fn_comm_join, [st, a0])
                return self._to_f64(builder, ret)

            if fn == "gmem_attach":
                if len(n.args) != 1:
                    raise ValueError("gmem_attach expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                ret = builder.call(self.fn_gmem_attach, [st, a0])
                return self._to_f64(builder, ret)

            if fn == "gmem_attach_size":
                if len(n.args) != 2:
                    raise ValueError("gmem_attach_size expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                ret = builder.call(self.fn_gmem_attach_size, [st, a0, a1])
                return self._to_f64(builder, ret)

            if fn == "gmem_size":
                if len(n.args) != 0:
                    raise ValueError("gmem_size expects 0 args")
                return builder.call(self.fn_gmem_size, [st])

            if fn == "gmem_get":
                if len(n.args) != 3:
                    raise ValueError("gmem_get expects 3 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                ret = builder.call(self.fn_gmem_get, [st, a0, a1, a2])
                return self._to_f64(builder, ret)

            if fn == "gmem_put":
                if len(n.args) != 3:
                    raise ValueError("gmem_put expects 3 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                ret = builder.call(self.fn_gmem_put, [st, a0, a1, a2])
                return self._to_f64(builder, ret)

            if fn == "gmem_fill":
                if len(n.args) != 3:
                    raise ValueError("gmem_fill expects 3 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                ret = builder.call(self.fn_gmem_fill, [st, a0, a1, a2])
                return self._to_f64(builder, ret)

            if fn == "gmem_zero":
                if len(n.args) != 2:
                    raise ValueError("gmem_zero expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                ret = builder.call(self.fn_gmem_zero, [st, a0, a1])
                return self._to_f64(builder, ret)

            if fn == "gmem_copy":
                if len(n.args) != 3:
                    raise ValueError("gmem_copy expects 3 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                ret = builder.call(self.fn_gmem_copy, [st, a0, a1, a2])
                return self._to_f64(builder, ret)

            if fn == "gmem_seq":
                if len(n.args) != 1:
                    raise ValueError("gmem_seq expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(self.fn_gmem_seq, [st, a0])

            if fn == "gmem_page":
                if len(n.args) != 1:
                    raise ValueError("gmem_page expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(self.fn_gmem_page, [st, a0])

            if fn == "msg_subscribe":
                if len(n.args) != 1:
                    raise ValueError("msg_subscribe expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                ret = builder.call(self.fn_msg_subscribe, [st, a0])
                return self._to_f64(builder, ret)

            if fn == "msg_unsubscribe":
                if len(n.args) != 1:
                    raise ValueError("msg_unsubscribe expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                ret = builder.call(self.fn_msg_unsubscribe, [st, a0])
                return self._to_f64(builder, ret)

            if fn == "msg_advertise":
                if len(n.args) != 2:
                    raise ValueError("msg_advertise expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                ret = builder.call(self.fn_msg_advertise, [st, a0, a1])
                return self._to_f64(builder, ret)

            if fn == "msg_send":
                if len(n.args) != 6:
                    raise ValueError("msg_send expects 6 args")
                argv = [st] + [self.emit_expr(builder, st, a) for a in n.args]
                ret = builder.call(self.fn_msg_send, argv)
                return self._to_f64(builder, ret)

            if fn == "msg_sendto":
                if len(n.args) != 7:
                    raise ValueError("msg_sendto expects 7 args")
                argv = [st] + [self.emit_expr(builder, st, a) for a in n.args]
                ret = builder.call(self.fn_msg_sendto, argv)
                return self._to_f64(builder, ret)

            if fn == "msg_avail":
                if len(n.args) != 1:
                    raise ValueError("msg_avail expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(self.fn_msg_avail, [st, a0])

            if fn == "msg_kind":
                if len(n.args) != 1:
                    raise ValueError("msg_kind expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(self.fn_msg_kind, [st, a0])

            if fn == "msg_recv":
                if len(n.args) != 7:
                    raise ValueError("msg_recv expects 7 args")
                ret = self._call_with_outputs(builder, st, n.args, self.fn_msg_recv, {1, 2, 3, 4, 5, 6}, "msg_recv")
                return self._to_f64(builder, ret)

            if fn == "msg_send_buf":
                if len(n.args) != 4:
                    raise ValueError("msg_send_buf expects 4 args")
                argv = [st] + [self.emit_expr(builder, st, a) for a in n.args]
                ret = builder.call(self.fn_msg_send_buf, argv)
                return self._to_f64(builder, ret)

            if fn == "msg_sendto_buf":
                if len(n.args) != 5:
                    raise ValueError("msg_sendto_buf expects 5 args")
                argv = [st] + [self.emit_expr(builder, st, a) for a in n.args]
                ret = builder.call(self.fn_msg_sendto_buf, argv)
                return self._to_f64(builder, ret)

            if fn == "msg_recv_buf":
                if len(n.args) != 5:
                    raise ValueError("msg_recv_buf expects 5 args")
                ret = self._call_with_outputs(builder, st, n.args, self.fn_msg_recv_buf, {1, 2}, "msg_recv_buf")
                return self._to_f64(builder, ret)

            if fn == "msg_length":
                if len(n.args) != 0:
                    raise ValueError("msg_length expects 0 args")
                return builder.call(self.fn_msg_length, [st])

            if fn == "msg_dropped":
                if len(n.args) != 1:
                    raise ValueError("msg_dropped expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(self.fn_msg_dropped, [st, a0])

            if fn == "msg_clear":
                if len(n.args) != 1:
                    raise ValueError("msg_clear expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                ret = builder.call(self.fn_msg_clear, [st, a0])
                return self._to_f64(builder, ret)

            if fn == "msg_peer_count":
                if len(n.args) != 2:
                    raise ValueError("msg_peer_count expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                return builder.call(self.fn_msg_peer_count, [st, a0, a1])

            if fn == "msg_peer_id":
                if len(n.args) != 3:
                    raise ValueError("msg_peer_id expects 3 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                return builder.call(self.fn_msg_peer_id, [st, a0, a1, a2])

            if fn == "msg_peer_name":
                if len(n.args) != 2:
                    raise ValueError("msg_peer_name expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                out_str = self._get_out_lvalue_ptr(builder, st, n.args[1], "msg_peer_name")
                ret = builder.call(self.fn_msg_peer_name, [st, a0, out_str])
                return self._to_f64(builder, ret)

            if fn == "msg_peer_uid":
                if len(n.args) != 2:
                    raise ValueError("msg_peer_uid expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                out_str = self._get_out_lvalue_ptr(builder, st, n.args[1], "msg_peer_uid")
                ret = builder.call(self.fn_msg_peer_uid, [st, a0, out_str])
                return self._to_f64(builder, ret)

            if fn == "msg_peer_caps":
                if len(n.args) != 1:
                    raise ValueError("msg_peer_caps expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(self.fn_msg_peer_caps, [st, a0])

            if fn == "msg_peer_alive":
                if len(n.args) != 1:
                    raise ValueError("msg_peer_alive expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.call(self.fn_msg_peer_alive, [st, a0])

            if fn == "midirecv":
                if len(n.args) not in (3, 4):
                    raise ValueError("midirecv expects 3 or 4 args")
                callee = self.fn_midirecv if len(n.args) == 4 else self.fn_midirecv_msg23
                ret = self._call_with_outputs(builder, st, n.args, callee,
                                              set(range(len(n.args))), "midirecv")
                return self._to_f64(builder, ret)

            if fn == "midirecv_buf":
                if len(n.args) != 3:
                    raise ValueError("midirecv_buf expects 3 args")
                ret = self._call_with_outputs(builder, st, n.args, self.fn_midirecv_buf, {0}, "midirecv_buf")
                return self._to_f64(builder, ret)

            if fn == "midirecv_str":
                if len(n.args) != 2:
                    raise ValueError("midirecv_str expects 2 args")
                ret = self._call_with_outputs(builder, st, n.args, self.fn_midirecv_str, {0, 1}, "midirecv_str")
                return self._to_f64(builder, ret)

            if fn == "midisend":
                if len(n.args) == 4:
                    a0 = self.emit_expr(builder, st, n.args[0])
                    a1 = self.emit_expr(builder, st, n.args[1])
                    a2 = self.emit_expr(builder, st, n.args[2])
                    a3 = self.emit_expr(builder, st, n.args[3])
                    ret = builder.call(self.fn_midisend, [st, a0, a1, a2, a3])
                    return self._to_f64(builder, ret)
                if len(n.args) == 3:
                    a0 = self.emit_expr(builder, st, n.args[0])
                    a1 = self.emit_expr(builder, st, n.args[1])
                    a2 = self.emit_expr(builder, st, n.args[2])
                    ret = builder.call(self.fn_midisend_msg23, [st, a0, a1, a2])
                    return self._to_f64(builder, ret)
                raise ValueError("midisend expects 3 or 4 args")

            if fn == "midisend_buf":
                if len(n.args) != 3:
                    raise ValueError("midisend_buf expects 3 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                ret = builder.call(self.fn_midisend_buf, [st, a0, a1, a2])
                return self._to_f64(builder, ret)

            if fn == "midisend_str":
                if len(n.args) != 2:
                    raise ValueError("midisend_str expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                ret = builder.call(self.fn_midisend_str, [st, a0, a1])
                return self._to_f64(builder, ret)

            if fn == "midisyx":
                if len(n.args) != 3:
                    raise ValueError("midisyx expects 3 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])
                ret = builder.call(self.fn_midisyx, [st, a0, a1, a2])
                return self._to_f64(builder, ret)

            if (self._emitting_gfx or self.native_gfx_legacy) and fn == "time_precise":
                symbol = "jsfx_native_time_precise"
                fdecl = self._buildins.get(symbol)
                if fdecl is None:
                    fdecl = self._function(self.module, ir.FunctionType(self.double, [self.state_ptr]), name=symbol)
                    self._buildins[symbol] = fdecl
                if len(n.args)>1:
                    raise ValueError("time_precise expects 0 or 1 args")
                output = None
                if n.args:
                    arg = n.args[0]
                    if isinstance(arg, Var) and not arg.name.startswith(("#", "$")) or isinstance(arg, Index) and not self._is_gmem_index(arg):
                        output = self._get_out_lvalue_ptr(builder, st, arg, fn)
                    else:
                        self.emit_expr(builder, st, arg)
                value = builder.call(fdecl, [st])
                if output is not None:
                    self._store_cell(builder, value, output)
                return value

            if self._emitting_gfx and fn in ("strcpy", "strcat", "strncpy", "strlen", "gfx_getchar", "gfx_showmenu"):
                values = [self.emit_expr(builder, st, arg) for arg in n.args]
                if fn == "gfx_getchar" and not values:
                    values = [self._const_f64(0)]
                symbol = "jsfx_native_" + fn
                fdecl = self._buildins.get(symbol)
                if fdecl is None:
                    fdecl = self._function(self.module, ir.FunctionType(self.double, [self.state_ptr] + [self.double] * len(values)), name=symbol)
                    self._buildins[symbol] = fdecl
                return builder.call(fdecl, [st] + values)
            if self._emitting_gfx and fn in ("sliderchange", "slider_automate"):
                mask, direct_idx = self._emit_slider_mask_selector(builder, st, n.args[0])
                end = self.emit_expr(builder, st, n.args[1]) if len(n.args) > 1 else self._const_f64(0)
                symbol = "jsfx_native_slider_event"
                fdecl = self._buildins.get(symbol)
                if fdecl is None:
                    fdecl = self._function(self.module, ir.FunctionType(self.double, [self.state_ptr, self.double, self.i32, self.i32, self.double]), name=symbol)
                    self._buildins[symbol] = fdecl
                return builder.call(fdecl, [st, mask, direct_idx, self._const_i32(1 if fn == "slider_automate" else 0), end])

            if fn == "strlen":
                if len(n.args) != 1:
                    raise ValueError("strlen expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                ret = builder.call(self.fn_strlen, [st, a0])
                return self._to_f64(builder, ret)

            if fn == "str_getchar":
                if len(n.args) != 2:
                    raise ValueError("str_getchar expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                ret = builder.call(self.fn_str_getchar, [st, a0, a1])
                return self._to_f64(builder, ret)

            if fn == "__memtop":
                if len(n.args) != 0:
                    raise ValueError("__memtop expects 0 args")
                return self._const_f64(float(self.jsfx_memtop_slots))

            # ------------------------------------------------------------
            # Runtime sample-pool API: large-bank float32 storage outside mem[].
            # ------------------------------------------------------------
            if fn in SAMPLE_POOL_FUNCTIONS:
                def get_decl(name, arg_count, returns_ptr=False):
                    fdecl = self._buildins.get(name)
                    if fdecl is None:
                        fnty = ir.FunctionType(self.double, [self.state_ptr] + [self.double] * arg_count)
                        fdecl = self._function(self.module, fnty, name=name)
                        self._buildins[name] = fdecl
                    return fdecl

                if fn == "sample_pool_from_slot":
                    if len(n.args) != 2:
                        raise ValueError("sample_pool_from_slot expects 2 args")
                    fdecl = get_decl("jsfx_sample_pool_from_slot", 2)
                    return builder.call(fdecl, [st, self.emit_expr(builder, st, n.args[0]), self.emit_expr(builder, st, n.args[1])])

                if fn in ("sample_pool_set_mode", "sample_pool_set_deferred", "sample_pool_set_budget_mb"):
                    if len(n.args) != 2:
                        raise ValueError(f"{fn} expects 2 args")
                    rt_name = "jsfx_" + fn
                    fdecl = get_decl(rt_name, 2)
                    return builder.call(fdecl, [st, self.emit_expr(builder, st, n.args[0]), self.emit_expr(builder, st, n.args[1])])

                if fn in ("sample_pool_adopt", "sample_pool_commit", "sample_pool_state", "sample_pool_selected", "sample_pool_loaded", "sample_pool_failed", "sample_pool_ram_mb", "sample_pool_generation"):
                    if len(n.args) != 1:
                        raise ValueError(f"{fn} expects 1 arg")
                    rt_name = "jsfx_" + fn
                    fdecl = get_decl(rt_name, 1)
                    return builder.call(fdecl, [st, self.emit_expr(builder, st, n.args[0])])

                if fn == "sample_get":
                    if len(n.args) != 2:
                        raise ValueError("sample_get expects 2 args")
                    fdecl = get_decl("jsfx_sample_get", 2)
                    return builder.call(fdecl, [st, self.emit_expr(builder, st, n.args[0]), self.emit_expr(builder, st, n.args[1])])

                if fn in ("sample_len", "sample_channels", "sample_srate", "sample_peak", "sample_rms", "sample_preview_bins"):
                    if len(n.args) != 2:
                        raise ValueError(f"{fn} expects 2 args")
                    rt_name = "jsfx_" + fn
                    fdecl = get_decl(rt_name, 2)
                    return builder.call(fdecl, [st, self.emit_expr(builder, st, n.args[0]), self.emit_expr(builder, st, n.args[1])])

                if fn in ("sample_read", "sample_read_interp"):
                    if len(n.args) != 4:
                        raise ValueError(f"{fn} expects 4 args")
                    rt_name = "jsfx_" + fn
                    fdecl = get_decl(rt_name, 4)
                    return builder.call(fdecl, [st] + [self.emit_expr(builder, st, a) for a in n.args])

                if fn in ("sample_read2", "sample_read2_interp"):
                    if len(n.args) != 5:
                        raise ValueError(f"{fn} expects 5 args")
                    rt_name = "jsfx_" + fn
                    fdecl = self._buildins.get(rt_name)
                    if fdecl is None:
                        fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double, self.double, self.double.as_pointer(), self.double.as_pointer()])
                        fdecl = self._function(self.module, fnty, name=rt_name)
                        self._buildins[rt_name] = fdecl
                    return self._call_with_outputs(builder, st, n.args, fdecl, {3, 4}, fn)

                if fn == "sample_name":
                    if len(n.args) != 3:
                        raise ValueError("sample_name expects 3 args")
                    rt_name = "jsfx_sample_name"
                    fdecl = self._buildins.get(rt_name)
                    if fdecl is None:
                        fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double, self.double.as_pointer()])
                        fdecl = self._function(self.module, fnty, name=rt_name)
                        self._buildins[rt_name] = fdecl
                    return builder.call(fdecl, [st, self.emit_expr(builder, st, n.args[0]), self.emit_expr(builder, st, n.args[1]), self._get_out_lvalue_ptr(builder, st, n.args[2], fn)])

                if fn == "sample_preview_read":
                    if len(n.args) != 6:
                        raise ValueError("sample_preview_read expects 6 args")
                    rt_name = "jsfx_sample_preview_read"
                    fdecl = self._buildins.get(rt_name)
                    if fdecl is None:
                        fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double, self.double, self.double.as_pointer(), self.double.as_pointer(), self.double.as_pointer()])
                        fdecl = self._function(self.module, fnty, name=rt_name)
                        self._buildins[rt_name] = fdecl
                    return self._call_with_outputs(builder, st, n.args, fdecl, {3, 4, 5}, fn)

                if fn in ("sample_export_mem", "sample_export_mem2"):
                    if len(n.args) != 5:
                        raise ValueError(f"{fn} expects 5 args")
                    rt_name = "jsfx_" + fn
                    fdecl = get_decl(rt_name, 5)
                    return builder.call(fdecl, [st] + [self.emit_expr(builder, st, a) for a in n.args])

            if self.native_gfx_legacy and fn in NATIVE_GFX_APIS and (self._emitting_gfx or fn not in NATIVE_FILE_APIS):
                # EEL builtins receive references, including aliased outputs.
                # Capture addresses in argument order. Only computed values
                # use private temporaries; guest references retain identity.
                count = len(n.args)
                argv = self._entry_alloca(builder, self.double.as_pointer(), size=self._const_i32(max(1, count)))
                for i, arg in enumerate(n.args):
                    if isinstance(arg, Var) and not arg.name.startswith(("#", "$")) and arg.name not in ("mem", "gmem"):
                        ptr = self._get_slot_ptr(builder, st, arg.name)
                    elif isinstance(arg, Index) and not self._is_gmem_index(arg):
                        ptr = self._mem_elem_ptr(builder, st, arg.base, arg.index)
                    elif isinstance(arg, Call) and arg.fn in ("slider", "spl") and len(arg.args) == 1:
                        target, valid = self._dyn_state_array_ptr(builder, st, arg.fn, arg.args[0])
                        dummy = self._entry_alloca(builder, self.double); dummy.align = 8
                        builder.store(self._const_f64(0), dummy)
                        ptr = builder.select(valid, target, dummy)
                    else:
                        value = self.emit_expr(builder, st, arg)
                        ptr = self._entry_alloca(builder, self.double); ptr.align = 8
                        builder.store(value, ptr)
                    builder.store(ptr, builder.gep(argv, [self._const_i32(i)]))
                symbol = "jsfx_native_gfx_dispatch"
                fdecl = self._buildins.get(symbol)
                if fdecl is None:
                    fdecl = self._function(self.module, ir.FunctionType(self.double,
                        [self.state_ptr, self.i32, self.double.as_pointer().as_pointer(), self.i32]), name=symbol)
                    self._buildins[symbol] = fdecl
                return builder.call(fdecl, [st, self._const_i32(list(NATIVE_GFX_APIS).index(fn)), argv, self._const_i32(count)])

            # ------------------------------------------------------------
            # File I/O (DSP-JSFX runtime)
            #
            # REAPER JSFX provides file_*() APIs. In DSP-JSFX we keep the API
            # surface, but route it to host-provided non-blocking runtime
            # helpers (disk I/O happens off the audio thread).
            #
            # Host extension for multi-file slots:
            #   h  = file_open_multi(slot[, mode])
            #   n  = file_multi_count(h)
            #   ok = file_multi_select(h, zero_based_index)
            #
            # file_multi_select() switches which selected file subsequent
            # file_avail/file_var/file_mem/file_riff/file_text/file_seek/
            # file_rewind calls operate on for that handle.
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
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0, a1])

            if fn == "file_open_multi":
                if len(n.args) < 1 or len(n.args) > 2:
                    raise ValueError("file_open_multi expects 1 or 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1]) if len(n.args) == 2 else self._const_f64(0.0)

                rt_name = "jsfx_file_open_multi"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double])
                    fdecl = self._function(self.module, fnty, name=rt_name)
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
                    fdecl = self._function(self.module, fnty, name=rt_name)
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
                    fdecl = self._function(self.module, fnty, name=rt_name)
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
                    fdecl = self._function(self.module, fnty, name=rt_name)
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
                    fdecl = self._function(self.module, fnty, name=rt_name)
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
                    fdecl = self._function(self.module, fnty, name=rt_name)
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
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0, a1, a2])

            if fn == "file_multi_count":
                if len(n.args) != 1:
                    raise ValueError("file_multi_count expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                rt_name = "jsfx_file_multi_count"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double])
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0])

            if fn == "file_multi_select":
                if len(n.args) != 2:
                    raise ValueError("file_multi_select expects 2 args")
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                rt_name = "jsfx_file_multi_select"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double])
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, a0, a1])

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
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl

                return builder.call(fdecl, [st, h, dst_ptr])

            if fn == "file_riff":
                if len(n.args) != 3:
                    raise ValueError("file_riff expects 3 args")

                rt_name = "jsfx_file_riff"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double.as_pointer(), self.double.as_pointer()])
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl

                return self._call_with_outputs(builder, st, n.args, fdecl, {1, 2}, fn, allow_null=True)


            if self._emitting_gfx and fn == "sprintf":
                dest = self.emit_expr(builder, st, n.args[0])
                fmt = self.emit_expr(builder, st, n.args[1])
                count = len(n.args) - 2
                args = self._entry_alloca(builder, self.double, size=self._const_i32(max(1, count)))
                for i, arg in enumerate(n.args[2:]):
                    builder.store(self.emit_expr(builder, st, arg), builder.gep(args, [self._const_i32(i)]))
                symbol = "jsfx_native_sprintf"
                fdecl = self._buildins.get(symbol)
                if fdecl is None:
                    fdecl = self._function(self.module, ir.FunctionType(self.double,
                                        [self.state_ptr, self.double, self.double, self.double.as_pointer(), self.i32]), name=symbol)
                    self._buildins[symbol] = fdecl
                return builder.call(fdecl, [st, dest, fmt, args, self._const_i32(count)])

            if self._emitting_gfx and fn == "gfx_measurestr":
                symbol = "jsfx_native_gfx_measurestr"
                fdecl = self._buildins.get(symbol)
                if fdecl is None:
                    fdecl = self._function(self.module, ir.FunctionType(self.double,
                                        [self.state_ptr, self.double, self.double.as_pointer(), self.double.as_pointer()]), name=symbol)
                    self._buildins[symbol] = fdecl
                text = self.emit_expr(builder, st, n.args[0])
                outputs = [self._get_out_lvalue_ptr(builder, st, a, fn) for a in n.args[1:]]
                outputs += [ir.Constant(self.double.as_pointer(), None)] * (2 - len(outputs))
                return builder.call(fdecl, [st, text] + outputs)

            if self._emitting_gfx and fn in ("gfx_set", "gfx_rect", "gfx_circle", "gfx_line", "gfx_setfont", "gfx_drawstr"):
                values = [self.emit_expr(builder, st, arg) for arg in n.args]
                if fn == "gfx_set":
                    # 1..4 argument form: grey, optional green/blue, alpha.
                    grey = values[0]
                    packed = [grey, values[1] if len(values) > 1 else grey,
                              values[2] if len(values) > 2 else grey,
                              values[3] if len(values) > 3 else self._const_f64(1.0)]
                    symbol = "jsfx_native_gfx_set"
                elif fn == "gfx_rect":
                    packed = values[:4] + [values[4] if len(values) > 4 else self._const_f64(1.0)]
                    symbol = "jsfx_native_gfx_rect"
                elif fn == "gfx_circle":
                    packed = values[:3] + [values[3] if len(values) > 3 else self._const_f64(0.0),
                                          values[4] if len(values) > 4 else self._const_f64(1.0)]
                    symbol = "jsfx_native_gfx_circle"
                elif fn == "gfx_line":
                    packed = values[:4] + [values[4] if len(values) > 4 else self._const_f64(1.0)]
                    symbol = "jsfx_native_gfx_line"
                elif fn == "gfx_setfont":
                    packed = values + [self._const_f64(0.0)] * (4 - len(values))
                    packed.append(self._const_f64(float(len(values))))
                    symbol = "jsfx_native_gfx_setfont"
                else:
                    packed = values
                    symbol = "jsfx_native_gfx_drawstr"
                fdecl = self._buildins.get(symbol)
                if fdecl is None:
                    fdecl = self._function(self.module,
                                        ir.FunctionType(self.double, [self.state_ptr] + [self.double] * len(packed)),
                                        name=symbol)
                    self._buildins[symbol] = fdecl
                return builder.call(fdecl, [st] + packed)


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
            if fn == "file_string" and self.native_gfx_legacy:
                args = [self.emit_expr(builder, st, arg) for arg in n.args]
                if len(args) != 2:
                    raise ValueError("file_string expects 2 arguments")
                rt_name = "jsfx_file_string"
                callee = self._buildins.get(rt_name)
                if callee is None:
                    callee = self._function(self.module, ir.FunctionType(self.double, [self.state_ptr, self.double, self.double]), name=rt_name)
                    self._buildins[rt_name] = callee
                return builder.call(callee, [st] + args)
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
                values = [self.emit_expr(builder, st, a) for a in n.args]
                expected = len(self.user_fn_defs[n.fn].params)
                if expected == 1 and not values:
                    values = [self._const_f64(0)]
                elif expected == 0 and len(values) == 1:
                    # WDL zero-parameter functions occupy the one-argument
                    # overload and evaluate/discard an explicit operand.
                    values = []
                if len(values) != expected:
                    raise ValueError(f"{n.fn} expects {expected} args, got {len(values)} at {n.span.line}:{n.span.col}")
                argv = [st] + values + [self._get_slot_ptr(builder, st, a.name) for a in n.cell_args]
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

            if fn == "sqr":
                if len(n.args) != 1:
                    raise ValueError("sqr expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                return builder.fmul(a0, a0)

            if fn == "sign":
                if len(n.args) != 1:
                    raise ValueError("sign expects 1 arg")
                a0 = self.emit_expr(builder, st, n.args[0])
                is_pos = builder.fcmp_ordered(">", a0, self._const_f64(0.0))
                is_neg = builder.fcmp_ordered("<", a0, self._const_f64(0.0))
                neg_or_zero = builder.select(is_neg, self._const_f64(-1.0), self._const_f64(0.0))
                return builder.select(is_pos, self._const_f64(1.0), neg_or_zero)

            if fn in ("sin", "cos", "sqrt", "fabs", "floor", "ceil"):
                if len(n.args) != 1:
                    raise ValueError(f"{fn} expects 1 arg")
                fdecl = self._declare_math(fn)
                a0 = self.emit_expr(builder, st, n.args[0])
                if fn == "sqrt":
                    # WDL/JSFX defines sqrt(x) as sqrt(abs(x)), including
                    # runtime inputs (not just constant-folded expressions).
                    a0 = builder.call(self._declare_math("fabs"), [a0])
                return builder.call(fdecl, [a0])

            if fn == "invsqrt":
                if len(n.args) != 1:
                    raise ValueError("invsqrt expects 1 arg")
                # Match EEL2/JSFX's classic fast inverse-square-root shape:
                #   y = bitcast_float(0x5f3759df - (bitcast_i32((float)x) >> 1))
                #   y * (1.5 - 0.5 * x * y * y)
                # This preserves the documented approximation behavior rather
                # than lowering to an exact reciprocal sqrt.
                a0 = self.emit_expr(builder, st, n.args[0])
                a0_f32 = builder.fptrunc(a0, self.float)
                bits = builder.bitcast(a0_f32, self.i32)
                half_bits = builder.ashr(bits, self._const_i32(1))
                magic = ir.Constant(self.i32, 0x5f3759df)
                approx_bits = builder.sub(magic, half_bits)
                approx_f32 = builder.bitcast(approx_bits, self.float)
                y0 = builder.fpext(approx_f32, self.double)
                y0_sq = builder.fmul(y0, y0)
                correction = builder.fsub(
                    self._const_f64(1.5),
                    builder.fmul(builder.fmul(self._const_f64(0.5), a0), y0_sq),
                )
                return builder.fmul(y0, correction)

            if fn in ("pow", "atan2"):
                if len(n.args) != 2:
                    raise ValueError(f"{fn} expects 2 args")
                fdecl = self._declare_math(fn)
                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                return builder.call(fdecl, [a0, a1])

            if fn in ("asin", "acos", "atan", "exp", "log", "tan", "log10"):
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
                mask, direct_idx = self._emit_slider_mask_selector(builder, st, n.args[0])
                ret = builder.call(self.fn_sliderchange, [st, mask, direct_idx])
                return self._to_f64(builder, ret)

            if fn == "slider_automate":
                # JSFX builtin: slider_automate(slider_mask_or_var, [is_end_gesture])
                #
                # Record host automation gesture hints for internally-driven slider
                # changes so the runtime can begin/end touch gestures correctly.
                if len(n.args) not in (1, 2):
                    raise ValueError("slider_automate expects 1 or 2 args")
                mask, direct_idx = self._emit_slider_mask_selector(builder, st, n.args[0])
                end_touch = self.emit_expr(builder, st, n.args[1]) if len(n.args) == 2 else self._const_f64(0.0)
                ret = builder.call(self.fn_slider_automate, [st, mask, direct_idx, end_touch])
                return self._to_f64(builder, ret)


            if fn == "slider_next_chg":
                # JSFX builtin: slider_next_chg(sliderX_or_index, out_value)
                #
                # Minimal AOT/JUCE compatibility implementation: expose the current
                # value and report no future sample-accurate change point in the
                # current block.
                if len(n.args) != 2:
                    raise ValueError("slider_next_chg expects 2 args")

                slider_idx = self.emit_expr(builder, st, n.args[0])

                out_ptr = None
                if isinstance(n.args[1], Var) and n.args[1].name != "mem":
                    out_ptr = self._get_slot_ptr(builder, st, n.args[1].name)
                elif isinstance(n.args[1], Index):
                    out_ptr = self._mem_elem_ptr(builder, st, n.args[1].base, n.args[1].index)
                else:
                    _ = self.emit_expr(builder, st, n.args[1])

                if out_ptr is None:
                    out_ptr = ir.Constant(self.double.as_pointer(), None)

                rt_name = "jsfx_slider_next_chg"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double.as_pointer()])
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl
                return builder.call(fdecl, [st, slider_idx, out_ptr])

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

                mask_f, direct_idx = self._emit_slider_mask_selector(builder, st, n.args[0])
                mode_v = self.emit_expr(builder, st, n.args[1]) if len(n.args) == 2 else self._const_f64(0.0)
                has_mode = self._const_i32(1 if len(n.args) == 2 else 0)
                return builder.call(self.fn_slider_show, [st, mask_f, direct_idx, mode_v, has_mode])

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
                dest_i64 = self._checked_nonnegative_index(builder, st, dest_sum)

                zero_i64 = self._const_i64(0)
                isneg = builder.icmp_signed("<", dest_i64, zero_i64)
                dest_i64 = builder.select(isneg, zero_i64, dest_i64)

                # length = max(0, trunc(length))
                len_i64 = self._checked_nonnegative_index(builder, st, len_v)
                isnegL = builder.icmp_signed("<", len_i64, zero_i64)
                len_i64 = builder.select(isnegL, zero_i64, len_i64)

                # Ensure memory for dest + len
                end_i64 = builder.add(dest_i64, len_i64)
                with builder.if_then(builder.icmp_signed("<", end_i64, dest_i64)):
                    self._memory_failure_return(builder, st)
                self._ensure_mem_end(builder, st, end_i64)

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
                self._store_cell(builder, value_v, ptr)
                i_next = builder.add(i_phi, self._const_i64(1))
                body_end = builder.block
                builder.branch(cond_bb)
                i_phi.add_incoming(i_next, body_end)

                builder.position_at_end(after_bb)
                return dest_v


            if fn == "memcpy":
                # JSFX builtin: memcpy(dest, src, length)
                # Copies length doubles within mem[] (overlap permitted). Returns 0.
                if len(n.args) != 3:
                    raise ValueError("memcpy expects 3 args")

                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])

                rt_name = "jsfx_memcpy"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double, self.double])
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl

                return builder.call(fdecl, [st, a0, a1, a2])


            if fn in ("fft", "ifft", "fft_real", "ifft_real", "fft_permute", "fft_ipermute"):
                # JSFX FFT helpers. Complex fft()/ifft() use interleaved
                # real/imag pairs in mem[]. Real fft_real()/ifft_real()
                # operate on size real samples packed into the same mem region
                # and expose size/2 complex bins (with DC/Nyquist packed in the
                # first pair, matching WDL/JSFX semantics).
                if len(n.args) != 2:
                    raise ValueError(f"{fn} expects 2 args")

                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])

                rt_name = {
                    "fft": "jsfx_fft",
                    "ifft": "jsfx_ifft",
                    "fft_real": "jsfx_fft_real",
                    "ifft_real": "jsfx_ifft_real",
                    "fft_permute": "jsfx_fft_permute",
                    "fft_ipermute": "jsfx_fft_ipermute",
                }[fn]

                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double])
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl

                return builder.call(fdecl, [st, a0, a1])

            if fn == "convolve_c":
                # JSFX builtin: convolve_c(dest, src, size)
                # Multiplies size complex pairs in dest by src in-place.
                if len(n.args) != 3:
                    raise ValueError("convolve_c expects 3 args")

                a0 = self.emit_expr(builder, st, n.args[0])
                a1 = self.emit_expr(builder, st, n.args[1])
                a2 = self.emit_expr(builder, st, n.args[2])

                rt_name = "jsfx_convolve_c"
                fdecl = self._buildins.get(rt_name)
                if fdecl is None:
                    fnty = ir.FunctionType(self.double, [self.state_ptr, self.double, self.double, self.double])
                    fdecl = self._function(self.module, fnty, name=rt_name)
                    self._buildins[rt_name] = fdecl

                return builder.call(fdecl, [st, a0, a1, a2])

            raise ValueError(f"Unknown function call {n.fn}")

        # sequence
        if isinstance(n, Seq):
            last = self._const_f64(0.0)
            for item in n.items:
                if isinstance(item, If):
                    self.emit_if(builder, st, item)
                    last = self._const_f64(0.0)
                elif isinstance(item, While):
                    last = self.emit_while(builder, st, item)
                else:
                    last = self.emit_expr(builder, st, item)
            return last

        # allow if/while as expression (returns 0)
        if isinstance(n, If):
            self.emit_if(builder, st, n)
            return self._const_f64(0.0)
        if isinstance(n, While):
            return self.emit_while(builder, st, n)

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
        # WDL converts to a signed 32-bit count, then caps at 2^20.
        valid = builder.and_(builder.fcmp_ordered(">=", count_v, self._const_f64(0)),
                             builder.fcmp_ordered("<", count_v, self._const_f64(2147483648.)))
        safe = builder.select(valid, count_v, self._const_f64(0))
        bounded = builder.select(builder.fcmp_ordered(">", safe, self._const_f64(1048576)), self._const_f64(1048576), safe)
        n_i = builder.fptosi(bounded, self.i64)

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
            if self.has_tasks:
                tasks_compiler.checkpoint(sys.modules[__name__], self, builder, st)
            v = self.emit_expr(builder, st, n.body)
            i_next = builder.add(phi_i, self._const_i64(1))
            latch_bb = builder.block
            if not builder.block.is_terminated:
                builder.branch(cond_bb)

            # Add incoming from latch
            phi_i.add_incoming(i_next, latch_bb)
            phi_last.add_incoming(v, latch_bb)

            builder.position_at_end(after_bb)
            # WDL's loop() is an execution primitive, whose numeric result is
            # always one, including when the count skips the body.
            return self._const_f64(1.0)
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

    fn = self._function(
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
    self._store_cell(builder, sb_val, sb_ptr)

    # mirror block metadata for runtime helpers
    fld_blocksize = ir.Constant(self.i32, 14)
    blocksize_ptr = builder.gep(st, [z, fld_blocksize], inbounds=True)
    builder.store(nSamp, blocksize_ptr)

    fld_currate = ir.Constant(self.i32, 15)
    currate_ptr = builder.gep(st, [z, fld_currate], inbounds=True)
    fld_srate = ir.Constant(self.i32, 5)
    srate_ptr = builder.gep(st, [z, fld_srate], inbounds=True)
    builder.store(self._load_cell(builder, srate_ptr), currate_ptr)

    # Call block 
    builder.call(fn_block, [st])

    # If @block changed sliders internally (via sliderchange/slider_automate),
    # run @slider once before @sample so derived state tracks the new values
    # within the same host block.
    fld_sliderchg = ir.Constant(self.i32, 23)
    fld_sliderauto = ir.Constant(self.i32, 24)
    fld_sliderautoend = ir.Constant(self.i32, 25)
    sliderchg_arr = builder.gep(st, [z, fld_sliderchg], inbounds=True)
    sliderauto_arr = builder.gep(st, [z, fld_sliderauto], inbounds=True)
    sliderautoend_arr = builder.gep(st, [z, fld_sliderautoend], inbounds=True)
    any_sliderchg = ir.Constant(self.i64, 0)
    for word in range(SLIDER_MASK_WORDS):
        wi = ir.Constant(self.i32, word)
        chg = builder.load(builder.gep(sliderchg_arr, [z, wi], inbounds=True))
        auto = builder.load(builder.gep(sliderauto_arr, [z, wi], inbounds=True))
        autoend = builder.load(builder.gep(sliderautoend_arr, [z, wi], inbounds=True))
        any_sliderchg = builder.or_(any_sliderchg, builder.or_(chg, builder.or_(auto, autoend)))
    have_sliderchg = builder.icmp_signed("!=", any_sliderchg, ir.Constant(self.i64, 0))
    with builder.if_then(have_sliderchg):
        builder.call(fn_slider, [st])

    # An empty @sample preserves input audio. Copy whole channel buffers so the
    # out-of-place API has the same behavior as the production in-place host,
    # without a per-sample guest-state loop for MIDI/controller effects.
    if not has_sample_work:
        copy_cond = fn.append_basic_block("copy_cond")
        copy_body = fn.append_basic_block("copy_body")
        copy_end = fn.append_basic_block("copy_end")
        copy_pre = builder.block
        builder.cbranch(builder.icmp_signed(">", nSamp, zero_i32), copy_cond, copy_end)
        builder.position_at_end(copy_cond)
        channel = builder.phi(self.i32, name="copy_channel")
        channel.add_incoming(zero_i32, copy_pre)
        builder.cbranch(builder.icmp_signed("<", channel, chLim), copy_body, copy_end)
        builder.position_at_end(copy_body)
        source = builder.load(builder.gep(inputs, [channel]))
        target = builder.load(builder.gep(outputs, [channel]))
        with builder.if_then(builder.icmp_unsigned("!=", source, target)):
            byte_ptr = ir.IntType(8).as_pointer()
            move = self.module.declare_intrinsic("llvm.memmove", [byte_ptr, byte_ptr, self.i64])
            size = builder.mul(builder.sext(nSamp, self.i64), ir.Constant(self.i64, 4))
            builder.call(move, [builder.bitcast(target, byte_ptr), builder.bitcast(source, byte_ptr),
                                size, ir.Constant(ir.IntType(1), 0)])
        next_channel = builder.add(channel, ir.Constant(self.i32, 1))
        channel.add_incoming(next_channel, builder.block)
        builder.branch(copy_cond)
        builder.position_at_end(copy_end)
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
    self._store_cell(builder, in_d, spl_ptr)

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
    out_d = self._load_cell(builder, spl2_ptr)
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


def compile_jsfx_to_ir(jsfx_text: str,
                       *,
                       enable_section_hoists: bool = False,
                       enable_loop_hoists: bool = False,
                       native_gfx_prototype: bool = False,
                       native_gfx_legacy: bool = False,
                       stream_functions: bool = False,
                       state_var_capacity: int = 0,
                       pipeline: Optional[Dict[str, Any]] = None) -> Tuple[ir.Module, Dict[str, Any]]:
    if pipeline is None:
        pipeline = prepare_jsfx_pipeline(
            jsfx_text,
            include_gfx=native_gfx_prototype,
            native_gfx_legacy=native_gfx_legacy,
            enable_section_hoists=enable_section_hoists,
            enable_loop_hoists=enable_loop_hoists,
            collect_opt_report=False,
        )
    return compile_pipeline_to_ir(jsfx_text, pipeline, native_gfx_prototype=native_gfx_prototype,
                                  native_gfx_legacy=native_gfx_legacy, stream_functions=stream_functions, state_var_capacity=state_var_capacity)



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
    comm_meta: Dict[str, Any] = dict(meta.get("comm", {}) or {})
    uses_gmem = 1 if comm_meta.get("uses_gmem") else 0
    uses_msg = 1 if comm_meta.get("uses_msg") else 0
    uses_msg_buffers = 1 if comm_meta.get("uses_msg_buffers") else 0
    sample_pool_meta: Dict[str, Any] = dict(meta.get("sample_pool", {}) or {})
    uses_sample_pool = 1 if sample_pool_meta.get("uses_sample_pool") else 0
    uses_raw_sample_read = 1 if sample_pool_meta.get("uses_raw_sample_read") else 0
    uses_sample_export_mem = 1 if sample_pool_meta.get("uses_export_mem") else 0
    uses_legacy_file_io = 1 if sample_pool_meta.get("uses_legacy_file_io") else 0

    in_ch = max(0, min(64, in_ch))
    out_ch = max(0, min(64, out_ch))
    proc_ch = max(0, min(64, proc_ch))


    def _c_escape(s: str) -> str:
        return s.replace('\\', r'\\').replace('"', r'\\"')

    lines = []
    lines.append("#pragma once")
    lines.append(f"#define DSPJSFX_HAS_TASKS {1 if meta.get('has_tasks') else 0}")
    lines.append(f"#define DSPJSFX_HAS_TASK_WORKERS {1 if meta.get('has_task_workers', meta.get('has_tasks')) else 0}")
    lines.append(f"#define DSPJSFX_HAS_FAUST {1 if meta.get('has_faust') else 0}")
    lines.append(f"#define DSPJSFX_EEL2_STORES {0 if meta.get('numeric_semantics')=='native' else 1}")
    lines.append("#include <stdint.h>")
    lines.append(f"#define DSPJSFX_NATIVE_GFX_LEGACY {1 if meta.get('native_gfx_legacy') else 0}")
    lines.append(f"#define DSPJSFX_MAX_MEM_CELLS {int(meta.get('memtop_slots', JSFX_DEFAULT_MEMTOP_SLOTS))}LL")
    lines.append('#if defined(__cplusplus) && DSPJSFX_NATIVE_GFX_LEGACY')
    lines.append('#include "JsfxSharedCells.h"')
    lines.append('#else')
    lines.append('typedef double DSPJSFX_Cell;')
    lines.append('#endif')

    lines.append("")
    lines.append("/* Inferred minimum I/O channel counts (from splN usage / pin declarations) */")
    lines.append(f"#define DSPJSFX_INPUT_CHANNELS {in_ch}")
    lines.append(f"#define DSPJSFX_OUTPUT_CHANNELS {out_ch}")
    lines.append(f"#define DSPJSFX_PROCESS_CHANNELS {proc_ch}")
    lines.append(f"#define DSPJSFX_USES_MIDI {uses_midi}")
    lines.append(f"#define DSPJSFX_USES_GMEM {uses_gmem}")
    lines.append(f"#define DSPJSFX_USES_MSG {uses_msg}")
    lines.append(f"#define DSPJSFX_USES_MSG_BUFFERS {uses_msg_buffers}")
    lines.append(f"#define DSPJSFX_USES_SAMPLE_POOL {uses_sample_pool}")
    lines.append(f"#define DSPJSFX_USES_RAW_SAMPLE_READ {uses_raw_sample_read}")
    lines.append(f"#define DSPJSFX_USES_SAMPLE_EXPORT_MEM {uses_sample_export_mem}")
    lines.append(f"#define DSPJSFX_USES_LEGACY_FILE_IO {uses_legacy_file_io}")
    lines.append("#define DSPJSFX_COMM_API_VERSION 1")
    lines.append("#define DSPJSFX_SAMPLE_POOL_API_VERSION 1")
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
    state_abi=(6 if meta.get('native_gfx_legacy') else 4) if meta.get('has_tasks') else (5 if meta.get('native_gfx_legacy') else 3)
    if meta.get('has_faust'): state_abi+=10
    if meta.get('native_gfx_legacy'): state_abi=17
    lines.append(f"#define DSPJSFX_RUNTIME_STATE_ABI {state_abi}")
    lines.append("#define DSPJSFX_MAX_SLIDERS 256")
    lines.append("#define DSPJSFX_SLIDER_MASK_WORDS 4")
    lines.append("typedef struct DSPJSFX_State {")
    lines.append("    DSPJSFX_Cell spl[64];")
    lines.append("    DSPJSFX_Cell sliders[DSPJSFX_MAX_SLIDERS];")
    lines.append(f"#define DSPJSFX_DYNAMIC_VARIABLES {1 if meta.get('native_gfx_legacy') else 0}")
    lines.append(f"#define DSPJSFX_STATE_VARS_CAPACITY {var_cap}")
    lines.append("    DSPJSFX_Cell* vars;" if meta.get('native_gfx_legacy') else f"    DSPJSFX_Cell vars[{var_cap}];")
    lines.append("    DSPJSFX_Cell* mem;")
    lines.append("    int64_t memN;")
    lines.append("    DSPJSFX_Cell srate;")
    lines.append("    DSPJSFX_Cell samplesblock;")
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
    lines.append("    uint64_t pendingSliderChangeMask[DSPJSFX_SLIDER_MASK_WORDS];")
    lines.append("    uint64_t pendingSliderAutomateMask[DSPJSFX_SLIDER_MASK_WORDS];")
    lines.append("    uint64_t pendingSliderAutomateEndMask[DSPJSFX_SLIDER_MASK_WORDS];")
    lines.append("    uint32_t randMT[624];")
    lines.append("    uint32_t randIndex;")
    lines.append("    uint64_t sliderVisibleMask[DSPJSFX_SLIDER_MASK_WORDS];")
    lines.append("    int32_t sliderVisibilityInit;")
    lines.append("    void* runtimeOpaque;")
    lines.append("    DSPJSFX_Cell midi_bus;")
    lines.append("    DSPJSFX_Cell ext_midi_bus;")
    lines.append("    void* hostOwner;")
    lines.append("    int64_t memUsed;")
    lines.append("    int32_t memoryFault;")
    if meta.get('native_gfx_legacy'):
        lines.append("    void* sharedState; /* null for DSP; GFX points at the DSP state */")
        lines.append("    void* atomicContext; /* instance mutex for explicit atomic_* calls */")
        lines.append("    void* nativeStrings; /* instance-owned native string context */")
    if meta.get('has_tasks') or meta.get('native_gfx_legacy'):
        lines.append("    void* taskContext; /* instance-owned structured task runtime */")
    if meta.get("has_faust") or meta.get("native_gfx_legacy"):
        lines.append("    void* faustContext; /* instance-owned mixed Faust pipeline */")
    if meta.get("native_gfx_legacy"): lines.append("    int64_t varsN; /* instance variable extent, independent of code loading */")
    lines.append("} DSPJSFX_State;")
    lines.append("")

    # Export the AOT compiler's user-var symbol mapping so host code can bind EEL/JSFX
    # variables by *name* to the correct vars[] index.
    #
    # NOTE: MSVC does not allow zero-sized arrays, so we emit a 1-element dummy when empty.
    items_by_index = sorted(user_vars.items(), key=lambda kv: int(kv[1]))
    count = len(items_by_index)
    gfx_var_flags_meta: Dict[str, int] = dict(meta.get("gfx_var_flags", {}) or {})
    gfx_var_sync_mode = str(meta.get("gfx_var_sync_mode", "legacy") or "legacy")
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

    if meta.get('native_gfx_legacy'):
        for opcode, name in enumerate(NATIVE_GFX_APIS):
            lines.append(f"#define DSPJSFX_{name.upper()} {opcode}")
        for opcode, name in enumerate(NATIVE_STRING_APIS):
            lines.append(f"#define DSPJSFX_STRING_{name.upper()} {opcode}")
        lines.append("#ifdef __cplusplus\nextern \"C\"\n#endif\ndouble jsfx_native_string_dispatch(DSPJSFX_State*, int32_t, const double*, int32_t);")
        lines.append("#ifdef __cplusplus\nextern \"C\"\n#endif\ndouble jsfx_native_gfx_dispatch(DSPJSFX_State* st, int32_t opcode, double** args, int32_t count);")
        names = meta.get("named_strings", {})
        lines.append("typedef struct DSPJSFX_NamedStringDesc { const char* name; int64_t handle; } DSPJSFX_NamedStringDesc;")
        lines.append(f"#define DSPJSFX_NAMED_STRINGS_COUNT {len(names)}")
        lines.append(f"static const DSPJSFX_NamedStringDesc DSPJSFX_NAMED_STRINGS[{max(1,len(names))}] = {{")
        for name, handle in names.items():
            lines.append(f'    {{"{_c_escape(name)}", {handle}LL}},')
        if not names:
            lines.append('    {"", -1LL},')
        lines.append("};")
        aliases = meta.get('slider_aliases', {})
        lines.append(f"static const int DSPJSFX_LEGACY_SLIDER_ALIASES[{arr_size}] = {{" +
                     ",".join(str(aliases.get(name, -1)) for name, _ in items_by_index or [("", 0)]) + "};")

    lines.append("/* @gfx user-var sync flags (vars[] index).")
    lines.append("   bit0: sync audio-owned updates into the @gfx VM (audio -> gfx)")
    lines.append("   bit1: feed @gfx-authored writes back into the DSP state (gfx -> audio)")
    lines.append(f"   Mode: {_c_escape(gfx_var_sync_mode)} */")
    lines.append("#define DSPJSFX_GFX_VAR_FLAG_TO_GFX 1u")
    lines.append("#define DSPJSFX_GFX_VAR_FLAG_FROM_GFX 2u")
    lines.append(f"#define DSPJSFX_GFX_VAR_FLAGS_COUNT {count}")
    lines.append(f"static const uint8_t DSPJSFX_GFX_VAR_FLAGS[{arr_size}] = {{")
    if count == 0:
        lines.append("    0,")
    else:
        for name, idx in items_by_index:
            flags = int(gfx_var_flags_meta.get(name, GFX_VAR_FLAG_TO_GFX | GFX_VAR_FLAG_FROM_GFX))
            lines.append(f"    {flags},")
    lines.append("};")
    lines.append("")

    string_literals_meta: List[Dict[str, Any]] = list(meta.get("string_literals") or [])
    literal_count = len(string_literals_meta)
    lines.append("/* String literals (opaque runtime handles used by MIDI/string helpers) */")
    lines.append("typedef struct DSPJSFX_StringLiteralDesc { int64_t handle; int32_t length; const uint8_t* data; } DSPJSFX_StringLiteralDesc;")
    lines.append(f"#define DSPJSFX_STRING_LITERALS_COUNT {literal_count}")
    if literal_count <= 0:
        lines.append("static const uint8_t DSPJSFX_STRING_LITERAL_BYTES_0[1] = { 0 };")
        lines.append("static const DSPJSFX_StringLiteralDesc DSPJSFX_STRING_LITERALS[1] = {")
        lines.append("    { 0LL, 0, DSPJSFX_STRING_LITERAL_BYTES_0 },")
        lines.append("};")
    else:
        for i, item in enumerate(string_literals_meta):
            raw = str(item.get("text", "")).encode("utf-8")
            raw_size = len(raw) if len(raw) > 0 else 1
            byte_list = ", ".join(f"0x{b:02x}" for b in raw) if raw else "0"
            lines.append(f"static const uint8_t DSPJSFX_STRING_LITERAL_BYTES_{i}[{raw_size}] = {{ {byte_list} }};")
        lines.append(f"static const DSPJSFX_StringLiteralDesc DSPJSFX_STRING_LITERALS[{literal_count}] = {{")
        for i, item in enumerate(string_literals_meta):
            raw = str(item.get("text", "")).encode("utf-8")
            handle = int(item.get("handle", 0))
            lines.append(f"    {{ {handle}LL, {len(raw)}, DSPJSFX_STRING_LITERAL_BYTES_{i} }},")
        lines.append("};")
    lines.append("")

    lines.append("/* Sections */")
    lines.append("void jsfx_init(DSPJSFX_State* st);")
    lines.append("void jsfx_slider(DSPJSFX_State* st);")
    lines.append("void jsfx_block(DSPJSFX_State* st);")
    lines.append("void jsfx_sample(DSPJSFX_State* st);")
    lines.append(f"#define DSPJSFX_HAS_NATIVE_GFX {1 if meta.get('native_gfx_prototype') else 0}")
    lines.append(f"#define DSPJSFX_NATIVE_GFX_PRESENT {1 if meta.get('sections_present', {}).get('gfx') else 0}")
    if meta.get("native_gfx_prototype"):
        size = meta["native_gfx_size"]
        lines.append(f"#define DSPJSFX_NATIVE_GFX_WIDTH {size[0]}")
        lines.append(f"#define DSPJSFX_NATIVE_GFX_HEIGHT {size[1]}")
        contract = meta.get("native_gfx_contract", {})
        lines.append(f"#define DSPJSFX_NATIVE_GFX_INTERACTIVE {1 if contract.get('interactive') else 0}")
        views = contract.get("views", [])
        lines.append(f"#define DSPJSFX_NATIVE_GFX_VIEW_COUNT {len(views)}")
        lines.append("typedef struct { int baseVar, countVar, strideVar, generationVar; int64_t base, count, stride, generation; int maxRecords, fieldCount; int fields[16]; } DSPJSFX_NativeView;")
        lines.append("static const DSPJSFX_NativeView DSPJSFX_NATIVE_VIEWS[] = {")
        for v in views:
            refs = [v.get(k, 1 if k == "stride" else 0) for k in ("base", "count", "stride", "generation")]
            indices = [meta["vars"][r] if isinstance(r, str) else -1 for r in refs]
            values = [r if isinstance(r, int) else 0 for r in refs]
            fields = v.get("fields", [0])
            lines.append("  {" + ",".join(map(str, indices + values + [v["max"], len(fields)])) + ",{" + ",".join(map(str, fields)) + "}},")
        if not views:
            lines.append("  {-1,-1,-1,-1,0,0,1,0,0,0,{0}},")
        lines.append("};")
        for label, key in (("COMMAND", "commands"), ("PERSIST", "persist")):
            ids = [meta["vars"][n] for n in contract.get(key, [])]
            lines.append(f"#define DSPJSFX_NATIVE_GFX_{label}_COUNT {len(ids)}")
            lines.append(f"static const int DSPJSFX_NATIVE_GFX_{label}_INDICES[] = {{" + ",".join(map(str, ids or [-1])) + "};")
        lines.append("double jsfx_native_read_mem(DSPJSFX_State*, int64_t);")
        for fn, argc in (("strcpy", 2), ("strcat", 2), ("strncpy", 3), ("strlen", 1), ("gfx_getchar", 1), ("gfx_showmenu", 1), ("time_precise", 0)):
            lines.append("double jsfx_native_" + fn + "(DSPJSFX_State*" + ", double" * argc + ");")
        lines.append("double jsfx_native_slider_event(DSPJSFX_State*, double, int32_t, int32_t, double);")
        lines.append("void jsfx_gfx_aot(DSPJSFX_State* st);")
        lines.append("double jsfx_native_gfx_set(DSPJSFX_State* st, double r, double g, double b, double a);")
        lines.append("double jsfx_native_gfx_rect(DSPJSFX_State* st, double x, double y, double w, double h, double fill);")
        lines.append("double jsfx_native_gfx_circle(DSPJSFX_State* st, double x, double y, double radius, double fill, double aa);")
        lines.append("double jsfx_native_gfx_line(DSPJSFX_State* st, double x, double y, double x2, double y2, double aa);")
        lines.append("double jsfx_native_gfx_measurestr(DSPJSFX_State* st, double text, double* width, double* height);")
        lines.append("double jsfx_native_gfx_setfont(DSPJSFX_State* st, double id, double name, double size, double flags, double argc);")
        lines.append("double jsfx_native_gfx_drawstr(DSPJSFX_State* st, double text);")
        lines.append("double jsfx_native_sprintf(DSPJSFX_State* st, double dest, double format, const double* args, int32_t count);")
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
    lines.append("int jsfx_midirecv_msg23(DSPJSFX_State* st, double* offset, double* msg1, double* msg23);")
    lines.append("int jsfx_midirecv_buf(DSPJSFX_State* st, double* offset, double buf, double maxlen);")
    lines.append("int jsfx_midirecv_str(DSPJSFX_State* st, double* offset, double* strSlot);")
    lines.append("int jsfx_midisend(DSPJSFX_State* st, double offset, double msg1, double msg2, double msg3);")
    lines.append("int jsfx_midisend_msg23(DSPJSFX_State* st, double offset, double msg1, double msg23);")
    lines.append("int jsfx_midisend_buf(DSPJSFX_State* st, double offset, double buf, double len);")
    lines.append("int jsfx_midisend_str(DSPJSFX_State* st, double offset, double strHandle);")
    lines.append("int jsfx_midisyx(DSPJSFX_State* st, double offset, double msgptr, double len);")
    lines.append("int jsfx_strlen(DSPJSFX_State* st, double strHandle);")
    lines.append("int jsfx_str_getchar(DSPJSFX_State* st, double strHandle, double index);")
    lines.append("int jsfx_sliderchange(DSPJSFX_State* st, double sliderMask, int directSliderIndex);")
    lines.append("int jsfx_slider_automate(DSPJSFX_State* st, double sliderMask, int directSliderIndex, double endTouch);")
    lines.append("double jsfx_slider_show(DSPJSFX_State* st, double sliderMask, int directSliderIndex, double mode, int hasMode);")
    lines.append("double jsfx_slider_next_chg(DSPJSFX_State* st, double sliderIndex, double* outValue);")
    lines.append("double jsfx_instance_id(DSPJSFX_State* st);")
    lines.append("int jsfx_instance_uid(DSPJSFX_State* st, double* outStr);")
    lines.append("int jsfx_instance_set_name(DSPJSFX_State* st, double strHandle);")
    lines.append("int jsfx_instance_get_name(DSPJSFX_State* st, double* outStr);")
    lines.append("int jsfx_track_name(DSPJSFX_State* st, double* outStr);")
    lines.append("double jsfx_track_name_available(DSPJSFX_State* st);")
    lines.append("double jsfx_track_name_seq(DSPJSFX_State* st);")
    lines.append("int jsfx_comm_join(DSPJSFX_State* st, double domainHandle);")
    lines.append("int jsfx_gmem_attach(DSPJSFX_State* st, double nameHandle);")
    lines.append("int jsfx_gmem_attach_size(DSPJSFX_State* st, double nameHandle, double cells);")
    lines.append("double jsfx_gmem_size(DSPJSFX_State* st);")
    lines.append("double jsfx_gmem_load(DSPJSFX_State* st, double idx);")
    lines.append("double jsfx_gmem_store(DSPJSFX_State* st, double idx, double value);")
    lines.append("int jsfx_gmem_get(DSPJSFX_State* st, double dstBase, double srcIdx, double count);")
    lines.append("int jsfx_gmem_put(DSPJSFX_State* st, double dstIdx, double srcBase, double count);")
    lines.append("int jsfx_gmem_fill(DSPJSFX_State* st, double dstIdx, double value, double count);")
    lines.append("int jsfx_gmem_zero(DSPJSFX_State* st, double dstIdx, double count);")
    lines.append("int jsfx_gmem_copy(DSPJSFX_State* st, double dstIdx, double srcIdx, double count);")
    lines.append("double jsfx_gmem_seq(DSPJSFX_State* st, double page);")
    lines.append("double jsfx_gmem_page(DSPJSFX_State* st, double idx);")
    lines.append("int jsfx_msg_subscribe(DSPJSFX_State* st, double chanHandle);")
    lines.append("int jsfx_msg_unsubscribe(DSPJSFX_State* st, double chanHandle);")
    lines.append("int jsfx_msg_advertise(DSPJSFX_State* st, double chanHandle, double caps);")
    lines.append("int jsfx_msg_send(DSPJSFX_State* st, double chanHandle, double tag, double a, double b, double c, double d);")
    lines.append("int jsfx_msg_sendto(DSPJSFX_State* st, double targetId, double chanHandle, double tag, double a, double b, double c, double d);")
    lines.append("double jsfx_msg_avail(DSPJSFX_State* st, double chanHandle);")
    lines.append("double jsfx_msg_kind(DSPJSFX_State* st, double chanHandle);")
    lines.append("int jsfx_msg_recv(DSPJSFX_State* st, double chanHandle, double* src, double* tag, double* a, double* b, double* c, double* d);")
    lines.append("int jsfx_msg_send_buf(DSPJSFX_State* st, double chanHandle, double tag, double srcBase, double len);")
    lines.append("int jsfx_msg_sendto_buf(DSPJSFX_State* st, double targetId, double chanHandle, double tag, double srcBase, double len);")
    lines.append("int jsfx_msg_recv_buf(DSPJSFX_State* st, double chanHandle, double* src, double* tag, double dstBase, double maxLen);")
    lines.append("double jsfx_msg_length(DSPJSFX_State* st);")
    lines.append("double jsfx_msg_dropped(DSPJSFX_State* st, double chanHandle);")
    lines.append("int jsfx_msg_clear(DSPJSFX_State* st, double chanHandle);")
    lines.append("double jsfx_msg_peer_count(DSPJSFX_State* st, double chanHandle, double role);")
    lines.append("double jsfx_msg_peer_id(DSPJSFX_State* st, double chanHandle, double role, double index);")
    lines.append("int jsfx_msg_peer_name(DSPJSFX_State* st, double peerId, double* outStr);")
    lines.append("int jsfx_msg_peer_uid(DSPJSFX_State* st, double peerId, double* outStr);")
    lines.append("double jsfx_msg_peer_caps(DSPJSFX_State* st, double peerId);")
    lines.append("double jsfx_msg_peer_alive(DSPJSFX_State* st, double peerId);")
    lines.append("double jsfx_sample_pool_from_slot(DSPJSFX_State* st, double slot, double nameHandle);")
    lines.append("double jsfx_sample_pool_set_deferred(DSPJSFX_State* st, double pool, double deferred);")
    lines.append("double jsfx_sample_pool_adopt(DSPJSFX_State* st, double pool);")
    lines.append("double jsfx_sample_pool_set_mode(DSPJSFX_State* st, double pool, double mode);")
    lines.append("double jsfx_sample_pool_set_budget_mb(DSPJSFX_State* st, double pool, double mb);")
    lines.append("double jsfx_sample_pool_commit(DSPJSFX_State* st, double pool);")
    lines.append("double jsfx_sample_pool_state(DSPJSFX_State* st, double pool);")
    lines.append("double jsfx_sample_pool_selected(DSPJSFX_State* st, double pool);")
    lines.append("double jsfx_sample_pool_loaded(DSPJSFX_State* st, double pool);")
    lines.append("double jsfx_sample_pool_failed(DSPJSFX_State* st, double pool);")
    lines.append("double jsfx_sample_pool_ram_mb(DSPJSFX_State* st, double pool);")
    lines.append("double jsfx_sample_pool_generation(DSPJSFX_State* st, double pool);")
    lines.append("double jsfx_sample_get(DSPJSFX_State* st, double pool, double index);")
    lines.append("double jsfx_sample_len(DSPJSFX_State* st, double pool, double sampleId);")
    lines.append("double jsfx_sample_channels(DSPJSFX_State* st, double pool, double sampleId);")
    lines.append("double jsfx_sample_srate(DSPJSFX_State* st, double pool, double sampleId);")
    lines.append("double jsfx_sample_peak(DSPJSFX_State* st, double pool, double sampleId);")
    lines.append("double jsfx_sample_rms(DSPJSFX_State* st, double pool, double sampleId);")
    lines.append("double jsfx_sample_name(DSPJSFX_State* st, double pool, double sampleId, double* outStr);")
    lines.append("double jsfx_sample_read(DSPJSFX_State* st, double pool, double sampleId, double channel, double frame);")
    lines.append("double jsfx_sample_read_interp(DSPJSFX_State* st, double pool, double sampleId, double channel, double phase);")
    lines.append("double jsfx_sample_read2(DSPJSFX_State* st, double pool, double sampleId, double phase, double* outL, double* outR);")
    lines.append("double jsfx_sample_read2_interp(DSPJSFX_State* st, double pool, double sampleId, double phase, double* outL, double* outR);")
    lines.append("double jsfx_sample_preview_bins(DSPJSFX_State* st, double pool, double sampleId);")
    lines.append("double jsfx_sample_preview_read(DSPJSFX_State* st, double pool, double sampleId, double bin, double* minValue, double* maxValue, double* rmsValue);")
    lines.append("double jsfx_sample_export_mem(DSPJSFX_State* st, double pool, double sampleId, double dstBase, double srcFrame, double frameCount);")
    lines.append("double jsfx_sample_export_mem2(DSPJSFX_State* st, double pool, double sampleId, double dstBase, double srcFrame, double frameCount);")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    lines.append(faust_compiler.header(meta))
    return "\n".join(lines)

def _aot_opt_and_emit(mod_ir: ir.Module,
                     opt_level: int,
                     emit_obj: Optional[str],
                     emit_asm: Optional[str],
                     target_triple: Optional[str] = None,
                     out_ll_unopt: Optional[str] = None,
                     out_ll_opt: Optional[str] = None,
                     position_independent: bool = False,
                     native_gfx_legacy: bool = False) -> str:
    """Return optimized LLVM IR text. Optionally emits object/asm and pre/post-opt IR."""
    triple = target_triple or llvm.get_default_triple()
    default_triple = llvm.get_default_triple()

    if target_triple and target_triple != default_triple:
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
    else:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

    target = llvm.Target.from_triple(triple)
    tm = target.create_target_machine(opt=opt_level, reloc="pic" if position_independent else "default")

    if isinstance(mod_ir, ir.Module):
        mod_ir.triple = triple
        mod_ir.data_layout = str(tm.target_data)
    serialized_ir = str(mod_ir)
    if os.environ.get("JSFX_AOT_TRACE_PHASES"):
        print(f"AOT: LLVM parse ({len(serialized_ir)} source bytes)", file=sys.stderr, flush=True)
    llvm_mod = llvm.parse_assembly(serialized_ir)
    llvm_mod.triple = triple
    llvm_mod.data_layout = str(tm.target_data)
    llvm_mod.verify()

    if out_ll_unopt:
        Path(out_ll_unopt).write_text(str(llvm_mod), encoding="utf-8")

    if opt_level > 0:
        # Use the default (no size optimization). Newer llvmlite versions no
        # longer accept size_level, while speed_level remains supported.
        pto = llvm.PipelineTuningOptions(speed_level=int(opt_level))
        if native_gfx_legacy and len(serialized_ir) > 8 * 1024 * 1024:
            # Receiver specialization already expands large instruments. A
            # bounded inliner avoids multiplying those graphs again. Keep the
            # full requested optimization level and normal small DSP inlining.
            pto.inlining_threshold = 32
        if os.environ.get("JSFX_AOT_TRACE_PHASES"):
            print(f"AOT: LLVM optimize O{opt_level}, inliner={pto.inlining_threshold}", file=sys.stderr, flush=True)
        pb = llvm.create_pass_builder(tm, pto)
        pm = pb.getModulePassManager()
        pm.run(llvm_mod, pb)

    if os.environ.get("JSFX_AOT_TRACE_PHASES"):
        print("AOT: optimization complete", file=sys.stderr, flush=True)
    post_opt_ir = str(llvm_mod)
    if out_ll_opt:
        Path(out_ll_opt).write_text(post_opt_ir, encoding="utf-8")

    if emit_obj:
        if target_triple and (("windows" in target_triple.lower()) or ("apple" in target_triple.lower())):
            clang = shutil.which("clang")
            if not clang:
                raise RuntimeError(
                    "clang not found on PATH. Install LLVM for Windows and ensure clang.exe is on PATH."
                )

            with tempfile.TemporaryDirectory() as td:
                ll_path = Path(td) / "aot.ll"
                ll_path.write_text(post_opt_ir, encoding="utf-8")

                cmd = [
                    clang,
                    f"--target={target_triple}",
                    f"-O{opt_level}",
                    # IR has already passed through our tuned LLVM pipeline.
                    # Select optimized machine-code generation without running
                    # another, unbounded IR inliner over large instruments.
                    "-Xclang", "-disable-llvm-passes",
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

    return post_opt_ir



def validate_slider_declarations(src: str) -> None:
    """Reject slider declarations outside REAPER's slider1..slider256 range.

    Slider declarations live in the JSFX preamble.  Preamble metadata such as
    ``provides: dependencies/*`` is *not* EEL code, so a wildcard there must not
    start a block comment and hide declarations that follow it.  Only comments
    whose first non-whitespace token is a comment delimiter are ignored here.

    Alias-only use can otherwise degrade silently into a normal user variable,
    so an invalid declaration is a compile-time error rather than a host-parser
    warning.
    """
    in_block_comment = False
    for lineno, raw in enumerate(src.splitlines(), 1):
        stripped = raw.lstrip()
        if stripped.startswith("@"):  # first executable section ends preamble
            break

        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue

        if stripped.startswith("//"):
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped[2:]:
                in_block_comment = True
            continue

        m = re.match(r"^slider(\d+)\s*:", stripped, flags=re.IGNORECASE)
        if not m:
            continue
        index = int(m.group(1))
        if index < 1 or index > MAX_JSFX_SLIDERS:
            raise ValueError(
                f"Invalid slider declaration slider{index} at line {lineno}: "
                f"REAPER JSFX supports slider1..slider{MAX_JSFX_SLIDERS}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="DSP-JSFX -> LLVM IR + AOT object + JUCE-callable entry point")
    ap.add_argument("input", help="Path to .jsfx file")
    ap.add_argument("--out-ll", default="", help="Write primary LLVM IR (.ll) to this path (default: stdout)")
    ap.add_argument("--out-ll-unopt", default="", help="Write pre-LLVM-optimization IR after JSFX custom passes")
    ap.add_argument("--out-ll-opt", default="", help="Write post-LLVM-optimization IR")
    ap.add_argument("--opt-report", default="", help="Write movement report (.txt or .json)")
    ap.add_argument("--opt-dump-dir", default="", help="Write original/lowered/optimized section dumps, raw/custom/optimized IR, and movement report into this directory")
    ap.add_argument("--enable-custom-opt", action="store_true", help="Enable all experimental JSFX custom optimization passes (default: off)")
    ap.add_argument("--enable-section-hoist", action="store_true", help="Enable experimental @sample/@block/@init section hoisting")
    ap.add_argument("--enable-loop-hoist", action="store_true", help="Enable experimental loop-invariant scalar hoisting")
    ap.add_argument("--no-section-hoist", action="store_true", help="Force-disable section hoisting even if a broader enable flag was supplied")
    ap.add_argument("--no-loop-hoist", action="store_true", help="Force-disable loop hoisting even if a broader enable flag was supplied")
    ap.add_argument("--out-obj", default="", help="Emit AOT object file (.o/.obj) to this path")
    ap.add_argument("--out-asm", default="", help="Emit AOT assembly (.s) to this path")
    ap.add_argument("--out-h", default="", help="Emit C/C++ header (.h) with State + prototypes")
    ap.add_argument("--meta", default="", help="Optional JSON metadata output")
    ap.add_argument("--opt", type=int, default=2, help="Optimization level for AOT (0-3). Default 2.")
    ap.add_argument("--target", default="", help="LLVM target triple for AOT object/asm (e.g. x86_64-pc-windows-msvc)")
    ap.add_argument("--faust-include", action="append", default=[], help="Faust library include directory (repeatable)")
    ap.add_argument("--native-gfx-prototype", action="store_true", help="Compile validated native @gfx with explicit interactive/publication contracts")
    ap.add_argument("--native-gfx-legacy", action="store_true", help="Compile native @gfx with live shared variables/RAM and a fixed maxmem heap (requires C++20)")
    args = ap.parse_args()
    if args.native_gfx_prototype and args.native_gfx_legacy:
        ap.error("Choose either --native-gfx-prototype or --native-gfx-legacy")
    args.native_gfx_prototype = args.native_gfx_prototype or args.native_gfx_legacy

    input_path = Path(args.input).resolve()
    txt = input_path.read_text(encoding="utf-8", errors="replace")
    txt = preprocess_jsfx_imports(txt, input_path)
    validate_slider_declarations(txt)

    enable_section_hoists = bool(args.enable_custom_opt or args.enable_section_hoist)
    enable_loop_hoists = bool(args.enable_custom_opt or args.enable_loop_hoist)
    if args.no_section_hoist:
        enable_section_hoists = False
    if args.no_loop_hoist:
        enable_loop_hoists = False
    want_opt_report = bool(args.opt_report or args.opt_dump_dir)

    pipeline = prepare_jsfx_pipeline(
        txt,
        include_gfx=args.native_gfx_prototype,
        native_gfx_legacy=args.native_gfx_legacy,
        enable_section_hoists=enable_section_hoists,
        enable_loop_hoists=enable_loop_hoists,
        collect_opt_report=want_opt_report,
    )
    if pipeline.get('faust_plan'):
        pipeline['faust_plan']['include_paths']=[str(input_path.parent),*args.faust_include]
    mod, meta = compile_jsfx_to_ir(
        txt,
        enable_section_hoists=enable_section_hoists,
        enable_loop_hoists=enable_loop_hoists,
        pipeline=pipeline,
        native_gfx_prototype=args.native_gfx_prototype,
        native_gfx_legacy=args.native_gfx_legacy,
        stream_functions=args.native_gfx_legacy,
    )

    dump_dir: Optional[Path] = None
    dump_unopt_path: Optional[str] = args.out_ll_unopt or None
    dump_opt_path: Optional[str] = args.out_ll_opt or None

    if args.opt_dump_dir:
        dump_dir = _ensure_dir(args.opt_dump_dir)
        (dump_dir / "01_sections_original.txt").write_text(
            _emit_original_sections_text(pipeline["sections"]),
            encoding="utf-8",
        )
        (dump_dir / "02_sections_lowered.txt").write_text(
            _emit_reconstructed_sections_text(pipeline["lowered_programs"], pipeline["fn_defs"], annotate=True),
            encoding="utf-8",
        )
        (dump_dir / "03_sections_after_custom_opt.txt").write_text(
            _emit_reconstructed_sections_text(pipeline["programs"], pipeline["fn_defs"], annotate=True),
            encoding="utf-8",
        )
        _write_opt_report(str(dump_dir / "10_opt_report.txt"), pipeline["opt_report"])
        _write_opt_report(str(dump_dir / "11_opt_report.json"), pipeline["opt_report"])

        raw_pipeline = prepare_jsfx_pipeline(
            txt,
            include_gfx=args.native_gfx_prototype,
            native_gfx_legacy=args.native_gfx_legacy,
            enable_section_hoists=False,
            enable_loop_hoists=False,
            collect_opt_report=False,
        )
        if raw_pipeline.get('faust_plan'):
            raw_pipeline['faust_plan']['include_paths']=[str(input_path.parent),*args.faust_include]
        raw_mod, _ = compile_jsfx_to_ir(
            txt,
            enable_section_hoists=False,
            enable_loop_hoists=False,
            pipeline=raw_pipeline,
            native_gfx_prototype=args.native_gfx_prototype,
            native_gfx_legacy=args.native_gfx_legacy,
            stream_functions=args.native_gfx_legacy,
        )
        _aot_opt_and_emit(
            mod_ir=raw_mod,
            opt_level=0,
            emit_obj=None,
            emit_asm=None,
            target_triple=(args.target or None),
            out_ll_unopt=str(dump_dir / "20_ir_before_custom_opt.ll"),
            out_ll_opt=None,
        )

        if not dump_unopt_path:
            dump_unopt_path = str(dump_dir / "30_ir_after_custom_opt_before_llvm.ll")
        if not dump_opt_path:
            dump_opt_path = str(dump_dir / "40_ir_after_llvm_opt.ll")

    opt_report_data = pipeline.get("opt_report")
    want_llvm_opt_output = bool(args.out_obj or args.out_asm or dump_unopt_path or dump_opt_path)
    if want_llvm_opt_output:
        # Large native GFX modules must not retain the Python instruction graph
        # and all AST specialization snapshots during LLVM optimization.
        serialized_mod = str(mod)
        del mod, pipeline
        if dump_dir:
            del raw_mod, raw_pipeline
        import gc
        gc.collect()
        meta["aot_inlining"] = "bounded-32" if args.native_gfx_legacy and len(serialized_mod) > 8*1024*1024 else "default"
        ir_text = _aot_opt_and_emit(
            mod_ir=serialized_mod,
            opt_level=max(0, min(3, int(args.opt))),
            emit_obj=args.out_obj or None,
            emit_asm=args.out_asm or None,
            target_triple=(args.target or None),
            out_ll_unopt=dump_unopt_path,
            out_ll_opt=dump_opt_path,
            position_independent=args.native_gfx_prototype,
            native_gfx_legacy=args.native_gfx_legacy,
        )
    else:
        ir_text = str(mod)

    if args.out_ll:
        Path(args.out_ll).write_text(ir_text, encoding="utf-8")
    else:
        print(ir_text)

    if args.opt_report:
        _write_opt_report(args.opt_report, opt_report_data)

    if args.out_h:
        Path(args.out_h).write_text(_emit_header(meta), encoding="utf-8")

    if args.meta:
        Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
