#include "Frontend.h"
#include <bit>
#include <charconv>
#include <map>
#include <locale.h>
#include <regex>
#include <set>

namespace za::compiler
{
double parseDecimal(const std::string& text, char** end)
{
    // A library caller's locale must not change JSFX's decimal point or rounding.
#ifdef _WIN32
    struct Locale { _locale_t value=_create_locale(LC_NUMERIC,"C"); ~Locale() { _free_locale(value); } };
    static const Locale locale;
    return _strtod_l(text.c_str(),end,locale.value);
#else
    struct Locale { locale_t value=newlocale(LC_NUMERIC_MASK,"C",nullptr); ~Locale() { freelocale(value); } };
    static const Locale locale;
    return strtod_l(text.c_str(),end,locale.value);
#endif
}
String decimalIntegerHex(String source)
{
    std::string result="0";
    for (auto c:source)
    {
        int carry= c>='0' && c<='9' ? int(c-'0') : c>='a' && c<='f' ? int(c-'a'+10) : int(c-'A'+10);
        for (auto i=result.rbegin();i!=result.rend();++i) { int n=(*i-'0')*16+carry; *i=char('0'+n%10); carry=n/10; }
        while (carry) { result.insert(result.begin(),char('0'+carry%10)); carry/=10; }
    }
    return String(result);
}
var object(std::initializer_list<std::pair<String, var>> fields)
{
    auto* result = new juce::DynamicObject;
    for (auto& [name, value] : fields) result->setProperty(juce::Identifier(name), value);
    return var(result);
}
namespace
{
struct Span { juce::int64 line; int col; var json() const { return object({{"line", line}, {"col", col}}); } };
struct Token { String kind, text; Span span; var json() const { return object({{"kind", kind}, {"text", text}, {"span", span.json()}}); } };
bool digit(juce::juce_wchar c) { return c >= '0' && c <= '9'; }
bool first(juce::juce_wchar c) { return c == '#' || c == '$' || c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool rest(juce::juce_wchar c) { return first(c) || digit(c); }
int hex(juce::juce_wchar c) { return digit(c) ? int(c - '0') : c >= 'a' && c <= 'f' ? int(c - 'a' + 10) : c >= 'A' && c <= 'F' ? int(c - 'A' + 10) : -1; }
String repr(String s)
{
    String q = s.containsChar('\'') && !s.containsChar('"') ? "\"" : "'", out = q;
    for (auto c : s)
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c == '\\' || String::charToString(c) == q) out += "\\" + String::charToString(c);
        else out += String::charToString(c);
    return out + q;
}
class Lexer
{
    std::vector<juce::juce_wchar> source;
    int pos = 0, col = 1, length;
    juce::int64 line;
    const std::atomic_bool* cancelled;
    juce::juce_wchar peek(int n = 0) const { return pos + n < length ? source[size_t(pos + n)] : 0; }
    String slice(int start, int end) const { return String(juce::CharPointer_UTF32(source.data()+start), juce::CharPointer_UTF32(source.data()+end)); }
    void advance(int n = 1) { while (n-- && pos < length) { if (source[pos++] == '\n') { ++line; col = 1; } else ++col; } }
    [[noreturn]] void fail(String message) { throw Failure("lexer", message, line, col); }
public:
    Lexer(String s, int base, const std::atomic_bool* c) : line(base), length(s.length()), cancelled(c) { for (auto ch:s) source.push_back(ch); source.push_back(0); }
    Token next()
    {
        while (true)
        {
            if (cancelled && cancelled->load()) throw Failure("cancelled", "Compilation cancelled");
            Span sp{line, col}; auto c = peek();
            if (pos >= length) return {"eof", "", sp};
            if (c == ' ' || c == '\t' || c == '\r') { advance(); continue; }
            if (c == '\n') { advance(); return {"eol", "\n", sp}; }
            if (c == '/' && peek(1) == '/') { while (peek() && peek() != '\n') advance(); continue; }
            if (c == '/' && peek(1) == '*')
            {
                advance(2);
                while (!(peek() == '*' && peek(1) == '/')) { if (!peek()) throw Failure("lexer", "Unterminated /* comment */"); advance(); }
                advance(2); continue;
            }
            if (c == '$' && peek(1) == '~' && digit(peek(2)))
            {
                advance(2); int bits = 0;
                while (digit(peek())) { bits = std::min(53, bits * 10 + int(peek() - '0')); advance(); }
                return {"num", String(juce::int64((uint64_t(1) << bits) - 1)), sp};
            }
            if (c == '$' && peek(1) == '\'' && peek(3) == '\'') { auto v = peek(2); advance(4); return {"num", String(int(v)), sp}; }
            if ((c == '0' || c == '$') && (peek(1) == 'x' || peek(1) == 'X') && hex(peek(2)) >= 0)
            {
                advance(2); int start = pos; while (hex(peek()) >= 0) advance();
                return {"num", decimalIntegerHex(slice(start, pos)), sp};
            }
            static const String ops[] = {"===", "!==", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "^=", "|=", "&=", "~=", "&&", "||", "<<", ">>"};
            for (auto op : ops) if (pos+op.length()<=length && slice(pos, pos + op.length()) == op) { advance(op.length()); return {"op", op, sp}; }
            if (digit(c) || (c == '.' && digit(peek(1))))
            {
                int start = pos;
                while (digit(peek())) advance();
                if (peek() == '.') { advance(); while (digit(peek())) advance(); }
                if ((peek() == 'e' || peek() == 'E') && (digit(peek(1)) || ((peek(1) == '+' || peek(1) == '-') && digit(peek(2)))))
                { advance(); if (peek() == '+' || peek() == '-') advance(); while (digit(peek())) advance(); }
                return {"num", slice(start, pos), sp};
            }
            if (first(c))
            {
                int start = pos; advance(); while (rest(peek())) advance();
                while (peek() == '.' && first(peek(1))) { advance(2); while (rest(peek())) advance(); }
                auto text = slice(start, pos).toLowerCase();
                return {text == "if" || text == "else" || text == "while" ? "kw" : "ident", text, sp};
            }
            if (c == '"' || c == '\'')
            {
                auto quote = c; std::string out; advance();
                while (true)
                {
                    auto ch = peek(); if (!ch) fail("Unterminated string literal");
                    if (ch == quote) { advance(); break; }
                    if (ch == '\\')
                    {
                        advance(); auto esc = peek(); if (!esc) fail("Unterminated string escape"); advance();
                        if (esc == 'n') out += '\n';
                        else if (esc == 'r') out += '\r';
                        else if (esc == 't') out += '\t';
                        else if ((esc == 'x' || esc == 'X') && hex(peek()) >= 0 && hex(peek(1)) >= 0)
                        { int value=hex(peek()) * 16 + hex(peek(1)); out += value==0 ? std::string(1,'\0') : String::charToString(juce::juce_wchar(value)).toStdString(); advance(2); }
                        else if (esc == '0') out += '\0';
                        else out += String::charToString(esc).toStdString();
                    }
                    else { out += String::charToString(ch).toStdString(); advance(); }
                }
                if (quote == '\'')
                {
                    if (out.size() > 4) fail("Packed character literal exceeds four bytes");
                    uint64_t value = 0; for (unsigned char b : out) value = (value << 8) | b;
                    return {"num", String(juce::int64(value)), sp};
                }
                return {"str", String::toHexString(out.data(),int(out.size()),0), sp};
            }
            if (String("()[]{},;:+-*/=<>&|!?:^%~").containsChar(c))
            {
                advance(); return {c == ';' ? "semi" : String("();,[]{}").containsChar(c) ? "punc" : "op", String::charToString(c), sp};
            }
            throw Failure("lexer", "Unexpected character " + repr(String::charToString(c)), sp.line, sp.col);
        }
    }
};
var node(String type, Span span, std::initializer_list<std::pair<String, var>> fields)
{
    int depth=1;
    for (auto& [_,value]:fields)
    {
        if (value.isObject() && value.hasProperty("__depth")) depth=std::max(depth,1+int(value["__depth"]));
        if (auto* list=value.getArray()) for (auto& child:*list) if (child.isObject() && child.hasProperty("__depth")) depth=std::max(depth,1+int(child["__depth"]));
    }
    if (depth>512) throw Failure("limits","AST nesting exceeds 512",span.line,span.col);
    auto n = object({{"type", type}, {"span", span.json()}});
    n.getDynamicObject()->setProperty("__depth",depth);
    for (auto& [name, value] : fields) n.getDynamicObject()->setProperty(juce::Identifier(name), value);
    return n;
}
Span spanOf(const var& n) { return {juce::int64(n["span"]["line"]), int(n["span"]["col"])}; }
var number(Span s, double value)
{
    auto bits = std::bit_cast<uint64_t>(value);
    auto text = String::toHexString(juce::int64(bits)).paddedLeft('0', 16);
    return node("Num", s, {{"value", text}});
}
class Parser
{
    Lexer lex;
    Token cur, nxt;
    int depth = 0;
    struct Depth { int& d; Depth(int& v) : d(v) { if (++d > 512) throw Failure("limits", "Parser nesting exceeds 512"); } ~Depth() { --d; } };
    void adv() { cur = nxt; nxt = lex.next(); }
    [[noreturn]] void fail(String msg) { throw Failure("parser", msg, cur.span.line, cur.span.col); }
    bool at(String kind, String text = {}) const { return cur.kind == kind && (text.isEmpty() || cur.text == text); }
    Token eat(String kind, String text = {})
    {
        if (cur.kind != kind) fail("Expected " + kind + ", got " + cur.kind + " " + repr(cur.text));
        if (text.isNotEmpty() && cur.text != text) fail("Expected " + repr(text) + ", got " + repr(cur.text));
        auto t = cur; adv(); return t;
    }
    void seps() { while (at("eol") || at("semi")) adv(); }
    void eols() { while (at("eol")) adv(); }
    static int precedence(String op)
    {
        static const std::map<String,int> table = {{"=",11},{"+=",11},{"-=",11},{"*=",11},{"/=",11},{"%=",11},{"^=",11},{"|=",11},{"&=",11},{"~=",11},{"?",1},{"||",2},{"&&",2},{"==",3},{"!=",3},{"===",3},{"!==",3},{"<",3},{"<=",3},{">",3},{">=",3},{"|",4},{"&",4},{"~",4},{"+",5},{"-",6},{"*",7},{"/",8},{"%",9},{"<<",9},{">>",9},{"^",10}};
        auto i = table.find(op); return i == table.end() ? -1 : i->second;
    }
    static bool assignment(String op) { return op == "=" || (op.length() == 2 && String("+-*/%^|&~").containsChar(op[0]) && op[1] == '='); }
    static bool target(const var& n)
    { return n["type"] == var("Var") || n["type"] == var("Index") || (n["type"] == var("Call") && (n["fn"] == var("slider") || n["fn"] == var("spl")) && n["args"].size() == 1); }
    var seqStatement() { return at("kw", "if") ? parseIf() : at("kw", "while") ? parseWhile() : expr(0); }
    var statement() { return at("ident", "function") ? function() : seqStatement(); }
    var parseIf()
    {
        auto t = eat("kw", "if"); eat("punc", "("); auto cond = expr(0); eat("punc", ")"); seps();
        auto then = expr(0); seps(); var els;
        if (at("kw", "else")) { adv(); seps(); els = expr(0); seps(); }
        return node("If", t.span, {{"cond", cond}, {"then", then}, {"els", els}});
    }
    var parseWhile()
    {
        auto t = eat("kw", "while"); eols(); eat("punc", "("); seps();
        if (at("punc", ")")) fail("while() requires an expression");
        Array<var> items; items.add(seqStatement());
        while (true)
        {
            if (at("punc", ")")) { adv(); break; }
            if (!at("eol") && !at("semi")) fail("Expected separator or ')' in while() expression");
            seps(); if (at("punc", ")")) { adv(); break; } items.add(seqStatement());
        }
        auto cond = items.size() == 1 ? items[0] : node("Seq", t.span, {{"items", var(items)}});
        eols(); auto body = at("punc", "(") ? primary() : number(t.span, 0);
        return node("While", t.span, {{"cond", cond}, {"body", body}});
    }
    Array<var> names(String label)
    {
        Array<var> out; eat("punc", "("); seps();
        while (!at("punc", ")"))
        {
            seps(); if (at("punc", ")")) break;
            if (at("punc", ",")) { adv(); continue; }
            if (!at("ident")) fail("Expected " + label + " name");
            out.add(eat("ident").text); seps();
            if (at("punc", ",")) { adv(); continue; }
            if (at("ident")) continue;
            break;
        }
        seps(); eat("punc", ")"); return out;
    }
    var function()
    {
        auto t = eat("ident", "function"); if (!at("ident")) fail("Expected function name after 'function'");
        auto name = eat("ident").text; auto params = names("parameter"); Array<var> locals, instances; seps();
        while (at("ident") && (cur.text == "local" || cur.text == "instance" || cur.text == "global" || cur.text == "globals"))
        {
            auto qual = cur.text; adv(); auto ns = names(qual + " variable");
            if (qual == "local") locals.addArray(ns); else if (qual == "instance") instances.addArray(ns); seps();
        }
        if (!at("punc", "(")) fail("Expected '(' to start function body");
        auto body = primary(); seps(); if (at("semi")) adv();
        return node("FunctionDef", t.span, {{"name", name}, {"params", var(params)}, {"locals", var(locals)}, {"instances", var(instances)}, {"body", body}, {"cell_params", var(Array<var>())}});
    }
    var call(Span s, String fn, Array<var> args) { return node("Call", s, {{"fn", fn}, {"args", var(args)}, {"cell_args", var(Array<var>())}}); }
    var deferred(String fn, Span s)
    {
        Array<var> args;
        int prefix = fn == "defer" ? 0 : fn == "defer_after" ? 1 : fn == "defer_reduce" ? 4 : 2;
        for (int i = 0; i < prefix; ++i)
        {
            seps(); auto arg = expr(0);
            if ((fn == "defer_for" || fn == "defer_reduce") && i == 0)
            {
                if (arg["type"] != var("Var")) throw Failure("parser", "Deferred iteration requires an index variable");
                auto name = arg["name"].toString();
                static const std::set<String> reserved = {"mem","gmem","srate","samplesblock","midi_bus","ext_midi_bus","task_invalid","task_busy","task_pending","task_running","task_succeeded","task_cancelled","task_failed"};
                auto numbered = [&](String prefix, int low, int high) { auto tail = name.substring(prefix.length()); if (!name.startsWith(prefix) || tail.isEmpty() || !tail.containsOnly("0123456789")) return false; auto n = tail.getIntValue(); return n >= low && n <= high; };
                if (reserved.count(name) || name.startsWith("#") || name.startsWith("gfx_") || name.startsWith("mouse_") || numbered("slider",1,256) || numbered("spl",0,63))
                    throw Failure("parser", "Deferred iteration requires an ordinary private index variable");
            }
            if (fn == "defer_reduce" && i == 2)
            {
                static const std::map<String,int> reductions = {{"sum",2},{"product",3},{"min",4},{"max",5},{"any",6},{"all",7}};
                auto it = reductions.find(arg["name"].toString());
                if (arg["type"] != var("Var") || it == reductions.end()) throw Failure("parser", "Reduction must be SUM, PRODUCT, MIN, MAX, ANY or ALL");
                arg = number(spanOf(arg), it->second);
            }
            args.add(arg); seps(); eat("punc", ",");
        }
        seps(); Array<var> body;
        while (!at("punc", ")")) { body.add(seqStatement()); seps(); }
        adv(); args.add(body.isEmpty() ? number(s, 0) : node("Seq", s, {{"items", var(body)}})); return call(s, fn, args);
    }
    var expr(int minPrec)
    {
        Depth guard(depth); eols(); var lhs;
        if (at("op") && (cur.text == "+" || cur.text == "-" || cur.text == "!")) { auto t = cur; adv(); lhs = node("Unary", t.span, {{"op", t.text}, {"a", expr(11)}}); }
        else lhs = postfix();
        while (true)
        {
            while (at("eol") && (nxt.kind == "eol" || (nxt.kind == "op" && precedence(nxt.text) >= minPrec))) adv();
            if (!at("op") || cur.text == "?" || cur.text == ":") break;
            auto op = cur.text; int prec = precedence(op); if (prec < minPrec) break;
            bool assign = assignment(op); adv(); auto rhs = expr(assign ? 0 : prec + 1); auto sp = spanOf(lhs);
            if (assign)
            {
                if (!target(lhs)) fail("Assignment target must be a variable, index, or slider()/spl() reference");
                if (lhs["type"] == var("Var") && lhs["name"].toString().startsWith("#") && (op == "=" || op == "+=")) lhs = call(sp, op == "=" ? "strcpy" : "strcat", {lhs, rhs});
                else lhs = node("Assign", sp, {{"op", op}, {"target", lhs}, {"value", rhs}});
            }
            else lhs = node("Binary", sp, {{"op", op}, {"l", lhs}, {"r", rhs}});
        }
        while (at("eol") && (nxt.kind == "eol" || (nxt.kind == "op" && nxt.text == "?"))) adv();
        if (at("op", "?") && 1 >= minPrec)
        {
            auto q = cur; adv(); seps(); auto then = at("op", ":") ? number(q.span, 0) : expr(0);
            while (at("eol") && (nxt.kind == "eol" || (nxt.kind == "op" && nxt.text == ":"))) adv();
            var els; if (at("op", ":")) { adv(); seps(); els = expr(0); } else els = number(q.span, 0);
            lhs = node("Ternary", q.span, {{"cond", lhs}, {"then", then}, {"els", els}});
        }
        return lhs;
    }
    var postfix()
    {
        auto n = primary();
        while (true)
        {
            if (at("punc", "("))
            {
                auto s = cur.span; adv(); if (n["type"] != var("Var")) fail("Can only call a named function"); auto fn = n["name"].toString();
                if (fn == "defer" || fn == "defer_after" || fn == "defer_for" || fn == "defer_reduce" || fn == "defer_arena") { n = deferred(fn, s); continue; }
                seps();
                if (fn == "loop")
                {
                    auto count = expr(0); seps(); if (at("punc", ",")) adv(); seps(); Array<var> items;
                    while (!at("punc", ")")) { items.add(seqStatement()); seps(); }
                    adv(); auto body = items.isEmpty() ? number(s,0) : items.size() == 1 ? items[0] : node("Seq", s, {{"items", var(items)}});
                    n = node("Loop", s, {{"count", count}, {"body", body}}); continue;
                }
                Array<var> args;
                if (!at("punc", ")")) while (true) { seps(); args.add(expr(0)); seps(); if (at("punc", ",")) { adv(); continue; } break; }
                seps(); eat("punc", ")"); n = call(s, fn, args); continue;
            }
            if (at("punc", "["))
            {
                auto s = cur.span; adv(); seps(); auto idx = at("punc", "]") ? number(s,0) : expr(0); seps(); eat("punc", "]");
                n = node("Index", s, {{"base", n}, {"index", idx}}); continue;
            }
            break;
        }
        return n;
    }
    var primary()
    {
        if (at("kw", "while")) return parseWhile();
        if (at("num")) { auto t = eat("num"); return number(t.span, parseDecimal(t.text.toStdString())); }
        if (at("str")) { auto t = eat("str"); return node("StrLit", t.span, {{"value", t.text}}); }
        if (at("ident")) { auto t = eat("ident"); return node("Var", t.span, {{"name", t.text}}); }
        if (at("punc", "("))
        {
            auto s = cur.span; adv(); seps(); Array<var> items;
            if (at("punc", ")")) { adv(); return node("Seq", s, {{"items", var(items)}}); }
            auto first = seqStatement(); if (at("punc", ")")) { adv(); return first; } items.add(first);
            while (true) { seps(); if (at("punc", ")")) { adv(); break; } items.add(seqStatement()); }
            return node("Seq", s, {{"items", var(items)}});
        }
        fail("Expected number, identifier, or '('");
    }
public:
    Parser(String source, int line, const std::atomic_bool* c) : lex(source, line, c), cur(lex.next()), nxt(lex.next()) {}
    var program()
    {
        Array<var> out; seps(); while (!at("eof")) out.add(statement()), seps();
        // Remove internal depth accounting iteratively before exposing the AST.
        std::vector<var> pending(out.begin(),out.end());
        while (!pending.empty())
        {
            auto value=pending.back(); pending.pop_back();
            if (auto* o=value.getDynamicObject()) { o->removeProperty("__depth"); for (auto& field:o->getProperties()) if (field.value.isObject() || field.value.isArray()) pending.push_back(field.value); }
            else if (auto* list=value.getArray()) for (auto& child:*list) if (child.isObject() || child.isArray()) pending.push_back(child);
        }
        return var(out);
    }
};
}
var frontend(const var& request, const std::atomic_bool* cancelled)
{
    const auto start = juce::Time::getMillisecondCounterHiRes();
    var result = object({{"protocol", protocolVersion}, {"backend", "cpp-experimental"}, {"stage", "frontend"}, {"ok", false}});
    try
    {
        if (!request.isObject() || !request["protocol"].isInt() || int(request["protocol"]) != protocolVersion) throw Failure("protocol", "Expected protocol 1");
        if (request["stage"] != var("frontend")) throw Failure("protocol", "Only frontend stage is implemented");
        if (!request["source"].isString()) throw Failure("protocol", "source must be a string");
        static const std::set<String> allowed={"protocol","stage","source","resolve","sourcePath","packageRoot","searchRoots","snippet","baseLine"};
        std::set<String> unknown;
        for (auto& field:request.getDynamicObject()->getProperties()) if (!allowed.count(field.name.toString())) unknown.insert(field.name.toString());
        if (!unknown.empty()) throw Failure("protocol","Unknown request field: "+*unknown.begin());
        for (String name:{"resolve","snippet"}) if (request.hasProperty(juce::Identifier(name)) && !request[juce::Identifier(name)].isBool()) throw Failure("protocol",name+" must be boolean");
        for (String name:{"sourcePath","packageRoot"}) if (request.hasProperty(juce::Identifier(name)) && (!request[juce::Identifier(name)].isString() || request[juce::Identifier(name)].toString().isEmpty())) throw Failure("protocol",name+" must be a nonempty string");
        if (request.hasProperty("baseLine") && (!request["baseLine"].isInt() || int(request["baseLine"])<1)) throw Failure("protocol","baseLine must be a positive 32-bit integer");
        if (request.hasProperty("searchRoots"))
        {
            if (!request["searchRoots"].isArray()) throw Failure("protocol","searchRoots must be an array");
            for (auto& root:*request["searchRoots"].getArray()) if (!root.isString() || root.toString().isEmpty()) throw Failure("protocol","searchRoots entries must be nonempty strings");
        }
        auto source = request["source"].toString();
        if (source.getNumBytesAsUTF8() > 4194304) throw Failure("limits", "Source exceeds 4 MiB");
        auto resolved = request["resolve"] == var(true) ? resolveSource(request, cancelled) : object({{"text", source}, {"dependencies", var(Array<var>())}, {"sectionSources", object({})}});
        source = resolved["text"].toString();
        if (source.getNumBytesAsUTF8() > 4194304) throw Failure("limits", "Expanded source exceeds 4 MiB");
        double resolvedAt = juce::Time::getMillisecondCounterHiRes();
        auto* properties = result.getDynamicObject(); properties->setProperty("resolution", resolved);
        Array<var> sections;
        if (request["snippet"] == var(true)) sections.add(object({{"name", "snippet"}, {"source", source}, {"line", request.hasProperty("baseLine") ? request["baseLine"] : var(1)}}));
        else
        {
            std::regex marker(R"(^\s*@([A-Za-z_][A-Za-z0-9_]*)\b.*$)");
            std::map<String,int> indexes; int current = -1, line = 1;
            auto text = source.toStdString(); size_t offset = 0;
            while (offset < text.size())
            {
                size_t end = text.find('\n',offset); if (end == std::string::npos) end = text.size(); else ++end;
                auto raw = text.substr(offset,end-offset); auto stripped = raw; while (!stripped.empty() && (stripped.back() == '\n' || stripped.back() == '\r')) stripped.pop_back();
                std::smatch m;
                if (std::regex_match(stripped,m,marker))
                {
                    String name = String(m[1].str()).toLowerCase(); auto i = indexes.find(name);
                    if (i == indexes.end()) { current = sections.size(); indexes[name] = current; sections.add(object({{"name",name},{"source",""},{"line",line+1}})); } else current = i->second;
                }
                else if (current >= 0) sections.getReference(current).getDynamicObject()->setProperty("source", sections[current]["source"].toString() + String::fromUTF8(raw.data(),int(raw.size())));
                offset = end; ++line;
            }
        }
        for (auto& section : sections)
        {
            auto name = section["name"].toString();
            if (name == "faust" || name.startsWith("faust_")) { section.getDynamicObject()->setProperty("opaque", true); continue; }
            auto code = section["source"].toString(); int line = int(section["line"]); Lexer lexer(code,line,cancelled); Array<var> tokens;
            while (true) { auto t = lexer.next(); tokens.add(t.json()); if (t.kind == "eof") break; }
            section.getDynamicObject()->setProperty("tokens", var(tokens));
            section.getDynamicObject()->setProperty("ast", Parser(code,line,cancelled).program());
        }
        properties->setProperty("sections",var(sections)); properties->setProperty("ok",true);
        properties->setProperty("phaseSeconds",object({{"resolution",(resolvedAt-start)/1000.0},{"lexerParser",(juce::Time::getMillisecondCounterHiRes()-resolvedAt)/1000.0}}));
    }
    catch (const Failure& e) { result.getDynamicObject()->setProperty("diagnostics", var(Array<var>{object({{"severity","error"},{"phase",e.phase},{"message",e.message},{"file",e.file.isNotEmpty()?var(e.file):request["sourcePath"]},{"line",e.line},{"column",e.column}})})); }
    catch (const std::exception& e) { result.getDynamicObject()->setProperty("diagnostics", var(Array<var>{object({{"severity","error"},{"phase","internal"},{"message",String(e.what())}})})); }
    return result;
}
}
