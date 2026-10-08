#include "Frontend.h"
#include "WDL/wdlstring.h"
#include "WDL/ptrlist.h"
#include "WDL/eel2/eel_pproc.h"
#include <filesystem>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <set>

void NSEEL_HOSTSTUB_EnterMutex() {}
void NSEEL_HOSTSTUB_LeaveMutex() {}
namespace za::compiler
{
namespace
{
namespace fs = std::filesystem;
using Text = std::string;
Text lower(Text s) { for (auto& c : s) if (c >= 'A' && c <= 'Z') c = char(c + 'a' - 'A'); return s; }
Text trim(Text s) { auto start = s.find_first_not_of(" \t\r\n"); if (start == Text::npos) return {}; return s.substr(start, s.find_last_not_of(" \t\r\n")-start+1); }
Text pathText(const fs::path& p) { auto s = p.u8string(); return Text(reinterpret_cast<const char*>(s.data()),s.size()); }
fs::path pathFrom(String s) { return fs::u8path(s.toStdString()); }
fs::path resolvedPath(const fs::path& p)
{
    // Windows' weakly_canonical can report access denied for a missing leaf.
    // Resolve the existing ancestor (including links), then append absent components.
    auto absolute=fs::absolute(p); std::error_code error;
    auto result=fs::weakly_canonical(absolute,error); if (!error) return result;
    std::vector<fs::path> tail; auto ancestor=absolute;
    while (ancestor!=ancestor.parent_path())
    {
        tail.push_back(ancestor.filename()); ancestor=ancestor.parent_path();
        error.clear(); result=fs::canonical(ancestor,error);
        if (!error) { for (auto i=tail.rbegin();i!=tail.rend();++i) result/=*i; return result.lexically_normal(); }
    }
    throw Failure("source","Cannot resolve source path: "+String(pathText(absolute)));
}
Text key(const fs::path& p)
{
    auto s = pathText(p);
#ifdef _WIN32
    return lower(s);
#else
    return s;
#endif
}
Text read(const fs::path& p)
{
    std::ifstream file(p,std::ios::binary); if (!file) throw Failure("source", "Cannot read " + String(pathText(p)));
    Text s((std::istreambuf_iterator<char>(file)),{});
    if (!juce::CharPointer_UTF8::isValidString(s.data(),int(s.size()))) throw Failure("source","Invalid UTF-8 in "+String(pathText(p)));
    if (s.starts_with("\xef\xbb\xbf")) s.erase(0,3);
    Text out; for (size_t i=0;i<s.size();++i) { if (s[i]=='\r') { out+='\n'; if (i+1<s.size() && s[i+1]=='\n') ++i; } else out+=s[i]; }
    return out;
}
std::vector<Text> lines(const Text& s)
{
    std::vector<Text> out; size_t offset=0;
    while (offset<s.size()) { size_t end=s.find('\n',offset); if (end==Text::npos) end=s.size(); else ++end; out.push_back(s.substr(offset,end-offset)); offset=end; } return out;
}
const std::regex section(R"(^\s*@([A-Za-z_][A-Za-z0-9_]*)\b)",std::regex::icase);
const std::regex import(R"(^\s*import\b)",std::regex::icase);
std::vector<std::pair<Text,Text>> codeLines(const Text& source)
{
    bool block=false, started=false, preBlock=false, metadata=false; char quote=0;
    std::vector<std::pair<Text,Text>> out;
    auto blank=[](Text s) { for (auto& c:s) if (c!='\n' && c!='\r') c=' '; return s; };
    for (auto raw:lines(source))
    {
        if (!started)
        {
            if (metadata) { if (trim(raw).empty() || std::isspace(static_cast<unsigned char>(raw[0]))) { out.emplace_back(raw,raw); continue; } metadata=false; }
            auto stripped=trim(raw);
            if (preBlock) { out.emplace_back(raw,blank(raw)); if (raw.find("*/")!=Text::npos) preBlock=false; continue; }
            if (stripped.starts_with("/*")) { out.emplace_back(raw,blank(raw)); if (stripped.find("*/",2)==Text::npos) preBlock=true; continue; }
            if (stripped.starts_with("//")) { out.emplace_back(raw,blank(raw)); continue; }
            std::smatch header;
            if (std::regex_search(raw,header,std::regex(R"(^\s*([A-Za-z_][A-Za-z0-9_]*):)")))
            {
                auto name=lower(header[1].str()); if ((name=="provides" || name=="about") && trim(raw.substr(header.length())).empty()) metadata=true;
                out.emplace_back(raw,raw); continue;
            }
            if (!std::regex_search(raw,section) && !std::regex_search(raw,import)) { out.emplace_back(raw,raw); continue; }
            if (std::regex_search(raw,section)) started=true;
        }
        Text mask=raw;
        for (size_t i=0;i<raw.size();)
        {
            char c=raw[i];
            if (block) { mask[i]=' '; if (raw.compare(i,2,"*/")==0) { mask[i+1]=' '; i+=2; block=false; } else ++i; }
            else if (quote) { mask[i]=' '; if (c=='\\' && i+1<raw.size()) { mask[i+1]=' '; i+=2; } else { if (c==quote) quote=0; ++i; } }
            else if (raw.compare(i,2,"//")==0) { std::fill(mask.begin()+i,mask.end(),' '); break; }
            else if (raw.compare(i,2,"/*")==0) { mask[i]=mask[i+1]=' '; block=true; i+=2; }
            else if (c=='"' || c=='\'') { quote=c; mask[i]=' '; ++i; }
            else ++i;
        }
        out.emplace_back(raw,mask);
    }
    return out;
}
struct Section { Text name, header, body; };
struct Unit { fs::path path; Text text,preamble; std::vector<Section> sections; std::vector<std::pair<int,Text>> imports; Section* get(const Text& name) { for (auto& s:sections) if (s.name==name) return &s; return nullptr; } };
Unit parse(const fs::path& path,const Text& source)
{
    Unit unit{path,source}; int current=-1,line=0; std::map<Text,int> repeated;
    bool mixed=std::regex_search(source,std::regex(R"((^|\n)\s*@faust\b)",std::regex::icase));
    for (auto& [raw,mask]:codeLines(source))
    {
        ++line; std::smatch sec,imp;
        if (std::regex_search(mask,imp,import) && !(current>=0 && unit.sections[current].name.starts_with("faust")))
        {
            size_t start=lower(mask).find("import"); Text remaining=raw.substr(start); std::smatch token;
            static const std::regex grammar(R"re(^import\s+(?:"([^"]+)"|'([^']+)'|([^\s;]+)))re",std::regex::icase);
            if (!std::regex_search(remaining,token,grammar)) throw Failure("source", String(pathText(path))+":"+String(line)+": invalid import directive: "+String(trim(raw)),line);
            Text tail=mask.substr(start+token.length()); for (auto& c:tail) if (c==';') c=' ';
            if (!trim(tail).empty()) throw Failure("source", String(pathText(path))+":"+String(line)+": invalid import directive: "+String(trim(raw)),line);
            Text value; for (int g=1;g<=3;++g) if (token[g].matched) value=token[g].str(); unit.imports.emplace_back(line,value);
            (current<0?unit.preamble:unit.sections[current].body)+="\n";
        }
        else if (std::regex_search(mask,sec,section))
        {
            auto name=lower(sec[1].str());
            if (mixed && (name=="faust" || name=="block" || name=="sample")) { int ordinal=repeated[name]++; if (ordinal) name+="_"+std::to_string(ordinal); }
            if (auto* existing=unit.get(name))
            {
                if (name=="init") { existing->body+="\n"; current=int(existing-unit.sections.data()); continue; }
                throw Failure("source", String(pathText(path))+":"+String(line)+": duplicate @"+String(name)+" section",line);
            }
            current=int(unit.sections.size()); unit.sections.push_back({name,raw.ends_with("\n")?raw:raw+"\n",{}});
        }
        else (current<0?unit.preamble:unit.sections[current].body)+=raw;
    }
    return unit;
}
std::vector<Text> shellWords(const Text& source)
{
    std::vector<Text> out; Text word; char quote=0; bool active=false;
    for (size_t i=0;i<source.size();++i)
    {
        char c=source[i];
        if (c=='\\' && quote!='\'') { if (++i>=source.size()) throw Failure("source","No escaped character"); char next=source[i]; if (quote=='"' && next!='"' && next!='\\') word+='\\'; word+=next; active=true; }
        else if (quote) { if (c==quote) quote=0; else word+=c; }
        else if (c=='\'' || c=='"') { quote=c; active=true; }
        else if (std::isspace(static_cast<unsigned char>(c))) { if (active) { out.push_back(word); word.clear(); active=false; } }
        else { word+=c; active=true; }
    }
    if (quote) throw Failure("source","No closing quotation"); if (active) out.push_back(word); return out;
}
double configNumber(Text token)
{
    auto low=lower(token); bool negative=low.starts_with("-"); auto unsignedToken=negative?low.substr(1):low;
    if (unsignedToken.starts_with("$~")) { auto n=unsignedToken.substr(2); if (n.empty() || n.find_first_not_of("0123456789")!=Text::npos) throw std::invalid_argument("mask"); int bits=std::stoi(n); if (bits>53) throw std::invalid_argument("mask"); double v=double((uint64_t(1)<<bits)-1); return negative?-v:v; }
    if (unsignedToken.starts_with("$x"))
    {
        auto digits=unsignedToken.substr(2); if (!std::regex_match(digits,std::regex("[0-9a-f]+"))) throw std::invalid_argument("hex");
        double v=parseDecimal(decimalIntegerHex(String(digits)).toStdString()); if (!std::isfinite(v)) throw std::invalid_argument("hex overflow"); return negative?-v:v;
    }
    if (unsignedToken.starts_with("0x") || unsignedToken.starts_with("+0x"))
    {
        size_t start=unsignedToken.starts_with("+")?3:2; auto digits=unsignedToken.substr(start);
        if (!std::regex_match(digits,std::regex("[0-9a-f]+"))) throw std::invalid_argument("hex");
        double v=parseDecimal(decimalIntegerHex(String(digits)).toStdString()); if (!std::isfinite(v)) throw std::invalid_argument("hex overflow"); return negative?-v:v;
    }
    char* end=nullptr; double value=parseDecimal(token,&end); if (end==token.c_str() || *end) throw std::invalid_argument("number"); return value;
}
std::map<Text,double> configs(const fs::path& path,const Text& source)
{
    std::map<Text,double> out; std::set<Text> seen; int line=0;
    for (auto raw:lines(source))
    {
        ++line; if (std::regex_search(raw,section)) break; std::smatch match;
        if (!std::regex_search(raw,match,std::regex(R"(^\s*config:\s*(.*?)\s*$)",std::regex::icase))) continue;
        auto parts=shellWords(match[1].str());
        if (parts.size()<3 || !std::regex_match(parts[0],std::regex(R"([A-Za-z_][A-Za-z0-9_.]*)"))) throw Failure("source",String(pathText(path))+":"+String(line)+": invalid config directive: "+String(trim(raw)),line);
        if (!seen.insert(lower(parts[0])).second) throw Failure("source",String(pathText(path))+":"+String(line)+": duplicate config variable '"+String(parts[0])+"'",line);
        try { out[parts[0]]=configNumber(parts[2]); } catch (const std::exception&) { throw Failure("source",String(pathText(path))+":"+String(line)+": invalid config default '"+String(parts[2])+"'",line); }
    }
    return out;
}
class Resolver
{
    std::vector<fs::path> roots, visiting, files;
    std::map<Text,Unit> units;
    std::vector<Text> ordered;
    std::map<Text,double> definitions;
    bool indexed=false;
    const std::atomic_bool* cancelled;
    void checkpoint() { if (cancelled && cancelled->load()) throw Failure("cancelled","Compilation cancelled"); }
    bool allowed(const fs::path& p)
    {
        for (auto& root:roots) { auto relative=p.lexically_relative(root); if (!relative.empty() && *relative.begin()!=".." && !relative.is_absolute()) return true; } return false;
    }
    fs::path find(Text token,const fs::path& importer,int line)
    {
        std::replace(token.begin(),token.end(),'\\','/'); fs::path relative=fs::u8path(token);
        if (token.empty() || token.starts_with("/") || std::regex_search(token,std::regex(R"(^[A-Za-z]:)"))) throw Failure("source",String(pathText(importer))+":"+String(line)+": import must be a package-relative path: '"+String(token)+"'",line);
        auto bases=roots; bases.insert(bases.begin(),importer.parent_path());
        for (auto& base:bases) { auto candidate=resolvedPath(base/relative); if (allowed(candidate) && fs::is_regular_file(candidate)) return candidate; }
        bool parent=false; for (auto& component:relative) if (component=="..") parent=true;
        if (!parent)
        {
            if (!indexed) { std::map<Text,fs::path> unique; for (auto& root:roots) for (auto& entry:fs::recursive_directory_iterator(root)) { checkpoint(); if (entry.is_regular_file()) { auto p=resolvedPath(entry.path()); if (allowed(p)) unique[key(p)]=p; } } for (auto& [_,p]:unique) files.push_back(p); indexed=true; }
            for (bool insensitive:{false,true})
            {
                std::vector<fs::path> hits; Text suffix="/"+token; if (insensitive) suffix=lower(suffix);
                for (auto& p:files) { auto raw=p.generic_u8string(); Text name(reinterpret_cast<const char*>(raw.data()),raw.size()); if (insensitive) name=lower(name); if (name.ends_with(suffix)) hits.push_back(p); }
                if (hits.size()==1) return hits[0];
                if (!hits.empty()) { String message=String(pathText(importer))+":"+String(line)+": ambiguous import '"+String(token)+"':"; for (auto& hit:hits) message+="\n  "+String(pathText(hit)); throw Failure("source",message,line); }
            }
        }
        throw Failure("source",String(pathText(importer))+":"+String(line)+": missing import '"+String(token)+"'",line);
    }
    Text preprocess(const fs::path& path,const Text& text)
    {
        if (text.find("<?")==Text::npos) return text;
        EEL2_PreProcessor processor(64<<20); auto dirs=roots;
        if (std::find(dirs.begin(),dirs.end(),path.parent_path())==dirs.end()) dirs.push_back(path.parent_path());
        for (auto& dir:dirs) processor.m_include_paths.Add(pathText(dir).c_str());
        std::vector<std::pair<Text,double>> defs(definitions.begin(),definitions.end());
        std::stable_sort(defs.begin(),defs.end(),[](auto& a,auto& b){ return lower(a.first)<lower(b.first); });
        for (auto& [name,value]:defs) { if (!std::isfinite(value)) throw Failure("source",String(pathText(path))+": invalid non-finite preprocessor definition "+String(name)); processor.define(name.c_str(),value); }
        WDL_FastString output; if (auto* error=processor.preprocess(text.c_str(),&output)) throw Failure("source",String(pathText(path))+": "+String(error));
        Text expanded(output.Get(),size_t(output.GetLength()));
#ifdef _WIN32
        // The reference executable writes via a Windows text-mode stdout stream.
        // Preserve that observable expansion until both backends deliberately change it.
        Text translated; for (char c:expanded) { if (c=='\n') translated+='\r'; translated+=c; } return translated;
#else
        return expanded;
#endif
    }
    void visit(const fs::path& path,const Text* supplied=nullptr)
    {
        checkpoint(); auto id=key(path);
        if (std::find(visiting.begin(),visiting.end(),path)!=visiting.end()) { String chain="Cyclic JSFX import: "; for (auto& p:visiting) chain+=String(pathText(p))+" -> "; throw Failure("source",chain+String(pathText(path))); }
        if (units.count(id)) return;
        if (visiting.size()>=32) throw Failure("source","JSFX import nesting exceeds 32: "+String(pathText(path)));
        if (!allowed(path)) throw Failure("source","Source is outside the package search roots: "+String(pathText(path)));
        auto raw=supplied?*supplied:read(path); auto unit=parse(path,preprocess(path,raw)); visiting.push_back(path);
        for (auto& [line,token]:unit.imports) visit(find(token,path,line));
        visiting.pop_back();
        if (!supplied && unit.sections.empty()) for (auto& [_,mask]:codeLines(unit.preamble)) if (!trim(mask).empty()) throw Failure("source",String(pathText(path))+": sectionless import; put library functions in @init");
        units.emplace(id,std::move(unit)); ordered.push_back(id);
    }
public:
    Resolver(std::vector<fs::path> paths,const std::atomic_bool* c):cancelled(c) { for (auto& p:paths) { auto root=resolvedPath(p); if (std::find(roots.begin(),roots.end(),root)==roots.end()) roots.push_back(root); } }
    var expand(const fs::path& path,const Text& source)
    {
        definitions=configs(path,source); visit(path,&source); auto& main=units.at(key(path)); Array<var> dependencies; auto owners=object({});
        for (auto& id:ordered) dependencies.add(String(pathText(units.at(id).path)));
        Text output;
        if (main.imports.empty()) { output=main.text; for (auto& s:main.sections) owners.getDynamicObject()->setProperty(juce::Identifier(s.name),var(Array<var>{String(pathText(path))})); }
        else
        {
            output=main.preamble; if (!output.empty() && !output.ends_with("\n")) output+='\n'; Array<var> initOwners;
            for (auto& id:ordered) if (auto* s=units.at(id).get("init"))
            {
                if (initOwners.isEmpty()) output+="@init\n"; else output+="\n;\n";
                output+=s->body+"\n"; initOwners.add(String(pathText(units.at(id).path)));
            }
            if (!initOwners.isEmpty()) owners.getDynamicObject()->setProperty("init",var(initOwners));
            auto priorities=ordered; priorities.pop_back(); priorities.insert(priorities.begin(),key(path)); std::set<Text> seen;
            for (auto& id:priorities) for (auto& s:units.at(id).sections) if (s.name!="init" && seen.insert(s.name).second)
            { output+=s.header+s.body+"\n"; owners.getDynamicObject()->setProperty(juce::Identifier(s.name),var(Array<var>{String(pathText(units.at(id).path))})); }
        }
        return object({{"text",String::fromUTF8(output.data(),int(output.size()))},{"dependencies",var(dependencies)},{"sectionSources",owners}});
    }
};
}
var resolveSource(const var& request,const std::atomic_bool* cancelled)
{
    // WDL's host mutex hooks above assume one compiler operation per process.
    // Serialize resolver/preprocessor calls when this library is used directly.
    static std::mutex resolutionMutex;
    std::lock_guard<std::mutex> lock(resolutionMutex);
    if (!request["sourcePath"].isString() || request["sourcePath"].toString().isEmpty()) throw Failure("protocol","Resolution requires sourcePath");
    auto path=resolvedPath(pathFrom(request["sourcePath"].toString())); std::vector<fs::path> roots;
    roots.push_back(request["packageRoot"].isString()?pathFrom(request["packageRoot"].toString()):path.parent_path());
    if (auto* extra=request["searchRoots"].getArray()) for (auto& p:*extra) { if (!p.isString()) throw Failure("protocol","searchRoots entries must be strings"); roots.push_back(pathFrom(p.toString())); }
    try { return Resolver(roots,cancelled).expand(path,request["source"].toString().toStdString()); }
    catch (Failure& error)
    {
        auto message=error.message.toStdString(); std::smatch match;
        if (std::regex_search(message,match,std::regex(R"(^(.*?):([0-9]+):)"))) { error.file=String(match[1].str()); error.line=std::stoi(match[2].str()); }
        else if (std::regex_search(message,match,std::regex(R"(^(.*?): )")) && fs::u8path(match[1].str()).is_absolute()) error.file=String(match[1].str());
        throw;
    }
}
}
