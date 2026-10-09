#include "Frontend.h"
#include <future>
#include <iostream>
#include <clocale>

using namespace za::compiler;
static void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
int main()
{
    try
    {
        auto request=object({{"protocol",1},{"stage","frontend"},{"snippet",true},{"source","x=1;defer_reduce(i,3,SUM,0,i);"}});
        auto first=frontend(request), second=frontend(request);
        require(bool(first["ok"]),"Valid program rejected");
        require(juce::JSON::toString(first["sections"])==juce::JSON::toString(second["sections"]),"Nondeterministic AST");
        std::atomic_bool cancelled{true}; auto stopped=frontend(request,&cancelled);
        require(!bool(stopped["ok"]) && stopped["diagnostics"][0]["phase"]==var("cancelled"),"Cancellation not reported");
        auto bad=object({{"protocol",2},{"stage","frontend"},{"source","x=1;"}});
        require(frontend(bad)["diagnostics"][0]["phase"]==var("protocol"),"Protocol mismatch accepted");
        auto oversized=object({{"protocol",1},{"stage","frontend"},{"source",String::repeatedString("x",4194305)}});
        require(frontend(oversized)["diagnostics"][0]["phase"]==var("limits"),"Source limit ignored");
        auto nested=object({{"protocol",1},{"stage","frontend"},{"snippet",true},{"source",String::repeatedString("(",600)+"x"+String::repeatedString(")",600)}});
        require(frontend(nested)["diagnostics"][0]["phase"]==var("limits"),"Nesting limit ignored");
        auto chain=object({{"protocol",1},{"stage","frontend"},{"snippet",true},{"source",String::repeatedString("x+",10000)+"1"}});
        require(frontend(chain)["diagnostics"][0]["phase"]==var("limits"),"Left-associated AST limit ignored");
        auto decimal=object({{"protocol",1},{"stage","frontend"},{"snippet",true},{"source","x=.25;y=1e-10;"}});
        auto before=juce::JSON::toString(frontend(decimal)["sections"]);
        std::string previous=std::setlocale(LC_NUMERIC,nullptr);
#if defined(_WIN32)
        constexpr auto commaLocale = "German_Germany.1252";
#else
        constexpr auto commaLocale = "de_DE.UTF-8";
#endif
        require(std::setlocale(LC_NUMERIC,commaLocale)!=nullptr,"Locale test could not establish comma decimal locale");
        auto after=juce::JSON::toString(frontend(decimal)["sections"]);
        std::setlocale(LC_NUMERIC,previous.c_str());
        require(before==after,"Host locale changed numeric parsing");
        std::vector<std::future<var>> futures;
        for (int i=0;i<8;++i) futures.push_back(std::async(std::launch::async,[request]{ return frontend(request); }));
        for (auto& f:futures) require(juce::JSON::toString(f.get()["sections"])==juce::JSON::toString(first["sections"]),"Concurrent parse mismatch");
        std::cout << "{\"passed\":true,\"checks\":[\"determinism\",\"cancellation\",\"protocol\",\"source-limit\",\"parser-nesting-limit\",\"AST-depth-limit\",\"locale-independence\",\"concurrent-parsing\"]}\n";
        return 0;
    }
    catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
