#include "Frontend.h"
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3) { std::cerr << "Usage: jsfx_frontend request.json result.json\n"; return 2; }
    const juce::File input(juce::String::fromUTF8(argv[1]));
    const juce::File output(juce::String::fromUTF8(argv[2]));
    if (!input.existsAsFile() || input.getSize() > 8 * 1024 * 1024) return 2;
    juce::var request;
    auto parse = juce::JSON::parse(input.loadFileAsString(), request);
    if (parse.failed()) { std::cerr << parse.getErrorMessage() << '\n'; return 2; }
    const auto result = za::compiler::frontend(request);
    if (!output.replaceWithText(juce::JSON::toString(result, true))) return 2;
    return bool(result["ok"]) ? 0 : 1;
}
