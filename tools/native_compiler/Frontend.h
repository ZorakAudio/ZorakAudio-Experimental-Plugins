#pragma once
#include <juce_core/juce_core.h>
#include <atomic>
#include <stdexcept>

namespace za::compiler
{
using juce::String;
using juce::var;
using juce::Array;
// v1 is a frontend-only protocol. Successful results do not contain executable DSP.
inline constexpr int protocolVersion = 1;
struct Failure : std::runtime_error
{
    String phase, message, file;
    juce::int64 line = 0;
    int column = 0;
    Failure(String p, String m, juce::int64 l = 0, int c = 0)
        : std::runtime_error(m.toStdString()), phase(p), message(m), line(l), column(c) {}
};
var object(std::initializer_list<std::pair<String, var>>);
double parseDecimal(const std::string& text, char** end = nullptr);
String decimalIntegerHex(String source);
var resolveSource(const var& request, const std::atomic_bool* cancelled);
// Runs resolution, tokenization and parsing only; never LLVM emission or execution.
// Call off the audio thread. Cancellation is polled between units and syntax operations.
var frontend(const var& request, const std::atomic_bool* cancelled = nullptr);
}
