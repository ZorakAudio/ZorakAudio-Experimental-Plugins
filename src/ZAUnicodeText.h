#pragma once

#include <juce_core/juce_core.h>

namespace za::text
{
    /**
        Constructs a JUCE String from a repository-authored UTF-8 literal.

        Use this for any non-ASCII UI text. JUCE's String (const char*)
        constructor treats the input as ASCII in debug builds and asserts on
        bytes >= 128. This helper makes the encoding explicit and keeps
        symbols such as …, ×, ≤, ≥, →, and • valid.
    */
    inline juce::String utf8 (const char* text)
    {
        return juce::String::fromUTF8 (text != nullptr ? text : "");
    }

    inline juce::String utf8 (const char* text, int numBytes)
    {
        return text != nullptr ? juce::String::fromUTF8 (text, numBytes) : juce::String();
    }
}
