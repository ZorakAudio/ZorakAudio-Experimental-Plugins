#pragma once
#include "JsfxSliderDeclarations.h"
#include <juce_data_structures/juce_data_structures.h>
#include <cmath>
#include <cstdlib>

namespace za::jsfx {
// Opt-in native factory defaults with explicit values for older saved states.
// The source slider declaration remains the standalone JSFX default.
struct NativeSliderDefault {
    int index0;
    juce::String id;
    float factory;
    float legacy;
    juce::String reseedVariable;
};

inline std::vector<NativeSliderDefault> parseNativeSliderDefaults(
    const char* source, const std::vector<JsfxSliderDecl>& declarations)
{
    std::vector<NativeSliderDefault> result;
    if (source == nullptr) return result;
    static const std::regex directive(
        R"(^\s*//\s*#NATIVE_DEFAULT:\s*slider([0-9]+)\s*=\s*([-+0-9.eE]+)\s+legacy\s*=\s*([-+0-9.eE]+)(?:\s+reseed\s*=\s*([A-Za-z_][A-Za-z_0-9]*))?\s*$)");
    std::istringstream lines(source);
    std::string line;
    while (std::getline(lines, line)) {
        std::smatch match;
        if (!std::regex_match(line, match, directive)) continue;
        const auto id = juce::String("slider") + juce::String(match[1].str());
        const auto found = std::find_if(declarations.begin(), declarations.end(),
            [&](const auto& declaration) { return declaration.id == id; });
        if (found == declarations.end() || found->isString || found->isChoice) continue;
        // Reject malformed numeric tokens rather than silently interpreting them as zero.
        char* end = nullptr;
        const auto factoryToken = match[2].str(), legacyToken = match[3].str();
        const auto factory = std::strtod(factoryToken.c_str(), &end);
        if (end != factoryToken.c_str() + factoryToken.size()) continue;
        const auto legacy = std::strtod(legacyToken.c_str(), &end);
        if (end != legacyToken.c_str() + legacyToken.size()) continue;
        if (!std::isfinite(factory) || !std::isfinite(legacy)
            || factory < found->min || factory > found->max
            || legacy < found->min || legacy > found->max) continue;
        if (std::any_of(result.begin(), result.end(),
            [&](const auto& rule) { return rule.id == id; })) continue;
        result.push_back({found->index0, id, float(factory), float(legacy),
                          juce::String(match[4].str())});
    }
    return result;
}

inline void applyNativeSliderDefaults(const std::vector<NativeSliderDefault>& rules,
                                     std::vector<JsfxSliderDecl>& declarations)
{
    for (const auto& rule : rules)
        for (auto& declaration : declarations)
            if (declaration.index0 == rule.index0) declaration.def = rule.factory;
}

inline bool restoreMissingNativeSliderDefaults(const std::vector<NativeSliderDefault>& rules,
                                              juce::ValueTree& state)
{
    if (!state.hasType("PARAMS")) return false;
    bool changed = false;
    for (const auto& rule : rules) {
        auto parameter = state.getChildWithProperty("id", rule.id);
        if (!parameter.isValid()) {
            parameter = juce::ValueTree("PARAM");
            parameter.setProperty("id", rule.id, nullptr);
            parameter.setProperty("value", rule.legacy, nullptr);
            state.addChild(parameter, -1, nullptr);
            changed = true;
        } else if (!parameter.hasProperty("value")) {
            parameter.setProperty("value", rule.legacy, nullptr);
            changed = true;
        }
    }
    return changed;
}
}
