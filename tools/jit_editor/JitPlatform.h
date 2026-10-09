#pragma once
#include <juce_core/juce_core.h>
#include <functional>

// Only OS integration lives here. Both platforms use the same compiler,
// generated state ABI, runtime services and program publication code.
namespace jit_platform {
using Module = void*;
juce::File findRuntime(const void* address);
Module loadLLVM(const juce::File& runtime);
void* symbol(Module module, const char* name);
juce::String runHelper(const juce::File& runtime, const juce::File& job,
                       const std::function<bool()>& cancelled);
}
