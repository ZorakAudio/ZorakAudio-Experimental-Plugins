// SPDX-License-Identifier: Zlib
#pragma once
namespace za::jsfx {
struct SilenceOnMemoryFault {
    DSPJSFX_State& state;
    juce::AudioBuffer<float>& audio;
    juce::MidiBuffer* midi=nullptr;
    ~SilenceOnMemoryFault() {
        if(state.memoryFault!=0){audio.clear();if(midi)midi->clear();}
    }
};
}
