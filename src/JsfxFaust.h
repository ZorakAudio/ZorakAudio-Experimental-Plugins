#pragma once
// Build-time Faust LLVM modules. No libfaust, JIT, allocation or compiler on the
// audio thread. Include after the generated JSFXDSP.h.
#if DSPJSFX_HAS_FAUST
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "JsfxFaustEngine.h"
namespace jsfx_faust {
class Engine final : public za::jsfx::BasicFaustEngine<jsfx_faust::Stage> {
public:
    Engine() : BasicFaustEngine(std::vector<jsfx_faust::Stage>(std::begin(jsfx_faust::stages),std::end(jsfx_faust::stages)),
                           DSPJSFX_PROCESS_CHANNELS,jsfx_faust::quantum,&jsfx_slider) {}
};
}
extern "C" void jsfx_faust_process(DSPJSFX_State*,const float* const*,float* const*,int32_t,int32_t);
#ifdef JSFX_FAUST_IMPLEMENTATION
extern "C" void jsfx_faust_process(DSPJSFX_State* s,const float* const* in,float* const* out,int32_t channels,int32_t count) {
    za::jsfx::processFaustContext<jsfx_faust::Engine>(s,in,out,channels,count);
}
#endif
#endif
