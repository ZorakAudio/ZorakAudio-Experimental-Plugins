// SPDX-License-Identifier: Zlib
#pragma once
#include <cstdlib>
#include <limits>
namespace za::jsfx {
inline void releaseHeap(DSPJSFX_State& state)noexcept {
    std::free(state.mem);state.mem=nullptr;state.memN=0;
}
inline bool allocateHeap(DSPJSFX_State& state,int64_t count)noexcept {
    if(count<0 || (uint64_t)count>std::numeric_limits<size_t>::max()/sizeof(DSPJSFX_Cell))return false;
    auto* memory=static_cast<DSPJSFX_Cell*>(std::calloc((size_t)count,sizeof(DSPJSFX_Cell)));
    if(!memory && count)return false;
    // Touch pages before publishing the instance to the audio thread.
#if DSPJSFX_NATIVE_GFX_LEGACY
    volatile unsigned char* pages=reinterpret_cast<volatile unsigned char*>(memory);
    for(size_t offset=0;offset<(size_t)count*sizeof(DSPJSFX_Cell);offset+=4096)pages[offset]=0;
#endif
    releaseHeap(state);state.mem=memory;state.memN=count;return true;
}
}
