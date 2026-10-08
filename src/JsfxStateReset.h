// SPDX-License-Identifier: Zlib
#pragma once
#include <cstring>
#include "JsfxStateVariables.h"
namespace za::jsfx {
inline void resetGuestStatePreservingHeap(DSPJSFX_State& state,StateVariables& variables,size_t count) {
    auto* heap=state.mem;const auto extent=state.memN;
    std::memset(&state,0,sizeof(state));
    variables.bind(state,count);
    state.mem=heap;state.memN=extent;
}
}
