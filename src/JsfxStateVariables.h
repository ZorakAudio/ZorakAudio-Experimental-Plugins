// SPDX-License-Identifier: Zlib
#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>
#ifndef DSPJSFX_DYNAMIC_VARIABLES
#define DSPJSFX_DYNAMIC_VARIABLES 0
#endif
namespace za::jsfx {
class StateVariables {
    std::vector<DSPJSFX_Cell> cells;
    size_t extent=0;
public:
    void bind(DSPJSFX_State& state,size_t count) {
#if DSPJSFX_DYNAMIC_VARIABLES
        cells.resize(std::max<size_t>(1,count));
        for(auto& cell:cells)cell=0.0;
        extent=count;state.vars=cells.data();state.varsN=(int64_t)extent;
#else
        (void)state;(void)count;
#endif
    }
    void rebind(DSPJSFX_State& state)noexcept {
#if DSPJSFX_DYNAMIC_VARIABLES
        state.vars=cells.data();state.varsN=(int64_t)extent;
#else
        (void)state;
#endif
    }
};
inline int64_t variableCount(const DSPJSFX_State& state)noexcept {
#if DSPJSFX_DYNAMIC_VARIABLES
    return state.varsN;
#else
    return (int64_t)std::size(state.vars);
#endif
}
}
