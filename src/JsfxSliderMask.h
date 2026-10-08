// SPDX-License-Identifier: Zlib
#pragma once
#include <cmath>
#include <cstdint>
namespace jsfx_gfx {
// Numeric masks use the legacy low 64 bits. Direct slider references carry
// their index separately, so sliders 65..256 do not lose bits in a double.
inline SliderMask sliderMask(double value,int directIndex) noexcept {
    SliderMask mask;
    if(directIndex>=0 && directIndex<DSPJSFX_MAX_SLIDERS){mask.set(directIndex);return mask;}
    if(!std::isfinite(value))return mask;
    const auto rounded=std::round(value);
    if(rounded>=0 && rounded<18446744073709551616.0)mask.words[0]=(uint64_t)rounded;
    else if(rounded<0 && rounded>=-9223372036854775808.0)mask.words[0]=(uint64_t)(int64_t)rounded;
    return mask;
}
}
