// SPDX-License-Identifier: Zlib
#pragma once
#include <array>
#include "JsfxSliderValues.h"
namespace za::jsfx {
// Host values and script-authored values are distinct. In the live native
// mode, an unchanged host value must not erase a slider written by guest code.
class ParameterState {
public:
    std::array<double,DSPJSFX_MAX_SLIDERS> lastSliders{};
    std::array<double,DSPJSFX_MAX_SLIDERS> internalSliderShadow{};
    jsfx_gfx::SliderMask internalSliderPendingMask;
    bool lastSlidersValid=false;
    void retainScriptSlider(int slot,double value) noexcept {
        internalSliderShadow[(size_t)slot]=value;
        internalSliderPendingMask.set(slot);
        lastSliders[(size_t)slot]=value;
    }
    template<class Declaration> double resolveHostSlider(int slot,const Declaration& info,double hostValue,double proposedValue) noexcept {
        if(internalSliderPendingMask.test(slot)) {
            const double shadow=internalSliderShadow[(size_t)slot];
            if(sliderValuesEquivalent(info,hostValue,shadow))internalSliderPendingMask.reset(slot);
            else return shadow;
        }
        return proposedValue;
    }
    template<class AliasWrite> bool applyHostSlider(DSPJSFX_State& state,int slot,double value,AliasWrite&& aliases) {
        const bool changed=!lastSlidersValid || value!=lastSliders[(size_t)slot];
#if DSPJSFX_NATIVE_GFX_LEGACY
        if(!changed)return false;
#endif
        lastSliders[(size_t)slot]=value;
        state.sliders[slot]=value;aliases(value);
        return changed;
    }
};
}
