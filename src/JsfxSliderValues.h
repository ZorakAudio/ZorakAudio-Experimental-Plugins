// SPDX-License-Identifier: Zlib
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace za::jsfx {
// JUCE choices carry an index; numeric controls carry their raw float value.
// Preserve the production double-precision mapping and JSFX step quantization.
template<class Declaration> double hostSliderValue(const Declaration& info,double value) {
    if(info.isChoice) {
        const auto index=(int64_t)std::llround(value);
        const double step=double(info.step>0 ? info.step : 1);
        value=double(info.rangeStart)+double(index)*(info.reversed?-step:step);
    }
    value=std::clamp(value,double(info.min),double(info.max));
    if(!info.isChoice && info.step>0) {
        const double step=info.reversed?-double(info.step):double(info.step);
        const double index=std::llround((value-double(info.rangeStart))/step);
        value=std::clamp(double(info.rangeStart)+index*step,double(info.min),double(info.max));
    }
    return value;
}
template<class Declaration> bool sliderValuesEquivalent(const Declaration& info,double a,double b) noexcept {
    if(!std::isfinite(a) || !std::isfinite(b))return false;
    if(info.isChoice){const double step=double(info.step>0?info.step:1);const double signedStep=info.reversed?-step:step;
        return (int64_t)std::llround((a-double(info.rangeStart))/signedStep)==(int64_t)std::llround((b-double(info.rangeStart))/signedStep);}
    const double tolerance=std::max(1.0e-6,info.step>0?double(info.step)*.25:1.0e-6);
    return std::abs(a-b)<=tolerance;
}
template<class Mask,class Used,class Equivalent> Mask actualSliderChanges(const Mask& requested,const double* before,const double* after,Used&& used,Equivalent&& equivalent){
    Mask result;for(int i=0;i<256;++i)if(requested.test(i) && used(i) && !equivalent(i,before[i],after[i]))result.set(i);return result;
}
template<class Mask> void bracketGfxGesture(const Mask& changes,Mask& automate,Mask& end,Mask& touch,bool pressed){
    if(pressed)automate.merge(changes);
    else{automate.merge(changes);end.merge(changes);end.merge(touch);}
    touch.merge(automate);touch.remove(end);
}
}
