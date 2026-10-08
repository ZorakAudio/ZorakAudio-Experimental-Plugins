// SPDX-License-Identifier: Zlib
#pragma once
#include <cmath>
#include <cstdint>
#include <atomic>
#include "JsfxSliderMask.h"

// --- Joep/JSFX compatibility: slider_next_chg() --------------------------
// Minimal runtime behavior:
//   - write the current slider value to *outValue
//   - return -1.0 when no sub-block automation change points are available
// Host automation is delivered at block boundaries. Reporting an exhausted
// iterator is essential: a zero offset makes consumers loop forever.
extern "C" double jsfx_slider_next_chg (DSPJSFX_State* st,
                                        double sliderIndex,
                                        double* outValue)
{
    const int idx1 = std::isfinite(sliderIndex) && sliderIndex >= 1.0 && sliderIndex <= DSPJSFX_MAX_SLIDERS
                        ? (int) (sliderIndex + 1.0e-5) : 0;
    const int idx0 = idx1 - 1;

    double current = 0.0;
    if (st != nullptr && idx0 >= 0 && idx0 < DSPJSFX_MAX_SLIDERS)
        current = st->sliders[(size_t) idx0];

    if (outValue != nullptr)
        jsfxCellStore (outValue, current);

    return -1.0;
}

static jsfx_gfx::SliderMask jsfxSliderMask (double value, int directSliderIndex) noexcept
{
    return jsfx_gfx::sliderMask(value,directSliderIndex);
}

static void mergeSliderMaskWords (uint64_t* dst, const jsfx_gfx::SliderMask& mask) noexcept
{
    if (dst == nullptr)
        return;
    for (int word = 0; word < DSPJSFX_SLIDER_MASK_WORDS; ++word)
    {
#if DSPJSFX_NATIVE_GFX_LEGACY
        std::atomic_ref<uint64_t>(dst[word]).fetch_or(mask.words[(size_t)word]);
#else
        dst[word] |= mask.words[(size_t)word];
#endif
    }
}

static void clearSliderMaskWords (uint64_t* dst) noexcept
{
    if (dst == nullptr)
        return;
    for (int word = 0; word < DSPJSFX_SLIDER_MASK_WORDS; ++word)
    {
#if DSPJSFX_NATIVE_GFX_LEGACY
        std::atomic_ref<uint64_t>(dst[word]).exchange(0);
#else
        dst[word] = 0;
#endif
    }
}

static bool anySliderMaskWords (const uint64_t* words) noexcept
{
    if (words == nullptr)
        return false;
    for (int word = 0; word < DSPJSFX_SLIDER_MASK_WORDS; ++word)
        if (
#if DSPJSFX_NATIVE_GFX_LEGACY
            std::atomic_ref<uint64_t>(const_cast<uint64_t&>(words[word])).load()
#else
            words[word]
#endif
            != 0)
            return true;
    return false;
}

static jsfx_gfx::SliderMask loadSliderMaskWords (const uint64_t* words) noexcept
{
    jsfx_gfx::SliderMask out;
    if (words != nullptr)
        for (int word = 0; word < DSPJSFX_SLIDER_MASK_WORDS; ++word)
            out.words[(size_t) word] =
#if DSPJSFX_NATIVE_GFX_LEGACY
                std::atomic_ref<uint64_t>(const_cast<uint64_t&>(words[word])).load();
#else
                words[word];
#endif
    return out;
}

extern "C" int jsfx_sliderchange (DSPJSFX_State* st, double sliderMask, int directSliderIndex)
{
    if (st == nullptr)
        return 0;

#if DSPJSFX_NATIVE_GFX_LEGACY
    if (st->sharedState != nullptr) st = static_cast<DSPJSFX_State*>(st->sharedState);
#endif

    const auto mask = jsfxSliderMask (sliderMask, directSliderIndex);
    if (! mask.any())
        return 0;

    mergeSliderMaskWords (st->pendingSliderChangeMask, mask);
    return 1;
}

extern "C" int jsfx_slider_automate (DSPJSFX_State* st, double sliderMask, int directSliderIndex, double endTouch)
{
    if (st == nullptr)
        return 0;

#if DSPJSFX_NATIVE_GFX_LEGACY
    if (st->sharedState != nullptr) st = static_cast<DSPJSFX_State*>(st->sharedState);
#endif

    const auto mask = jsfxSliderMask (sliderMask, directSliderIndex);
    if (! mask.any())
        return 0;

    mergeSliderMaskWords (st->pendingSliderChangeMask, mask);
    if (endTouch != 0.0)
        mergeSliderMaskWords (st->pendingSliderAutomateEndMask, mask);
    else
        mergeSliderMaskWords (st->pendingSliderAutomateMask, mask);
    return 1;
}

extern "C" double jsfx_slider_show (DSPJSFX_State* st, double sliderMask, int directSliderIndex,
                                      double mode, int hasMode)
{
    if (st == nullptr)
        return 0.0;

#if DSPJSFX_NATIVE_GFX_LEGACY
    if (st->sharedState != nullptr) st = static_cast<DSPJSFX_State*>(st->sharedState);
#endif

    const auto mask = jsfxSliderMask (sliderMask, directSliderIndex);
    if (! mask.any())
        return 0.0;

    if (st->sliderVisibilityInit == 0)
    {
        for (int word = 0; word < DSPJSFX_SLIDER_MASK_WORDS; ++word)
            st->sliderVisibleMask[word] = ~UINT64_C (0);
        st->sliderVisibilityInit = 1;
    }

    if (hasMode != 0)
    {
        for (int word = 0; word < DSPJSFX_SLIDER_MASK_WORDS; ++word)
        {
            const uint64_t bits = mask.words[(size_t) word];
            if (mode == -1.0)
                st->sliderVisibleMask[word] ^= bits;
            else if (mode <= 0.0)
                st->sliderVisibleMask[word] &= ~bits;
            else
                st->sliderVisibleMask[word] |= bits;
        }
    }

    if (directSliderIndex >= 64)
        return mask.test (directSliderIndex)
            && (st->sliderVisibleMask[(size_t) directSliderIndex >> 6] & (UINT64_C (1) << (directSliderIndex & 63))) != 0
            ? 1.0 : 0.0;

    return (double) (st->sliderVisibleMask[0] & mask.words[0]);
}

