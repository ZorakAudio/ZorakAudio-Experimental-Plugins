// SPDX-License-Identifier: Zlib
// Amalgamated CPU-only LICE backend. Include from the single processor TU only.
// WDL's original copyright/license notices remain in the included source files.
#pragma once
#define _LICE_NO_SYSBITMAPS_
// This project deliberately does not instantiate OS-backed LICE surfaces/fonts.
#include "WDL/lice/lice.cpp"
#include "WDL/lice/lice_line.cpp"
#include "WDL/lice/lice_arc.cpp"
#include "WDL/lice/lice_text.cpp"
#undef _LICE_NO_SYSBITMAPS_
// LICE implementation macros must not leak into the amalgamated JUCE processor.
#undef IGNORE_SCALING
#undef DO_RECT_SC
#undef A
#undef AF
#undef DEF_ALPHAS
#undef _PI
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
