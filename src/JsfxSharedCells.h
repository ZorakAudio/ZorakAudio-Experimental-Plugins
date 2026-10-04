// SPDX-License-Identifier: Zlib
#pragma once
#include <atomic>
#include <cstddef>
#include <cstring>
#include <type_traits>

// The generated LLVM ABI remains an aligned double per cell. Every C++ access
// to a live guest cell must use atomic_ref too, including output lvalues.
inline double jsfxCellLoad(const double *p) noexcept {
#if DSPJSFX_NATIVE_GFX_LEGACY
  return std::atomic_ref<double>(*const_cast<double *>(p))
      .load(std::memory_order_relaxed);
#else
  return *p;
#endif
}
inline void jsfxCellStore(double *p, double v) noexcept {
#if DSPJSFX_NATIVE_GFX_LEGACY
  std::atomic_ref<double>(*p).store(v, std::memory_order_relaxed);
#else
  *p = v;
#endif
}
#if DSPJSFX_NATIVE_GFX_LEGACY
struct alignas(8) DSPJSFX_Cell {
  double value;
  DSPJSFX_Cell() = default;
  DSPJSFX_Cell(const DSPJSFX_Cell &v) noexcept : value(double(v)) {}
  operator double() const noexcept { return jsfxCellLoad(&value); }
  DSPJSFX_Cell &operator=(double v) noexcept {
    jsfxCellStore(&value, v);
    return *this;
  }
  DSPJSFX_Cell &operator=(const DSPJSFX_Cell &v) noexcept {
    return *this = double(v);
  }
  DSPJSFX_Cell &operator+=(double v) noexcept {
    return *this = double(*this) + v;
  }
  DSPJSFX_Cell &operator-=(double v) noexcept {
    return *this = double(*this) - v;
  }
  DSPJSFX_Cell &operator*=(double v) noexcept {
    return *this = double(*this) * v;
  }
  DSPJSFX_Cell &operator/=(double v) noexcept {
    return *this = double(*this) / v;
  }
  double *operator&() noexcept { return &value; }
  const double *operator&() const noexcept { return &value; }
};
static_assert(sizeof(DSPJSFX_Cell) == sizeof(double) &&
              alignof(DSPJSFX_Cell) == 8);
static_assert(std::is_standard_layout_v<DSPJSFX_Cell>);
static_assert(std::is_trivially_default_constructible_v<DSPJSFX_Cell>);
static_assert(std::atomic_ref<double>::required_alignment <= 8);
static_assert(std::atomic_ref<double>::is_always_lock_free,
              "Native legacy JSFX requires lock-free aligned 64-bit cells");
#else
using DSPJSFX_Cell = double;
#endif
inline void jsfxCopyCells(double *dst, const DSPJSFX_Cell *src,
                          size_t n) noexcept {
#if DSPJSFX_NATIVE_GFX_LEGACY
  for (size_t i = 0; i < n; ++i)
    dst[i] = double(src[i]);
#else
  std::memcpy(dst, src, n * sizeof(double));
#endif
}
inline void jsfxWriteCells(DSPJSFX_Cell *dst, const double *src,
                           size_t n) noexcept {
#if DSPJSFX_NATIVE_GFX_LEGACY
  for (size_t i = 0; i < n; ++i)
    dst[i] = src[i];
#else
  std::memcpy(dst, src, n * sizeof(double));
#endif
}
