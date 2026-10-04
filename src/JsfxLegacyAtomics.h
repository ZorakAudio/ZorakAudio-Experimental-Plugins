// SPDX-License-Identifier: Zlib
// Include once in the runtime TU, after JSFXDSP.h. WDL eel_atomic.h serializes
// explicit atomic calls with an instance mutex; ordinary guest access stays
// live.
#pragma once
#include <cmath>
#include <mutex>
#if DSPJSFX_NATIVE_GFX_LEGACY
namespace jsfx_legacy {
inline std::mutex &atomicMutex(DSPJSFX_State *st) {
  return *static_cast<std::mutex *>(st->atomicContext);
}
} // namespace jsfx_legacy
extern "C" double jsfx_atomic_get(DSPJSFX_State *st, double *p) {
  const std::lock_guard<std::mutex> lock(jsfx_legacy::atomicMutex(st));
  return jsfxCellLoad(p);
}
extern "C" double jsfx_atomic_set(DSPJSFX_State *st, double *p, double *value) {
  const std::lock_guard<std::mutex> lock(jsfx_legacy::atomicMutex(st));
  const double v = jsfxCellLoad(value);
  jsfxCellStore(p, v);
  return v;
}
extern "C" double jsfx_atomic_add(DSPJSFX_State *st, double *p, double *value) {
  const std::lock_guard<std::mutex> lock(jsfx_legacy::atomicMutex(st));
  const double next = jsfxCellLoad(p) + jsfxCellLoad(value);
  jsfxCellStore(p, next);
  return next;
}
extern "C" double jsfx_atomic_setifequal(DSPJSFX_State *st, double *p,
                                         double *expected, double *value) {
  const std::lock_guard<std::mutex> lock(jsfx_legacy::atomicMutex(st));
  const double old = jsfxCellLoad(p);
  if (std::fabs(old - jsfxCellLoad(expected)) < 0.00001)
    jsfxCellStore(p, jsfxCellLoad(value));
  return old;
}
extern "C" double jsfx_atomic_exch(DSPJSFX_State *st, double *a, double *b) {
  const std::lock_guard<std::mutex> lock(jsfx_legacy::atomicMutex(st));
  const double old = jsfxCellLoad(a), next = jsfxCellLoad(b);
  jsfxCellStore(a, next);
  jsfxCellStore(b, old);
  return next;
}
#endif
