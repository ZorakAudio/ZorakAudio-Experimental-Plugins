// SPDX-License-Identifier: Zlib
#pragma once
extern "C" void jsfx_ensure_mem (DSPJSFX_State* st, int64_t needed)
{
#if DSPJSFX_HAS_TASKS
    if (jsfx_tasks::Runtime::privateHeap(st)) {
        if (needed<=0 || needed>st->memN || !st->mem) st->memoryFault=1;
        return;
    }
#endif
#if DSPJSFX_NATIVE_GFX_LEGACY
    if (st != nullptr && (needed <= 0 || needed > st->memN || st->mem == nullptr)) st->memoryFault = 1;
    return;
#else
    if (st == nullptr)
        return;
    if (needed <= 0)
    {
        st->memoryFault = 1;
        return;
    }
    if (needed <= st->memN && st->mem != nullptr)
    {
        noteTrackedJsfxMemUsed (st, needed);
        return;
    }
    if (st->memoryFault != 0)
        return; // Retry only after a state reset, not once per audio sample.

    const uint64_t maxCells = std::min<uint64_t> (
        (uint64_t) std::numeric_limits<int64_t>::max(),
        (uint64_t) (std::numeric_limits<size_t>::max() / sizeof (double)));
    if ((uint64_t) needed > maxCells || st->memN < 0)
    {
        st->memoryFault = 1;
        return;
    }

    uint64_t newN = std::max<uint64_t> ((uint64_t) st->memN, 1024u);
    while (newN < (uint64_t) needed)
    {
        const uint64_t extra = newN / 2u + 64u;
        if (newN > maxCells - std::min (extra, maxCells))
        {
            newN = (uint64_t) needed;
            break;
        }
        newN += extra;
    }
    newN = std::min (newN, maxCells);
    const size_t oldBytes = (size_t) st->memN * sizeof (double);
    const size_t newBytes = (size_t) newN * sizeof (double);
    void* p = std::realloc (st->mem, newBytes);
    if (p == nullptr)
    {
        st->memoryFault = 1;
        return;
    }
    st->mem = static_cast<double*> (p);
    if (newBytes > oldBytes)
        std::memset (static_cast<uint8_t*> (p) + oldBytes, 0, newBytes - oldBytes);
    st->memN = (int64_t) newN;
    // Keep bounded GFX mirroring aware of newly allocated slack, as before.
    noteTrackedJsfxMemUsed (st, st->memN);
#endif
}


