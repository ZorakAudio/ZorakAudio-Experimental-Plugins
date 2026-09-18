// SPDX-License-Identifier: Zlib
// GFX memory transfer policy. No JUCE dependency; no allocation in range builders.
#ifndef ZA_JSFX_GFX_MEMORY_SYNC_H
#define ZA_JSFX_GFX_MEMORY_SYNC_H
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// @gfx only needs a bounded shared-state view, not the entire DSP heap.
//
// A simple low-prefix cap fixed the soft-lock, but it regressed scripts that
// intentionally keep UI-visible summaries at very high addresses (Texture keeps
// its waveform summary far away from the raw sample buffer). To preserve that
// usage pattern without reintroducing catastrophic whole-heap mirroring, we
// snapshot two bounded windows:
//   1) a low shared prefix, where most UI state lives
//   2) a high shared suffix, which catches far-away summaries near the heap end
//
// That keeps copy cost constant while still making "summary-at-the-top" layouts
// visible to @gfx.

static constexpr int kGfxSharedPrefixDoubles = 262144; // ~= 2 MiB
static constexpr int kGfxSharedSuffixDoubles = 262144 * 8; // ~= 16 MiB
static constexpr int kMaxGfxMemSpans = 64; // ZA-GFX-MEM-SYNC: auto prefix/suffix + explicit sparse ranges

static constexpr uint8_t kGfxSyncToGfx   = 1u;
static constexpr uint8_t kGfxSyncFromGfx = 2u;

struct GfxMirrorRange
{
    int64_t base = 0;
    int count = 0;
};

struct GfxSyncMemRange
{
    int64_t base = 0;
    int64_t count = 0;
    uint8_t flags = kGfxSyncToGfx;
};

static inline uint8_t parseGfxSyncMemDirectionToken (std::string token) noexcept
{
    for (auto& c : token)
        c = (char) std::toupper ((unsigned char) c);

    if (token == "FROM_GFX" || token == "GFX_TO_DSP")
        return kGfxSyncFromGfx;

    if (token == "BIDIR" || token == "BIDIRECTIONAL" || token == "BOTH")
        return (uint8_t) (kGfxSyncToGfx | kGfxSyncFromGfx);

    return kGfxSyncToGfx;
}

static inline int64_t gfxSafeEndExclusive (int64_t base, int64_t count) noexcept
{
    if (count <= 0)
        return base;

    if (base > std::numeric_limits<int64_t>::max() - count)
        return std::numeric_limits<int64_t>::max();

    return base + count;
}

static inline int appendGfxMirrorRange (std::array<GfxMirrorRange, kMaxGfxMemSpans>& out,
                                        int n,
                                        int64_t base,
                                        int64_t count,
                                        int64_t memN) noexcept
{
    if (memN <= 0 || count <= 0)
        return n;

    if (n >= (int) out.size())
        return n;

    base = std::max<int64_t> ((int64_t) 0, base);

    if (base >= memN)
        return n;

    const int64_t end = std::min<int64_t> (memN, gfxSafeEndExclusive (base, count));

    if (end <= base)
        return n;

    const int64_t safeCount = std::min<int64_t> (end - base,
                                                 (int64_t) std::numeric_limits<int>::max());

    out[(size_t) n++] = GfxMirrorRange { base, (int) safeCount };
    return n;
}

static inline int sortAndMergeGfxMirrorRanges (std::array<GfxMirrorRange, kMaxGfxMemSpans>& ranges,
                                               int n) noexcept
{
    n = std::max (0, std::min (n, (int) ranges.size()));

    std::sort (ranges.begin(), ranges.begin() + n,
               [] (const GfxMirrorRange& a, const GfxMirrorRange& b)
               {
                   return a.base < b.base;
               });

    int outN = 0;

    for (int i = 0; i < n; ++i)
    {
        auto r = ranges[(size_t) i];

        if (r.count <= 0)
            continue;

        if (outN == 0)
        {
            ranges[(size_t) outN++] = r;
            continue;
        }

        auto& prev = ranges[(size_t) (outN - 1)];
        const int64_t prevEnd = gfxSafeEndExclusive (prev.base, (int64_t) prev.count);
        const int64_t rEnd = gfxSafeEndExclusive (r.base, (int64_t) r.count);

        if (r.base <= prevEnd)
        {
            prev.count = (int) std::min<int64_t> (std::max<int64_t> (prevEnd, rEnd) - prev.base,
                                                 (int64_t) std::numeric_limits<int>::max());
        }
        else if (outN < (int) ranges.size())
        {
            ranges[(size_t) outN++] = r;
        }
    }

    return outN;
}

static inline int buildGfxMirrorRanges (int64_t memN,
                                        std::array<GfxMirrorRange, kMaxGfxMemSpans>& out) noexcept
{
    int n = 0;

    if (memN <= 0)
        return 0;

    const int64_t prefixCount = std::min<int64_t> (memN, (int64_t) kGfxSharedPrefixDoubles);
    n = appendGfxMirrorRange (out, n, 0, prefixCount, memN);

    if (memN > prefixCount)
    {
        const int64_t remaining = memN - prefixCount;
        const int64_t suffixCount = std::min<int64_t> (remaining,
                                                       (int64_t) kGfxSharedSuffixDoubles);

        if (suffixCount > 0)
            n = appendGfxMirrorRange (out, n, memN - suffixCount, suffixCount, memN);
    }

    return sortAndMergeGfxMirrorRanges (out, n);
}

static inline bool isGfxMirroredMemIndex (int64_t index, int64_t memN) noexcept
{
    if (index < 0 || index >= memN)
        return false;

    std::array<GfxMirrorRange, kMaxGfxMemSpans> ranges {};
    const int count = buildGfxMirrorRanges (memN, ranges);

    for (int i = 0; i < count; ++i)
    {
        const auto& r = ranges[(size_t) i];

        if (index >= r.base && index < gfxSafeEndExclusive (r.base, (int64_t) r.count))
            return true;
    }

    return false;
}

// An explicit declaration overrides automatic mirroring for the covered cells.
// Overlapping explicit declarations union their direction flags. Undeclared
// cells retain the historical automatic policy, unless EXPLICIT was selected.
// 30 declarations can create at most 63 intervals, within kMaxGfxMemSpans.
static constexpr int kMaxGfxSyncDeclarations = 30;

static inline int buildDirectionalGfxRanges (
    int64_t automaticMemN, int64_t allocatedMemN, bool explicitOnly,
    const std::vector<GfxSyncMemRange>& declarations, uint8_t direction,
    std::array<GfxMirrorRange, kMaxGfxMemSpans>& out) noexcept
{
    if (allocatedMemN <= 0 || declarations.size() > (size_t) kMaxGfxSyncDeclarations)
        return 0; // Fail closed, never broaden a range on overflow.

    if (declarations.empty())
        return explicitOnly ? 0 : buildGfxMirrorRanges (
            std::max<int64_t> (0, std::min (automaticMemN, allocatedMemN)), out);

    std::array<GfxMirrorRange, kMaxGfxMemSpans> automatic {};
    const int autoCount = explicitOnly ? 0 : buildGfxMirrorRanges (
        std::max<int64_t> (0, std::min (automaticMemN, allocatedMemN)), automatic);
    std::array<int64_t, kMaxGfxSyncDeclarations * 2 + 6> boundaries {};
    int boundaryCount = 0;
    for (int i = 0; i < autoCount; ++i)
    {
        boundaries[(size_t) boundaryCount++] = automatic[(size_t) i].base;
        boundaries[(size_t) boundaryCount++] = gfxSafeEndExclusive (
            automatic[(size_t) i].base, automatic[(size_t) i].count);
    }
    for (const auto& r : declarations)
    {
        if (r.base < 0 || r.count <= 0 || r.base >= allocatedMemN)
            continue;
        boundaries[(size_t) boundaryCount++] = r.base;
        boundaries[(size_t) boundaryCount++] = std::min (
            allocatedMemN, gfxSafeEndExclusive (r.base, r.count));
    }
    std::sort (boundaries.begin(), boundaries.begin() + boundaryCount);
    boundaryCount = (int) (std::unique (boundaries.begin(), boundaries.begin() + boundaryCount)
                           - boundaries.begin());
    int n = 0;
    for (int i = 0; i + 1 < boundaryCount; ++i)
    {
        const int64_t base = boundaries[(size_t) i];
        const int64_t end = boundaries[(size_t) i + 1];
        bool explicitlyCovered = false;
        uint8_t flags = 0;
        for (const auto& r : declarations)
        {
            if (r.base >= 0 && r.count > 0 && base >= r.base
                && base < gfxSafeEndExclusive (r.base, r.count))
            {
                explicitlyCovered = true;
                flags = (uint8_t) (flags | r.flags);
            }
        }
        if (! explicitlyCovered)
            for (int j = 0; j < autoCount; ++j)
                if (base >= automatic[(size_t) j].base
                    && base < gfxSafeEndExclusive (automatic[(size_t) j].base,
                                                   automatic[(size_t) j].count))
                    flags = (uint8_t) (kGfxSyncToGfx | kGfxSyncFromGfx);
        if ((flags & direction) == 0u || end <= base)
            continue;
        if (end - base > (int64_t) std::numeric_limits<int>::max())
            return 0;
        if (n > 0 && gfxSafeEndExclusive (out[(size_t) n - 1].base,
                                         out[(size_t) n - 1].count) == base
            && end - out[(size_t) n - 1].base <= (int64_t) std::numeric_limits<int>::max())
            out[(size_t) n - 1].count = (int) (end - out[(size_t) n - 1].base);
        else
        {
            if (n >= (int) out.size()) return 0;
            out[(size_t) n++] = GfxMirrorRange { base, (int) (end - base) };
        }
    }
    return n;
}

static inline bool isDirectionalGfxMemIndex (
    int64_t index, int64_t automaticMemN, int64_t allocatedMemN, bool explicitOnly,
    const std::vector<GfxSyncMemRange>& declarations, uint8_t direction) noexcept
{
    if (index < 0 || index >= allocatedMemN
        || declarations.size() > (size_t) kMaxGfxSyncDeclarations) return false;
    uint8_t flags = 0;
    bool covered = false;
    for (const auto& r : declarations)
        if (r.base >= 0 && r.count > 0 && index >= r.base
            && index < gfxSafeEndExclusive (r.base, r.count))
        {
            covered = true;
            flags = (uint8_t) (flags | r.flags);
        }
    if (covered) return (flags & direction) != 0u;
    if (explicitOnly) return false;
    // Membership in the two automatic windows without constructing/sorting an array.
    const int64_t n = std::min (automaticMemN, allocatedMemN);
    const int64_t prefix = std::min<int64_t> (n, kGfxSharedPrefixDoubles);
    const int64_t suffix = std::min<int64_t> (std::max<int64_t> (0, n - prefix), kGfxSharedSuffixDoubles);
    return index < prefix || (index < n && index >= n - suffix);
}
#endif
