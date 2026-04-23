#include "DspJsfxGmem.h"
#include "JSFXDSP.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

extern "C" void jsfx_ensure_mem(DSPJSFX_State* st, int64_t needed);

namespace za::jsfx
{

namespace
{
static std::uint64_t doubleToBits(double v) noexcept
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static double bitsToDouble(std::uint64_t bits) noexcept
{
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

static std::uint64_t requiredPageCount(std::uint64_t cellCount) noexcept
{
    return (cellCount + static_cast<std::uint64_t> (kDspJsfxGmemPageCells) - 1ull) / static_cast<std::uint64_t> (kDspJsfxGmemPageCells);
}

static std::size_t requiredBytes(std::uint64_t cellCount) noexcept
{
    const std::uint64_t pageCount = requiredPageCount(cellCount);
    return sizeof(DspJsfxGmemHeader)
        + static_cast<std::size_t> (pageCount) * sizeof(DspJsfxGmemPageHeader)
        + static_cast<std::size_t> (cellCount) * sizeof(std::atomic<std::uint64_t>);
}

static std::string objectStem(std::uint64_t domainHash, std::uint64_t namespaceHash)
{
    char buf[96] = {};
    std::snprintf(buf, sizeof(buf), "gmem_%016llx_%016llx",
                  static_cast<unsigned long long> (domainHash),
                  static_cast<unsigned long long> (namespaceHash));
    return std::string(buf);
}
} // namespace

std::uint64_t mixHashes(std::uint64_t a, std::uint64_t b) noexcept
{
    std::uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

std::uint64_t clampCellIndex(double idx) noexcept
{
    if (! std::isfinite(idx) || idx <= 0.0)
        return 0;
    const double truncated = std::floor(idx + 1.0e-5);
    if (truncated <= 0.0)
        return 0;
    if (truncated >= static_cast<double> (std::numeric_limits<std::uint64_t>::max()))
        return std::numeric_limits<std::uint64_t>::max();
    return static_cast<std::uint64_t> (truncated);
}

bool DspJsfxGmemAttachment::attach(std::uint64_t domainHash, std::uint64_t namespaceHash, std::uint64_t requestedCells, std::uint64_t writerId)
{
    detach();

    domainHash_ = domainHash;
    namespaceHash_ = namespaceHash;
    const std::uint64_t cellCount = std::max<std::uint64_t> (requestedCells, kDspJsfxDefaultGmemCells);

    bool created = false;
    if (! segment_.openOrCreate(objectStem(domainHash, namespaceHash), requiredBytes(cellCount), &created))
        return false;

    auto* base = static_cast<std::uint8_t*> (segment_.data());
    if (base == nullptr)
        return false;

    header_ = reinterpret_cast<DspJsfxGmemHeader*> (base);
    if (created || header_->magic != kDspJsfxGmemMagic)
    {
        std::memset(base, 0, requiredBytes(cellCount));
        header_->magic = kDspJsfxGmemMagic;
        header_->abiVersion = kDspJsfxGmemAbiVersion;
        header_->domainHash = domainHash;
        header_->namespaceHash = namespaceHash;
        header_->cellCount = cellCount;
        header_->pageCellCount = kDspJsfxGmemPageCells;
        header_->pageCount = static_cast<std::uint32_t> (requiredPageCount(cellCount));
        header_->globalSeq.store(0, std::memory_order_release);
        header_->refCount.store(0, std::memory_order_release);
    }

    if (! validateLayout())
    {
        detach();
        return false;
    }

    pages_ = reinterpret_cast<DspJsfxGmemPageHeader*> (base + sizeof(DspJsfxGmemHeader));
    cells_ = reinterpret_cast<std::atomic<std::uint64_t>*> (
        base + sizeof(DspJsfxGmemHeader) + sizeof(DspJsfxGmemPageHeader) * header_->pageCount);

    header_->refCount.fetch_add(1u, std::memory_order_acq_rel);
    header_->globalSeq.fetch_add(1u, std::memory_order_acq_rel);
    if (header_->pageCount > 0)
        bumpPage(0, writerId);
    return true;
}

void DspJsfxGmemAttachment::detach() noexcept
{
    if (header_ != nullptr)
        header_->refCount.fetch_sub(1u, std::memory_order_acq_rel);
    header_ = nullptr;
    pages_ = nullptr;
    cells_ = nullptr;
    segment_.close();
    domainHash_ = 0;
    namespaceHash_ = 0;
}

bool DspJsfxGmemAttachment::validateLayout() const noexcept
{
    if (header_ == nullptr)
        return false;
    if (header_->magic != kDspJsfxGmemMagic || header_->abiVersion != kDspJsfxGmemAbiVersion)
        return false;
    if (header_->pageCellCount != kDspJsfxGmemPageCells)
        return false;
    if (header_->cellCount == 0 || header_->pageCount == 0)
        return false;
    if (header_->domainHash != domainHash_ || header_->namespaceHash != namespaceHash_)
        return false;
    return true;
}

std::uint64_t DspJsfxGmemAttachment::cellCount() const noexcept
{
    return header_ != nullptr ? header_->cellCount : 0u;
}

std::uint64_t DspJsfxGmemAttachment::pageIndexForCell(std::uint64_t idx) const noexcept
{
    return static_cast<std::uint64_t> (kDspJsfxGmemPageCells) > 0u ? (idx / static_cast<std::uint64_t> (kDspJsfxGmemPageCells)) : 0u;
}

std::atomic<std::uint64_t>* DspJsfxGmemAttachment::cellPtr(std::uint64_t idx) const noexcept
{
    if (cells_ == nullptr || header_ == nullptr || idx >= header_->cellCount)
        return nullptr;
    return cells_ + idx;
}

DspJsfxGmemPageHeader* DspJsfxGmemAttachment::pagePtr(std::uint64_t page) const noexcept
{
    if (pages_ == nullptr || header_ == nullptr || page >= header_->pageCount)
        return nullptr;
    return pages_ + page;
}

void DspJsfxGmemAttachment::bumpPage(std::uint64_t page, std::uint64_t writerId) noexcept
{
    if (auto* p = pagePtr(page); p != nullptr)
    {
        p->lastWriterId.store(writerId, std::memory_order_release);
        p->seq.fetch_add(1u, std::memory_order_acq_rel);
    }
    if (header_ != nullptr)
        header_->globalSeq.fetch_add(1u, std::memory_order_acq_rel);
}

double DspJsfxGmemAttachment::load(double idx) const noexcept
{
    const std::uint64_t cell = clampCellIndex(idx);
    if (const auto* ptr = cellPtr(cell); ptr != nullptr)
        return bitsToDouble(ptr->load(std::memory_order_relaxed));
    return 0.0;
}

double DspJsfxGmemAttachment::store(double idx, double value, std::uint64_t writerId) noexcept
{
    const std::uint64_t cell = clampCellIndex(idx);
    if (auto* ptr = cellPtr(cell); ptr != nullptr)
    {
        ptr->store(doubleToBits(value), std::memory_order_relaxed);
        bumpPage(pageIndexForCell(cell), writerId);
        return value;
    }
    return 0.0;
}

int DspJsfxGmemAttachment::bulkGet(DSPJSFX_State& st, int dstBase, int srcIdx, int count) const noexcept
{
    if (count <= 0 || st.mem == nullptr || ! isAttached())
        return 0;
    if (dstBase < 0 || srcIdx < 0)
        return 0;
    const std::uint64_t src = static_cast<std::uint64_t> (srcIdx);
    const std::uint64_t n = static_cast<std::uint64_t> (count);
    if (src >= cellCount())
        return 0;
    const std::uint64_t available = std::min<std::uint64_t> (n, cellCount() - src);
    const std::uint64_t dstEnd = static_cast<std::uint64_t> (dstBase) + available;
    if (dstEnd > static_cast<std::uint64_t> (st.memN))
        jsfx_ensure_mem(&st, static_cast<int64_t> (dstEnd));
    if (dstEnd > static_cast<std::uint64_t> (st.memN) || st.mem == nullptr)
        return 0;
    for (std::uint64_t i = 0; i < available; ++i)
        st.mem[static_cast<std::size_t> (dstBase) + i] = bitsToDouble(cells_[src + i].load(std::memory_order_relaxed));
    return static_cast<int> (available);
}

int DspJsfxGmemAttachment::bulkPut(const DSPJSFX_State& st, int dstIdx, int srcBase, int count, std::uint64_t writerId) noexcept
{
    if (count <= 0 || st.mem == nullptr || ! isAttached())
        return 0;
    if (dstIdx < 0 || srcBase < 0)
        return 0;
    const std::uint64_t dst = static_cast<std::uint64_t> (dstIdx);
    const std::uint64_t src = static_cast<std::uint64_t> (srcBase);
    const std::uint64_t n = static_cast<std::uint64_t> (count);
    if (src + n > static_cast<std::uint64_t> (st.memN) || dst >= cellCount())
        return 0;
    const std::uint64_t available = std::min<std::uint64_t> (n, cellCount() - dst);
    std::uint64_t lastPage = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t i = 0; i < available; ++i)
    {
        cells_[dst + i].store(doubleToBits(st.mem[src + i]), std::memory_order_relaxed);
        const std::uint64_t page = pageIndexForCell(dst + i);
        if (page != lastPage)
        {
            bumpPage(page, writerId);
            lastPage = page;
        }
    }
    return static_cast<int> (available);
}

int DspJsfxGmemAttachment::fill(int dstIdx, double value, int count, std::uint64_t writerId) noexcept
{
    if (count <= 0 || ! isAttached() || dstIdx < 0)
        return 0;
    const std::uint64_t dst = static_cast<std::uint64_t> (dstIdx);
    if (dst >= cellCount())
        return 0;
    const std::uint64_t available = std::min<std::uint64_t> (static_cast<std::uint64_t> (count), cellCount() - dst);
    const auto bits = doubleToBits(value);
    std::uint64_t lastPage = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t i = 0; i < available; ++i)
    {
        cells_[dst + i].store(bits, std::memory_order_relaxed);
        const std::uint64_t page = pageIndexForCell(dst + i);
        if (page != lastPage)
        {
            bumpPage(page, writerId);
            lastPage = page;
        }
    }
    return static_cast<int> (available);
}

int DspJsfxGmemAttachment::zero(int dstIdx, int count, std::uint64_t writerId) noexcept
{
    return fill(dstIdx, 0.0, count, writerId);
}

int DspJsfxGmemAttachment::copy(int dstIdx, int srcIdx, int count, std::uint64_t writerId) noexcept
{
    if (count <= 0 || ! isAttached() || dstIdx < 0 || srcIdx < 0)
        return 0;
    const std::uint64_t dst = static_cast<std::uint64_t> (dstIdx);
    const std::uint64_t src = static_cast<std::uint64_t> (srcIdx);
    if (dst >= cellCount() || src >= cellCount())
        return 0;
    const std::uint64_t available = std::min<std::uint64_t> ({ static_cast<std::uint64_t> (count), cellCount() - dst, cellCount() - src });
    std::vector<std::uint64_t> temp;
    temp.reserve(static_cast<std::size_t> (available));
    for (std::uint64_t i = 0; i < available; ++i)
        temp.push_back(cells_[src + i].load(std::memory_order_relaxed));
    std::uint64_t lastPage = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t i = 0; i < available; ++i)
    {
        cells_[dst + i].store(temp[static_cast<std::size_t> (i)], std::memory_order_relaxed);
        const std::uint64_t page = pageIndexForCell(dst + i);
        if (page != lastPage)
        {
            bumpPage(page, writerId);
            lastPage = page;
        }
    }
    return static_cast<int> (available);
}

double DspJsfxGmemAttachment::pageSeq(int page) const noexcept
{
    if (page < 0)
        return header_ != nullptr ? static_cast<double> (header_->globalSeq.load(std::memory_order_acquire)) : 0.0;
    if (const auto* p = pagePtr(static_cast<std::uint64_t> (page)); p != nullptr)
        return static_cast<double> (p->seq.load(std::memory_order_acquire));
    return 0.0;
}

} // namespace za::jsfx
