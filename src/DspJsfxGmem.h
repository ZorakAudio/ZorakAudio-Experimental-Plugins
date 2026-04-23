#pragma once

#include "DspJsfxSharedMemory.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

struct DSPJSFX_State;

namespace za::jsfx
{

static constexpr std::uint32_t kDspJsfxGmemMagic = 0x474D454Du; // 'GMEM'
static constexpr std::uint32_t kDspJsfxGmemAbiVersion = 1u;
static constexpr std::uint64_t kDspJsfxDefaultGmemCells = 1024ull * 1024ull;
static constexpr std::uint32_t kDspJsfxGmemPageCells = 1024u;

struct DspJsfxGmemHeader
{
    std::uint32_t magic = 0;
    std::uint32_t abiVersion = 0;
    std::uint64_t domainHash = 0;
    std::uint64_t namespaceHash = 0;
    std::uint64_t cellCount = 0;
    std::uint32_t pageCellCount = 0;
    std::uint32_t pageCount = 0;
    std::atomic<std::uint64_t> globalSeq { 0 };
    std::atomic<std::uint32_t> refCount { 0 };
    std::uint32_t reserved = 0;
};

struct DspJsfxGmemPageHeader
{
    std::atomic<std::uint64_t> seq { 0 };
    std::atomic<std::uint64_t> lastWriterId { 0 };
};

class DspJsfxGmemAttachment
{
public:
    DspJsfxGmemAttachment() = default;
    ~DspJsfxGmemAttachment() = default;

    bool attach(std::uint64_t domainHash, std::uint64_t namespaceHash, std::uint64_t requestedCells, std::uint64_t writerId);
    void detach() noexcept;

    bool isAttached() const noexcept { return header_ != nullptr && cells_ != nullptr; }
    std::uint64_t cellCount() const noexcept;
    std::uint64_t pageCellCount() const noexcept { return kDspJsfxGmemPageCells; }
    std::uint64_t pageIndexForCell(std::uint64_t idx) const noexcept;

    double load(double idx) const noexcept;
    double store(double idx, double value, std::uint64_t writerId) noexcept;

    int bulkGet(DSPJSFX_State& st, int dstBase, int srcIdx, int count) const noexcept;
    int bulkPut(const DSPJSFX_State& st, int dstIdx, int srcBase, int count, std::uint64_t writerId) noexcept;
    int fill(int dstIdx, double value, int count, std::uint64_t writerId) noexcept;
    int zero(int dstIdx, int count, std::uint64_t writerId) noexcept;
    int copy(int dstIdx, int srcIdx, int count, std::uint64_t writerId) noexcept;

    double pageSeq(int page) const noexcept;

private:
    bool validateLayout() const noexcept;
    std::atomic<std::uint64_t>* cellPtr(std::uint64_t idx) const noexcept;
    DspJsfxGmemPageHeader* pagePtr(std::uint64_t page) const noexcept;
    void bumpPage(std::uint64_t page, std::uint64_t writerId) noexcept;

    DspJsfxSharedMemorySegment segment_;
    DspJsfxGmemHeader* header_ = nullptr;
    DspJsfxGmemPageHeader* pages_ = nullptr;
    std::atomic<std::uint64_t>* cells_ = nullptr;
    std::uint64_t domainHash_ = 0;
    std::uint64_t namespaceHash_ = 0;
};

std::uint64_t mixHashes(std::uint64_t a, std::uint64_t b) noexcept;
std::uint64_t clampCellIndex(double idx) noexcept;

} // namespace za::jsfx
