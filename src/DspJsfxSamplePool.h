#pragma once

#ifndef NOMINMAX
#define NOMINMAX 1
#endif

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace za::jsfx
{

enum DspJsfxSamplePoolState : int
{
    kSamplePoolEmpty = 0,
    kSamplePoolScanning = 1,
    kSamplePoolLoading = 2,
    kSamplePoolReady = 3,
    kSamplePoolPartial = 4,
    kSamplePoolFailed = 5,
};

enum DspJsfxSamplePoolMode : int
{
    kSamplePoolModeResident = 0,
    kSamplePoolModeBudgeted = 1,
    kSamplePoolModeLazy = 2,
    kSamplePoolModeStream = 3,
};

struct DspJsfxSamplePreviewBin
{
    float minValue = 0.0f;
    float maxValue = 0.0f;
    float rms = 0.0f;
};

struct DspJsfxSamplePoolEntry
{
    std::uint64_t id = 0;        // 1-based stable within the active generation
    std::uint64_t offsetItems = 0;
    std::uint32_t frames = 0;
    std::uint32_t sampleRate = 0;
    std::uint16_t channels = 0;
    std::uint16_t reserved = 0;
    std::uint32_t previewOffset = 0;
    std::uint32_t previewCount = 0;
    float peak = 0.0f;
    float rms = 0.0f;
};

struct DspJsfxSamplePoolGeneration
{
    std::uint64_t sourceGeneration = 0;
    std::vector<DspJsfxSamplePoolEntry> entries;
    std::vector<float> audio; // packed float32, interleaved per entry
    std::vector<DspJsfxSamplePreviewBin> previews;
    std::vector<std::string> names;
    int selectedCount = 0;
    int failedCount = 0;
    std::uint64_t decodedBytes = 0;
};

class DspJsfxSamplePool
{
public:
    DspJsfxSamplePool();
    ~DspJsfxSamplePool();

    DspJsfxSamplePool(const DspJsfxSamplePool&) = delete;
    DspJsfxSamplePool& operator=(const DspJsfxSamplePool&) = delete;

    void setMode(int mode) noexcept;
    void setBudgetMB(double mb) noexcept;

    int mode() const noexcept { return mode_.load(std::memory_order_acquire); }
    double budgetMB() const noexcept { return static_cast<double>(budgetBytes_.load(std::memory_order_acquire)) / (1024.0 * 1024.0); }

    // Schedules a background scan/decode if paths or settings changed.
    // The completed generation is immutable and atomically published.
    bool commitFromPaths(const std::vector<juce::String>& paths, std::uint64_t sourceGeneration);

    int state() const noexcept { return state_.load(std::memory_order_acquire); }
    int selected() const noexcept { return selected_.load(std::memory_order_acquire); }
    int loaded() const noexcept;
    int failed() const noexcept { return failed_.load(std::memory_order_acquire); }
    double ramMB() const noexcept;
    std::uint64_t generation() const noexcept;

    std::uint64_t sampleIdAt(int index) const noexcept;
    int length(std::uint64_t sampleId) const noexcept;
    int channels(std::uint64_t sampleId) const noexcept;
    int sampleRate(std::uint64_t sampleId) const noexcept;
    double peak(std::uint64_t sampleId) const noexcept;
    double rms(std::uint64_t sampleId) const noexcept;
    bool name(std::uint64_t sampleId, std::string& out) const;

    double read(std::uint64_t sampleId, int channel, double frame) const noexcept;
    double readInterp(std::uint64_t sampleId, int channel, double phase) const noexcept;
    bool read2(std::uint64_t sampleId, double phase, double* outL, double* outR, bool interp) const noexcept;

    int previewBins(std::uint64_t sampleId) const noexcept;
    bool previewRead(std::uint64_t sampleId, int bin, double* minValue, double* maxValue, double* rmsValue) const noexcept;

private:
    struct Request
    {
        std::vector<juce::String> paths;
        std::uint64_t sourceGeneration = 0;
        int mode = kSamplePoolModeResident;
        std::uint64_t budgetBytes = 0;
        std::uint64_t requestId = 0;
    };

    static std::shared_ptr<DspJsfxSamplePoolGeneration> buildGeneration(const Request& request);
    void ensureWorker();
    void workerMain();
    void publishGeneration(std::shared_ptr<DspJsfxSamplePoolGeneration> gen, std::uint64_t requestId);
    const DspJsfxSamplePoolGeneration* active() const noexcept { return activeRaw_.load(std::memory_order_acquire); }
    const DspJsfxSamplePoolEntry* entryFor(const DspJsfxSamplePoolGeneration* gen, std::uint64_t sampleId) const noexcept;

    std::atomic<int> mode_ { kSamplePoolModeResident };
    std::atomic<std::uint64_t> budgetBytes_ { 0 }; // 0 = unlimited in resident mode
    std::atomic<int> state_ { kSamplePoolEmpty };
    std::atomic<int> selected_ { 0 };
    std::atomic<int> failed_ { 0 };
    std::atomic<const DspJsfxSamplePoolGeneration*> activeRaw_ { nullptr };
    std::atomic<std::uint64_t> publishedGeneration_ { 0 };
    std::atomic<std::uint64_t> decodedBytes_ { 0 };
    std::atomic<std::uint64_t> nextRequestId_ { 1 };
    std::atomic<std::uint64_t> lastCommittedSourceGeneration_ { 0 };
    std::atomic<std::uint64_t> lastCommittedBudgetBytes_ { 0 };
    std::atomic<int> lastCommittedMode_ { -1 };

    mutable std::mutex generationMutex_;
    std::vector<std::shared_ptr<const DspJsfxSamplePoolGeneration>> generations_; // retains immutable generations for lock-free readers

    std::mutex workerMutex_;
    std::condition_variable workerCv_;
    bool workerExit_ = false;
    bool requestPending_ = false;
    Request pendingRequest_;
    std::thread worker_;
};

} // namespace za::jsfx
