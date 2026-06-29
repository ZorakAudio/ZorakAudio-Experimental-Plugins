#include "DspJsfxSamplePool.h"
#include "DspJsfxAudioFilePreflight.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>

namespace za::jsfx
{
namespace
{
static constexpr int kPreviewBinsPerSample = 256;

static bool validSampleId(std::uint64_t id) noexcept
{
    return id > 0 && id <= static_cast<std::uint64_t>(std::numeric_limits<int>::max());
}

static std::uint64_t safeAudioBytes(std::int64_t frames, int channels) noexcept
{
    if (frames <= 0 || channels <= 0)
        return 0;
    const auto f = static_cast<std::uint64_t>(frames);
    const auto c = static_cast<std::uint64_t>(channels);
    if (f > std::numeric_limits<std::uint64_t>::max() / c)
        return std::numeric_limits<std::uint64_t>::max();
    const auto items = f * c;
    if (items > std::numeric_limits<std::uint64_t>::max() / sizeof(float))
        return std::numeric_limits<std::uint64_t>::max();
    return items * sizeof(float);
}

static bool shouldResampleToTarget(double sourceRate, double targetRate) noexcept
{
    return std::isfinite(sourceRate)
        && std::isfinite(targetRate)
        && sourceRate > 1000.0
        && targetRate > 1000.0
        && std::abs(sourceRate - targetRate) > 1.0;
}

static bool resampleInterleavedLinear(const float* src,
                                      std::uint64_t srcFrames,
                                      int channels,
                                      double sourceRate,
                                      double targetRate,
                                      std::vector<float>& out,
                                      std::uint64_t& outFrames)
{
    out.clear();
    outFrames = 0;

    if (src == nullptr || srcFrames == 0 || channels <= 0)
        return false;

    if (! shouldResampleToTarget(sourceRate, targetRate))
        return false;

    const double ratio = targetRate / sourceRate;
    if (! std::isfinite(ratio) || ratio <= 0.0)
        return false;

    const double dstFramesD = std::max(1.0, std::round(static_cast<double>(srcFrames) * ratio));
    if (! std::isfinite(dstFramesD) || dstFramesD > static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
        return false;

    outFrames = static_cast<std::uint64_t>(dstFramesD);
    const auto bytes = safeAudioBytes(static_cast<std::int64_t>(outFrames), channels);
    if (bytes == 0 || bytes == std::numeric_limits<std::uint64_t>::max())
        return false;

    const auto totalItems = outFrames * static_cast<std::uint64_t>(channels);
    if (totalItems > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float)))
        return false;

    try
    {
        out.resize(static_cast<std::size_t>(totalItems));
    }
    catch (...)
    {
        out.clear();
        outFrames = 0;
        return false;
    }

    for (std::uint64_t frame = 0; frame < outFrames; ++frame)
    {
        const double srcPos = static_cast<double>(frame) * sourceRate / targetRate;
        const auto pos0 = static_cast<std::uint64_t>(std::min<double>(static_cast<double>(srcFrames - 1), std::floor(srcPos)));
        const auto pos1 = std::min<std::uint64_t>(srcFrames - 1, pos0 + 1);
        const float frac = static_cast<float>(std::max(0.0, std::min(1.0, srcPos - static_cast<double>(pos0))));

        const auto dstBase = static_cast<std::size_t>(frame * static_cast<std::uint64_t>(channels));
        const auto srcBase0 = static_cast<std::size_t>(pos0 * static_cast<std::uint64_t>(channels));
        const auto srcBase1 = static_cast<std::size_t>(pos1 * static_cast<std::uint64_t>(channels));

        for (int ch = 0; ch < channels; ++ch)
        {
            const float a = src[srcBase0 + static_cast<std::size_t>(ch)];
            const float b = src[srcBase1 + static_cast<std::size_t>(ch)];
            out[dstBase + static_cast<std::size_t>(ch)] = a + (b - a) * frac;
        }
    }

    return true;
}

static void buildPreviewForEntry(DspJsfxSamplePoolGeneration& gen, DspJsfxSamplePoolEntry& entry)
{
    if (entry.frames == 0 || entry.channels == 0)
        return;

    const int bins = std::min<int>(kPreviewBinsPerSample, std::max<int>(1, static_cast<int>(entry.frames)));
    entry.previewOffset = static_cast<std::uint32_t>(gen.previews.size());
    entry.previewCount = static_cast<std::uint32_t>(bins);
    gen.previews.resize(gen.previews.size() + static_cast<std::size_t>(bins));

    for (int b = 0; b < bins; ++b)
    {
        const std::uint64_t start = (static_cast<std::uint64_t>(b) * entry.frames) / static_cast<std::uint64_t>(bins);
        std::uint64_t end = (static_cast<std::uint64_t>(b + 1) * entry.frames) / static_cast<std::uint64_t>(bins);
        if (end <= start)
            end = start + 1;
        end = std::min<std::uint64_t>(end, entry.frames);

        float mn = 1.0f;
        float mx = -1.0f;
        double sumSq = 0.0;
        std::uint64_t count = 0;

        for (std::uint64_t f = start; f < end; ++f)
        {
            float mono = 0.0f;
            for (int ch = 0; ch < static_cast<int>(entry.channels); ++ch)
                mono += gen.audio[static_cast<std::size_t>(entry.offsetItems + f * entry.channels + static_cast<std::uint64_t>(ch))];
            mono /= static_cast<float>(std::max<int>(1, entry.channels));
            mn = std::min(mn, mono);
            mx = std::max(mx, mono);
            sumSq += static_cast<double>(mono) * static_cast<double>(mono);
            ++count;
        }

        auto& pb = gen.previews[static_cast<std::size_t>(entry.previewOffset + static_cast<std::uint32_t>(b))];
        pb.minValue = count > 0 ? mn : 0.0f;
        pb.maxValue = count > 0 ? mx : 0.0f;
        pb.rms = count > 0 ? static_cast<float>(std::sqrt(sumSq / static_cast<double>(count))) : 0.0f;
    }
}
}

DspJsfxSamplePool::DspJsfxSamplePool() = default;

DspJsfxSamplePool::~DspJsfxSamplePool()
{
    {
        std::lock_guard<std::mutex> lock(workerMutex_);
        workerExit_ = true;
        requestPending_ = false;
    }
    workerCv_.notify_all();
    if (worker_.joinable())
        worker_.join();
    activeRaw_.store(nullptr, std::memory_order_release);
}

void DspJsfxSamplePool::setMode(int mode) noexcept
{
    if (mode < kSamplePoolModeResident || mode > kSamplePoolModeStream)
        mode = kSamplePoolModeResident;
    mode_.store(mode, std::memory_order_release);
}

void DspJsfxSamplePool::setBudgetMB(double mb) noexcept
{
    if (! std::isfinite(mb) || mb <= 0.0)
    {
        budgetBytes_.store(0, std::memory_order_release);
        return;
    }

    const double bytes = mb * 1024.0 * 1024.0;
    const auto clamped = bytes >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
        ? std::numeric_limits<std::uint64_t>::max()
        : static_cast<std::uint64_t>(bytes);
    budgetBytes_.store(clamped, std::memory_order_release);
}

void DspJsfxSamplePool::setTargetSampleRate(double sampleRate) noexcept
{
    if (! std::isfinite(sampleRate) || sampleRate <= 1000.0)
        sampleRate = 0.0;

    targetSampleRate_.store(sampleRate, std::memory_order_release);
}

void DspJsfxSamplePool::setCompletionCallback(CompletionCallback callback)
{
    std::lock_guard<std::mutex> lock(callbackMutex_);
    completionCallback_ = std::move(callback);
}

bool DspJsfxSamplePool::commitFromPaths(const std::vector<juce::String>& paths, std::uint64_t sourceGeneration)
{
    const int currentMode = mode_.load(std::memory_order_acquire);
    const auto currentBudget = budgetBytes_.load(std::memory_order_acquire);
    const double currentTargetRate = targetSampleRate_.load(std::memory_order_acquire);
    const double committedTargetRate = lastCommittedTargetSampleRate_.load(std::memory_order_acquire);

    if (sourceGeneration != 0
        && lastCommittedSourceGeneration_.load(std::memory_order_acquire) == sourceGeneration
        && lastCommittedMode_.load(std::memory_order_acquire) == currentMode
        && lastCommittedBudgetBytes_.load(std::memory_order_acquire) == currentBudget
        && lastCommittedSourceKind_.load(std::memory_order_acquire) == 0
        && std::abs(committedTargetRate - currentTargetRate) <= 0.5)
        return true;

    ensureWorker();

    Request req;
    req.paths = paths;
    req.sourceGeneration = sourceGeneration;
    req.mode = currentMode;
    req.budgetBytes = currentBudget;
    req.targetSampleRate = currentTargetRate;
    req.requestId = nextRequestId_.fetch_add(1, std::memory_order_acq_rel);

    selected_.store(static_cast<int>(req.paths.size()), std::memory_order_release);
    failed_.store(0, std::memory_order_release);
    state_.store(req.paths.empty() ? kSamplePoolEmpty : kSamplePoolScanning, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(workerMutex_);
        pendingRequest_ = std::move(req);
        requestPending_ = true;
    }
    workerCv_.notify_one();

    lastCommittedSourceGeneration_.store(sourceGeneration, std::memory_order_release);
    lastCommittedMode_.store(currentMode, std::memory_order_release);
    lastCommittedBudgetBytes_.store(currentBudget, std::memory_order_release);
    lastCommittedSourceKind_.store(0, std::memory_order_release);
    lastCommittedTargetSampleRate_.store(currentTargetRate, std::memory_order_release);
    return true;
}

bool DspJsfxSamplePool::commitFromMemory(std::shared_ptr<const DspJsfxSamplePoolMemorySourceList> sources,
                                         std::uint64_t sourceGeneration)
{
    const int currentMode = mode_.load(std::memory_order_acquire);
    const auto currentBudget = budgetBytes_.load(std::memory_order_acquire);
    const double currentTargetRate = targetSampleRate_.load(std::memory_order_acquire);
    const double committedTargetRate = lastCommittedTargetSampleRate_.load(std::memory_order_acquire);

    if (sourceGeneration != 0
        && lastCommittedSourceGeneration_.load(std::memory_order_acquire) == sourceGeneration
        && lastCommittedMode_.load(std::memory_order_acquire) == currentMode
        && lastCommittedBudgetBytes_.load(std::memory_order_acquire) == currentBudget
        && lastCommittedSourceKind_.load(std::memory_order_acquire) == 1
        && std::abs(committedTargetRate - currentTargetRate) <= 0.5)
        return true;

    ensureWorker();

    Request req;
    req.memorySources = std::move(sources);
    req.sourceGeneration = sourceGeneration;
    req.mode = currentMode;
    req.budgetBytes = currentBudget;
    req.targetSampleRate = currentTargetRate;
    req.requestId = nextRequestId_.fetch_add(1, std::memory_order_acq_rel);

    const int selected = req.memorySources != nullptr ? static_cast<int>(req.memorySources->size()) : 0;
    selected_.store(selected, std::memory_order_release);
    failed_.store(0, std::memory_order_release);
    state_.store(selected > 0 ? kSamplePoolLoading : kSamplePoolEmpty, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(workerMutex_);
        pendingRequest_ = std::move(req);
        requestPending_ = true;
    }
    workerCv_.notify_one();

    lastCommittedSourceGeneration_.store(sourceGeneration, std::memory_order_release);
    lastCommittedMode_.store(currentMode, std::memory_order_release);
    lastCommittedBudgetBytes_.store(currentBudget, std::memory_order_release);
    lastCommittedSourceKind_.store(1, std::memory_order_release);
    lastCommittedTargetSampleRate_.store(currentTargetRate, std::memory_order_release);
    return true;
}

int DspJsfxSamplePool::loaded() const noexcept
{
    if (const auto* gen = active())
        return static_cast<int>(gen->entries.size());
    return 0;
}

double DspJsfxSamplePool::ramMB() const noexcept
{
    return static_cast<double>(decodedBytes_.load(std::memory_order_acquire)) / (1024.0 * 1024.0);
}

std::uint64_t DspJsfxSamplePool::generation() const noexcept
{
    return publishedGeneration_.load(std::memory_order_acquire);
}

const DspJsfxSamplePoolEntry* DspJsfxSamplePool::entryFor(const DspJsfxSamplePoolGeneration* gen, std::uint64_t sampleId) const noexcept
{
    if (gen == nullptr || ! validSampleId(sampleId))
        return nullptr;
    const auto idx = static_cast<std::size_t>(sampleId - 1);
    if (idx >= gen->entries.size())
        return nullptr;
    return &gen->entries[idx];
}

std::uint64_t DspJsfxSamplePool::sampleIdAt(int index) const noexcept
{
    const auto* gen = active();
    if (gen == nullptr || index < 0 || index >= static_cast<int>(gen->entries.size()))
        return 0;
    return gen->entries[static_cast<std::size_t>(index)].id;
}

int DspJsfxSamplePool::length(std::uint64_t sampleId) const noexcept
{
    if (auto* e = entryFor(active(), sampleId))
        return static_cast<int>(e->frames);
    return 0;
}

int DspJsfxSamplePool::channels(std::uint64_t sampleId) const noexcept
{
    if (auto* e = entryFor(active(), sampleId))
        return static_cast<int>(e->channels);
    return 0;
}

int DspJsfxSamplePool::sampleRate(std::uint64_t sampleId) const noexcept
{
    if (auto* e = entryFor(active(), sampleId))
        return static_cast<int>(e->sampleRate);
    return 0;
}

double DspJsfxSamplePool::peak(std::uint64_t sampleId) const noexcept
{
    if (auto* e = entryFor(active(), sampleId))
        return static_cast<double>(e->peak);
    return 0.0;
}

double DspJsfxSamplePool::rms(std::uint64_t sampleId) const noexcept
{
    if (auto* e = entryFor(active(), sampleId))
        return static_cast<double>(e->rms);
    return 0.0;
}

bool DspJsfxSamplePool::name(std::uint64_t sampleId, std::string& out) const
{
    out.clear();
    const auto* gen = active();
    if (gen == nullptr || ! validSampleId(sampleId))
        return false;
    const auto idx = static_cast<std::size_t>(sampleId - 1);
    if (idx >= gen->names.size())
        return false;
    out = gen->names[idx];
    return true;
}

double DspJsfxSamplePool::read(std::uint64_t sampleId, int channel, double frame) const noexcept
{
    const auto* gen = active();
    const auto* e = entryFor(gen, sampleId);
    if (gen == nullptr || e == nullptr || e->frames == 0 || e->channels == 0)
        return 0.0;

    if (! std::isfinite(frame))
        frame = 0.0;
    auto f = static_cast<std::int64_t>(std::llround(frame));
    if (f < 0 || f >= static_cast<std::int64_t>(e->frames))
        return 0.0;

    if (channel < 0)
        channel = 0;
    if (channel >= static_cast<int>(e->channels))
        channel = static_cast<int>(e->channels) - 1;

    const auto idx = static_cast<std::size_t>(e->offsetItems + static_cast<std::uint64_t>(f) * e->channels + static_cast<std::uint64_t>(channel));
    if (idx >= gen->audio.size())
        return 0.0;
    return static_cast<double>(gen->audio[idx]);
}

double DspJsfxSamplePool::readInterp(std::uint64_t sampleId, int channel, double phase) const noexcept
{
    if (! std::isfinite(phase))
        phase = 0.0;
    const auto base = std::floor(phase);
    const double frac = phase - base;
    const double x0 = read(sampleId, channel, base);
    const double x1 = read(sampleId, channel, base + 1.0);
    return x0 + (x1 - x0) * frac;
}

bool DspJsfxSamplePool::read2(std::uint64_t sampleId, double phase, double* outL, double* outR, bool interp) const noexcept
{
    const auto* gen = active();
    const auto* e = entryFor(gen, sampleId);
    if (gen == nullptr || e == nullptr || e->frames == 0 || e->channels == 0)
    {
        if (outL) *outL = 0.0;
        if (outR) *outR = 0.0;
        return false;
    }

    // Hard sample-boundary rule. A rendered segment is a complete pseudo-file;
    // reads beyond its frame range must not report success or bleed into the
    // following packed entry in the generation. read() already returns zero
    // out-of-range, but returning false lets JSFX voices terminate cleanly.
    if (! std::isfinite(phase) || phase < 0.0 || phase > static_cast<double>(e->frames - 1))
    {
        if (outL) *outL = 0.0;
        if (outR) *outR = 0.0;
        return false;
    }

    const double l = interp ? readInterp(sampleId, 0, phase) : read(sampleId, 0, phase);
    const double r = e->channels >= 2
        ? (interp ? readInterp(sampleId, 1, phase) : read(sampleId, 1, phase))
        : l;
    if (outL) *outL = l;
    if (outR) *outR = r;
    return true;
}

int DspJsfxSamplePool::previewBins(std::uint64_t sampleId) const noexcept
{
    if (auto* e = entryFor(active(), sampleId))
        return static_cast<int>(e->previewCount);
    return 0;
}

bool DspJsfxSamplePool::previewRead(std::uint64_t sampleId, int bin, double* minValue, double* maxValue, double* rmsValue) const noexcept
{
    const auto* gen = active();
    const auto* e = entryFor(gen, sampleId);
    if (gen == nullptr || e == nullptr || bin < 0 || bin >= static_cast<int>(e->previewCount))
        return false;
    const auto idx = static_cast<std::size_t>(e->previewOffset + static_cast<std::uint32_t>(bin));
    if (idx >= gen->previews.size())
        return false;
    const auto& p = gen->previews[idx];
    if (minValue) *minValue = p.minValue;
    if (maxValue) *maxValue = p.maxValue;
    if (rmsValue) *rmsValue = p.rms;
    return true;
}

void DspJsfxSamplePool::ensureWorker()
{
    std::lock_guard<std::mutex> lock(workerMutex_);
    if (! worker_.joinable())
        worker_ = std::thread([this] { workerMain(); });
}

void DspJsfxSamplePool::workerMain()
{
    for (;;)
    {
        Request req;
        {
            std::unique_lock<std::mutex> lock(workerMutex_);
            workerCv_.wait(lock, [this] { return workerExit_ || requestPending_; });
            if (workerExit_)
                return;
            req = std::move(pendingRequest_);
            requestPending_ = false;
        }

        state_.store(kSamplePoolLoading, std::memory_order_release);
        auto gen = buildGeneration(req);
        publishGeneration(std::move(gen), req.requestId);
    }
}

std::shared_ptr<DspJsfxSamplePoolGeneration> DspJsfxSamplePool::buildGeneration(const Request& request)
{
    auto gen = std::make_shared<DspJsfxSamplePoolGeneration>();
    gen->sourceGeneration = request.sourceGeneration;

    std::uint64_t usedBytes = 0;
    const bool budgeted = (request.mode == kSamplePoolModeBudgeted || request.mode == kSamplePoolModeLazy || request.mode == kSamplePoolModeStream) && request.budgetBytes > 0;

    if (request.memorySources != nullptr)
    {
        gen->selectedCount = static_cast<int>(request.memorySources->size());

        for (const auto& src : *request.memorySources)
        {
            const int channels = std::max<int>(1, std::min<int>(64, static_cast<int>(src.channels)));
            const std::uint64_t totalItems = static_cast<std::uint64_t>(src.audio.size());
            const std::uint64_t frames64 = totalItems / static_cast<std::uint64_t>(channels);

            if (frames64 == 0 || totalItems == 0 || totalItems % static_cast<std::uint64_t>(channels) != 0)
            {
                ++gen->failedCount;
                continue;
            }

            std::vector<float> resampledAudio;
            std::uint64_t resampledFrames = 0;
            const float* sourceAudio = src.audio.data();
            std::uint64_t sourceFrames = frames64;
            std::uint32_t sourceSampleRate = std::max<std::uint32_t>(1u, src.sampleRate);

            if (resampleInterleavedLinear(src.audio.data(),
                                          frames64,
                                          channels,
                                          static_cast<double>(sourceSampleRate),
                                          request.targetSampleRate,
                                          resampledAudio,
                                          resampledFrames)
                && resampledFrames > 0)
            {
                sourceAudio = resampledAudio.data();
                sourceFrames = resampledFrames;
                sourceSampleRate = static_cast<std::uint32_t>(std::max<int>(1, static_cast<int>(std::llround(request.targetSampleRate))));
            }

            const auto bytes = safeAudioBytes(static_cast<std::int64_t>(sourceFrames), channels);
            if (bytes == 0 || bytes == std::numeric_limits<std::uint64_t>::max())
            {
                ++gen->failedCount;
                continue;
            }

            if (budgeted && usedBytes + bytes > request.budgetBytes)
            {
                ++gen->failedCount;
                continue;
            }

            DspJsfxSamplePoolEntry entry;
            entry.id = static_cast<std::uint64_t>(gen->entries.size() + 1);
            entry.offsetItems = static_cast<std::uint64_t>(gen->audio.size());
            entry.frames = static_cast<std::uint32_t>(std::min<std::uint64_t>(sourceFrames, std::numeric_limits<std::uint32_t>::max()));
            entry.channels = static_cast<std::uint16_t>(channels);
            entry.sampleRate = sourceSampleRate;

            const std::uint64_t keptItems = static_cast<std::uint64_t>(entry.frames) * static_cast<std::uint64_t>(entry.channels);
            if (keptItems > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))
                || keptItems > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
            {
                ++gen->failedCount;
                continue;
            }

            try
            {
                gen->audio.insert(gen->audio.end(), sourceAudio, sourceAudio + static_cast<std::ptrdiff_t>(keptItems));
            }
            catch (...)
            {
                ++gen->failedCount;
                continue;
            }

            double sumSq = 0.0;
            float peak = 0.0f;
            for (std::uint64_t i = 0; i < keptItems; ++i)
            {
                const float v = gen->audio[static_cast<std::size_t>(entry.offsetItems + i)];
                peak = std::max(peak, std::abs(v));
                sumSq += static_cast<double>(v) * static_cast<double>(v);
            }

            entry.peak = peak;
            entry.rms = keptItems > 0 ? static_cast<float>(std::sqrt(sumSq / static_cast<double>(keptItems))) : 0.0f;
            buildPreviewForEntry(*gen, entry);
            gen->names.push_back(src.name.empty() ? std::string("memory sample") : src.name);
            gen->entries.push_back(entry);
            usedBytes += bytes;
        }

        gen->decodedBytes = static_cast<std::uint64_t>(gen->audio.size()) * sizeof(float);
        return gen;
    }

    gen->selectedCount = static_cast<int>(request.paths.size());

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();

    for (const auto& path : request.paths)
    {
        const juce::File file(path);
        std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(file));
        if (! reader)
        {
            ++gen->failedCount;
            continue;
        }

        const int channels = std::max<int>(1, std::min<int>(64, static_cast<int>(reader->numChannels)));

        bool skipUnsafeMalformedFile = false;
        const auto safeFrames = chooseSafeDecodedFrameCount(file, *reader, skipUnsafeMalformedFile);
        if (skipUnsafeMalformedFile || safeFrames <= 0)
        {
            ++gen->failedCount;
            continue;
        }

        const auto nativeFrames64 = std::min<std::int64_t>(safeFrames, std::numeric_limits<std::uint32_t>::max());
        const auto nativeItems = static_cast<std::uint64_t>(nativeFrames64) * static_cast<std::uint64_t>(channels);
        if (nativeItems == 0 || nativeItems > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float)))
        {
            ++gen->failedCount;
            continue;
        }

        std::vector<float> decoded;
        try
        {
            decoded.resize(static_cast<std::size_t>(nativeItems));
        }
        catch (...)
        {
            ++gen->failedCount;
            continue;
        }

        juce::AudioBuffer<float> temp;
        constexpr int chunk = 65536;
        bool ok = true;

        for (std::int64_t pos = 0; pos < nativeFrames64; pos += chunk)
        {
            const int toRead = static_cast<int>(std::min<std::int64_t>(chunk, nativeFrames64 - pos));
            temp.setSize(channels, toRead, false, false, true);
            temp.clear();
            if (! reader->read(&temp, 0, toRead, pos, true, true))
            {
                ok = false;
                break;
            }

            for (int i = 0; i < toRead; ++i)
            {
                const auto frame = static_cast<std::uint64_t>(pos + i);
                const auto base = static_cast<std::size_t>(frame * static_cast<std::uint64_t>(channels));
                for (int ch = 0; ch < channels; ++ch)
                    decoded[base + static_cast<std::size_t>(ch)] = temp.getSample(ch, i);
            }
        }

        if (! ok)
        {
            ++gen->failedCount;
            continue;
        }

        const double nativeSampleRate = reader->sampleRate > 1000.0 ? reader->sampleRate : 48000.0;
        std::vector<float> resampledAudio;
        std::uint64_t resampledFrames = 0;
        const float* sourceAudio = decoded.data();
        std::uint64_t sourceFrames = static_cast<std::uint64_t>(nativeFrames64);
        std::uint32_t sourceSampleRate = static_cast<std::uint32_t>(std::max<int>(1, static_cast<int>(std::llround(nativeSampleRate))));

        if (resampleInterleavedLinear(decoded.data(),
                                      sourceFrames,
                                      channels,
                                      nativeSampleRate,
                                      request.targetSampleRate,
                                      resampledAudio,
                                      resampledFrames)
            && resampledFrames > 0)
        {
            sourceAudio = resampledAudio.data();
            sourceFrames = resampledFrames;
            sourceSampleRate = static_cast<std::uint32_t>(std::max<int>(1, static_cast<int>(std::llround(request.targetSampleRate))));
        }

        const auto entryFrames64 = std::min<std::uint64_t>(sourceFrames, std::numeric_limits<std::uint32_t>::max());
        const auto bytes = safeAudioBytes(static_cast<std::int64_t>(entryFrames64), channels);
        if (bytes == 0 || bytes == std::numeric_limits<std::uint64_t>::max())
        {
            ++gen->failedCount;
            continue;
        }

        if (budgeted && usedBytes + bytes > request.budgetBytes)
        {
            ++gen->failedCount;
            continue;
        }

        DspJsfxSamplePoolEntry entry;
        entry.id = static_cast<std::uint64_t>(gen->entries.size() + 1);
        entry.offsetItems = static_cast<std::uint64_t>(gen->audio.size());
        entry.frames = static_cast<std::uint32_t>(entryFrames64);
        entry.channels = static_cast<std::uint16_t>(channels);
        entry.sampleRate = sourceSampleRate;

        const auto totalItems = static_cast<std::uint64_t>(entry.frames) * static_cast<std::uint64_t>(entry.channels);
        if (totalItems > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))
            || totalItems > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        {
            ++gen->failedCount;
            continue;
        }

        const auto oldSize = gen->audio.size();
        try
        {
            gen->audio.insert(gen->audio.end(), sourceAudio, sourceAudio + static_cast<std::ptrdiff_t>(totalItems));
        }
        catch (...)
        {
            gen->audio.resize(oldSize);
            ++gen->failedCount;
            continue;
        }

        double sumSq = 0.0;
        float peak = 0.0f;
        for (std::uint64_t i = 0; i < totalItems; ++i)
        {
            const float v = gen->audio[static_cast<std::size_t>(entry.offsetItems + i)];
            peak = std::max(peak, std::abs(v));
            sumSq += static_cast<double>(v) * static_cast<double>(v);
        }

        entry.peak = peak;
        entry.rms = totalItems > 0 ? static_cast<float>(std::sqrt(sumSq / static_cast<double>(totalItems))) : 0.0f;
        buildPreviewForEntry(*gen, entry);
        gen->names.push_back(file.getFileName().toStdString());
        gen->entries.push_back(entry);
        usedBytes += bytes;
    }

    gen->decodedBytes = static_cast<std::uint64_t>(gen->audio.size()) * sizeof(float);
    return gen;
}

void DspJsfxSamplePool::publishGeneration(std::shared_ptr<DspJsfxSamplePoolGeneration> gen, std::uint64_t)
{
    if (! gen)
    {
        state_.store(kSamplePoolFailed, std::memory_order_release);
        CompletionCallback callback;
        {
            std::lock_guard<std::mutex> lock(callbackMutex_);
            callback = completionCallback_;
        }
        if (callback)
            callback(0, kSamplePoolFailed);
        return;
    }

    const auto* raw = gen.get();
    {
        std::lock_guard<std::mutex> lock(generationMutex_);
        // Audio/GFX reads are intentionally lock-free against activeRaw_. That means
        // an old generation must not be freed while a thread could still be inside a
        // read helper. Retain immutable generations for the plugin lifetime; sample
        // bank reloads are user-scale events, not per-block allocations.
        generations_.push_back(gen);
    }

    activeRaw_.store(raw, std::memory_order_release);
    selected_.store(gen->selectedCount, std::memory_order_release);
    failed_.store(gen->failedCount, std::memory_order_release);
    decodedBytes_.store(gen->decodedBytes, std::memory_order_release);
    publishedGeneration_.store(gen->sourceGeneration, std::memory_order_release);

    int finalState = kSamplePoolReady;
    if (gen->entries.empty())
        finalState = gen->selectedCount > 0 ? kSamplePoolFailed : kSamplePoolEmpty;
    else if (gen->failedCount > 0 || static_cast<int>(gen->entries.size()) < gen->selectedCount)
        finalState = kSamplePoolPartial;

    state_.store(finalState, std::memory_order_release);

    CompletionCallback callback;
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        callback = completionCallback_;
    }
    if (callback)
        callback(gen->sourceGeneration, finalState);
}

} // namespace za::jsfx
