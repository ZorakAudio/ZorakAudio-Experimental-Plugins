#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace za::jsfx
{
struct HostTransportObservation
{
    bool valid = false, playing = false, recording = false;
    bool hasSamples = false, hasSeconds = false;
    std::int64_t samples = 0;
    double seconds = 0.0;
};

class HostTransportTracker
{
public:
    void reset() noexcept { *this = {}; }
    bool update(const HostTransportObservation& now, int blockSamples, double sampleRate) noexcept
    {
        if (!now.valid)
        {
            haveSamples_ = haveSeconds_ = false;
            return false; // unknown is not a manufactured Stop
        }
        const double rate = std::isfinite(sampleRate) && sampleRate > 0.0 ? sampleRate : 1.0;
        const bool hasSeconds = now.hasSeconds && std::isfinite(now.seconds);
        bool discontinuity = haveState_ && wasPlaying_ && !now.playing;
        if (haveState_ && wasPlaying_ && now.playing)
        {
            if (now.hasSamples && haveSamples_)
            {
                // Integer host clock is authoritative. Allow one rounding sample,
                // not float-second jitter, and avoid signed overflow at extremes.
                const auto delta = static_cast<long double>(now.samples) - expectedSamples_;
                discontinuity = std::abs(delta) > 1.0L;
            }
            else if (hasSeconds && haveSeconds_)
            {
                // Coarser hosts cannot reliably identify tiny seeks from seconds.
                // Use the preceding block's prediction and a conservative fallback.
                const double tolerance = std::max(0.050, 3.0 * std::max(previousBlock_, blockSamples) / rate);
                discontinuity = std::abs(now.seconds - expectedSeconds_) > tolerance;
            }
        }
        haveState_ = true;
        wasPlaying_ = now.playing;
        haveSamples_ = now.hasSamples;
        haveSeconds_ = hasSeconds;
        previousBlock_ = std::max(0, blockSamples);
        expectedSamples_ = static_cast<long double>(now.samples) + previousBlock_;
        expectedSeconds_ = now.seconds + previousBlock_ / rate;
        return discontinuity;
    }
private:
    bool haveState_ = false, wasPlaying_ = false, haveSamples_ = false, haveSeconds_ = false;
    int previousBlock_ = 0;
    long double expectedSamples_ = 0.0L;
    double expectedSeconds_ = 0.0;
};
} // namespace za::jsfx
