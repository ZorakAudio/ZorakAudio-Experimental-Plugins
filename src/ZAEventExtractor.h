#pragma once

// Deterministic, offline-only event analysis. No JUCE, model download, audio-thread
// work, or global state. Feature/profile version changes must be explicit.
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace za::fileimport::extractor
{
constexpr int featureVersion = 1;
constexpr int bandCount = 16;
constexpr int maxExampleFrames = 64;
constexpr int maxExamples = 32;
using Cancel = std::function<bool()>;
inline bool cancelled (const Cancel& c) { return c && c(); }
inline float unit (float v) { return std::clamp (v, 0.0f, 1.0f); }
inline float db (double power) { return (float) (10.0 * std::log10 (std::max (power, 1.0e-12))); }

struct AudioView
{
    const float* const* data = nullptr;
    int channels = 0, samples = 0;
    double rate = 0.0;
    bool valid() const noexcept { return data != nullptr && channels > 0 && samples > 0 && std::isfinite (rate) && rate > 0.0; }
    float at (int ch, int sample) const noexcept
    {
        const float v = data[ch][sample];
        return std::isfinite (v) ? v : 0.0f;
    }
};

struct Peak
{
    float low = 0.0f, high = 0.0f;
    void add (float v) { low = std::min (low, v); high = std::max (high, v); }
    void add (Peak p) { low = std::min (low, p.low); high = std::max (high, p.high); }
};

// Exact min/max range queries: cached aligned blocks plus at most two 32-sample
// edges. Work during paint is bounded by pixel width, not source duration.
class WaveformPyramid
{
public:
    bool build (AudioView audio, const Cancel& cancel = {})
    {
        levels.clear(); length = audio.samples;
        if (! audio.valid()) return true;
        levels.emplace_back ((size_t) (((int64_t) audio.samples + base - 1) / base));
        for (int i = 0; i < audio.samples; ++i)
        {
            if ((i & 16383) == 0 && cancelled (cancel)) { levels.clear(); return false; }
            auto& p = levels.front()[(size_t) (i / base)];
            for (int ch = 0; ch < audio.channels; ++ch) p.add (audio.at (ch, i));
        }
        while (levels.back().size() > 1)
        {
            const auto& previous = levels.back();
            std::vector<Peak> next ((previous.size() + 1) / 2);
            for (size_t i = 0; i < previous.size(); ++i) next[i / 2].add (previous[i]);
            levels.push_back (std::move (next));
        }
        return ! cancelled (cancel);
    }

    Peak range (AudioView audio, int begin, int end) const
    {
        Peak result;
        if (! audio.valid()) return result;
        begin = std::clamp (begin, 0, audio.samples);
        end = std::clamp (end, begin, audio.samples);
        if (levels.empty() || length != audio.samples)
        {
            // Used only for an empty/uninitialised cache. Do not scan long files
            // on the UI thread; the worker publishes the cache and audio together.
            if (end - begin > base * 2) return result;
            for (int i = begin; i < end; ++i)
                for (int ch = 0; ch < audio.channels; ++ch) result.add (audio.at (ch, i));
            return result;
        }
        auto addSample = [&] (int i)
        {
            for (int ch = 0; ch < audio.channels; ++ch) result.add (audio.at (ch, i));
        };
        while (begin < end && (begin % base != 0 || end - begin < base)) addSample (begin++);
        while (end > begin && end % base != 0) addSample (--end);
        int first = begin / base, last = end / base;
        size_t level = 0;
        while (first < last && level < levels.size())
        {
            if (first & 1) result.add (levels[level][(size_t) first++]);
            if (last & 1) result.add (levels[level][(size_t) --last]);
            first /= 2; last /= 2; ++level;
        }
        return result;
    }
    size_t bytes() const
    {
        size_t size = 0;
        for (const auto& l : levels) size += l.size() * sizeof (Peak);
        return size;
    }
private:
    static constexpr int base = 32;
    int length = 0;
    std::vector<std::vector<Peak>> levels;
};

struct Frame
{
    float energyDb = -120.0f;
    float peakDb = -120.0f;
    float noiseDb = -120.0f;
    float onset = 0.0f;
    float change = 0.0f;
    float zcr = 0.0f;
    // Square-root normalised spectral power. Channel powers are combined, never
    // channel waveforms: anti-phase stereo cannot cancel out of the detector.
    std::array<float, bandCount> bands {};
};

struct Analysis
{
    int version = featureVersion;
    double rate = 0.0;
    int samples = 0, hop = 1;
    std::vector<Frame> frames;
    int sampleAt (int frame) const { return (int) std::clamp ((int64_t) frame * hop, int64_t { 0 }, (int64_t) samples); }
    int frameAt (int sample) const { return std::clamp (sample / std::max (1, hop), 0, std::max (0, (int) frames.size() - 1)); }
};

class SmallFFT
{
public:
    static constexpr int size = 512;
    SmallFFT()
    {
        constexpr double pi = 3.14159265358979323846;
        for (int i = 0; i < size; ++i)
        {
            int v = i, r = 0;
            for (int b = 0; b < 9; ++b) { r = (r << 1) | (v & 1); v >>= 1; }
            reverse[(size_t) i] = r;
            window[(size_t) i] = (float) (0.5 - 0.5 * std::cos (2.0 * pi * i / (size - 1)));
        }
        for (int i = 0; i < size / 2; ++i)
            twiddle[(size_t) i] = std::polar (1.0f, (float) (-2.0 * pi * i / size));
    }
    void power (const float* input, std::array<float, size / 2 + 1>& powers) const
    {
        std::array<std::complex<float>, size> values {};
        for (int i = 0; i < size; ++i) values[(size_t) reverse[(size_t) i]] = input[i] * window[(size_t) i];
        for (int width = 2; width <= size; width *= 2)
            for (int start = 0; start < size; start += width)
                for (int j = 0; j < width / 2; ++j)
                {
                    const auto v = values[(size_t) (start + j + width / 2)] * twiddle[(size_t) (j * size / width)];
                    const auto u = values[(size_t) (start + j)];
                    values[(size_t) (start + j)] = u + v;
                    values[(size_t) (start + j + width / 2)] = u - v;
                }
        for (int i = 0; i <= size / 2; ++i) powers[(size_t) i] = std::norm (values[(size_t) i]);
    }
private:
    std::array<int, size> reverse {};
    std::array<float, size> window {};
    std::array<std::complex<float>, size / 2> twiddle {};
};

inline float bandDistance (const Frame& a, const Frame& b)
{
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (int k = 0; k < bandCount; ++k)
    {
        dot += a.bands[(size_t) k] * b.bands[(size_t) k];
        normA += a.bands[(size_t) k] * a.bands[(size_t) k];
        normB += b.bands[(size_t) k] * b.bands[(size_t) k];
    }
    if (normA < 1.0e-12f && normB < 1.0e-12f) return 0.0f;
    return unit (1.0f - dot);
}

inline Analysis analyse (AudioView audio, const Cancel& cancel = {})
{
    Analysis out;
    out.rate = audio.rate; out.samples = audio.samples;
    if (! audio.valid()) return out;
    out.hop = std::max (1, (int) std::llround (audio.rate * 0.010));
    const int count = (int) (((int64_t) audio.samples + out.hop - 1) / out.hop);
    out.frames.resize ((size_t) count);

    // Energy at the original sample rate. Spectral analysis uses a box-averaged
    // low-rate analysis copy ONLY; imported/rendered audio is not downsampled.
    const int stride = std::max (1, (int) std::ceil (audio.rate / 16000.0));
    const double spectralRate = audio.rate / stride;
    const int smallCount = (int) (((int64_t) audio.samples + stride - 1) / stride);
    std::vector<std::vector<float>> lowRate ((size_t) audio.channels, std::vector<float> ((size_t) smallCount));
    for (int ch = 0; ch < audio.channels; ++ch)
    {
        for (int i = 0; i < smallCount; ++i)
        {
            if ((i & 4095) == 0 && cancelled (cancel)) { out.frames.clear(); return out; }
            const int begin = i * stride, end = std::min (audio.samples, begin + stride);
            double sum = 0.0;
            for (int s = begin; s < end; ++s)
            {
                const float v = audio.at (ch, s);
                sum += v;
                // Accumulate separate channel energy, including opposite polarity.
            }
            lowRate[(size_t) ch][(size_t) i] = (float) (sum / std::max (1, end - begin));
        }
    }
    SmallFFT fft;
    std::array<int, SmallFFT::size / 2 + 1> bandForBin {};
    for (int bin = 1; bin <= SmallFFT::size / 2; ++bin)
    {
        const double hz = spectralRate * bin / SmallFFT::size;
        bandForBin[(size_t) bin] = std::clamp ((int) std::floor (std::log2 (std::max (40.0, hz) / 40.0) * 2.0), 0, bandCount - 1);
    }
    for (int f = 0; f < count; ++f)
    {
        if (cancelled (cancel)) { out.frames.clear(); return out; }
        auto& frame = out.frames[(size_t) f];
        const int begin = f * out.hop, end = std::min (audio.samples, begin + out.hop);
        double energy = 0.0, peak = 0.0;
        int crossings = 0;
        std::array<double, bandCount> powers {};
        const int centre = (begin + (end - begin) / 2) / stride;
        for (int ch = 0; ch < audio.channels; ++ch)
        {
            float previous = audio.at (ch, begin);
            for (int i = begin; i < end; ++i)
            {
                const float v = audio.at (ch, i);
                energy += (double) v * v;
                peak = std::max (peak, (double) v * v);
                if ((v > 0.0f) != (previous > 0.0f)) ++crossings;
                previous = v;
            }
            std::array<float, SmallFFT::size> input {};
            for (int i = 0; i < SmallFFT::size; ++i)
            {
                const int s = centre + i - SmallFFT::size / 2;
                if (s >= 0 && s < smallCount) input[(size_t) i] = lowRate[(size_t) ch][(size_t) s];
            }
            std::array<float, SmallFFT::size / 2 + 1> spectrum {};
            fft.power (input.data(), spectrum);
            for (int bin = 1; bin <= SmallFFT::size / 2; ++bin)
                powers[(size_t) bandForBin[(size_t) bin]] += spectrum[(size_t) bin];
        }
        const double denominator = (double) std::max (1, end - begin) * audio.channels;
        frame.energyDb = db (energy / denominator); frame.peakDb = db (peak);
        frame.zcr = (float) (crossings / denominator);
        const double total = std::accumulate (powers.begin(), powers.end(), 0.0);
        if (total > 1.0e-18)
            for (int k = 0; k < bandCount; ++k) frame.bands[(size_t) k] = (float) std::sqrt (powers[(size_t) k] / total);
        if (f > 0)
        {
            const auto& previous = out.frames[(size_t) f - 1];
            const float rise = unit ((frame.energyDb - previous.energyDb) / 18.0f);
            frame.onset = unit (0.65f * rise + 0.35f * bandDistance (frame, previous));
            frame.change = bandDistance (frame, out.frames[(size_t) std::max (0, f - 8)]);
        }
    }

    // Local lower-quantile background, constrained during sustained activity.
    // A long scrape must not become background solely because it is long.
    const int context = std::max (8, (int) std::llround (2.0 * audio.rate / out.hop));
    std::vector<float> scratch;
    scratch.reserve ((size_t) context);
    float floor = -120.0f;
    for (int f = 0; f < count; ++f)
    {
        if ((f % 25) == 0)
        {
            if (cancelled (cancel)) { out.frames.clear(); return out; }
            scratch.clear();
            const int left = std::max (0, f - context / 2), right = std::min (count, left + context);
            for (int j = left; j < right; ++j) scratch.push_back (out.frames[(size_t) j].energyDb);
            const auto q = scratch.begin() + (std::ptrdiff_t) (scratch.size() / 5);
            std::nth_element (scratch.begin(), q, scratch.end());
            const float estimate = *q;
            if (f == 0 || estimate < floor) floor = estimate;
            else if (out.frames[(size_t) f].energyDb < floor + 5.0f) floor = std::min (estimate, floor + 0.25f);
            else floor = std::min (estimate, floor + 0.025f);
        }
        out.frames[(size_t) f].noiseDb = floor;
    }
    return out;
}

enum class Reason : int { Quiet = 0, Attack = 1, Texture = 2, Example = 3, LongSpan = 4, Manual = 5, PossibleMiss = 6 };
struct Region
{
    int start = 0, end = 0;
    float event = 0.8f, startQuality = 0.8f, endQuality = 0.8f;
    Reason reason = Reason::Quiet;
};
struct Settings
{
    bool adaptive = true;
    float silenceDb = -50.0f;
    float sensitivity = 0.5f; // fewer -> more cuts
    float gesture = 0.65f;   // individual events -> whole gestures
    float tails = 0.65f;
    double quietMs = 100.0, minimumMs = 25.0, maximumMs = 30000.0;
    double preMs = 5.0, postMs = 15.0;
};
struct Detection
{
    std::vector<Region> regions;
    std::vector<Region> reviewSpans; // visible, auditionable, not exported until accepted
};
inline float entryThreshold (const Frame& f, const Settings& s)
{
    return s.adaptive ? std::max (-100.0f, f.noiseDb + 13.0f - 8.0f * unit (s.sensitivity)) : s.silenceDb;
}
inline float boundaryQuality (const Analysis& a, int sample, bool start)
{
    if (a.frames.empty()) return 0.0f;
    const int f = a.frameAt (sample);
    const int outside = std::clamp (f + (start ? -1 : 1), 0, (int) a.frames.size() - 1);
    const int inside = std::clamp (f + (start ? 1 : -1), 0, (int) a.frames.size() - 1);
    const auto& quiet = a.frames[(size_t) outside];
    const float contrast = a.frames[(size_t) inside].energyDb - quiet.energyDb;
    // A boundary already in the local background is a usable edge even if
    // pre/post-roll puts both adjacent frames in the same quiet gap.
    const float clearance = quiet.noiseDb + 6.0f - quiet.energyDb;
    return unit (0.50f + 0.030f * std::max (0.0f, contrast) + 0.050f * clearance);
}

inline Detection detect (const Analysis& a, const Settings& settings, const Cancel& cancel = {})
{
    Detection out;
    if (a.frames.empty() || a.rate <= 0.0) return out;
    const int count = (int) a.frames.size();
    const auto framesForMs = [&] (double ms) { return std::max (1, (int) std::llround (ms * a.rate / (1000.0 * a.hop))); };
    const int quietFrames = framesForMs (settings.quietMs * (0.55 + 0.9 * unit (settings.tails)));
    const int minFrames = framesForMs (settings.minimumMs);
    const int preSamples = (int) std::llround (settings.preMs * a.rate / 1000.0);
    const int postSamples = (int) std::llround ((settings.postMs + 40.0 * unit (settings.tails)) * a.rate / 1000.0);
    int start = -1, quietStart = -1;
    auto append = [&] (int first, int last)
    {
        if (last - first < minFrames) return;
        Region r;
        r.start = std::max (0, a.sampleAt (first) - preSamples);
        r.end = std::min (a.samples, a.sampleAt (last) + postSamples);
        r.startQuality = boundaryQuality (a, r.start, true);
        r.endQuality = boundaryQuality (a, r.end, false);
        r.event = 0.85f;
        if (settings.maximumMs > 0.0 && (r.end - r.start) / a.rate * 1000.0 > settings.maximumMs)
        { r.event = 0.35f; r.reason = Reason::LongSpan; }
        out.regions.push_back (r);
    };
    for (int f = 0; f < count; ++f)
    {
        if ((f & 255) == 0 && cancelled (cancel)) return {};
        const auto& frame = a.frames[(size_t) f];
        const float threshold = entryThreshold (frame, settings);
        if (start < 0)
        {
            if (frame.energyDb > threshold) { start = f; quietStart = -1; }
        }
        else if (frame.energyDb < threshold - (2.0f + 4.0f * unit (settings.tails)))
        {
            if (quietStart < 0) quietStart = f;
            if (f - quietStart + 1 >= quietFrames)
            { append (start, quietStart); start = -1; quietStart = -1; }
        }
        else quietStart = -1;
    }
    if (start >= 0) append (start, quietStart >= 0 ? quietStart : count);

    // Strong internal attacks and texture transitions can propose splits without
    // requiring literal silence. Gesture intent raises the grouping threshold.
    std::vector<Region> split;
    for (const auto& r : out.regions)
    {
        if (cancelled (cancel)) return {};
        int previous = r.start;
        const int first = a.frameAt (r.start), last = a.frameAt (std::max (r.start, r.end - 1));
        const int minGap = std::max (minFrames, framesForMs (80.0 + 360.0 * unit (settings.gesture)));
        for (int f = first + minGap; f < last - minGap; ++f)
        {
            if ((f & 255) == 0 && cancelled (cancel)) return {};
            const auto& frame = a.frames[(size_t) f];
            const float strength = 0.8f * frame.onset + 0.2f * frame.change;
            const float threshold = 0.28f + 0.30f * unit (settings.gesture) - 0.12f * unit (settings.sensitivity);
            if (strength < threshold || frame.energyDb < entryThreshold (frame, settings)) continue;
            if (strength < 0.8f * a.frames[(size_t) f - 1].onset + 0.2f * a.frames[(size_t) f - 1].change) continue;
            int valley = f;
            for (int j = std::max (first, f - 5); j < f; ++j)
                if (a.frames[(size_t) j].energyDb < a.frames[(size_t) valley].energyDb) valley = j;
            const int cut = a.sampleAt (valley);
            if (cut - previous < minGap * a.hop || r.end - cut < minFrames * a.hop) continue;
            Region part = r; part.start = previous; part.end = cut;
            part.startQuality = boundaryQuality (a, previous, true);
            part.reason = frame.onset >= frame.change ? Reason::Attack : Reason::Texture;
            part.endQuality = boundaryQuality (a, cut, false);
            part.event = std::min (0.8f, 0.45f + strength);
            split.push_back (part); previous = cut; f += minGap - 1;
        }
        Region tail = r; tail.start = previous;
        if (previous != r.start) tail.startQuality = boundaryQuality (a, previous, true);
        split.push_back (tail);
    }
    out.regions = std::move (split);
    for (size_t i = 1; i < out.regions.size(); ++i)
        if (out.regions[i - 1].end > out.regions[i].start)
        {
            const int middle = out.regions[i].start + (out.regions[i - 1].end - out.regions[i].start) / 2;
            out.regions[i - 1].end = middle; out.regions[i].start = middle;
            out.regions[i - 1].endQuality = boundaryQuality (a, middle, false);
            out.regions[i].startQuality = boundaryQuality (a, middle, true);
        }

    // Review weak *uncut* activity too; these are not silently made export clips.
    int run = -1;
    size_t regionIndex = 0;
    for (int f = 0; f <= count; ++f)
    {
        if ((f & 255) == 0 && cancelled (cancel)) return {};
        const int sample = a.sampleAt (f);
        while (regionIndex < out.regions.size() && out.regions[regionIndex].end <= sample) ++regionIndex;
        const bool covered = regionIndex < out.regions.size() && out.regions[regionIndex].start <= sample;
        bool suspicious = false;
        if (f < count && ! covered)
        {
            const auto& frame = a.frames[(size_t) f];
            suspicious = frame.energyDb > -90.0f && (frame.energyDb > frame.noiseDb + 3.0f || frame.onset > 0.2f);
        }
        if (suspicious && run < 0) run = f;
        if (! suspicious && run >= 0)
        {
            if (f - run >= minFrames)
                out.reviewSpans.push_back ({ a.sampleAt (run), sample, 0.35f, 0.3f, 0.3f, Reason::PossibleMiss });
            run = -1;
        }
    }
    if (out.regions.empty() && out.reviewSpans.empty())
    {
        const auto loudest = std::max_element (a.frames.begin(), a.frames.end(), [] (const Frame& x, const Frame& y) { return x.energyDb < y.energyDb; });
        if (loudest->energyDb > settings.silenceDb)
            out.reviewSpans.push_back ({ 0, a.samples, 0.2f, 0.3f, 0.3f, Reason::LongSpan });
    }
    return out;
}

struct TemplateFrame
{
    std::array<float, bandCount> bands {};
    float energy = 0.0f, onset = 0.0f, zcr = 0.0f;
};
struct Example
{
    std::string name, sourceId;
    int sourceStart = 0, sourceEnd = 0;
    double sourceRate = 0.0, duration = 0.0;
    bool negative = false;
    float leadFraction = 0.0f, tailFraction = 0.0f;
    std::vector<TemplateFrame> frames;
};
struct Profile
{
    int version = featureVersion;
    std::string name;
    std::vector<Example> examples;
    int positives() const { return (int) std::count_if (examples.begin(), examples.end(), [] (const Example& e) { return ! e.negative; }); }
};

inline Example makeExample (const Analysis& a, int start, int end, bool negative = false)
{
    Example out;
    if (a.frames.empty()) return out;
    start = std::clamp (start, 0, a.samples); end = std::clamp (end, start, a.samples);
    if (end <= start || a.rate <= 0.0) return out;
    out.sourceStart = start; out.sourceEnd = end; out.sourceRate = a.rate;
    out.duration = (end - start) / a.rate; out.negative = negative;
    const int first = a.frameAt (start), last = a.frameAt (end - 1);
    const int n = std::clamp (last - first + 1, 4, maxExampleFrames);
    float peak = -120.0f;
    for (int f = first; f <= last; ++f) peak = std::max (peak, a.frames[(size_t) f].energyDb);
    if (peak <= -119.0f) return out; // Digital silence is not a useful event prototype.
    out.frames.resize ((size_t) n);
    int activeFirst = first, activeLast = last;
    while (activeFirst < last && a.frames[(size_t) activeFirst].energyDb < peak - 30.0f) ++activeFirst;
    while (activeLast > first && a.frames[(size_t) activeLast].energyDb < peak - 30.0f) --activeLast;
    out.leadFraction = unit ((a.sampleAt (activeFirst) - start) / (float) std::max (1, end - start));
    out.tailFraction = unit ((end - a.sampleAt (activeLast + 1)) / (float) std::max (1, end - start));
    for (int i = 0; i < n; ++i)
    {
        // Equal-duration bins preserve internal event order; amplitude is relative
        // to the selected event, not the microphone's absolute gain.
        const int lo = first + (int) ((int64_t) i * (last - first + 1) / n);
        const int hi = std::max (lo + 1, first + (int) ((int64_t) (i + 1) * (last - first + 1) / n));
        auto& t = out.frames[(size_t) i];
        for (int f = lo; f < std::min (hi, last + 1); ++f)
        {
            const auto& v = a.frames[(size_t) f];
            for (int k = 0; k < bandCount; ++k) t.bands[(size_t) k] += v.bands[(size_t) k];
            t.energy += std::clamp ((v.energyDb - peak) / 45.0f, -1.0f, 0.0f);
            t.onset = std::max (t.onset, v.onset); t.zcr += v.zcr;
        }
        const float denom = (float) std::max (1, std::min (hi, last + 1) - lo);
        t.energy /= denom; t.zcr /= denom;
        float norm = 0.0f;
        for (float v : t.bands) norm += v * v;
        if (norm > 1.0e-12f) for (auto& v : t.bands) v /= std::sqrt (norm);
    }
    return out;
}
inline float frameCost (const TemplateFrame& a, const TemplateFrame& b)
{
    float dot = 0.0f;
    for (int k = 0; k < bandCount; ++k) dot += a.bands[(size_t) k] * b.bands[(size_t) k];
    // Silence shape participates in alignment, but its undefined spectrum does not.
    const float audible = unit (1.0f + std::min (a.energy, b.energy));
    return 0.52f * audible * unit (1.0f - dot) + 0.32f * std::abs (a.energy - b.energy)
         + 0.12f * std::abs (a.onset - b.onset) + 0.04f * std::abs (a.zcr - b.zcr);
}
inline float coarseCost (const Example& a, const Example& b)
{
    if (a.frames.empty() || b.frames.empty()) return 1.0f;
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i)
    {
        const size_t ai = std::min (a.frames.size() - 1, (size_t) ((i + 0.5) * a.frames.size() / 8.0));
        const size_t bi = std::min (b.frames.size() - 1, (size_t) ((i + 0.5) * b.frames.size() / 8.0));
        sum += frameCost (a.frames[ai], b.frames[bi]);
    }
    return sum / 8.0f;
}
inline float dtwCost (const Example& a, const Example& b)
{
    if (a.frames.empty() || b.frames.empty() || a.frames.size() > maxExampleFrames || b.frames.size() > maxExampleFrames) return 1.0f;
    const int n = (int) a.frames.size(), m = (int) b.frames.size();
    constexpr float inf = 1.0e9f;
    std::array<float, maxExampleFrames + 1> previous {}, row {};
    previous.fill (inf); previous[0] = 0.0f;
    const int band = std::max (3, std::max (n, m) / 4);
    for (int i = 1; i <= n; ++i)
    {
        row.fill (inf);
        const int centre = (int) std::llround ((double) i * m / n);
        for (int j = std::max (1, centre - band); j <= std::min (m, centre + band); ++j)
        {
            const float cost = frameCost (a.frames[(size_t) i - 1], b.frames[(size_t) j - 1]);
            // Penalty discourages stretching one frame across a whole gesture.
            row[(size_t) j] = std::min ({ previous[(size_t) j - 1] + 2.0f * cost,
                                         previous[(size_t) j] + cost + 0.025f,
                                         row[(size_t) j - 1] + cost + 0.025f });
        }
        previous = row;
    }
    return previous[(size_t) m] / (float) (n + m);
}
inline float similarity (const Example& a, const Example& b)
{
    if (a.duration <= 0.0 || b.duration <= 0.0) return 0.0f;
    const double ratio = b.duration / a.duration;
    if (ratio < 0.45 || ratio > 2.2) return 0.0f;
    // An averaged alignment can hide a missing attack in a short part of a
    // long template. Preserve the strongest onset as a separate constraint;
    // a decay alone must not pass as a complete struck event.
    float attackA = 0.0f, attackB = 0.0f;
    for (const auto& f : a.frames) attackA = std::max (attackA, f.onset);
    for (const auto& f : b.frames) attackB = std::max (attackB, f.onset);
    const float missingAttack = 0.42f * std::abs (attackA - attackB);
    const float occupancy = 0.25f * (std::abs (a.leadFraction - b.leadFraction)
                                    + std::abs (a.tailFraction - b.tailFraction));
    return unit (1.0f - 2.3f * dtwCost (a, b) - missingAttack - occupancy
                 - 0.06f * (float) std::abs (std::log2 (ratio)));
}
inline float contrastiveScore (float positive, float negative)
{
    // Equal positive/negative evidence is ambiguous, not an automatic match.
    // Apply the strongest negative once; duplicate negatives must not stack.
    return negative > positive - 0.04f
        ? unit (positive - 0.45f * unit ((negative - positive + 0.10f) / 0.10f))
        : unit (positive);
}
inline float overlap (const Region& a, const Region& b)
{
    const int shared = std::max (0, std::min (a.end, b.end) - std::max (a.start, b.start));
    return shared / (float) std::max (1, std::min (a.end - a.start, b.end - b.start));
}

// Each positive prototype searches the whole source (including uncut spans).
// A cheap temporal prefilter precedes bounded DTW. Negative examples affect
// identity only; start/end quality remain independent measurements.
inline std::vector<Region> findMatches (const Analysis& a, const Profile& profile, float threshold = 0.68f,
                                        const Cancel& cancel = {})
{
    std::vector<Region> matches;
    if (profile.version != featureVersion || a.frames.empty() || profile.positives() == 0) return matches;
    for (const auto& example : profile.examples)
    {
        if (example.negative || example.frames.empty() || example.duration <= 0.0) continue;
        const int nominal = std::max (2, (int) std::llround (example.duration * a.rate / a.hop));
        const int step = std::max (1, std::min (10, nominal / 8));
        struct Candidate { Region region; float score = 0.0f; };
        std::vector<Candidate> candidates;
        // Best candidate per neighbourhood limits expensive comparisons without
        // throwing away all but a fixed number of matches in a long recording.
        for (int first = 0; first < (int) a.frames.size(); first += step)
        {
            if (cancelled (cancel)) return {};
            Candidate best; best.score = -1.0f;
            for (double scale : { 0.60, 0.78, 1.0, 1.28, 1.65 })
            {
                const int length = std::max (2, (int) std::llround (nominal * scale));
                if (first + length > (int) a.frames.size()) continue;
                const int end = a.sampleAt (first + length);
                auto candidate = makeExample (a, a.sampleAt (first), end);
                const float coarse = coarseCost (example, candidate);
                if (coarse > 0.27f) continue;
                // A profile must not propose stretches of silence.
                float highest = -120.0f;
                for (int f = first; f < first + length; f += std::max (1, length / 8)) highest = std::max (highest, a.frames[(size_t) f].energyDb);
                if (highest < -95.0f) continue;
                const float score = 1.0f - coarse;
                if (score > best.score) best = { { a.sampleAt (first), end, 0.0f, 0.0f, 0.0f, Reason::Example }, score };
            }
            if (best.score >= 0.0f) candidates.push_back (best);
        }
        std::stable_sort (candidates.begin(), candidates.end(), [] (const Candidate& x, const Candidate& y) { return x.score > y.score; });
        std::vector<Region> acceptedForExample;
        for (const auto& candidate : candidates)
        {
            if (cancelled (cancel)) return {};
            bool alreadyCovered = false;
            for (const auto& r : acceptedForExample)
                if (overlap (r, candidate.region) > 0.65f) { alreadyCovered = true; break; }
            if (alreadyCovered) continue;
            Region best = candidate.region; best.event = 0.0f;
            // Refine both temporal endpoints around the coarse window. DTW finds
            // the identity; local energy/contrast separately assess the edges.
            const int radius = std::max (1, step);
            for (int offset : { -radius, 0, radius })
                for (double scale : { 0.90, 1.0, 1.10 })
                {
                    const int begin = std::clamp (candidate.region.start + offset * a.hop, 0, a.samples - 1);
                    const int end = std::clamp (begin + (int) std::llround ((candidate.region.end - candidate.region.start) * scale), begin + 1, a.samples);
                    const auto proposed = makeExample (a, begin, end);
                    const float positive = similarity (example, proposed);
                    float negativeScore = 0.0f;
                    for (const auto& negative : profile.examples)
                        if (negative.negative && ! negative.frames.empty())
                            negativeScore = std::max (negativeScore, similarity (negative, proposed));
                    const float score = contrastiveScore (positive, negativeScore);
                    if (score > best.event) { best.start = begin; best.end = end; best.event = score; }
                }
            if (best.event < threshold) continue;
            best.startQuality = boundaryQuality (a, best.start, true);
            best.endQuality = boundaryQuality (a, best.end, false);
            acceptedForExample.push_back (best);
        }
        matches.insert (matches.end(), acceptedForExample.begin(), acceptedForExample.end());
    }
    std::stable_sort (matches.begin(), matches.end(), [] (const Region& a, const Region& b) { return a.event > b.event; });
    std::vector<Region> unique;
    for (const auto& match : matches)
    {
        if (cancelled (cancel)) return {};
        bool duplicate = false;
        for (const auto& kept : unique) if (overlap (match, kept) > 0.45f) { duplicate = true; break; }
        if (! duplicate) unique.push_back (match);
    }
    std::sort (unique.begin(), unique.end(), [] (const Region& a, const Region& b) { return a.start < b.start; });
    return unique;
}

// Apply authoritative source intervals without losing unrelated audio on
// either side of a broad proposal. Clipped fragments are explicitly uncertain.
// The region type is deliberately generic so this state rule is unit-testable
// without JUCE; both the editor and final renderer use this implementation.
template <typename RegionType>
inline std::vector<RegionType> preserveConstraints (std::vector<RegionType> automatic,
                                                    const std::vector<RegionType>& existing)
{
    std::vector<RegionType> result;
    for (const auto& candidate : automatic)
    {
        std::vector<RegionType> fragments { candidate };
        for (const auto& constraint : existing)
        {
            if (! constraint.locked || constraint.endSample <= constraint.startSample) continue;
            std::vector<RegionType> next;
            for (const auto& piece : fragments)
            {
                if (piece.endSample <= constraint.startSample || piece.startSample >= constraint.endSample)
                    next.push_back (piece);
                else
                {
                    if (piece.startSample < constraint.startSample)
                    {
                        auto left = piece; left.endSample = constraint.startSample;
                        left.endBoundaryScore = std::min (left.endBoundaryScore, 0.49f);
                        next.push_back (left);
                    }
                    if (piece.endSample > constraint.endSample)
                    {
                        auto right = piece; right.startSample = constraint.endSample;
                        right.startBoundaryScore = std::min (right.startBoundaryScore, 0.49f);
                        next.push_back (right);
                    }
                }
            }
            fragments = std::move (next);
            if (fragments.empty()) break;
        }
        result.insert (result.end(), fragments.begin(), fragments.end());
    }
    for (const auto& constraint : existing) if (constraint.locked) result.push_back (constraint);
    std::stable_sort (result.begin(), result.end(), [] (const RegionType& a, const RegionType& b) { return a.startSample < b.startSample; });
    return result;
}

// Multi-channel low-discontinuity snap. Never sums anti-phase waveforms. Search
// is bounded to the requested neighbourhood; Alt in the editor bypasses this.
inline int snapBoundary (AudioView audio, int wanted, int radius)
{
    if (! audio.valid()) return wanted;
    wanted = std::clamp (wanted, 0, audio.samples);
    if (wanted == 0 || wanted == audio.samples || radius <= 0) return wanted;
    const int first = std::max (1, wanted - radius), last = std::min (audio.samples - 1, wanted + radius);
    int best = wanted; double bestCost = std::numeric_limits<double>::max();
    for (int i = first; i <= last; ++i)
    {
        double cost = 0.0;
        for (int ch = 0; ch < audio.channels; ++ch)
        {
            const double x = audio.at (ch, i), previous = audio.at (ch, i - 1);
            cost += x * x + previous * previous + 0.5 * (x - previous) * (x - previous);
        }
        cost *= 1.0 + 0.5 * std::abs (i - wanted) / std::max (1, radius);
        if (cost < bestCost) { bestCost = cost; best = i; }
    }
    return best;
}

inline const char* qualityName (float value) { return value >= 0.75f ? "Strong" : (value >= 0.50f ? "Review" : "Weak"); }
inline const char* reasonName (Reason reason)
{
    switch (reason)
    {
        case Reason::Attack: return "new attack";
        case Reason::Texture: return "texture change";
        case Reason::Example: return "example match";
        case Reason::LongSpan: return "long uncut span";
        case Reason::Manual: return "user edit";
        case Reason::PossibleMiss: return "possible missed event";
        default: return "quiet gap";
    }
}
} // namespace za::fileimport::extractor
