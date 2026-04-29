#pragma once

#if defined(__has_include)
 #if __has_include(<JuceHeader.h>)
  #include <JuceHeader.h>
 #else
  #include <juce_core/juce_core.h>
  #include <juce_audio_formats/juce_audio_formats.h>
  #include <juce_gui_basics/juce_gui_basics.h>
 #endif
#else
 #include <juce_core/juce_core.h>
 #include <juce_audio_formats/juce_audio_formats.h>
 #include <juce_gui_basics/juce_gui_basics.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <thread>
#include <vector>

namespace za::fileimport
{

enum class IngressSource
{
    FileDialog,
    DragDrop,
    ClipboardTextUri,
    Recent,
    Favorite,
    Recipe
};

enum class ImportAction
{
    LoadSeparate = 1,
    AppendRawAsSingle = 2,
    BuildMegaTexture = 3,
    SegmentLongFile = 4,
    ModifyExisting = 5,
    SegmentThenMegaTexture = 6
};

enum class RenderedLoadMode
{
    SeparateEntries,
    AppendAsSingleFile
};

struct SourceFingerprint
{
    juce::String path;
    int64_t sizeBytes = 0;
    int64_t modifiedUtcMs = 0;
    uint64_t quickHash = 0;
};

struct SegmentRegion
{
    int startSample = 0;
    int endSample = 0;
    double rmsDb = -120.0;
    double peakDb = -120.0;
    double spectralFlux = 0.0;
    double novelty = 0.0;
    bool enabled = true;

    int length() const noexcept { return juce::jmax (0, endSample - startSample); }
};

struct AudioFeatureVector
{
    double rmsDb = -120.0;
    double peakDb = -120.0;
    double spectralFlux = 0.0;
    double novelty = 0.0;
    double zcr = 0.0;
    std::array<double, 16> bands {};
};

struct ImportRules
{
    int version = 1;

    bool trimEdges = true;
    bool stripInternalSilence = false;
    bool segmentBySilence = false;

    // Absolute silence gate used for segmentation and pruning. A direct dBFS
    // threshold is easier to reason about than the old RMS-ratio-only gate.
    double silenceThresholdDb = -50.0;
    float silenceThresholdRatio = 0.10f;
    bool useRelativeRmsThreshold = false;
    double silenceAnalysisWindowMs = 5.0;
    double minSilenceMs = 100.0;
    double preRollMs = 5.0;
    double postRollMs = 15.0;
    double minSegmentMs = 25.0;
    double maxSegmentMs = 30000.0;
    double edgeFadeMs = 5.0;

    bool removeLowRms = false;
    double minRmsDb = -65.0;

    bool rejectNearDuplicates = false;
    double duplicateSimilarityThreshold = 0.92;

    bool preferNovelSamples = false;
    double minSpectralFlux = 0.0;

    bool randomize = false;
    uint32_t randomSeed = 0;

    double gapMs = 0.0;
    double crossfadeMs = 5.0;

    bool normalizeClipsRms = false;
    double clipTargetRmsDb = -24.0;

    bool normalizeFinalRms = false;
    double finalTargetRmsDb = -24.0;

    int outputChannels = 2;
    double outputSampleRate = 0.0; // 0 == first source rate

    double previewSeconds = 30.0;
};

struct ImportRecipe
{
    int version = 1;
    ImportAction action = ImportAction::LoadSeparate;
    std::vector<SourceFingerprint> inputs;
    ImportRules rules;
    uint32_t seed = 0;
    juce::String displayName;
};

struct AudioFileData
{
    juce::AudioBuffer<float> buffer;
    double sampleRate = 0.0;
    juce::String sourceName;
};

struct RenderResult
{
    bool ok = false;
    juce::String message;
    std::vector<juce::File> files;
    std::vector<AudioFileData> renderedAudio;
    RenderedLoadMode loadMode = RenderedLoadMode::SeparateEntries;
    ImportRecipe recipe;
};

static inline bool isSupportedAudioExtension (const juce::String& pathOrName)
{
    const auto ext = juce::File (pathOrName).getFileExtension().toLowerCase();
    return ext == ".wav" || ext == ".wave" || ext == ".aif" || ext == ".aiff" ||
           ext == ".flac" || ext == ".ogg" || ext == ".mp3" || ext == ".m4a" ||
           ext == ".caf" || ext == ".w64";
}

static inline std::vector<juce::File> filterSupportedExistingFiles (const std::vector<juce::File>& files)
{
    std::vector<juce::File> out;
    std::set<juce::String> seen;

    for (const auto& f : files)
    {
        if (! f.existsAsFile())
            continue;

        if (! isSupportedAudioExtension (f.getFullPathName()))
            continue;

        const auto key = f.getFullPathName().toLowerCase();
        if (seen.insert (key).second)
            out.push_back (f);
    }

    return out;
}

static inline bool containsSupportedFileExtension (const juce::StringArray& names)
{
    for (const auto& name : names)
        if (isSupportedAudioExtension (name))
            return true;
    return false;
}

static inline juce::String uriDecode (juce::String s)
{
    juce::String out;
    for (int i = 0; i < s.length(); ++i)
    {
        const auto c = s[i];
        if (c == '%' && i + 2 < s.length())
        {
            const auto hex = s.substring (i + 1, i + 3);
            const int value = hex.getHexValue32();
            out << juce::String::charToString (static_cast<juce::juce_wchar> (value));
            i += 2;
        }
        else if (c == '+')
        {
            out << ' ';
        }
        else
        {
            out << c;
        }
    }
    return out;
}

static inline juce::String normaliseFileUriToPath (juce::String text)
{
    text = text.trim().unquoted();
    if (text.startsWithIgnoreCase ("file://"))
    {
        juce::String path = text.fromFirstOccurrenceOf ("file://", false, true);

        // file:///C:/x.wav -> /C:/x.wav.  Windows wants C:/x.wav.
        if (path.startsWithChar ('/') && path.length() > 3 && ((path[1] >= 'A' && path[1] <= 'Z') || (path[1] >= 'a' && path[1] <= 'z')) && path[2] == ':')
            path = path.substring (1);

        return uriDecode (path).replaceCharacter ('/', juce::File::getSeparatorChar());
    }

    return text;
}

static inline void addPathTokenIfFile (std::vector<juce::File>& out, juce::String token)
{
    token = token.trim().unquoted();
    token = token.trimCharactersAtStart ("{").trimCharactersAtEnd ("}");
    if (token.isEmpty() || token.startsWithChar ('#'))
        return;

    juce::File f (normaliseFileUriToPath (token));
    if (f.existsAsFile())
        out.push_back (f);
}

static inline std::vector<juce::File> parseFilesFromClipboardText (juce::String text)
{
    std::vector<juce::File> out;

    text = text.trim();
    if (text.isEmpty())
        return out;

    // text/uri-list and newline-separated lists.
    auto lines = juce::StringArray::fromLines (text);
    bool consumedLineList = false;
    for (auto line : lines)
    {
        line = line.trim();
        if (line.isEmpty() || line.startsWithChar ('#'))
            continue;

        if (line.startsWithIgnoreCase ("file://") || juce::File (normaliseFileUriToPath (line)).existsAsFile())
        {
            addPathTokenIfFile (out, line);
            consumedLineList = true;
        }
    }

    if (consumedLineList)
        return filterSupportedExistingFiles (out);

    // Quoted or semicolon-delimited path lists, including Soundly/Windows style snippets.
    juce::StringArray tokens;
    tokens.addTokens (text, ";\n\r\t", "\"'");
    for (auto token : tokens)
        addPathTokenIfFile (out, token);

    // Fallback: one raw path.
    if (out.empty())
        addPathTokenIfFile (out, text);

    return filterSupportedExistingFiles (out);
}

static inline uint64_t fnv1a64 (const void* data, size_t bytes, uint64_t h = 1469598103934665603ull)
{
    const auto* p = static_cast<const uint8_t*> (data);
    for (size_t i = 0; i < bytes; ++i)
    {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static inline uint64_t quickHashFile (const juce::File& file)
{
    std::unique_ptr<juce::FileInputStream> in (file.createInputStream());
    if (in == nullptr || ! in->openedOk())
        return 0;

    constexpr int kChunk = 4096;
    std::array<char, kChunk> block {};
    uint64_t h = 1469598103934665603ull;

    const auto size = file.getSize();
    const int n1 = in->read (block.data(), kChunk);
    if (n1 > 0)
        h = fnv1a64 (block.data(), (size_t) n1, h);

    if (size > kChunk)
    {
        in->setPosition (juce::jmax<int64_t> (0, size - kChunk));
        const int n2 = in->read (block.data(), kChunk);
        if (n2 > 0)
            h = fnv1a64 (block.data(), (size_t) n2, h);
    }

    return h;
}

static inline SourceFingerprint fingerprintForFile (const juce::File& file)
{
    SourceFingerprint fp;
    fp.path = file.getFullPathName();
    fp.sizeBytes = file.getSize();
    fp.modifiedUtcMs = file.getLastModificationTime().toMilliseconds();
    fp.quickHash = quickHashFile (file);
    return fp;
}

static inline juce::ValueTree rulesToValueTree (const ImportRules& r)
{
    juce::ValueTree t ("RULES");
    t.setProperty ("version", r.version, nullptr);
    t.setProperty ("trimEdges", r.trimEdges, nullptr);
    t.setProperty ("stripInternalSilence", r.stripInternalSilence, nullptr);
    t.setProperty ("segmentBySilence", r.segmentBySilence, nullptr);
    t.setProperty ("silenceThresholdDb", r.silenceThresholdDb, nullptr);
    t.setProperty ("silenceThresholdRatio", r.silenceThresholdRatio, nullptr);
    t.setProperty ("useRelativeRmsThreshold", r.useRelativeRmsThreshold, nullptr);
    t.setProperty ("silenceAnalysisWindowMs", r.silenceAnalysisWindowMs, nullptr);
    t.setProperty ("minSilenceMs", r.minSilenceMs, nullptr);
    t.setProperty ("preRollMs", r.preRollMs, nullptr);
    t.setProperty ("postRollMs", r.postRollMs, nullptr);
    t.setProperty ("minSegmentMs", r.minSegmentMs, nullptr);
    t.setProperty ("maxSegmentMs", r.maxSegmentMs, nullptr);
    t.setProperty ("edgeFadeMs", r.edgeFadeMs, nullptr);
    t.setProperty ("removeLowRms", r.removeLowRms, nullptr);
    t.setProperty ("minRmsDb", r.minRmsDb, nullptr);
    t.setProperty ("rejectNearDuplicates", r.rejectNearDuplicates, nullptr);
    t.setProperty ("duplicateSimilarityThreshold", r.duplicateSimilarityThreshold, nullptr);
    t.setProperty ("preferNovelSamples", r.preferNovelSamples, nullptr);
    t.setProperty ("minSpectralFlux", r.minSpectralFlux, nullptr);
    t.setProperty ("randomize", r.randomize, nullptr);
    t.setProperty ("randomSeed", (int64_t) r.randomSeed, nullptr);
    t.setProperty ("gapMs", r.gapMs, nullptr);
    t.setProperty ("crossfadeMs", r.crossfadeMs, nullptr);
    t.setProperty ("normalizeClipsRms", r.normalizeClipsRms, nullptr);
    t.setProperty ("clipTargetRmsDb", r.clipTargetRmsDb, nullptr);
    t.setProperty ("normalizeFinalRms", r.normalizeFinalRms, nullptr);
    t.setProperty ("finalTargetRmsDb", r.finalTargetRmsDb, nullptr);
    t.setProperty ("outputChannels", r.outputChannels, nullptr);
    t.setProperty ("outputSampleRate", r.outputSampleRate, nullptr);
    return t;
}

static inline ImportRules rulesFromValueTree (const juce::ValueTree& t)
{
    ImportRules r;
    if (! t.isValid())
        return r;

    r.version = (int) t.getProperty ("version", r.version);
    r.trimEdges = (bool) t.getProperty ("trimEdges", r.trimEdges);
    r.stripInternalSilence = (bool) t.getProperty ("stripInternalSilence", r.stripInternalSilence);
    r.segmentBySilence = (bool) t.getProperty ("segmentBySilence", r.segmentBySilence);
    r.silenceThresholdDb = (double) t.getProperty ("silenceThresholdDb", r.silenceThresholdDb);
    r.silenceThresholdRatio = (float) (double) t.getProperty ("silenceThresholdRatio", r.silenceThresholdRatio);
    r.useRelativeRmsThreshold = (bool) t.getProperty ("useRelativeRmsThreshold", r.useRelativeRmsThreshold);
    r.silenceAnalysisWindowMs = (double) t.getProperty ("silenceAnalysisWindowMs", r.silenceAnalysisWindowMs);
    r.minSilenceMs = (double) t.getProperty ("minSilenceMs", r.minSilenceMs);
    r.preRollMs = (double) t.getProperty ("preRollMs", r.preRollMs);
    r.postRollMs = (double) t.getProperty ("postRollMs", r.postRollMs);
    r.minSegmentMs = (double) t.getProperty ("minSegmentMs", r.minSegmentMs);
    r.maxSegmentMs = (double) t.getProperty ("maxSegmentMs", r.maxSegmentMs);
    r.edgeFadeMs = (double) t.getProperty ("edgeFadeMs", r.edgeFadeMs);
    r.removeLowRms = (bool) t.getProperty ("removeLowRms", r.removeLowRms);
    r.minRmsDb = (double) t.getProperty ("minRmsDb", r.minRmsDb);
    r.rejectNearDuplicates = (bool) t.getProperty ("rejectNearDuplicates", r.rejectNearDuplicates);
    r.duplicateSimilarityThreshold = (double) t.getProperty ("duplicateSimilarityThreshold", r.duplicateSimilarityThreshold);
    r.preferNovelSamples = (bool) t.getProperty ("preferNovelSamples", r.preferNovelSamples);
    r.minSpectralFlux = (double) t.getProperty ("minSpectralFlux", r.minSpectralFlux);
    r.randomize = (bool) t.getProperty ("randomize", r.randomize);
    r.randomSeed = (uint32_t) (int64_t) t.getProperty ("randomSeed", (int64_t) r.randomSeed);
    r.gapMs = (double) t.getProperty ("gapMs", r.gapMs);
    r.crossfadeMs = (double) t.getProperty ("crossfadeMs", r.crossfadeMs);
    r.normalizeClipsRms = (bool) t.getProperty ("normalizeClipsRms", r.normalizeClipsRms);
    r.clipTargetRmsDb = (double) t.getProperty ("clipTargetRmsDb", r.clipTargetRmsDb);
    r.normalizeFinalRms = (bool) t.getProperty ("normalizeFinalRms", r.normalizeFinalRms);
    r.finalTargetRmsDb = (double) t.getProperty ("finalTargetRmsDb", r.finalTargetRmsDb);
    r.outputChannels = (int) t.getProperty ("outputChannels", r.outputChannels);
    r.outputSampleRate = (double) t.getProperty ("outputSampleRate", r.outputSampleRate);
    return r;
}

static inline juce::ValueTree recipeToValueTree (const ImportRecipe& recipe)
{
    juce::ValueTree t ("ZA_IMPORT_RECIPE");
    t.setProperty ("version", recipe.version, nullptr);
    t.setProperty ("action", (int) recipe.action, nullptr);
    t.setProperty ("seed", (int64_t) recipe.seed, nullptr);
    t.setProperty ("displayName", recipe.displayName, nullptr);
    t.addChild (rulesToValueTree (recipe.rules), -1, nullptr);

    juce::ValueTree inputs ("INPUTS");
    for (const auto& fp : recipe.inputs)
    {
        juce::ValueTree in ("INPUT");
        in.setProperty ("path", fp.path, nullptr);
        in.setProperty ("sizeBytes", fp.sizeBytes, nullptr);
        in.setProperty ("modifiedUtcMs", fp.modifiedUtcMs, nullptr);
        in.setProperty ("quickHash", (int64_t) fp.quickHash, nullptr);
        inputs.addChild (in, -1, nullptr);
    }
    t.addChild (inputs, -1, nullptr);
    return t;
}

static inline ImportRecipe recipeFromValueTree (const juce::ValueTree& t)
{
    ImportRecipe recipe;
    if (! t.isValid())
        return recipe;

    recipe.version = (int) t.getProperty ("version", recipe.version);
    recipe.action = (ImportAction) (int) t.getProperty ("action", (int) recipe.action);
    recipe.seed = (uint32_t) (int64_t) t.getProperty ("seed", (int64_t) recipe.seed);
    recipe.displayName = t.getProperty ("displayName", recipe.displayName).toString();
    recipe.rules = rulesFromValueTree (t.getChildWithName ("RULES"));

    if (auto inputs = t.getChildWithName ("INPUTS"); inputs.isValid())
    {
        for (int i = 0; i < inputs.getNumChildren(); ++i)
        {
            auto in = inputs.getChild (i);
            SourceFingerprint fp;
            fp.path = in.getProperty ("path", {}).toString();
            fp.sizeBytes = (int64_t) in.getProperty ("sizeBytes", (int64_t) 0);
            fp.modifiedUtcMs = (int64_t) in.getProperty ("modifiedUtcMs", (int64_t) 0);
            fp.quickHash = (uint64_t) (int64_t) in.getProperty ("quickHash", (int64_t) 0);
            recipe.inputs.push_back (std::move (fp));
        }
    }

    return recipe;
}

static inline double linearToDb (double x) noexcept
{
    return x <= 1.0e-12 ? -120.0 : 20.0 * std::log10 (x);
}

static inline double dbToLinear (double db) noexcept
{
    return std::pow (10.0, db / 20.0);
}

static inline double computeRmsLinear (const juce::AudioBuffer<float>& b, int start = 0, int num = -1)
{
    const int n = b.getNumSamples();
    const int chs = b.getNumChannels();
    if (n <= 0 || chs <= 0)
        return 0.0;

    start = juce::jlimit (0, n, start);
    if (num < 0)
        num = n - start;
    num = juce::jlimit (0, n - start, num);
    if (num <= 0)
        return 0.0;

    long double sum = 0.0;
    for (int ch = 0; ch < chs; ++ch)
    {
        const auto* p = b.getReadPointer (ch, start);
        for (int i = 0; i < num; ++i)
            sum += (long double) p[i] * (long double) p[i];
    }

    return std::sqrt ((double) (sum / (long double) (num * chs)));
}

static inline double computePeakLinear (const juce::AudioBuffer<float>& b, int start = 0, int num = -1)
{
    const int n = b.getNumSamples();
    const int chs = b.getNumChannels();
    if (n <= 0 || chs <= 0)
        return 0.0;

    start = juce::jlimit (0, n, start);
    if (num < 0)
        num = n - start;
    num = juce::jlimit (0, n - start, num);
    if (num <= 0)
        return 0.0;

    double peak = 0.0;
    for (int ch = 0; ch < chs; ++ch)
    {
        const auto* p = b.getReadPointer (ch, start);
        for (int i = 0; i < num; ++i)
            peak = std::max (peak, (double) std::abs (p[i]));
    }
    return peak;
}

static inline float sampleRmsAt (const juce::AudioBuffer<float>& b, int i) noexcept
{
    const int chs = b.getNumChannels();
    if (chs <= 0 || i < 0 || i >= b.getNumSamples())
        return 0.0f;

    float sum = 0.0f;
    for (int ch = 0; ch < chs; ++ch)
    {
        const float x = b.getSample (ch, i);
        sum += x * x;
    }
    return std::sqrt (sum / (float) chs);
}

struct SilenceAnalysis
{
    std::vector<uint8_t> silent;
    std::vector<float> envelope;
    float threshold = 0.0f;
};

static inline std::vector<float> computeRmsEnvelopeLinear (const juce::AudioBuffer<float>& b, double sr, double windowMs)
{
    const int n = b.getNumSamples();
    const int chs = b.getNumChannels();
    std::vector<float> envelope ((size_t) n, 0.0f);
    if (n <= 0 || chs <= 0)
        return envelope;

    std::vector<float> meanSquares ((size_t) n, 0.0f);
    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;
        for (int ch = 0; ch < chs; ++ch)
        {
            const double x = (double) b.getSample (ch, i);
            sum += x * x;
        }
        meanSquares[(size_t) i] = (float) (sum / (double) chs);
    }

    const int window = juce::jmax (1, (int) std::llround (sr * juce::jlimit (0.0, 100.0, windowMs) / 1000.0));
    if (window <= 1)
    {
        for (int i = 0; i < n; ++i)
            envelope[(size_t) i] = std::sqrt (meanSquares[(size_t) i]);
        return envelope;
    }

    const int radius = juce::jmax (0, window / 2);
    double sum = 0.0;
    int lo = 0;
    int hi = 0;

    for (int i = 0; i < n; ++i)
    {
        const int targetLo = juce::jmax (0, i - radius);
        const int targetHi = juce::jmin (n, i + radius + 1);

        while (hi < targetHi)
            sum += (double) meanSquares[(size_t) hi++];
        while (lo < targetLo)
            sum -= (double) meanSquares[(size_t) lo++];

        const int count = juce::jmax (1, hi - lo);
        envelope[(size_t) i] = (float) std::sqrt (juce::jmax (0.0, sum / (double) count));
    }

    return envelope;
}

static inline SilenceAnalysis analyseSilence (const juce::AudioBuffer<float>& b, const ImportRules& rules, double sr)
{
    SilenceAnalysis a;
    const int n = b.getNumSamples();
    a.silent.assign ((size_t) n, 1u);
    a.envelope.assign ((size_t) n, 0.0f);
    if (n <= 0)
        return a;

    const auto globalRms = computeRmsLinear (b);
    const auto globalPeak = computePeakLinear (b);
    if (globalRms <= 1.0e-10 && globalPeak <= 1.0e-10)
        return a;

    a.envelope = computeRmsEnvelopeLinear (b, sr, rules.silenceAnalysisWindowMs);

    double threshold = dbToLinear (juce::jlimit (-120.0, 0.0, rules.silenceThresholdDb));
    if (rules.useRelativeRmsThreshold)
        threshold = juce::jmax (threshold, globalRms * (double) juce::jlimit (0.0f, 4.0f, rules.silenceThresholdRatio));

    a.threshold = (float) juce::jlimit (1.0e-8, 4.0, threshold);

    for (int i = 0; i < n; ++i)
        a.silent[(size_t) i] = a.envelope[(size_t) i] <= a.threshold ? 1u : 0u;

    // Bridge microscopic non-silent spikes inside a quiet run. This makes the
    // detector behave like an RMS-pruning gate rather than a brittle sample-by-
    // sample zero detector.
    const int bridge = juce::jmax (1, (int) std::llround (sr * 2.0 / 1000.0));
    int i = 0;
    while (i < n)
    {
        if (a.silent[(size_t) i] != 0u)
        {
            ++i;
            continue;
        }

        int j = i;
        while (j < n && a.silent[(size_t) j] == 0u)
            ++j;

        const bool surroundedBySilence = i > 0 && j < n && a.silent[(size_t) (i - 1)] != 0u && a.silent[(size_t) j] != 0u;
        if (surroundedBySilence && (j - i) <= bridge)
            for (int k = i; k < j; ++k)
                a.silent[(size_t) k] = 1u;

        i = j;
    }

    return a;
}

static inline std::vector<uint8_t> computeSilenceMask (const juce::AudioBuffer<float>& b, const ImportRules& rules, double sr)
{
    return analyseSilence (b, rules, sr).silent;
}

static inline int findQuietestSampleInRun (const std::vector<float>& envelope, int start, int end)
{
    if (envelope.empty())
        return (start + end) / 2;

    start = juce::jlimit (0, (int) envelope.size(), start);
    end = juce::jlimit (start, (int) envelope.size(), end);
    if (end <= start)
        return start;

    int best = start;
    float bestValue = envelope[(size_t) start];
    for (int i = start + 1; i < end; ++i)
    {
        const float v = envelope[(size_t) i];
        if (v < bestValue)
        {
            bestValue = v;
            best = i;
        }
    }
    return best;
}

static inline std::vector<SegmentRegion> detectSegmentsBySilence (const juce::AudioBuffer<float>& b, double sr, const ImportRules& rules)
{
    std::vector<SegmentRegion> segments;
    const int n = b.getNumSamples();
    if (n <= 0 || sr <= 0.0)
        return segments;

    const auto analysis = analyseSilence (b, rules, sr);
    const auto& silent = analysis.silent;
    const int minSilence = juce::jmax (1, (int) std::llround (sr * rules.minSilenceMs / 1000.0));
    const int pre = juce::jmax (0, (int) std::llround (sr * rules.preRollMs / 1000.0));
    const int post = juce::jmax (0, (int) std::llround (sr * rules.postRollMs / 1000.0));
    const int minLen = juce::jmax (1, (int) std::llround (sr * rules.minSegmentMs / 1000.0));
    const int maxLen = juce::jmax (minLen, (int) std::llround (sr * rules.maxSegmentMs / 1000.0));

    auto addSegment = [&] (int rawStart, int rawEnd)
    {
        int start = juce::jlimit (0, n, rawStart);
        int end = juce::jlimit (start, n, rawEnd);
        if (end - start < minLen)
            return;

        while (end - start > maxLen)
        {
            const int chunkEnd = start + maxLen;
            const double rmsDb = linearToDb (computeRmsLinear (b, start, chunkEnd - start));
            if (! rules.removeLowRms || rmsDb >= rules.minRmsDb)
                segments.push_back ({ start, chunkEnd, rmsDb, linearToDb (computePeakLinear (b, start, chunkEnd - start)), 0.0, 0.0, true });
            start = chunkEnd;
        }

        if (end - start >= minLen)
        {
            const double rmsDb = linearToDb (computeRmsLinear (b, start, end - start));
            if (! rules.removeLowRms || rmsDb >= rules.minRmsDb)
                segments.push_back ({ start, end, rmsDb, linearToDb (computePeakLinear (b, start, end - start)), 0.0, 0.0, true });
        }
    };

    int firstSound = 0;
    while (firstSound < n && silent[(size_t) firstSound] != 0u)
        ++firstSound;

    if (firstSound >= n)
        return segments;

    int segStart = juce::jmax (0, firstSound - pre);
    int i = firstSound;

    while (i < n)
    {
        if (silent[(size_t) i] == 0u)
        {
            ++i;
            continue;
        }

        int j = i;
        while (j < n && silent[(size_t) j] != 0u)
            ++j;

        if (j - i >= minSilence)
        {
            const int cut = findQuietestSampleInRun (analysis.envelope, i, j);

            // Hard boundary rule: post-roll may keep quiet tail, and pre-roll may
            // keep quiet lead-in, but neither side is allowed to cross the chosen
            // cut point. A segmented pseudo-file therefore cannot bleed into the
            // following pseudo-file.
            const int cutCap = juce::jmax (segStart, cut);
            const int segEnd = juce::jlimit (segStart, cutCap, i + post);
            addSegment (segStart, segEnd);

            int nextSound = j;
            while (nextSound < n && silent[(size_t) nextSound] != 0u)
                ++nextSound;

            segStart = juce::jmax (cut, nextSound - pre);
            i = nextSound;
            continue;
        }

        i = j;
    }

    addSegment (segStart, n);

    if (segments.empty() && computeRmsLinear (b) > 0.0)
    {
        const double rmsDb = linearToDb (computeRmsLinear (b));
        if (! rules.removeLowRms || rmsDb >= rules.minRmsDb)
            segments.push_back ({ 0, n, rmsDb, linearToDb (computePeakLinear (b)), 0.0, 0.0, true });
    }

    return segments;
}

static inline void applyEdgeFades (juce::AudioBuffer<float>& b, double sr, double fadeMs)
{
    const int n = b.getNumSamples();
    const int chs = b.getNumChannels();
    const int fade = juce::jlimit (0, n / 2, (int) std::llround (sr * fadeMs / 1000.0));
    if (fade <= 1)
        return;

    for (int ch = 0; ch < chs; ++ch)
    {
        auto* p = b.getWritePointer (ch);
        for (int i = 0; i < fade; ++i)
        {
            const float gIn = (float) i / (float) fade;
            const float gOut = (float) (fade - i) / (float) fade;
            p[i] *= gIn;
            p[n - 1 - i] *= gOut;
        }
    }
}

static inline juce::AudioBuffer<float> copyRange (const juce::AudioBuffer<float>& b, int start, int end)
{
    start = juce::jlimit (0, b.getNumSamples(), start);
    end = juce::jlimit (start, b.getNumSamples(), end);
    juce::AudioBuffer<float> out (b.getNumChannels(), end - start);
    for (int ch = 0; ch < b.getNumChannels(); ++ch)
        out.copyFrom (ch, 0, b, ch, start, end - start);
    return out;
}

static inline juce::AudioBuffer<float> concatenateRanges (const juce::AudioBuffer<float>& b, const std::vector<SegmentRegion>& segments, double sr, const ImportRules& rules)
{
    int total = 0;
    for (const auto& s : segments)
        if (s.enabled)
            total += s.length();

    juce::AudioBuffer<float> out (b.getNumChannels(), total);
    int at = 0;
    for (const auto& s : segments)
    {
        if (! s.enabled || s.length() <= 0)
            continue;

        for (int ch = 0; ch < b.getNumChannels(); ++ch)
            out.copyFrom (ch, at, b, ch, s.startSample, s.length());
        at += s.length();
    }

    applyEdgeFades (out, sr, rules.edgeFadeMs);
    return out;
}

static inline juce::AudioBuffer<float> processBufferByRules (const juce::AudioBuffer<float>& b, double sr, const ImportRules& rules)
{
    if (b.getNumSamples() <= 0)
        return {};

    juce::AudioBuffer<float> out;

    if (rules.stripInternalSilence)
    {
        auto segments = detectSegmentsBySilence (b, sr, rules);
        out = concatenateRanges (b, segments, sr, rules);
    }
    else if (rules.trimEdges)
    {
        auto segments = detectSegmentsBySilence (b, sr, rules);
        if (! segments.empty())
        {
            const int start = segments.front().startSample;
            const int end = segments.back().endSample;
            out = copyRange (b, start, end);
            applyEdgeFades (out, sr, rules.edgeFadeMs);
        }
        else
        {
            out = b;
        }
    }
    else
    {
        out = b;
    }

    if (rules.normalizeClipsRms)
    {
        const auto rms = computeRmsLinear (out);
        if (rms > 1.0e-9)
        {
            const auto g = dbToLinear (rules.clipTargetRmsDb) / rms;
            out.applyGain ((float) g);
        }
    }

    return out;
}

static inline juce::AudioBuffer<float> convertChannels (const juce::AudioBuffer<float>& in, int targetChannels)
{
    targetChannels = juce::jlimit (1, 32, targetChannels);
    if (in.getNumChannels() == targetChannels)
        return in;

    juce::AudioBuffer<float> out (targetChannels, in.getNumSamples());
    out.clear();

    if (in.getNumChannels() <= 0)
        return out;

    if (targetChannels == 1)
    {
        for (int ch = 0; ch < in.getNumChannels(); ++ch)
            out.addFrom (0, 0, in, ch, 0, in.getNumSamples(), 1.0f / (float) in.getNumChannels());
    }
    else if (in.getNumChannels() == 1)
    {
        for (int ch = 0; ch < targetChannels; ++ch)
            out.copyFrom (ch, 0, in, 0, 0, in.getNumSamples());
    }
    else
    {
        for (int ch = 0; ch < targetChannels; ++ch)
            out.copyFrom (ch, 0, in, juce::jmin (ch, in.getNumChannels() - 1), 0, in.getNumSamples());
    }

    return out;
}

static inline juce::AudioBuffer<float> resampleLinear (const juce::AudioBuffer<float>& in, double sourceRate, double targetRate)
{
    if (sourceRate <= 0.0 || targetRate <= 0.0 || std::abs (sourceRate - targetRate) < 1.0e-6)
        return in;

    const int inN = in.getNumSamples();
    const int64_t outN64 = (int64_t) std::llround ((double) inN * targetRate / sourceRate);
    const int outN = (int) juce::jlimit<int64_t> (0, (int64_t) std::numeric_limits<int>::max() / 4, outN64);
    juce::AudioBuffer<float> out (in.getNumChannels(), outN);

    const double step = sourceRate / targetRate;
    for (int ch = 0; ch < in.getNumChannels(); ++ch)
    {
        const auto* src = in.getReadPointer (ch);
        auto* dst = out.getWritePointer (ch);
        for (int i = 0; i < outN; ++i)
        {
            const double pos = (double) i * step;
            const int i0 = juce::jlimit (0, inN - 1, (int) pos);
            const int i1 = juce::jmin (i0 + 1, inN - 1);
            const float frac = (float) (pos - (double) i0);
            dst[i] = src[i0] + (src[i1] - src[i0]) * frac;
        }
    }

    return out;
}

static inline std::optional<AudioFileData> readAudioFile (const juce::File& file, int targetChannels, double targetRate, double maxSeconds, juce::String& error)
{
    juce::AudioFormatManager fm;
    fm.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor (file));
    if (reader == nullptr)
    {
        error = "Could not create audio reader for: " + file.getFullPathName();
        return std::nullopt;
    }

    const int64_t srcLen64 = reader->lengthInSamples;
    if (srcLen64 <= 0)
    {
        error = "Empty audio file: " + file.getFileName();
        return std::nullopt;
    }

    int64_t readLen64 = srcLen64;
    if (maxSeconds > 0.0 && reader->sampleRate > 0.0)
        readLen64 = juce::jmin (readLen64, (int64_t) std::llround (reader->sampleRate * maxSeconds));

    if (readLen64 > (int64_t) std::numeric_limits<int>::max() / 4)
    {
        error = "File is too long for in-memory import preview/render: " + file.getFileName();
        return std::nullopt;
    }

    const int readLen = (int) readLen64;
    const int initialCh = juce::jlimit (1, 8, (int) reader->numChannels);
    juce::AudioBuffer<float> buffer (initialCh, readLen);
    buffer.clear();

    const bool ok = reader->read (&buffer, 0, readLen, 0, true, true);
    if (! ok)
    {
        error = "Read failed: " + file.getFileName();
        return std::nullopt;
    }

    const double sourceRate = reader->sampleRate > 0.0 ? reader->sampleRate : 44100.0;
    const double outRate = targetRate > 0.0 ? targetRate : sourceRate;
    auto converted = resampleLinear (convertChannels (buffer, targetChannels), sourceRate, outRate);

    AudioFileData data;
    data.buffer = std::move (converted);
    data.sampleRate = outRate;
    data.sourceName = file.getFileNameWithoutExtension();
    return data;
}

 static inline juce::String paddedImportIndex (int index)
{
    juce::String text (index);
    while (text.length() < 3)
        text = "0" + text;
    return text;
}

static inline juce::String sanitiseRecipeFileStem (juce::String stem)
{
    stem = stem.trim();
    if (stem.isEmpty())
        stem = "audio";

    const juce::String illegalChars ("\\/:*?\"<>|");
    for (int i = 0; i < illegalChars.length(); ++i)
        stem = stem.replaceCharacter (illegalChars[i], '_');

    while (stem.contains (".."))
        stem = stem.replace ("..", "_");

    return stem.substring (0, 96);
}

static inline double goertzelPower (const float* x, int n, double normalisedFreq)
{
    normalisedFreq = juce::jlimit (0.0001, 0.499, normalisedFreq);
    const double w = 2.0 * juce::MathConstants<double>::pi * normalisedFreq;
    const double coeff = 2.0 * std::cos (w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < n; ++i)
    {
        s0 = (double) x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

static inline AudioFeatureVector analyseAudioFeatures (const juce::AudioBuffer<float>& buffer, double sr)
{
    AudioFeatureVector f;
    f.rmsDb = linearToDb (computeRmsLinear (buffer));
    f.peakDb = linearToDb (computePeakLinear (buffer));

    if (buffer.getNumSamples() <= 0 || buffer.getNumChannels() <= 0)
        return f;

    juce::AudioBuffer<float> mono = convertChannels (buffer, 1);
    const auto* x = mono.getReadPointer (0);
    const int n = mono.getNumSamples();

    int zc = 0;
    for (int i = 1; i < n; ++i)
        if ((x[i - 1] < 0.0f && x[i] >= 0.0f) || (x[i - 1] >= 0.0f && x[i] < 0.0f))
            ++zc;
    f.zcr = n > 1 ? (double) zc / (double) (n - 1) : 0.0;

    constexpr int kBands = 16;
    const int frame = juce::jlimit (256, 4096, n);
    const int hop = juce::jmax (128, frame / 2);
    std::array<double, kBands> prev {};
    bool hasPrev = false;
    int frameCount = 0;
    double fluxSum = 0.0;

    for (int start = 0; start + frame <= n; start += hop)
    {
        std::array<double, kBands> cur {};
        for (int b = 0; b < kBands; ++b)
        {
            const double hz = 60.0 * std::pow (2.0, (double) b * 0.5);
            const double nf = juce::jlimit (0.0001, 0.49, hz / juce::jmax (1.0, sr));
            cur[(size_t) b] = std::sqrt (goertzelPower (x + start, frame, nf) / (double) frame);
            f.bands[(size_t) b] += cur[(size_t) b];
        }

        if (hasPrev)
        {
            double local = 0.0;
            double denom = 1.0e-12;
            for (int b = 0; b < kBands; ++b)
            {
                local += std::max (0.0, cur[(size_t) b] - prev[(size_t) b]);
                denom += cur[(size_t) b] + prev[(size_t) b];
            }
            fluxSum += local / denom;
        }

        prev = cur;
        hasPrev = true;
        ++frameCount;
    }

    if (frameCount > 0)
    {
        for (double& band : f.bands)
            band /= (double) frameCount;
        f.spectralFlux = fluxSum / juce::jmax (1, frameCount - 1);
        f.novelty = f.spectralFlux + 0.1 * f.zcr;
    }

    return f;
}

static inline double cosineSimilarity (const AudioFeatureVector& a, const AudioFeatureVector& b)
{
    std::array<double, 20> va {};
    std::array<double, 20> vb {};

    va[0] = dbToLinear (a.rmsDb);
    vb[0] = dbToLinear (b.rmsDb);
    va[1] = dbToLinear (a.peakDb);
    vb[1] = dbToLinear (b.peakDb);
    va[2] = a.spectralFlux;
    vb[2] = b.spectralFlux;
    va[3] = a.zcr;
    vb[3] = b.zcr;
    for (size_t i = 0; i < a.bands.size(); ++i)
    {
        va[i + 4] = a.bands[i];
        vb[i + 4] = b.bands[i];
    }

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < va.size(); ++i)
    {
        dot += va[i] * vb[i];
        na += va[i] * va[i];
        nb += vb[i] * vb[i];
    }

    if (na <= 1.0e-20 || nb <= 1.0e-20)
        return 0.0;
    return dot / std::sqrt (na * nb);
}

static inline void appendBuffer (juce::AudioBuffer<float>& dest, const juce::AudioBuffer<float>& clip, double sr, const ImportRules& rules)
{
    if (clip.getNumSamples() <= 0)
        return;

    if (dest.getNumSamples() <= 0)
    {
        dest = clip;
        return;
    }

    const int chs = juce::jmin (dest.getNumChannels(), clip.getNumChannels());
    const int gap = juce::jmax (0, (int) std::llround (sr * rules.gapMs / 1000.0));
    const int cross = gap > 0 ? 0 : juce::jmax (0, (int) std::llround (sr * rules.crossfadeMs / 1000.0));
    const int overlap = juce::jlimit (0, juce::jmin (dest.getNumSamples(), clip.getNumSamples()), cross);
    const int oldN = dest.getNumSamples();
    const int newN = oldN + gap + clip.getNumSamples() - overlap;

    juce::AudioBuffer<float> out (dest.getNumChannels(), newN);
    out.clear();
    for (int ch = 0; ch < dest.getNumChannels(); ++ch)
        out.copyFrom (ch, 0, dest, ch, 0, oldN);

    const int clipStartInOut = oldN + gap - overlap;

    for (int ch = 0; ch < chs; ++ch)
    {
        const auto* src = clip.getReadPointer (ch);
        auto* dst = out.getWritePointer (ch);

        for (int i = 0; i < overlap; ++i)
        {
            const float t = (float) (i + 1) / (float) (overlap + 1);
            const int outIndex = oldN - overlap + i;
            dst[outIndex] = dst[outIndex] * (1.0f - t) + src[i] * t;
        }

        for (int i = overlap; i < clip.getNumSamples(); ++i)
            dst[clipStartInOut + i] += src[i];
    }

    dest = std::move (out);
}

struct ProcessedClip
{
    juce::AudioBuffer<float> buffer;
    double sampleRate = 0.0;
    juce::String sourceName;
    AudioFeatureVector features;
};

static inline std::vector<ProcessedClip> preprocessClips (const std::vector<juce::File>& files, const ImportRules& rules, juce::String& error)
{
    std::vector<ProcessedClip> clips;
    if (files.empty())
        return clips;

    double targetRate = rules.outputSampleRate;
    if (targetRate <= 0.0)
    {
        juce::AudioFormatManager fm;
        fm.registerBasicFormats();
        if (auto reader = std::unique_ptr<juce::AudioFormatReader> (fm.createReaderFor (files.front())))
            targetRate = reader->sampleRate;
    }
    if (targetRate <= 0.0)
        targetRate = 48000.0;

    const int targetChannels = juce::jlimit (1, 8, rules.outputChannels <= 0 ? 2 : rules.outputChannels);

    for (const auto& f : files)
    {
        auto data = readAudioFile (f, targetChannels, targetRate, 0.0, error);
        if (! data.has_value())
            continue;

        auto processed = processBufferByRules (data->buffer, data->sampleRate, rules);
        if (processed.getNumSamples() <= 0)
            continue;

        auto features = analyseAudioFeatures (processed, data->sampleRate);
        if (rules.removeLowRms && features.rmsDb < rules.minRmsDb)
            continue;

        if (rules.preferNovelSamples && features.spectralFlux < rules.minSpectralFlux)
            continue;

        bool duplicate = false;
        if (rules.rejectNearDuplicates)
        {
            for (const auto& existing : clips)
            {
                if (cosineSimilarity (features, existing.features) >= rules.duplicateSimilarityThreshold)
                {
                    duplicate = true;
                    break;
                }
            }
        }
        if (duplicate)
            continue;

        clips.push_back ({ std::move (processed), data->sampleRate, data->sourceName, features });
    }

    if (rules.preferNovelSamples)
        std::stable_sort (clips.begin(), clips.end(), [] (const auto& a, const auto& b) { return a.features.novelty > b.features.novelty; });

    if (rules.randomize)
    {
        std::mt19937 rng (rules.randomSeed != 0 ? rules.randomSeed : 0x5eed1234u);
        std::shuffle (clips.begin(), clips.end(), rng);
    }

    return clips;
}

static inline uint32_t deterministicSeedForImport (const std::vector<juce::File>& files, ImportAction action)
{
    uint64_t h = 1469598103934665603ull;
    const auto actionInt = (uint32_t) action;
    h = fnv1a64 (&actionInt, sizeof (actionInt), h);

    for (const auto& file : files)
    {
        const auto fp = fingerprintForFile (file);
        const auto pathUtf8 = fp.path.toRawUTF8();
        h = fnv1a64 (pathUtf8, std::strlen (pathUtf8), h);
        h = fnv1a64 (&fp.sizeBytes, sizeof (fp.sizeBytes), h);
        h = fnv1a64 (&fp.modifiedUtcMs, sizeof (fp.modifiedUtcMs), h);
        h = fnv1a64 (&fp.quickHash, sizeof (fp.quickHash), h);
    }

    const auto folded = (uint32_t) (h ^ (h >> 32));
    return folded != 0 ? folded : 0x5eed1234u;
}

static inline ImportRules makeDefaultRulesForAction (ImportAction action)
{
    ImportRules rules;
    rules.stripInternalSilence = (action == ImportAction::BuildMegaTexture
                                  || action == ImportAction::ModifyExisting
                                  || action == ImportAction::SegmentThenMegaTexture);
    rules.segmentBySilence = (action == ImportAction::SegmentLongFile
                              || action == ImportAction::SegmentThenMegaTexture);
    rules.trimEdges = true;
    rules.rejectNearDuplicates = (action == ImportAction::BuildMegaTexture
                                  || action == ImportAction::SegmentThenMegaTexture);
    rules.preferNovelSamples = (action == ImportAction::BuildMegaTexture);
    rules.randomSeed = 0; // Resolved from source fingerprints at render time for deterministic replay.
    return rules;
}

static inline AudioFileData makeRenderedAudioData (juce::AudioBuffer<float> buffer, double sampleRate, juce::String sourceName)
{
    AudioFileData data;
    data.buffer = std::move (buffer);
    data.sampleRate = sampleRate;
    data.sourceName = std::move (sourceName);
    return data;
}

static inline bool clipPassesRules (const AudioFeatureVector& features, const std::vector<ProcessedClip>& existing, const ImportRules& rules)
{
    if (rules.removeLowRms && features.rmsDb < rules.minRmsDb)
        return false;

    if (rules.preferNovelSamples && features.spectralFlux < rules.minSpectralFlux)
        return false;

    if (rules.rejectNearDuplicates)
    {
        for (const auto& clip : existing)
            if (cosineSimilarity (features, clip.features) >= rules.duplicateSimilarityThreshold)
                return false;
    }

    return true;
}

static inline void finaliseClipOrdering (std::vector<ProcessedClip>& clips, const ImportRules& rules)
{
    if (rules.preferNovelSamples)
        std::stable_sort (clips.begin(), clips.end(), [] (const auto& a, const auto& b) { return a.features.novelty > b.features.novelty; });

    if (rules.randomize)
    {
        std::mt19937 rng (rules.randomSeed != 0 ? rules.randomSeed : 0x5eed1234u);
        std::shuffle (clips.begin(), clips.end(), rng);
    }
}

static inline RenderResult renderImportAction (const std::vector<juce::File>& inputFiles, ImportAction action, ImportRules rules)
{
    RenderResult result;
    auto files = filterSupportedExistingFiles (inputFiles);
    if (files.empty())
    {
        result.message = "No supported audio files were provided.";
        return result;
    }

    if (rules.randomSeed == 0)
        rules.randomSeed = deterministicSeedForImport (files, action);

    result.recipe.action = action;
    result.recipe.rules = rules;
    result.recipe.seed = rules.randomSeed;
    result.recipe.displayName = "File Import Recipe";
    for (const auto& f : files)
        result.recipe.inputs.push_back (fingerprintForFile (f));

    result.files = files; // Source paths are retained for deterministic recipe replay and favorites.

    if (action == ImportAction::LoadSeparate)
    {
        result.ok = true;
        result.loadMode = RenderedLoadMode::SeparateEntries;
        result.message = "Loaded source files.";
        return result;
    }

    if (action == ImportAction::AppendRawAsSingle)
    {
        juce::String error;
        double targetRate = rules.outputSampleRate;
        if (targetRate <= 0.0)
        {
            juce::AudioFormatManager fm;
            fm.registerBasicFormats();
            if (auto reader = std::unique_ptr<juce::AudioFormatReader> (fm.createReaderFor (files.front())))
                targetRate = reader->sampleRate;
        }
        if (targetRate <= 0.0)
            targetRate = 48000.0;

        ImportRules rawRules = rules;
        rawRules.trimEdges = false;
        rawRules.stripInternalSilence = false;
        rawRules.removeLowRms = false;
        rawRules.rejectNearDuplicates = false;
        rawRules.preferNovelSamples = false;
        rawRules.crossfadeMs = 0.0;
        rawRules.gapMs = 0.0;

        juce::AudioBuffer<float> appended;
        juce::String name = files.size() == 1 ? sanitiseRecipeFileStem (files.front().getFileNameWithoutExtension())
                                              : juce::String ("RawAppend");

        for (const auto& f : files)
        {
            auto data = readAudioFile (f, rawRules.outputChannels <= 0 ? 2 : rawRules.outputChannels, targetRate, 0.0, error);
            if (! data.has_value())
                continue;
            appendBuffer (appended, data->buffer, data->sampleRate, rawRules);
        }

        if (appended.getNumSamples() <= 0)
        {
            result.message = error.isNotEmpty() ? error : "Raw append produced no audio.";
            return result;
        }

        result.renderedAudio.push_back (makeRenderedAudioData (std::move (appended), targetRate, name));
        result.ok = true;
        result.loadMode = RenderedLoadMode::SeparateEntries;
        result.message = "Raw append rendered in memory.";
        return result;
    }

    juce::String error;

    if (action == ImportAction::ModifyExisting)
    {
        auto clips = preprocessClips (files, rules, error);
        if (clips.empty())
        {
            result.message = error.isNotEmpty() ? error : "Modify Existing produced no non-silent clips.";
            return result;
        }

        int idx = 1;
        for (auto& c : clips)
        {
            result.renderedAudio.push_back (makeRenderedAudioData (std::move (c.buffer), c.sampleRate,
                                                                   paddedImportIndex (idx++) + "_" + sanitiseRecipeFileStem (c.sourceName) + "_modified"));
        }

        result.ok = ! result.renderedAudio.empty();
        result.loadMode = RenderedLoadMode::SeparateEntries;
        result.message = result.ok ? "Modified files rendered in memory." : "Modify Existing produced no output.";
        return result;
    }

    if (action == ImportAction::SegmentLongFile)
    {
        int idx = 1;
        for (const auto& f : files)
        {
            auto data = readAudioFile (f, rules.outputChannels <= 0 ? 2 : rules.outputChannels, rules.outputSampleRate, 0.0, error);
            if (! data.has_value())
                continue;

            auto segments = detectSegmentsBySilence (data->buffer, data->sampleRate, rules);
            for (const auto& s : segments)
            {
                if (! s.enabled || s.length() <= 0)
                    continue;

                auto part = copyRange (data->buffer, s.startSample, s.endSample);
                applyEdgeFades (part, data->sampleRate, rules.edgeFadeMs);
                result.renderedAudio.push_back (makeRenderedAudioData (std::move (part), data->sampleRate,
                                                                       sanitiseRecipeFileStem (data->sourceName) + "_part" + paddedImportIndex (idx++)));
            }
        }

        result.ok = ! result.renderedAudio.empty();
        result.loadMode = RenderedLoadMode::SeparateEntries;
        result.message = result.ok ? "Segments rendered in memory." : (error.isNotEmpty() ? error : "No segments detected.");
        return result;
    }

    std::vector<ProcessedClip> clips;

    if (action == ImportAction::SegmentThenMegaTexture)
    {
        for (const auto& f : files)
        {
            auto data = readAudioFile (f, rules.outputChannels <= 0 ? 2 : rules.outputChannels, rules.outputSampleRate, 0.0, error);
            if (! data.has_value())
                continue;

            auto segments = detectSegmentsBySilence (data->buffer, data->sampleRate, rules);
            int localPart = 1;
            for (const auto& s : segments)
            {
                if (! s.enabled || s.length() <= 0)
                    continue;

                auto part = copyRange (data->buffer, s.startSample, s.endSample);
                applyEdgeFades (part, data->sampleRate, rules.edgeFadeMs);
                auto features = analyseAudioFeatures (part, data->sampleRate);
                if (! clipPassesRules (features, clips, rules))
                    continue;

                clips.push_back ({ std::move (part), data->sampleRate,
                                   sanitiseRecipeFileStem (data->sourceName) + "_part" + paddedImportIndex (localPart++),
                                   features });
            }
        }

        finaliseClipOrdering (clips, rules);
    }
    else if (action == ImportAction::BuildMegaTexture)
    {
        clips = preprocessClips (files, rules, error);
    }

    if (action == ImportAction::BuildMegaTexture || action == ImportAction::SegmentThenMegaTexture)
    {
        if (clips.empty())
        {
            result.message = error.isNotEmpty() ? error : "Mega Texture produced no clips after pruning.";
            return result;
        }

        juce::AudioBuffer<float> mega;
        double sr = clips.front().sampleRate > 0.0 ? clips.front().sampleRate : 48000.0;
        for (const auto& c : clips)
            appendBuffer (mega, c.buffer, sr, rules);

        if (rules.normalizeFinalRms)
        {
            const auto rms = computeRmsLinear (mega);
            if (rms > 1.0e-9)
                mega.applyGain ((float) (dbToLinear (rules.finalTargetRmsDb) / rms));
        }

        result.renderedAudio.push_back (makeRenderedAudioData (std::move (mega), sr, "MegaTexture"));
        result.ok = true;
        result.loadMode = RenderedLoadMode::SeparateEntries;
        result.message = "Mega Texture rendered in memory.";
        return result;
    }

    result.message = "Unsupported import action.";
    return result;
}


class ImportLandingPad final : public juce::Component
{
public:
    ImportLandingPad()
    {
        setInterceptsMouseClicks (false, false);
    }

    ImportAction actionForPoint (juce::Point<int> p, bool multipleFiles) const
    {
        const auto area = getLocalBounds().reduced (juce::jlimit (12, 44, getWidth() / 20));
        const int rowH = area.getHeight() / 4;
        const int row = juce::jlimit (0, 3, (p.y - area.getY()) / juce::jmax (1, rowH));
        switch (row)
        {
            case 0: return multipleFiles ? ImportAction::LoadSeparate : ImportAction::LoadSeparate;
            case 1: return multipleFiles ? ImportAction::BuildMegaTexture : ImportAction::AppendRawAsSingle;
            case 2: return ImportAction::SegmentLongFile;
            default: return ImportAction::ModifyExisting;
        }
    }

    void setHoverPoint (juce::Point<int> p)
    {
        hoverPoint = p;
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colours::black.withAlpha (0.62f));

        auto outer = getLocalBounds().reduced (juce::jlimit (12, 44, getWidth() / 20));
        const int gap = 10;
        const int rowH = juce::jmax (54, (outer.getHeight() - gap * 3) / 4);

        const std::array<juce::String, 4> titles {
            "Load Directly",
            "Build Mega Texture / Append Raw",
            "Segment / Auto-Segment",
            "Modify / Preprocess"
        };
        const std::array<juce::String, 4> subtitles {
            "Single or multiple files into the current file slot",
            "Multiple files to one texture with silence/RMS/novelty rules",
            "Show cut marks, then expose segments as logical entries",
            "Trim, strip silence, normalize, then load or export later"
        };

        for (int i = 0; i < 4; ++i)
        {
            auto r = outer.removeFromTop (rowH);
            outer.removeFromTop (gap);
            const bool hot = r.contains (hoverPoint);
            auto rf = r.toFloat();
            g.setColour (hot ? juce::Colour (0xff1687ff) : juce::Colour (0xff0f66d0));
            g.fillRoundedRectangle (rf, 10.0f);
            g.setColour (juce::Colours::white.withAlpha (hot ? 0.95f : 0.45f));
            g.drawRoundedRectangle (rf.reduced (1.0f), 10.0f, hot ? 2.0f : 1.0f);

            auto text = r.reduced (18, 8);
            g.setColour (juce::Colours::white);
            g.setFont (juce::Font (16.0f, juce::Font::bold));
            g.drawText (titles[(size_t) i], text.removeFromTop (24), juce::Justification::centredLeft, true);
            g.setColour (juce::Colours::white.withAlpha (0.82f));
            g.setFont (juce::Font (13.5f));
            g.drawFittedText (subtitles[(size_t) i], text, juce::Justification::centredLeft, 2);
        }
    }

private:
    juce::Point<int> hoverPoint { -10000, -10000 };
};

class WaveformPreview final : public juce::Component
{
public:
    void setBuffers (juce::AudioBuffer<float> originalIn,
                     juce::AudioBuffer<float> processedIn,
                     std::vector<SegmentRegion> segmentsIn = {},
                     double sampleRateIn = 0.0,
                     bool segmentationPreviewIn = false,
                     juce::String statusIn = {})
    {
        original = std::move (originalIn);
        processed = std::move (processedIn);
        segments = std::move (segmentsIn);
        sampleRate = sampleRateIn;
        segmentationPreview = segmentationPreviewIn;
        status = std::move (statusIn);
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colour (0xff15191d));
        auto r = getLocalBounds().reduced (8);
        auto top = r.removeFromTop (r.getHeight() / 2 - 4);
        r.removeFromTop (8);

        drawWave (g, top, original, segmentationPreview ? "Source / Proposed Cuts" : "Before", true);
        drawWave (g, r, processed, segmentationPreview ? "After / Kept Audio" : "After", false);
    }

private:
    void drawWave (juce::Graphics& g, juce::Rectangle<int> area, const juce::AudioBuffer<float>& b, const juce::String& label, bool drawSegments)
    {
        g.setColour (juce::Colour (0xff0f1318));
        g.fillRoundedRectangle (area.toFloat(), 8.0f);
        g.setColour (juce::Colours::white.withAlpha (0.16f));
        g.drawRoundedRectangle (area.toFloat().reduced (0.5f), 8.0f, 1.0f);

        auto header = area.removeFromTop (22).reduced (8, 0);
        g.setColour (juce::Colours::white.withAlpha (0.84f));
        g.setFont (13.0f);
        juce::String text = label;
        if (drawSegments && segmentationPreview)
        {
            int enabledCount = 0;
            for (const auto& s : segments)
                if (s.enabled && s.length() > 0)
                    ++enabledCount;
            text << "  |  " << enabledCount << " segment" << (enabledCount == 1 ? "" : "s");
            if (sampleRate > 0.0 && original.getNumSamples() > 0)
                text << "  |  " << juce::String ((double) original.getNumSamples() / sampleRate, 2) << "s full source";
        }
        if (drawSegments && status.isNotEmpty())
            text << "  |  " << status;
        g.drawText (text, header, juce::Justification::centredLeft, true);

        auto wave = area.reduced (8, 6);
        if (b.getNumSamples() <= 0 || b.getNumChannels() <= 0 || wave.getWidth() <= 1)
        {
            g.setColour (juce::Colours::white.withAlpha (0.4f));
            g.drawText (status.isNotEmpty() ? status : juce::String ("No preview data"), wave, juce::Justification::centred, true);
            return;
        }

        if (drawSegments && segmentationPreview && ! segments.empty())
            drawSegmentOverlay (g, wave, b.getNumSamples());

        const float mid = (float) wave.getCentreY();
        const float half = (float) wave.getHeight() * 0.45f;
        juce::Path path;
        const int width = juce::jmax (1, wave.getWidth());
        const int n = b.getNumSamples();

        for (int x = 0; x < width; ++x)
        {
            const int start = (int) ((int64_t) x * n / width);
            const int end = (int) ((int64_t) (x + 1) * n / width);
            float mn = 0.0f;
            float mx = 0.0f;
            for (int ch = 0; ch < b.getNumChannels(); ++ch)
            {
                const auto* p = b.getReadPointer (ch);
                for (int i = start; i < juce::jmax (start + 1, end); ++i)
                {
                    const float v = p[juce::jlimit (0, n - 1, i)];
                    mn = juce::jmin (mn, v);
                    mx = juce::jmax (mx, v);
                }
            }

            const float y1 = mid - mx * half;
            const float y2 = mid - mn * half;
            path.startNewSubPath ((float) wave.getX() + (float) x, y1);
            path.lineTo ((float) wave.getX() + (float) x, y2);
        }

        g.setColour (juce::Colour (0xff7cc7ff));
        g.strokePath (path, juce::PathStrokeType (1.0f));
    }

    void drawSegmentOverlay (juce::Graphics& g, juce::Rectangle<int> wave, int totalSamples)
    {
        if (totalSamples <= 0)
            return;

        for (const auto& s : segments)
        {
            if (! s.enabled || s.length() <= 0)
                continue;

            const int x1 = wave.getX() + (int) std::llround ((double) s.startSample * (double) wave.getWidth() / (double) totalSamples);
            const int x2 = wave.getX() + (int) std::llround ((double) s.endSample * (double) wave.getWidth() / (double) totalSamples);
            const auto region = juce::Rectangle<int> (x1, wave.getY(), juce::jmax (1, x2 - x1), wave.getHeight());
            g.setColour (juce::Colour (0xff34d399).withAlpha (0.13f));
            g.fillRect (region);
            g.setColour (juce::Colour (0xffffd166).withAlpha (0.86f));
            g.drawLine ((float) x1, (float) wave.getY(), (float) x1, (float) wave.getBottom(), 1.2f);
            g.drawLine ((float) x2, (float) wave.getY(), (float) x2, (float) wave.getBottom(), 1.2f);
        }
    }

    juce::AudioBuffer<float> original;
    juce::AudioBuffer<float> processed;
    std::vector<SegmentRegion> segments;
    double sampleRate = 0.0;
    bool segmentationPreview = false;
    juce::String status;
};

class ResettableSlider final : public juce::Slider
{
public:
    void setResetValue (double v)
    {
        resetValue = v;
        setDoubleClickReturnValue (true, resetValue);
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        if (e.mods.isRightButtonDown())
        {
            setValue (resetValue, juce::sendNotificationAsync);
            return;
        }

        juce::Slider::mouseDown (e);
    }

private:
    double resetValue = 0.0;
};

class ImportPreviewComponent final : public juce::Component, private juce::Slider::Listener
{
public:
    using ApplyCallback = std::function<void (ImportRules)>;

    ImportPreviewComponent (std::vector<juce::File> inputFiles, ImportAction actionIn, ImportRules initialRules, ApplyCallback cb)
        : files (std::move (inputFiles)), action (actionIn), rules (initialRules), onApply (std::move (cb))
    {
        title.setText ((action == ImportAction::SegmentLongFile || action == ImportAction::SegmentThenMegaTexture)
                         ? "Segmentation Preview"
                         : "Preprocess Preview",
                       juce::dontSendNotification);
        title.setFont (juce::Font (17.0f, juce::Font::bold));
        title.setJustificationType (juce::Justification::centredLeft);
        addAndMakeVisible (title);

        defaultRules = makeDefaultRulesForAction (action);

        configureSlider (silenceDb, "Silence threshold dBFS", -90.0, -6.0, 0.5, rules.silenceThresholdDb, defaultRules.silenceThresholdDb, -50.0);
        configureSlider (threshold, "Relative RMS ×", 0.0, 2.0, 0.01, rules.silenceThresholdRatio, defaultRules.silenceThresholdRatio, 0.25);
        configureSlider (minSilence, "Min quiet gap ms", 1.0, 5000.0, 1.0, rules.minSilenceMs, defaultRules.minSilenceMs, 100.0);
        configureSlider (minSegment, "Min segment ms", 1.0, 10000.0, 1.0, rules.minSegmentMs, defaultRules.minSegmentMs, 250.0);
        configureSlider (preRoll, "Pre-roll ms", 0.0, 500.0, 1.0, rules.preRollMs, defaultRules.preRollMs, 20.0);
        configureSlider (postRoll, "Post-roll ms", 0.0, 1000.0, 1.0, rules.postRollMs, defaultRules.postRollMs, 25.0);
        configureSlider (fade, "Fade ms", 0.0, 100.0, 0.5, rules.edgeFadeMs, defaultRules.edgeFadeMs, 10.0);
        configureSlider (rmsReject, "Reject below dB RMS", -120.0, -12.0, 0.5, rules.minRmsDb, defaultRules.minRmsDb, -65.0);

        relativeToggle.setButtonText ("Also use relative RMS gate");
        relativeToggle.setToggleState (rules.useRelativeRmsThreshold, juce::dontSendNotification);
        relativeToggle.onClick = [this] { updateRulesFromUi(); updateControlEnablement(); refreshPreview(); };
        addAndMakeVisible (relativeToggle);

        stripToggle.setButtonText ("Strip internal silence");
        stripToggle.setToggleState (rules.stripInternalSilence, juce::dontSendNotification);
        stripToggle.onClick = [this] { updateRulesFromUi(); refreshPreview(); };
        addAndMakeVisible (stripToggle);

        trimToggle.setButtonText ("Trim leading/trailing silence");
        trimToggle.setToggleState (rules.trimEdges, juce::dontSendNotification);
        trimToggle.onClick = [this] { updateRulesFromUi(); refreshPreview(); };
        addAndMakeVisible (trimToggle);

        rejectToggle.setButtonText ("Enable low RMS pruning");
        rejectToggle.setToggleState (rules.removeLowRms, juce::dontSendNotification);
        rejectToggle.onClick = [this] { updateRulesFromUi(); updateControlEnablement(); refreshPreview(); };
        addAndMakeVisible (rejectToggle);

        normalizeToggle.setButtonText ("Normalize clips to -24 dB RMS");
        normalizeToggle.setToggleState (rules.normalizeClipsRms, juce::dontSendNotification);
        normalizeToggle.onClick = [this] { updateRulesFromUi(); refreshPreview(); };
        addAndMakeVisible (normalizeToggle);

        addAndMakeVisible (waveform);

        apply.setButtonText ("Apply");
        apply.onClick = [this]
        {
            updateRulesFromUi();
            if (onApply)
                onApply (rules);
            if (auto* dw = findParentComponentOfClass<juce::DialogWindow>())
                dw->exitModalState (1);
        };
        addAndMakeVisible (apply);

        cancel.setButtonText ("Cancel");
        cancel.onClick = [this]
        {
            if (auto* dw = findParentComponentOfClass<juce::DialogWindow>())
                dw->exitModalState (0);
        };
        addAndMakeVisible (cancel);

        resetDefaults.setButtonText ("Reset");
        resetDefaults.setTooltip ("Reset all import/segmentation controls to defaults. Right-click any slider to reset only that control.");
        resetDefaults.onClick = [this]
        {
            rules = defaultRules;
            syncUiFromRules();
            refreshPreview();
        };
        addAndMakeVisible (resetDefaults);

        updateControlEnablement();

        setSize (980, 700);

        waveform.setBuffers (juce::AudioBuffer<float>(), juce::AudioBuffer<float>(), {}, 0.0,
                             action == ImportAction::SegmentLongFile || action == ImportAction::SegmentThenMegaTexture,
                             files.empty() ? juce::String ("No input file was passed to the preview.") : juce::String ("Loading preview…"));

        juce::Component::SafePointer<ImportPreviewComponent> safeThis (this);
        juce::MessageManager::callAsync ([safeThis]
        {
            if (safeThis != nullptr)
                safeThis->refreshPreview();
        });
    }

    void resized() override
    {
        auto r = getLocalBounds().reduced (14);
        title.setBounds (r.removeFromTop (28));
        r.removeFromTop (8);

        auto bottom = r.removeFromBottom (36);
        cancel.setBounds (bottom.removeFromRight (100));
        bottom.removeFromRight (8);
        apply.setBounds (bottom.removeFromRight (100));
        bottom.removeFromRight (8);
        resetDefaults.setBounds (bottom.removeFromRight (90));

        auto left = r.removeFromLeft (260);
        r.removeFromLeft (12);
        auto sliderH = 47;
        silenceDb.setBounds (left.removeFromTop (sliderH));
        threshold.setBounds (left.removeFromTop (sliderH));
        minSilence.setBounds (left.removeFromTop (sliderH));
        minSegment.setBounds (left.removeFromTop (sliderH));
        preRoll.setBounds (left.removeFromTop (sliderH));
        postRoll.setBounds (left.removeFromTop (sliderH));
        fade.setBounds (left.removeFromTop (sliderH));
        rmsReject.setBounds (left.removeFromTop (sliderH));
        left.removeFromTop (6);
        relativeToggle.setBounds (left.removeFromTop (26));
        trimToggle.setBounds (left.removeFromTop (26));
        stripToggle.setBounds (left.removeFromTop (26));
        rejectToggle.setBounds (left.removeFromTop (26));
        normalizeToggle.setBounds (left.removeFromTop (26));

        waveform.setBounds (r);
    }

private:
    void configureSlider (ResettableSlider& s, const juce::String& label, double min, double max, double step, double value, double defaultValue, double midpoint = 0.0)
    {
        s.setTextValueSuffix (juce::String ("  ") + label);
        s.setRange (min, max, step);
        s.setValue (value, juce::dontSendNotification);
        s.setResetValue (defaultValue);
        s.setSliderStyle (juce::Slider::LinearHorizontal);
        s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 104, 18);
        if (midpoint > min && midpoint < max)
            s.setSkewFactorFromMidPoint (midpoint);
        s.addListener (this);
        addAndMakeVisible (s);
    }

    void sliderValueChanged (juce::Slider*) override
    {
        updateRulesFromUi();
        refreshPreview();
    }

    void updateRulesFromUi()
    {
        rules.silenceThresholdDb = silenceDb.getValue();
        rules.silenceThresholdRatio = (float) threshold.getValue();
        rules.useRelativeRmsThreshold = relativeToggle.getToggleState();
        rules.minSilenceMs = minSilence.getValue();
        rules.minSegmentMs = minSegment.getValue();
        rules.preRollMs = preRoll.getValue();
        rules.postRollMs = postRoll.getValue();
        rules.edgeFadeMs = fade.getValue();
        rules.minRmsDb = rmsReject.getValue();
        rules.stripInternalSilence = stripToggle.getToggleState();
        rules.trimEdges = trimToggle.getToggleState();
        rules.removeLowRms = rejectToggle.getToggleState();
        rules.normalizeClipsRms = normalizeToggle.getToggleState();
    }

    void syncUiFromRules()
    {
        silenceDb.setValue (rules.silenceThresholdDb, juce::dontSendNotification);
        threshold.setValue (rules.silenceThresholdRatio, juce::dontSendNotification);
        relativeToggle.setToggleState (rules.useRelativeRmsThreshold, juce::dontSendNotification);
        minSilence.setValue (rules.minSilenceMs, juce::dontSendNotification);
        minSegment.setValue (rules.minSegmentMs, juce::dontSendNotification);
        preRoll.setValue (rules.preRollMs, juce::dontSendNotification);
        postRoll.setValue (rules.postRollMs, juce::dontSendNotification);
        fade.setValue (rules.edgeFadeMs, juce::dontSendNotification);
        rmsReject.setValue (rules.minRmsDb, juce::dontSendNotification);
        stripToggle.setToggleState (rules.stripInternalSilence, juce::dontSendNotification);
        trimToggle.setToggleState (rules.trimEdges, juce::dontSendNotification);
        rejectToggle.setToggleState (rules.removeLowRms, juce::dontSendNotification);
        normalizeToggle.setToggleState (rules.normalizeClipsRms, juce::dontSendNotification);
        updateControlEnablement();
    }

    void updateControlEnablement()
    {
        threshold.setEnabled (relativeToggle.getToggleState());
        rmsReject.setEnabled (rejectToggle.getToggleState());
    }

    void refreshPreview()
    {
        const bool segmentationMode = (action == ImportAction::SegmentLongFile
                                       || action == ImportAction::SegmentThenMegaTexture);

        if (files.empty())
        {
            waveform.setBuffers (juce::AudioBuffer<float>(), juce::AudioBuffer<float>(), {}, 0.0, segmentationMode,
                                 "No input file was passed to the preview.");
            return;
        }

        juce::String error;
        auto previewRules = rules;
        const double maxPreviewSeconds = segmentationMode ? 0.0 : previewRules.previewSeconds;
        const auto data = readAudioFile (files.front(), previewRules.outputChannels <= 0 ? 2 : previewRules.outputChannels,
                                         previewRules.outputSampleRate, maxPreviewSeconds, error);
        if (! data.has_value())
        {
            waveform.setBuffers (juce::AudioBuffer<float>(), juce::AudioBuffer<float>(), {}, 0.0, segmentationMode,
                                 error.isNotEmpty() ? error : juce::String ("Could not read preview audio."));
            return;
        }

        std::vector<SegmentRegion> segments;
        juce::AudioBuffer<float> processed;

        if (segmentationMode)
        {
            segments = detectSegmentsBySilence (data->buffer, data->sampleRate, previewRules);
            processed = concatenateRanges (data->buffer, segments, data->sampleRate, previewRules);
        }
        else
        {
            processed = processBufferByRules (data->buffer, data->sampleRate, previewRules);
            if (previewRules.stripInternalSilence || previewRules.trimEdges)
                segments = detectSegmentsBySilence (data->buffer, data->sampleRate, previewRules);
        }

        juce::String status;
        if (data->sampleRate > 0.0)
            status << files.front().getFileName() << " | " << juce::String ((double) data->buffer.getNumSamples() / data->sampleRate, 2) << "s";
        else
            status << files.front().getFileName();

        if (segmentationMode)
        {
            status << " | " << (int) segments.size() << " segment" << (segments.size() == 1 ? "" : "s")
                   << " | silence≤" << juce::String (previewRules.silenceThresholdDb, 1) << " dBFS"
                   << " | gap≥" << juce::String (previewRules.minSilenceMs, 0) << " ms"
                   << " | minLen≥" << juce::String (previewRules.minSegmentMs, 0) << " ms";
            if (previewRules.removeLowRms)
                status << " | reject<" << juce::String (previewRules.minRmsDb, 1) << " dB RMS";
        }

        waveform.setBuffers (data->buffer, std::move (processed), std::move (segments), data->sampleRate, segmentationMode, status);
    }

    std::vector<juce::File> files;
    ImportAction action;
    ImportRules rules;
    ImportRules defaultRules;
    ApplyCallback onApply;

    juce::Label title;
    ResettableSlider silenceDb, threshold, minSilence, minSegment, preRoll, postRoll, fade, rmsReject;
    juce::ToggleButton relativeToggle, stripToggle, trimToggle, rejectToggle, normalizeToggle;
    WaveformPreview waveform;
    juce::TextButton apply, cancel, resetDefaults;
};

static inline void showImportPreviewDialog (juce::Component& parent, std::vector<juce::File> files, ImportAction action, ImportRules rules, ImportPreviewComponent::ApplyCallback onApply)
{
    juce::DialogWindow::LaunchOptions opts;
    opts.dialogTitle = (action == ImportAction::SegmentLongFile || action == ImportAction::SegmentThenMegaTexture) ? "Segmentation Preview" : "Import / Preprocess";
    opts.dialogBackgroundColour = juce::Colour (0xff20272d);
    opts.escapeKeyTriggersCloseButton = true;
    opts.useNativeTitleBar = true;
    opts.resizable = true;
    opts.content.setOwned (new ImportPreviewComponent (std::move (files), action, rules, std::move (onApply)));
    opts.launchAsync();
    juce::ignoreUnused (parent);
}

} // namespace za::fileimport

