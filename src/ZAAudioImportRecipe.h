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

#include "ZAUnicodeText.h"

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

    // Non-destructive preview/editor state. Disabled inputs are skipped by
    // recipe rendering but retained in the recipe so the user can restore them
    // when editing the import again. Manual segments are indexed by input file
    // order after supported-file filtering. Segment samples are expressed in
    // the post-read/post-resample preview domain used by renderImportAction().
    std::vector<int> disabledInputIndices;
    std::vector<std::vector<SegmentRegion>> manualSegmentsByInput;
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
    t.setProperty ("randomSeed", (juce::int64) r.randomSeed, nullptr);
    t.setProperty ("gapMs", r.gapMs, nullptr);
    t.setProperty ("crossfadeMs", r.crossfadeMs, nullptr);
    t.setProperty ("normalizeClipsRms", r.normalizeClipsRms, nullptr);
    t.setProperty ("clipTargetRmsDb", r.clipTargetRmsDb, nullptr);
    t.setProperty ("normalizeFinalRms", r.normalizeFinalRms, nullptr);
    t.setProperty ("finalTargetRmsDb", r.finalTargetRmsDb, nullptr);
    t.setProperty ("outputChannels", r.outputChannels, nullptr);
    t.setProperty ("outputSampleRate", r.outputSampleRate, nullptr);

    if (! r.disabledInputIndices.empty())
    {
        juce::ValueTree disabled ("DISABLED_INPUTS");
        for (const auto index : r.disabledInputIndices)
        {
            juce::ValueTree item ("INPUT");
            item.setProperty ("index", index, nullptr);
            disabled.addChild (item, -1, nullptr);
        }
        t.addChild (disabled, -1, nullptr);
    }

    if (! r.manualSegmentsByInput.empty())
    {
        juce::ValueTree manual ("MANUAL_SEGMENTS");
        for (int fileIndex = 0; fileIndex < (int) r.manualSegmentsByInput.size(); ++fileIndex)
        {
            const auto& segments = r.manualSegmentsByInput[(size_t) fileIndex];
            if (segments.empty())
                continue;

            juce::ValueTree fileNode ("FILE");
            fileNode.setProperty ("index", fileIndex, nullptr);
            for (const auto& segment : segments)
            {
                juce::ValueTree seg ("SEGMENT");
                seg.setProperty ("startSample", segment.startSample, nullptr);
                seg.setProperty ("endSample", segment.endSample, nullptr);
                seg.setProperty ("enabled", segment.enabled, nullptr);
                seg.setProperty ("rmsDb", segment.rmsDb, nullptr);
                seg.setProperty ("peakDb", segment.peakDb, nullptr);
                fileNode.addChild (seg, -1, nullptr);
            }
            manual.addChild (fileNode, -1, nullptr);
        }

        if (manual.getNumChildren() > 0)
            t.addChild (manual, -1, nullptr);
    }

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
    r.randomSeed = (uint32_t) (juce::int64) t.getProperty ("randomSeed", (juce::int64) r.randomSeed);
    r.gapMs = (double) t.getProperty ("gapMs", r.gapMs);
    r.crossfadeMs = (double) t.getProperty ("crossfadeMs", r.crossfadeMs);
    r.normalizeClipsRms = (bool) t.getProperty ("normalizeClipsRms", r.normalizeClipsRms);
    r.clipTargetRmsDb = (double) t.getProperty ("clipTargetRmsDb", r.clipTargetRmsDb);
    r.normalizeFinalRms = (bool) t.getProperty ("normalizeFinalRms", r.normalizeFinalRms);
    r.finalTargetRmsDb = (double) t.getProperty ("finalTargetRmsDb", r.finalTargetRmsDb);
    r.outputChannels = (int) t.getProperty ("outputChannels", r.outputChannels);
    r.outputSampleRate = (double) t.getProperty ("outputSampleRate", r.outputSampleRate);

    if (auto disabled = t.getChildWithName ("DISABLED_INPUTS"); disabled.isValid())
    {
        for (int i = 0; i < disabled.getNumChildren(); ++i)
        {
            const int index = (int) disabled.getChild (i).getProperty ("index", -1);
            if (index >= 0 && std::find (r.disabledInputIndices.begin(), r.disabledInputIndices.end(), index) == r.disabledInputIndices.end())
                r.disabledInputIndices.push_back (index);
        }
        std::sort (r.disabledInputIndices.begin(), r.disabledInputIndices.end());
    }

    if (auto manual = t.getChildWithName ("MANUAL_SEGMENTS"); manual.isValid())
    {
        for (int fileNodeIndex = 0; fileNodeIndex < manual.getNumChildren(); ++fileNodeIndex)
        {
            const auto fileNode = manual.getChild (fileNodeIndex);
            const int fileIndex = (int) fileNode.getProperty ("index", -1);
            if (fileIndex < 0)
                continue;

            if ((int) r.manualSegmentsByInput.size() <= fileIndex)
                r.manualSegmentsByInput.resize ((size_t) fileIndex + 1);

            auto& outSegments = r.manualSegmentsByInput[(size_t) fileIndex];
            outSegments.clear();

            for (int segIndex = 0; segIndex < fileNode.getNumChildren(); ++segIndex)
            {
                const auto segNode = fileNode.getChild (segIndex);
                SegmentRegion segment;
                segment.startSample = (int) segNode.getProperty ("startSample", 0);
                segment.endSample = (int) segNode.getProperty ("endSample", 0);
                segment.enabled = (bool) segNode.getProperty ("enabled", true);
                segment.rmsDb = (double) segNode.getProperty ("rmsDb", segment.rmsDb);
                segment.peakDb = (double) segNode.getProperty ("peakDb", segment.peakDb);
                outSegments.push_back (segment);
            }
        }
    }

    return r;
}

static inline juce::ValueTree recipeToValueTree (const ImportRecipe& recipe)
{
    juce::ValueTree t ("ZA_IMPORT_RECIPE");
    t.setProperty ("version", recipe.version, nullptr);
    t.setProperty ("action", (int) recipe.action, nullptr);
    t.setProperty ("seed", (juce::int64) recipe.seed, nullptr);
    t.setProperty ("displayName", recipe.displayName, nullptr);
    t.addChild (rulesToValueTree (recipe.rules), -1, nullptr);

    juce::ValueTree inputs ("INPUTS");
    for (const auto& fp : recipe.inputs)
    {
        juce::ValueTree in ("INPUT");
        in.setProperty ("path", fp.path, nullptr);
        in.setProperty ("sizeBytes", (juce::int64) fp.sizeBytes, nullptr);
        in.setProperty ("modifiedUtcMs", (juce::int64) fp.modifiedUtcMs, nullptr);
        in.setProperty ("quickHash", (juce::int64) fp.quickHash, nullptr);
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
    recipe.seed = (uint32_t) (juce::int64) t.getProperty ("seed", (juce::int64) recipe.seed);
    recipe.displayName = t.getProperty ("displayName", recipe.displayName).toString();
    recipe.rules = rulesFromValueTree (t.getChildWithName ("RULES"));

    if (auto inputs = t.getChildWithName ("INPUTS"); inputs.isValid())
    {
        for (int i = 0; i < inputs.getNumChildren(); ++i)
        {
            auto in = inputs.getChild (i);
            SourceFingerprint fp;
            fp.path = in.getProperty ("path", {}).toString();
            fp.sizeBytes = (juce::int64) in.getProperty ("sizeBytes", (juce::int64) 0);
            fp.modifiedUtcMs = (juce::int64) in.getProperty ("modifiedUtcMs", (juce::int64) 0);
            fp.quickHash = (uint64_t) (juce::int64) in.getProperty ("quickHash", (juce::int64) 0);
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

static inline bool isInputIndexDisabled (const ImportRules& rules, int inputIndex) noexcept
{
    return inputIndex >= 0
        && std::find (rules.disabledInputIndices.begin(), rules.disabledInputIndices.end(), inputIndex) != rules.disabledInputIndices.end();
}

static inline void setInputIndexDisabled (ImportRules& rules, int inputIndex, bool shouldDisable)
{
    if (inputIndex < 0)
        return;

    auto& disabled = rules.disabledInputIndices;
    auto it = std::find (disabled.begin(), disabled.end(), inputIndex);

    if (shouldDisable)
    {
        if (it == disabled.end())
        {
            disabled.push_back (inputIndex);
            std::sort (disabled.begin(), disabled.end());
        }
    }
    else if (it != disabled.end())
    {
        disabled.erase (it);
    }
}

static inline void setManualSegmentsForInput (ImportRules& rules, int inputIndex, std::vector<SegmentRegion> segments)
{
    if (inputIndex < 0)
        return;

    if ((int) rules.manualSegmentsByInput.size() <= inputIndex)
        rules.manualSegmentsByInput.resize ((size_t) inputIndex + 1);

    rules.manualSegmentsByInput[(size_t) inputIndex] = std::move (segments);
}

static inline void clearManualSegmentsForInput (ImportRules& rules, int inputIndex)
{
    if (inputIndex < 0 || inputIndex >= (int) rules.manualSegmentsByInput.size())
        return;

    rules.manualSegmentsByInput[(size_t) inputIndex].clear();
}

static inline std::vector<SegmentRegion> sanitiseSegmentsForBuffer (const juce::AudioBuffer<float>& b, std::vector<SegmentRegion> segments)
{
    const int n = b.getNumSamples();
    if (n <= 0)
        return {};

    for (auto& segment : segments)
    {
        segment.startSample = juce::jlimit (0, n, segment.startSample);
        segment.endSample = juce::jlimit (segment.startSample, n, segment.endSample);
        if (segment.length() <= 0)
            segment.enabled = false;

        if (segment.enabled)
        {
            segment.rmsDb = linearToDb (computeRmsLinear (b, segment.startSample, segment.length()));
            segment.peakDb = linearToDb (computePeakLinear (b, segment.startSample, segment.length()));
        }
    }

    std::stable_sort (segments.begin(), segments.end(), [] (const auto& a, const auto& b) { return a.startSample < b.startSample; });
    return segments;
}

static inline std::vector<SegmentRegion> segmentsForInput (const ImportRules& rules,
                                                           int inputIndex,
                                                           const juce::AudioBuffer<float>& b,
                                                           double sr)
{
    if (inputIndex >= 0 && inputIndex < (int) rules.manualSegmentsByInput.size())
    {
        const auto& manual = rules.manualSegmentsByInput[(size_t) inputIndex];
        if (! manual.empty())
            return sanitiseSegmentsForBuffer (b, manual);
    }

    return detectSegmentsBySilence (b, sr, rules);
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

    for (int fileIndex = 0; fileIndex < (int) files.size(); ++fileIndex)
    {
        if (isInputIndexDisabled (rules, fileIndex))
            continue;

        const auto& f = files[(size_t) fileIndex];
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

        for (int fileIndex = 0; fileIndex < (int) files.size(); ++fileIndex)
        {
            if (isInputIndexDisabled (rules, fileIndex))
                continue;

            const auto& f = files[(size_t) fileIndex];
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
        for (int fileIndex = 0; fileIndex < (int) files.size(); ++fileIndex)
        {
            if (isInputIndexDisabled (rules, fileIndex))
                continue;

            const auto& f = files[(size_t) fileIndex];
            auto data = readAudioFile (f, rules.outputChannels <= 0 ? 2 : rules.outputChannels, rules.outputSampleRate, 0.0, error);
            if (! data.has_value())
                continue;

            auto segments = segmentsForInput (rules, fileIndex, data->buffer, data->sampleRate);
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
        for (int fileIndex = 0; fileIndex < (int) files.size(); ++fileIndex)
        {
            if (isInputIndexDisabled (rules, fileIndex))
                continue;

            const auto& f = files[(size_t) fileIndex];
            auto data = readAudioFile (f, rules.outputChannels <= 0 ? 2 : rules.outputChannels, rules.outputSampleRate, 0.0, error);
            if (! data.has_value())
                continue;

            auto segments = segmentsForInput (rules, fileIndex, data->buffer, data->sampleRate);
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
    using SegmentSelectCallback = std::function<void (int)>;
    using SegmentBoundaryCallback = std::function<void (int, bool, int)>;
    using SegmentDragFinishedCallback = std::function<void()>;
    using SegmentCreatedCallback = std::function<void (int, int)>;

    void setCallbacks (SegmentSelectCallback selectCb,
                       SegmentBoundaryCallback boundaryCb,
                       SegmentDragFinishedCallback dragFinishedCb = {},
                       SegmentCreatedCallback createdCb = {})
    {
        onSegmentSelected = std::move (selectCb);
        onSegmentBoundaryMoved = std::move (boundaryCb);
        onSegmentDragFinished = std::move (dragFinishedCb);
        onSegmentCreated = std::move (createdCb);
    }

    void setSelectedSegment (int index)
    {
        selectedSegment = index;
        repaint();
    }

    void setBuffers (juce::AudioBuffer<float> originalIn,
                     juce::AudioBuffer<float> processedIn,
                     std::vector<SegmentRegion> segmentsIn = {},
                     double sampleRateIn = 0.0,
                     bool segmentationPreviewIn = false,
                     juce::String statusIn = {})
    {
        const int oldNumSamples = original.getNumSamples();
        const int newNumSamples = originalIn.getNumSamples();
        const double oldSampleRate = sampleRate;

        original = std::move (originalIn);
        processed = std::move (processedIn);
        segments = std::move (segmentsIn);
        sampleRate = sampleRateIn;
        segmentationPreview = segmentationPreviewIn;
        status = std::move (statusIn);

        if (oldNumSamples != newNumSamples || std::abs (oldSampleRate - sampleRateIn) > 1.0 || visibleEndSample <= visibleStartSample)
            resetZoomToFull();
        else
            clampZoomRange();

        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colour (0xff15191d));

        auto r = getLocalBounds().reduced (8);
        auto kept = getKeptPanelBoundsFor (r);
        auto source = r;
        source.removeFromBottom (kept.getHeight());
        source.removeFromBottom (8);

        drawWave (g, source, original, segmentationPreview ? "Source / Editable Cuts" : "Before", true);
        drawWave (g, kept, processed, segmentationPreview ? "Kept Audio" : "After", false);
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        draggingSegment = -1;
        draggingStart = false;
        creatingSegment = false;
        scrollingZoom = false;

        if (! segmentationPreview || original.getNumSamples() <= 0)
            return;

        const auto wave = getSourceWaveBounds();
        if (! wave.contains (e.getPosition()))
            return;

        if (e.mods.isMiddleButtonDown())
        {
            const int start = getVisibleStartSample();
            const int end = getVisibleEndSample();
            if (end > start && end - start < original.getNumSamples())
            {
                scrollingZoom = true;
                scrollDragStartX = e.position.x;
                scrollDragStartSample = start;
                scrollDragVisibleSpan = end - start;
            }
            return;
        }

        if (e.mods.isCtrlDown() || e.mods.isCommandDown())
        {
            creatingSegment = true;
            createStartSample = sampleFromX (e.position.x);
            createEndSample = createStartSample;
            repaint();
            return;
        }

        const int hit = findSegmentAtX (e.position.x);
        if (hit >= 0)
        {
            selectedSegment = hit;
            if (onSegmentSelected)
                onSegmentSelected (hit);

            const int x = (int) std::lround (e.position.x);
            const int startX = xFromSample (segments[(size_t) hit].startSample);
            const int endX = xFromSample (segments[(size_t) hit].endSample);
            const int handleSlop = juce::jlimit (5, 12, getWidth() / 120);

            if (std::abs (x - startX) <= handleSlop)
            {
                draggingSegment = hit;
                draggingStart = true;
            }
            else if (std::abs (x - endX) <= handleSlop)
            {
                draggingSegment = hit;
                draggingStart = false;
            }

            repaint();
        }
    }

    void mouseDoubleClick (const juce::MouseEvent& e) override
    {
        if (segmentationPreview && getSourceWaveBounds().contains (e.getPosition()))
            resetZoomToFull();
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (scrollingZoom)
        {
            dragZoomScrollTo (e.position.x);
            return;
        }

        if (creatingSegment)
        {
            createEndSample = sampleFromX (e.position.x);
            repaint();
            return;
        }

        if (draggingSegment < 0 || ! onSegmentBoundaryMoved)
            return;

        onSegmentBoundaryMoved (draggingSegment, draggingStart, sampleFromX (e.position.x));
    }

    void mouseUp (const juce::MouseEvent&) override
    {
        const bool wasCreating = creatingSegment;
        const bool wasScrolling = scrollingZoom;
        const bool wasDragging = draggingSegment >= 0;
        const int start = juce::jmin (createStartSample, createEndSample);
        const int end = juce::jmax (createStartSample, createEndSample);

        draggingSegment = -1;
        creatingSegment = false;
        scrollingZoom = false;

        if (wasCreating)
        {
            if (onSegmentCreated && end > start)
                onSegmentCreated (start, end);
            repaint();
            return;
        }

        if (wasScrolling)
        {
            repaint();
            return;
        }

        if (wasDragging && onSegmentDragFinished)
            onSegmentDragFinished();
    }

    void mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override
    {
        if (! segmentationPreview || original.getNumSamples() <= 0 || ! getSourceWaveBounds().contains (e.getPosition()))
        {
            juce::Component::mouseWheelMove (e, wheel);
            return;
        }

        const float dominantDelta = std::abs (wheel.deltaX) > std::abs (wheel.deltaY) ? wheel.deltaX : wheel.deltaY;
        if (std::abs (dominantDelta) <= 0.000001f)
            return;

        if (e.mods.isShiftDown() || std::abs (wheel.deltaX) > std::abs (wheel.deltaY))
            panZoomedView (dominantDelta);
        else
            zoomAtX (e.position.x, dominantDelta);
    }

private:
    int getKeptPanelHeightForTotal (int totalHeight) const noexcept
    {
        if (! segmentationPreview)
            return juce::jmax (48, totalHeight / 2 - 4);

        constexpr int minKept = 72;
        constexpr int minSource = 140;
        const int preferred = (int) std::llround ((double) totalHeight * 0.20);
        const int maxKept = juce::jmax (minKept, totalHeight - minSource - 8);
        return juce::jlimit (minKept, maxKept, preferred);
    }

    juce::Rectangle<int> getKeptPanelBoundsFor (juce::Rectangle<int> area) const noexcept
    {
        return area.removeFromBottom (getKeptPanelHeightForTotal (area.getHeight()));
    }

    juce::Rectangle<int> getSourceWaveBounds() const
    {
        auto r = getLocalBounds().reduced (8);
        const auto kept = getKeptPanelBoundsFor (r);
        juce::ignoreUnused (kept);
        r.removeFromBottom (getKeptPanelHeightForTotal (r.getHeight()));
        r.removeFromBottom (8);
        r.removeFromTop (22);
        return r.reduced (8, 6);
    }

    int getVisibleStartSample() const noexcept
    {
        const int n = original.getNumSamples();
        if (n <= 0)
            return 0;
        return juce::jlimit (0, juce::jmax (0, n - 1), visibleStartSample);
    }

    int getVisibleEndSample() const noexcept
    {
        const int n = original.getNumSamples();
        if (n <= 0)
            return 0;
        return juce::jlimit (getVisibleStartSample() + 1, n, visibleEndSample);
    }

    int minimumVisibleSamples() const noexcept
    {
        const int n = original.getNumSamples();
        if (n <= 0)
            return 0;

        const int timeFloor = sampleRate > 0.0 ? (int) std::llround (sampleRate * 0.020) : 64;
        return juce::jlimit (1, n, juce::jmax (64, timeFloor));
    }

    void resetZoomToFull()
    {
        visibleStartSample = 0;
        visibleEndSample = juce::jmax (0, original.getNumSamples());
        repaint();
    }

    void clampZoomRange()
    {
        const int n = original.getNumSamples();
        if (n <= 0)
        {
            visibleStartSample = 0;
            visibleEndSample = 0;
            return;
        }

        const int minVisible = minimumVisibleSamples();
        int span = juce::jlimit (minVisible, n, visibleEndSample - visibleStartSample);
        visibleStartSample = juce::jlimit (0, juce::jmax (0, n - span), visibleStartSample);
        visibleEndSample = visibleStartSample + span;
    }

    void zoomAtX (float x, float wheelDelta)
    {
        const int n = original.getNumSamples();
        if (n <= 0)
            return;

        const auto wave = getSourceWaveBounds();
        const int oldStart = getVisibleStartSample();
        const int oldEnd = getVisibleEndSample();
        const int oldSpan = juce::jmax (1, oldEnd - oldStart);
        const int minVisible = minimumVisibleSamples();

        const double norm = juce::jlimit (0.0, 1.0, ((double) x - (double) wave.getX()) / (double) juce::jmax (1, wave.getWidth()));
        const int anchor = juce::jlimit (0, n, (int) std::llround ((double) oldStart + norm * (double) oldSpan));
        const double factor = juce::jlimit (0.20, 5.0, std::exp ((double) -wheelDelta * 1.75));
        const int newSpan = juce::jlimit (minVisible, n, (int) std::llround ((double) oldSpan * factor));
        int newStart = (int) std::llround ((double) anchor - norm * (double) newSpan);
        newStart = juce::jlimit (0, juce::jmax (0, n - newSpan), newStart);

        visibleStartSample = newStart;
        visibleEndSample = newStart + newSpan;
        repaint();
    }

    void panZoomedView (float wheelDelta)
    {
        const int n = original.getNumSamples();
        const int oldStart = getVisibleStartSample();
        const int oldEnd = getVisibleEndSample();
        const int span = oldEnd - oldStart;
        if (n <= 0 || span <= 0 || span >= n)
            return;

        const int step = juce::jmax (1, (int) std::llround ((double) span * 0.18 * (double) wheelDelta));
        int newStart = oldStart - step;
        newStart = juce::jlimit (0, juce::jmax (0, n - span), newStart);
        visibleStartSample = newStart;
        visibleEndSample = newStart + span;
        repaint();
    }

    void dragZoomScrollTo (float x)
    {
        const int n = original.getNumSamples();
        const auto wave = getSourceWaveBounds();
        const int span = scrollDragVisibleSpan;
        if (n <= 0 || span <= 0 || span >= n || wave.getWidth() <= 1)
            return;

        const double samplesPerPixel = (double) span / (double) wave.getWidth();
        const int deltaSamples = (int) std::llround (((double) x - (double) scrollDragStartX) * samplesPerPixel);
        const int newStart = juce::jlimit (0, juce::jmax (0, n - span), scrollDragStartSample - deltaSamples);
        visibleStartSample = newStart;
        visibleEndSample = newStart + span;
        repaint();
    }

    int xFromSample (int sample) const
    {
        const auto wave = getSourceWaveBounds();
        const int start = getVisibleStartSample();
        const int end = getVisibleEndSample();
        const int span = juce::jmax (1, end - start);
        return wave.getX() + (int) std::llround ((double) (sample - start) * (double) wave.getWidth() / (double) span);
    }

    int sampleFromX (float x) const
    {
        const auto wave = getSourceWaveBounds();
        const int n = juce::jmax (1, original.getNumSamples());
        const int start = getVisibleStartSample();
        const int end = getVisibleEndSample();
        const int span = juce::jmax (1, end - start);
        const double norm = ((double) x - (double) wave.getX()) / (double) juce::jmax (1, wave.getWidth());
        return juce::jlimit (0, n, (int) std::llround ((double) start + norm * (double) span));
    }

    int findSegmentAtX (float x) const
    {
        const int xi = (int) std::lround (x);
        int bodyHit = -1;
        int bestHandle = -1;
        int bestDistance = 1000000;
        const int handleSlop = juce::jlimit (5, 12, getWidth() / 120);
        const auto wave = getSourceWaveBounds();

        for (int i = 0; i < (int) segments.size(); ++i)
        {
            const auto& s = segments[(size_t) i];
            if (! s.enabled || s.length() <= 0)
                continue;

            if (s.endSample < getVisibleStartSample() || s.startSample > getVisibleEndSample())
                continue;

            const int sx = xFromSample (s.startSample);
            const int ex = xFromSample (s.endSample);
            if (juce::jmax (sx, ex) < wave.getX() - handleSlop || juce::jmin (sx, ex) > wave.getRight() + handleSlop)
                continue;

            const int ds = std::abs (xi - sx);
            const int de = std::abs (xi - ex);
            const int d = juce::jmin (ds, de);
            if (d <= handleSlop && d < bestDistance)
            {
                bestDistance = d;
                bestHandle = i;
            }

            if (xi >= juce::jmin (sx, ex) && xi <= juce::jmax (sx, ex))
                bodyHit = i;
        }

        return bestHandle >= 0 ? bestHandle : bodyHit;
    }

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
            {
                text << "  |  " << juce::String ((double) original.getNumSamples() / sampleRate, 2) << "s full source";
                const int visibleSpan = juce::jmax (1, getVisibleEndSample() - getVisibleStartSample());
                const double zoom = (double) original.getNumSamples() / (double) visibleSpan;
                if (zoom > 1.01)
                    text << "  |  zoom " << juce::String (zoom, 1) << "x";
            }
            text << "  |  wheel zoom, Shift+wheel/MMB-drag pan, Ctrl+drag new, Tab/Shift+Tab nav, Space play/pause, Delete remove, Ctrl+Z undo";
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
        const bool useZoom = drawSegments && segmentationPreview && (&b == &original);
        const int viewStart = useZoom ? getVisibleStartSample() : 0;
        const int viewEnd = useZoom ? getVisibleEndSample() : n;
        const int viewSpan = juce::jmax (1, viewEnd - viewStart);

        for (int x = 0; x < width; ++x)
        {
            const int start = viewStart + (int) ((int64_t) x * viewSpan / width);
            const int end = viewStart + (int) ((int64_t) (x + 1) * viewSpan / width);
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

        if (! drawSegments && segmentationPreview && ! segments.empty())
            drawKeptSegmentOverlay (g, wave, b.getNumSamples());
    }

    void drawKeptSegmentOverlay (juce::Graphics& g, juce::Rectangle<int> wave, int totalSamples)
    {
        if (totalSamples <= 0 || wave.getWidth() <= 1)
            return;

        int cursor = 0;
        for (int i = 0; i < (int) segments.size(); ++i)
        {
            const auto& s = segments[(size_t) i];
            if (! s.enabled || s.length() <= 0)
                continue;

            const int start = cursor;
            const int end = juce::jmin (totalSamples, cursor + s.length());
            cursor = end;

            if (end <= start)
                continue;

            const int x1 = wave.getX() + (int) std::llround ((double) start * (double) wave.getWidth() / (double) totalSamples);
            const int x2 = wave.getX() + (int) std::llround ((double) end * (double) wave.getWidth() / (double) totalSamples);
            const auto region = juce::Rectangle<int> (juce::jmin (x1, x2), wave.getY(), juce::jmax (1, std::abs (x2 - x1)), wave.getHeight());
            const bool selected = i == selectedSegment;

            g.setColour ((selected ? juce::Colour (0xff60a5fa) : juce::Colour (0xff34d399)).withAlpha (selected ? 0.16f : 0.07f));
            g.fillRect (region);
            g.setColour ((selected ? juce::Colour (0xff93c5fd) : juce::Colour (0xffffd166)).withAlpha (0.75f));
            g.drawLine ((float) x1, (float) wave.getY(), (float) x1, (float) wave.getBottom(), selected ? 2.0f : 1.0f);
            g.drawLine ((float) x2, (float) wave.getY(), (float) x2, (float) wave.getBottom(), selected ? 2.0f : 1.0f);
        }
    }

    void drawSegmentOverlay (juce::Graphics& g, juce::Rectangle<int> wave, int totalSamples)
    {
        if (totalSamples <= 0)
            return;

        for (int i = 0; i < (int) segments.size(); ++i)
        {
            const auto& s = segments[(size_t) i];
            if (! s.enabled || s.length() <= 0)
                continue;

            const int rawX1 = xFromSample (s.startSample);
            const int rawX2 = xFromSample (s.endSample);
            if (juce::jmax (rawX1, rawX2) < wave.getX() || juce::jmin (rawX1, rawX2) > wave.getRight())
                continue;

            const int x1 = juce::jlimit (wave.getX(), wave.getRight(), rawX1);
            const int x2 = juce::jlimit (wave.getX(), wave.getRight(), rawX2);
            const auto region = juce::Rectangle<int> (juce::jmin (x1, x2), wave.getY(), juce::jmax (1, std::abs (x2 - x1)), wave.getHeight());
            const bool selected = i == selectedSegment;
            g.setColour ((selected ? juce::Colour (0xff60a5fa) : juce::Colour (0xff34d399)).withAlpha (selected ? 0.22f : 0.13f));
            g.fillRect (region);
            g.setColour ((selected ? juce::Colour (0xff93c5fd) : juce::Colour (0xffffd166)).withAlpha (0.92f));
            if (rawX1 >= wave.getX() && rawX1 <= wave.getRight())
                g.drawLine ((float) rawX1, (float) wave.getY(), (float) rawX1, (float) wave.getBottom(), selected ? 2.0f : 1.2f);
            if (rawX2 >= wave.getX() && rawX2 <= wave.getRight())
                g.drawLine ((float) rawX2, (float) wave.getY(), (float) rawX2, (float) wave.getBottom(), selected ? 2.0f : 1.2f);
            if (selected)
            {
                g.setColour (juce::Colour (0xff93c5fd).withAlpha (0.75f));
                g.drawRect (region, 1);
            }
        }

        drawPendingCreatedSegment (g, wave);
    }

    void drawPendingCreatedSegment (juce::Graphics& g, juce::Rectangle<int> wave)
    {
        if (! creatingSegment)
            return;

        const int start = juce::jmin (createStartSample, createEndSample);
        const int end = juce::jmax (createStartSample, createEndSample);
        if (end <= start)
            return;

        const int rawX1 = xFromSample (start);
        const int rawX2 = xFromSample (end);
        if (juce::jmax (rawX1, rawX2) < wave.getX() || juce::jmin (rawX1, rawX2) > wave.getRight())
            return;

        const int x1 = juce::jlimit (wave.getX(), wave.getRight(), rawX1);
        const int x2 = juce::jlimit (wave.getX(), wave.getRight(), rawX2);
        const auto region = juce::Rectangle<int> (juce::jmin (x1, x2), wave.getY(), juce::jmax (1, std::abs (x2 - x1)), wave.getHeight());
        g.setColour (juce::Colour (0xffffd166).withAlpha (0.24f));
        g.fillRect (region);
        g.setColour (juce::Colour (0xffffe6a8).withAlpha (0.92f));
        g.drawRect (region, 2);
    }

    juce::AudioBuffer<float> original;
    juce::AudioBuffer<float> processed;
    std::vector<SegmentRegion> segments;
    double sampleRate = 0.0;
    bool segmentationPreview = false;
    juce::String status;
    int visibleStartSample = 0;
    int visibleEndSample = 0;
    int selectedSegment = -1;
    int draggingSegment = -1;
    bool draggingStart = false;
    bool creatingSegment = false;
    int createStartSample = 0;
    int createEndSample = 0;
    bool scrollingZoom = false;
    float scrollDragStartX = 0.0f;
    int scrollDragStartSample = 0;
    int scrollDragVisibleSpan = 0;
    SegmentSelectCallback onSegmentSelected;
    SegmentBoundaryCallback onSegmentBoundaryMoved;
    SegmentDragFinishedCallback onSegmentDragFinished;
    SegmentCreatedCallback onSegmentCreated;
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

class ImportPreviewComponent final : public juce::Component,
                                     private juce::Slider::Listener,
                                     private juce::Timer,
                                     private juce::KeyListener
{
public:
    using ApplyCallback = std::function<void (ImportRules)>;
    using AuditionCallback = std::function<void (juce::AudioBuffer<float>, double)>;
    using StopAuditionCallback = std::function<void()>;
    using PauseAuditionCallback = std::function<void (bool)>;

    ImportPreviewComponent (std::vector<juce::File> inputFiles,
                            ImportAction actionIn,
                            ImportRules initialRules,
                            ApplyCallback cb,
                            AuditionCallback auditionCb = {},
                            StopAuditionCallback stopAuditionCb = {},
                            PauseAuditionCallback pauseAuditionCb = {})
        : files (filterSupportedExistingFiles (inputFiles)),
          action (actionIn),
          rules (std::move (initialRules)),
          onApply (std::move (cb)),
          onAudition (std::move (auditionCb)),
          onStopAudition (std::move (stopAuditionCb)),
          onPauseAudition (std::move (pauseAuditionCb))
    {
        setWantsKeyboardFocus (true);
        setMouseClickGrabsKeyboardFocus (true);
        title.setText (isSegmentationMode() ? "Segmentation Preview" : "Preprocess Preview", juce::dontSendNotification);
        title.setFont (juce::Font (17.0f, juce::Font::bold));
        title.setJustificationType (juce::Justification::centredLeft);
        addAndMakeVisible (title);

        defaultRules = makeDefaultRulesForAction (action);

        sourceLabel.setText ("Preview source", juce::dontSendNotification);
        sourceLabel.setJustificationType (juce::Justification::centredLeft);
        addAndMakeVisible (sourceLabel);

        sourceSelector.setTextWhenNothingSelected ("No source files");
        sourceSelector.onChange = [this]
        {
            const int index = sourceSelector.getSelectedId() - 1;
            if (index >= 0 && index < (int) files.size() && index != previewFileIndex)
            {
                stopAuditionNow();
                previewFileIndex = index;
                selectedSegment = -1;
                refreshPreview();
                maybeAutoPlaySelectedSegment();
            }
        };
        addAndMakeVisible (sourceSelector);

        configureSlider (silenceDb, "Silence threshold dBFS", -90.0, -6.0, 0.5, rules.silenceThresholdDb, defaultRules.silenceThresholdDb, -50.0);
        configureSlider (threshold, za::text::utf8 ("Relative RMS ×"), 0.0, 2.0, 0.01, rules.silenceThresholdRatio, defaultRules.silenceThresholdRatio, 0.25);
        configureSlider (minSilence, "Min quiet gap ms", 1.0, 5000.0, 1.0, rules.minSilenceMs, defaultRules.minSilenceMs, 100.0);
        configureSlider (minSegment, "Min segment ms", 1.0, 10000.0, 1.0, rules.minSegmentMs, defaultRules.minSegmentMs, 250.0);
        configureSlider (preRoll, "Pre-roll ms", 0.0, 500.0, 1.0, rules.preRollMs, defaultRules.preRollMs, 20.0);
        configureSlider (postRoll, "Post-roll ms", 0.0, 1000.0, 1.0, rules.postRollMs, defaultRules.postRollMs, 25.0);
        configureSlider (fade, "Fade ms", 0.0, 100.0, 0.5, rules.edgeFadeMs, defaultRules.edgeFadeMs, 10.0);
        configureSlider (rmsReject, "Reject below dB RMS", -120.0, -12.0, 0.5, rules.minRmsDb, defaultRules.minRmsDb, -65.0);

        configureSlider (segmentStart, "Segment start sec", 0.0, 1.0, 0.0001, 0.0, 0.0);
        configureSlider (segmentEnd, "Segment end sec", 0.0, 1.0, 0.0001, 1.0, 1.0);

        relativeToggle.setButtonText ("Also use relative RMS gate");
        relativeToggle.setToggleState (rules.useRelativeRmsThreshold, juce::dontSendNotification);
        relativeToggle.onClick = [this] { captureUndoState(); updateRulesFromUi(); updateControlEnablement(); refreshPreview(); };
        addAndMakeVisible (relativeToggle);

        trimToggle.setButtonText ("Trim leading/trailing silence");
        trimToggle.setToggleState (rules.trimEdges, juce::dontSendNotification);
        trimToggle.onClick = [this] { captureUndoState(); updateRulesFromUi(); refreshPreview(); };
        addAndMakeVisible (trimToggle);

        stripToggle.setButtonText ("Strip internal silence");
        stripToggle.setToggleState (rules.stripInternalSilence, juce::dontSendNotification);
        stripToggle.onClick = [this] { captureUndoState(); updateRulesFromUi(); refreshPreview(); };
        addAndMakeVisible (stripToggle);

        rejectToggle.setButtonText ("Reject quiet clips");
        rejectToggle.setToggleState (rules.removeLowRms, juce::dontSendNotification);
        rejectToggle.onClick = [this] { captureUndoState(); updateRulesFromUi(); updateControlEnablement(); refreshPreview(); };
        addAndMakeVisible (rejectToggle);

        normalizeToggle.setButtonText ("Normalize clip RMS");
        normalizeToggle.setToggleState (rules.normalizeClipsRms, juce::dontSendNotification);
        normalizeToggle.onClick = [this] { captureUndoState(); updateRulesFromUi(); refreshPreview(); };
        addAndMakeVisible (normalizeToggle);

        segmentLabel.setText ("Segment edit", juce::dontSendNotification);
        segmentLabel.setJustificationType (juce::Justification::centredLeft);
        addAndMakeVisible (segmentLabel);

        prevSegment.setButtonText ("Prev");
        prevSegment.onClick = [this] { selectAdjacentSegment (-1); };
        addAndMakeVisible (prevSegment);

        nextSegment.setButtonText ("Next");
        nextSegment.onClick = [this] { selectAdjacentSegment (1); };
        addAndMakeVisible (nextSegment);

        playSegment.setButtonText ("Play seg");
        playSegment.setTooltip ("Audition the currently selected segment through the plugin output.");
        playSegment.onClick = [this] { toggleSelectedSegmentAudition(); };
        addAndMakeVisible (playSegment);

        stopAudition.setButtonText ("Stop");
        stopAudition.setTooltip ("Stop the current segment audition.");
        stopAudition.onClick = [this] { stopAuditionNow(); };
        addAndMakeVisible (stopAudition);

        autoPlaySegment.setButtonText ("Auto-play selected");
        autoPlaySegment.setTooltip ("Automatically audition a segment when it is selected with Tab, click, or newly created.");
        autoPlaySegment.onClick = [this]
        {
            segmentAutoPlay = autoPlaySegment.getToggleState();
            if (segmentAutoPlay)
                auditionSelectedSegment();
        };
        addAndMakeVisible (autoPlaySegment);

        deleteSegment.setButtonText ("Delete seg");
        deleteSegment.setTooltip ("Remove the selected segment from the rendered import. This does not delete anything on disk.");
        deleteSegment.onClick = [this] { deleteSelectedSegment(); };
        addAndMakeVisible (deleteSegment);

        restoreSegments.setButtonText ("Auto cuts");
        restoreSegments.setTooltip ("Discard manual cuts for this source and re-run auto segmentation with the current rules.");
        restoreSegments.onClick = [this]
        {
            captureUndoState();
            clearManualSegmentsForInput (rules, previewFileIndex);
            selectedSegment = -1;
            refreshPreview();
        };
        addAndMakeVisible (restoreSegments);

        removeCurrentFile.setButtonText ("Remove source");
        removeCurrentFile.setTooltip ("Remove/restore this source from the rendered import. The source file remains on disk and stays restorable in this recipe.");
        removeCurrentFile.onClick = [this]
        {
            captureUndoState();
            stopAuditionNow();
            setInputIndexDisabled (rules, previewFileIndex, ! isCurrentInputDisabled());
            syncSourceSelector();
            refreshPreview();
        };
        addAndMakeVisible (removeCurrentFile);

        waveform.setCallbacks ([this] (int index)
        {
            selectedSegment = index;
            syncSegmentControls();
            maybeAutoPlaySelectedSegment();
        },
        [this] (int index, bool isStart, int sample)
        {
            editSegmentBoundary (index, isStart, sample);
        },
        [this]
        {
            boundaryEditUndoCaptured = false;
            boundaryEditSegment = -1;
        },
        [this] (int startSample, int endSample)
        {
            createSegmentFromDrag (startSample, endSample);
        });
        waveform.setWantsKeyboardFocus (true);
        waveform.setMouseClickGrabsKeyboardFocus (true);
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
        resetDefaults.setTooltip ("Reset import/segmentation controls and clear manual cuts/source removals. Right-click any slider to reset only that control.");
        resetDefaults.onClick = [this]
        {
            captureUndoState();
            stopAuditionNow();
            rules = defaultRules;
            syncUiFromRules();
            selectedSegment = -1;
            refreshPreview();
        };
        addAndMakeVisible (resetDefaults);

        syncSourceSelector();
        updateControlEnablement();
        setSegmentationControlsVisible (isSegmentationMode());

        installShortcutKeyListeners();

        setSize (1120, 740);

        waveform.setBuffers (juce::AudioBuffer<float>(), juce::AudioBuffer<float>(), {}, 0.0,
                             isSegmentationMode(),
                             files.empty() ? juce::String ("No input file was passed to the preview.") : za::text::utf8 ("Loading preview…"));

        juce::Component::SafePointer<ImportPreviewComponent> safeThis (this);
        juce::MessageManager::callAsync ([safeThis]
        {
            if (safeThis != nullptr)
            {
                safeThis->refreshPreview();
                safeThis->grabKeyboardFocus();
            }
        });
    }

    ~ImportPreviewComponent() override
    {
        removeShortcutKeyListeners();
        stopTimer();
        stopAuditionNow();
    }

    void resized() override
    {
        auto r = getLocalBounds().reduced (14);
        title.setBounds (r.removeFromTop (28));
        r.removeFromTop (6);

        auto sourceRow = r.removeFromTop (30);
        sourceLabel.setBounds (sourceRow.removeFromLeft (110));
        sourceRow.removeFromLeft (8);
        sourceSelector.setBounds (sourceRow);
        r.removeFromTop (8);

        auto bottom = r.removeFromBottom (36);
        cancel.setBounds (bottom.removeFromRight (100));
        bottom.removeFromRight (8);
        apply.setBounds (bottom.removeFromRight (100));
        bottom.removeFromRight (8);
        resetDefaults.setBounds (bottom.removeFromRight (90));

        auto left = r.removeFromLeft (300);
        r.removeFromLeft (12);
        auto sliderH = 39;
        silenceDb.setBounds (left.removeFromTop (sliderH));
        threshold.setBounds (left.removeFromTop (sliderH));
        minSilence.setBounds (left.removeFromTop (sliderH));
        minSegment.setBounds (left.removeFromTop (sliderH));
        preRoll.setBounds (left.removeFromTop (sliderH));
        postRoll.setBounds (left.removeFromTop (sliderH));
        fade.setBounds (left.removeFromTop (sliderH));
        rmsReject.setBounds (left.removeFromTop (sliderH));
        left.removeFromTop (4);
        relativeToggle.setBounds (left.removeFromTop (24));
        trimToggle.setBounds (left.removeFromTop (24));
        stripToggle.setBounds (left.removeFromTop (24));
        rejectToggle.setBounds (left.removeFromTop (24));
        normalizeToggle.setBounds (left.removeFromTop (24));

        left.removeFromTop (8);
        segmentLabel.setBounds (left.removeFromTop (24));
        segmentStart.setBounds (left.removeFromTop (sliderH));
        segmentEnd.setBounds (left.removeFromTop (sliderH));
        auto segButtons1 = left.removeFromTop (28);
        prevSegment.setBounds (segButtons1.removeFromLeft (54));
        segButtons1.removeFromLeft (6);
        nextSegment.setBounds (segButtons1.removeFromLeft (54));
        segButtons1.removeFromLeft (6);
        playSegment.setBounds (segButtons1.removeFromLeft (82));
        segButtons1.removeFromLeft (6);
        stopAudition.setBounds (segButtons1);
        left.removeFromTop (6);
        auto segButtons2 = left.removeFromTop (30);
        restoreSegments.setBounds (segButtons2.removeFromLeft (92));
        segButtons2.removeFromLeft (8);
        removeCurrentFile.setBounds (segButtons2.removeFromLeft (120));
        segButtons2.removeFromLeft (8);
        deleteSegment.setBounds (segButtons2);
        left.removeFromTop (6);
        autoPlaySegment.setBounds (left.removeFromTop (24));

        waveform.setBounds (r);
    }

private:
    bool isSegmentationMode() const noexcept
    {
        return action == ImportAction::SegmentLongFile || action == ImportAction::SegmentThenMegaTexture;
    }

    struct UndoState
    {
        ImportRules rules;
        int previewFileIndex = 0;
        int selectedSegment = -1;
    };

    UndoState currentUndoState() const
    {
        return { rules, previewFileIndex, selectedSegment };
    }

    void captureUndoState()
    {
        if (restoringUndoState)
            return;

        undoStack.push_back (currentUndoState());
        if (undoStack.size() > 64)
            undoStack.erase (undoStack.begin());
        redoStack.clear();
    }

    void applyUndoState (const UndoState& state)
    {
        restoringUndoState = true;
        stopAuditionNow();
        rules = state.rules;
        previewFileIndex = juce::jlimit (files.empty() ? -1 : 0,
                                           juce::jmax (-1, (int) files.size() - 1),
                                           state.previewFileIndex);
        selectedSegment = state.selectedSegment;
        syncUiFromRules();
        refreshPreview();
        restoringUndoState = false;
    }

    void undoLastEdit()
    {
        if (undoStack.empty())
            return;

        redoStack.push_back (currentUndoState());
        const auto state = undoStack.back();
        undoStack.pop_back();
        applyUndoState (state);
    }

    void redoLastEdit()
    {
        if (redoStack.empty())
            return;

        undoStack.push_back (currentUndoState());
        const auto state = redoStack.back();
        redoStack.pop_back();
        applyUndoState (state);
    }

    bool isTextEditorFocused() const
    {
        if (auto* focused = juce::Component::getCurrentlyFocusedComponent())
            return dynamic_cast<juce::TextEditor*> (focused) != nullptr;
        return false;
    }

    bool handleShortcutKey (const juce::KeyPress& key)
    {
        if (! isSegmentationMode())
            return false;

        const auto mods = key.getModifiers();
        const int code = key.getKeyCode();
        const bool ctrlOrCmd = mods.isCtrlDown() || mods.isCommandDown();

        if (ctrlOrCmd && (code == 'z' || code == 'Z'))
        {
            if (mods.isShiftDown())
                redoLastEdit();
            else
                undoLastEdit();
            return true;
        }

        if (isTextEditorFocused())
            return false;

        if (! ctrlOrCmd && code == juce::KeyPress::tabKey)
        {
            selectAdjacentSegment (mods.isShiftDown() ? -1 : 1);
            return true;
        }

        if (code == juce::KeyPress::spaceKey)
        {
            toggleSelectedSegmentAudition();
            return true;
        }

        if (code == juce::KeyPress::deleteKey || code == juce::KeyPress::backspaceKey)
        {
            deleteSelectedSegment();
            return true;
        }

        return false;
    }

    bool keyPressed (const juce::KeyPress& key) override
    {
        return handleShortcutKey (key);
    }

    bool keyPressed (const juce::KeyPress& key, juce::Component*) override
    {
        return handleShortcutKey (key);
    }

    void installShortcutKeyListeners()
    {
        setWantsKeyboardFocus (true);
        addKeyListener (this);
        for (int i = 0; i < getNumChildComponents(); ++i)
            if (auto* child = getChildComponent (i))
                child->addKeyListener (this);
    }

    void removeShortcutKeyListeners()
    {
        removeKeyListener (this);
        for (int i = 0; i < getNumChildComponents(); ++i)
            if (auto* child = getChildComponent (i))
                child->removeKeyListener (this);
    }

    void timerCallback() override
    {
        if (segmentAuditionActive && ! segmentAuditionPaused
            && juce::Time::getMillisecondCounterHiRes() >= segmentAuditionEndMs)
        {
            clearSegmentAuditionUiState();
        }
    }

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

    void sliderValueChanged (juce::Slider* slider) override
    {
        if (updatingUi)
            return;

        if (! sliderEditUndoCaptured)
        {
            captureUndoState();
            sliderEditUndoCaptured = true;
        }

        const bool mouseDriven = slider != nullptr && slider->isMouseButtonDown();

        if (slider == &segmentStart || slider == &segmentEnd)
        {
            updateSelectedSegmentFromSliders();
            if (! mouseDriven)
                sliderEditUndoCaptured = false;
            return;
        }

        updateRulesFromUi();
        clearManualSegmentsForInput (rules, previewFileIndex);
        selectedSegment = -1;
        refreshPreview();
        if (! mouseDriven)
            sliderEditUndoCaptured = false;
    }

    void sliderDragStarted (juce::Slider*) override
    {
        if (! updatingUi && ! sliderEditUndoCaptured)
        {
            captureUndoState();
            sliderEditUndoCaptured = true;
        }
    }

    void sliderDragEnded (juce::Slider*) override
    {
        sliderEditUndoCaptured = false;
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
        updatingUi = true;
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
        updatingUi = false;
        syncSourceSelector();
        updateControlEnablement();
        syncSegmentControls();
    }

    void updateControlEnablement()
    {
        threshold.setEnabled (relativeToggle.getToggleState());
        rmsReject.setEnabled (rejectToggle.getToggleState());
        apply.setEnabled (! allInputsDisabled());
        removeCurrentFile.setEnabled (! files.empty());
    }

    void setSegmentationControlsVisible (bool shouldShow)
    {
        segmentLabel.setVisible (shouldShow);
        segmentStart.setVisible (shouldShow);
        segmentEnd.setVisible (shouldShow);
        prevSegment.setVisible (shouldShow);
        nextSegment.setVisible (shouldShow);
        playSegment.setVisible (shouldShow);
        stopAudition.setVisible (shouldShow);
        autoPlaySegment.setVisible (shouldShow);
        deleteSegment.setVisible (shouldShow);
        restoreSegments.setVisible (shouldShow);
    }

    bool allInputsDisabled() const noexcept
    {
        if (files.empty())
            return true;

        for (int i = 0; i < (int) files.size(); ++i)
            if (! isInputIndexDisabled (rules, i))
                return false;

        return true;
    }

    bool isCurrentInputDisabled() const noexcept
    {
        return isInputIndexDisabled (rules, previewFileIndex);
    }

    void syncSourceSelector()
    {
        updatingUi = true;
        sourceSelector.clear (juce::dontSendNotification);
        for (int i = 0; i < (int) files.size(); ++i)
        {
            juce::String label = files[(size_t) i].getFileName();
            if (isInputIndexDisabled (rules, i))
                label << "  (removed)";
            sourceSelector.addItem (label, i + 1);
        }

        if (previewFileIndex < 0 || previewFileIndex >= (int) files.size())
            previewFileIndex = files.empty() ? -1 : 0;

        sourceSelector.setSelectedId (previewFileIndex + 1, juce::dontSendNotification);
        updatingUi = false;
        removeCurrentFile.setButtonText (isCurrentInputDisabled() ? "Restore source" : "Remove source");
        updateControlEnablement();
    }

    int minEditableSegmentSamples() const noexcept
    {
        if (previewSampleRate <= 0.0)
            return 1;
        return juce::jmax (1, (int) std::llround (previewSampleRate * juce::jlimit (1.0, 10000.0, rules.minSegmentMs) / 1000.0));
    }

    int enabledSegmentCount() const noexcept
    {
        int count = 0;
        for (const auto& segment : previewSegments)
            if (segment.enabled && segment.length() > 0)
                ++count;
        return count;
    }

    int firstEnabledSegment() const noexcept
    {
        for (int i = 0; i < (int) previewSegments.size(); ++i)
            if (previewSegments[(size_t) i].enabled && previewSegments[(size_t) i].length() > 0)
                return i;
        return -1;
    }

    void syncSegmentControls()
    {
        const bool canEdit = isSegmentationMode()
                          && ! isCurrentInputDisabled()
                          && previewSampleRate > 0.0
                          && selectedSegment >= 0
                          && selectedSegment < (int) previewSegments.size()
                          && previewSegments[(size_t) selectedSegment].enabled
                          && previewSegments[(size_t) selectedSegment].length() > 0;

        segmentStart.setEnabled (canEdit);
        segmentEnd.setEnabled (canEdit);
        prevSegment.setEnabled (isSegmentationMode() && enabledSegmentCount() > 1);
        nextSegment.setEnabled (isSegmentationMode() && enabledSegmentCount() > 1);
        playSegment.setEnabled (canEdit && onAudition != nullptr);
        stopAudition.setEnabled (isSegmentationMode() && onStopAudition != nullptr && segmentAuditionActive);
        autoPlaySegment.setEnabled (isSegmentationMode() && onAudition != nullptr);
        updateAuditionButtonText();
        deleteSegment.setEnabled (canEdit);
        restoreSegments.setEnabled (isSegmentationMode() && previewFileIndex >= 0);

        if (! canEdit)
        {
            updatingUi = true;
            segmentStart.setRange (0.0, 1.0, 0.0001);
            segmentEnd.setRange (0.0, 1.0, 0.0001);
            segmentStart.setValue (0.0, juce::dontSendNotification);
            segmentEnd.setValue (0.0, juce::dontSendNotification);
            segmentLabel.setText (isCurrentInputDisabled() ? "Source removed from import" : "No editable segment selected", juce::dontSendNotification);
            updatingUi = false;
            waveform.setSelectedSegment (selectedSegment);
            return;
        }

        const auto& s = previewSegments[(size_t) selectedSegment];
        const double duration = previewSampleRate > 0.0 ? (double) previewOriginal.getNumSamples() / previewSampleRate : 1.0;
        const double startSec = (double) s.startSample / previewSampleRate;
        const double endSec = (double) s.endSample / previewSampleRate;

        updatingUi = true;
        segmentStart.setRange (0.0, duration, 0.0001);
        segmentEnd.setRange (0.0, duration, 0.0001);
        segmentStart.setResetValue (startSec);
        segmentEnd.setResetValue (endSec);
        segmentStart.setValue (startSec, juce::dontSendNotification);
        segmentEnd.setValue (endSec, juce::dontSendNotification);
        segmentLabel.setText ("Segment " + juce::String (selectedSegment + 1) + " / " + juce::String ((int) previewSegments.size())
                                + "  |  " + juce::String (endSec - startSec, 3) + " s",
                              juce::dontSendNotification);
        updatingUi = false;
        waveform.setSelectedSegment (selectedSegment);
    }

    void selectAdjacentSegment (int direction)
    {
        if (previewSegments.empty())
            return;

        int index = selectedSegment;
        for (int guard = 0; guard < (int) previewSegments.size(); ++guard)
        {
            index += direction;
            if (index < 0)
                index = (int) previewSegments.size() - 1;
            else if (index >= (int) previewSegments.size())
                index = 0;

            if (previewSegments[(size_t) index].enabled && previewSegments[(size_t) index].length() > 0)
            {
                selectedSegment = index;
                syncSegmentControls();
                maybeAutoPlaySelectedSegment();
                return;
            }
        }
    }

    void maybeAutoPlaySelectedSegment()
    {
        if (segmentAutoPlay && isSegmentationMode())
            auditionSelectedSegment();
    }

    void createSegmentFromDrag (int startSample, int endSample)
    {
        if (! isSegmentationMode()
            || isCurrentInputDisabled()
            || previewOriginal.getNumSamples() <= 0
            || previewSampleRate <= 0.0)
            return;

        const int n = previewOriginal.getNumSamples();
        int start = juce::jlimit (0, juce::jmax (0, n - 1), juce::jmin (startSample, endSample));
        int end = juce::jlimit (start + 1, n, juce::jmax (startSample, endSample));
        const int minLen = juce::jmin (minEditableSegmentSamples(), juce::jmax (1, n));
        if (end - start < minLen)
        {
            end = juce::jlimit (start + 1, n, start + minLen);
            if (end - start < minLen)
                start = juce::jlimit (0, juce::jmax (0, end - 1), end - minLen);
        }

        if (end <= start)
            return;

        captureUndoState();
        stopAuditionNow();

        SegmentRegion created;
        created.startSample = start;
        created.endSample = end;
        created.enabled = true;
        created.rmsDb = linearToDb (computeRmsLinear (previewOriginal, created.startSample, created.length()));
        created.peakDb = linearToDb (computePeakLinear (previewOriginal, created.startSample, created.length()));

        std::vector<SegmentRegion> updated;
        updated.reserve (previewSegments.size() + 2);
        for (auto segment : previewSegments)
        {
            if (! segment.enabled || segment.length() <= 0 || segment.endSample <= start || segment.startSample >= end)
            {
                updated.push_back (segment);
                continue;
            }

            const int oldStart = segment.startSample;
            const int oldEnd = segment.endSample;

            if (oldStart < start)
            {
                segment.endSample = start;
                if (segment.length() > 0)
                    updated.push_back (segment);
            }

            if (oldEnd > end)
            {
                SegmentRegion tail = segment;
                tail.startSample = end;
                tail.endSample = oldEnd;
                if (tail.length() > 0)
                    updated.push_back (tail);
            }
        }

        updated.push_back (created);
        std::stable_sort (updated.begin(), updated.end(), [] (const auto& a, const auto& b) { return a.startSample < b.startSample; });

        previewSegments = sanitiseSegmentsForBuffer (previewOriginal, std::move (updated));
        selectedSegment = -1;
        for (int i = 0; i < (int) previewSegments.size(); ++i)
        {
            const auto& segment = previewSegments[(size_t) i];
            if (segment.enabled && segment.startSample == start && segment.endSample == end)
            {
                selectedSegment = i;
                break;
            }
        }
        if (selectedSegment < 0)
            selectedSegment = firstEnabledSegment();

        saveManualSegmentsForCurrentFile();
        refreshProcessedPreviewOnly();
        maybeAutoPlaySelectedSegment();
    }

    void repairNeighbourOverlap (int index)
    {
        if (index < 0 || index >= (int) previewSegments.size())
            return;

        auto& s = previewSegments[(size_t) index];
        const int n = previewOriginal.getNumSamples();
        const int minLen = juce::jmin (minEditableSegmentSamples(), juce::jmax (1, n));
        s.startSample = juce::jlimit (0, juce::jmax (0, n - 1), s.startSample);
        s.endSample = juce::jlimit (s.startSample + 1, n, s.endSample);

        if (s.endSample - s.startSample < minLen)
            s.endSample = juce::jlimit (s.startSample + 1, n, s.startSample + minLen);

        if (index > 0)
        {
            auto& prev = previewSegments[(size_t) index - 1];
            if (prev.enabled && prev.endSample > s.startSample)
                prev.endSample = juce::jmax (prev.startSample, s.startSample);
        }

        if (index + 1 < (int) previewSegments.size())
        {
            auto& next = previewSegments[(size_t) index + 1];
            if (next.enabled && next.startSample < s.endSample)
                next.startSample = juce::jmin (next.endSample, s.endSample);
        }

        for (auto& segment : previewSegments)
            if (segment.enabled && segment.length() <= 0)
                segment.enabled = false;
    }

    void editSegmentBoundary (int index, bool isStart, int sample)
    {
        if (index < 0 || index >= (int) previewSegments.size() || previewOriginal.getNumSamples() <= 0)
            return;

        selectedSegment = index;
        if (! boundaryEditUndoCaptured || boundaryEditSegment != index)
        {
            captureUndoState();
            boundaryEditUndoCaptured = true;
            boundaryEditSegment = index;
        }

        auto& s = previewSegments[(size_t) index];
        const int n = previewOriginal.getNumSamples();
        const int minLen = juce::jmin (minEditableSegmentSamples(), juce::jmax (1, n));

        if (isStart)
            s.startSample = juce::jlimit (0, juce::jmax (0, s.endSample - minLen), sample);
        else
            s.endSample = juce::jlimit (juce::jmin (n, s.startSample + minLen), n, sample);

        repairNeighbourOverlap (index);
        saveManualSegmentsForCurrentFile();
        refreshProcessedPreviewOnly();
    }

    void updateSelectedSegmentFromSliders()
    {
        if (selectedSegment < 0 || selectedSegment >= (int) previewSegments.size() || previewSampleRate <= 0.0)
            return;

        auto& s = previewSegments[(size_t) selectedSegment];
        const int n = previewOriginal.getNumSamples();
        int start = juce::jlimit (0, juce::jmax (0, n - 1), (int) std::llround (segmentStart.getValue() * previewSampleRate));
        int end = juce::jlimit (start + 1, n, (int) std::llround (segmentEnd.getValue() * previewSampleRate));
        if (end <= start)
            end = juce::jlimit (start + 1, n, start + juce::jmin (minEditableSegmentSamples(), juce::jmax (1, n)));

        s.startSample = start;
        s.endSample = end;
        repairNeighbourOverlap (selectedSegment);
        saveManualSegmentsForCurrentFile();
        refreshProcessedPreviewOnly();
    }

    void auditionSelectedSegment()
    {
        if (onAudition == nullptr
            || selectedSegment < 0
            || selectedSegment >= (int) previewSegments.size()
            || previewSampleRate <= 0.0
            || previewOriginal.getNumSamples() <= 0
            || isCurrentInputDisabled())
            return;

        const auto& s = previewSegments[(size_t) selectedSegment];
        if (! s.enabled || s.length() <= 0)
            return;

        auto clip = copyRange (previewOriginal, s.startSample, s.endSample);
        applyEdgeFades (clip, previewSampleRate, rules.edgeFadeMs);
        const int clipSamples = clip.getNumSamples();
        onAudition (std::move (clip), previewSampleRate);

        segmentAuditionActive = true;
        segmentAuditionPaused = false;
        segmentAuditionFileIndex = previewFileIndex;
        segmentAuditionSegment = selectedSegment;
        segmentAuditionRemainingMs = juce::jmax (1.0, 1000.0 * (double) clipSamples / previewSampleRate);
        segmentAuditionEndMs = juce::Time::getMillisecondCounterHiRes() + segmentAuditionRemainingMs;
        startTimerHz (20);
        updateAuditionButtonText();
        syncSegmentControls();
    }

    void toggleSelectedSegmentAudition()
    {
        const bool sameSegment = segmentAuditionActive
                              && segmentAuditionFileIndex == previewFileIndex
                              && segmentAuditionSegment == selectedSegment;

        if (! sameSegment)
        {
            auditionSelectedSegment();
            return;
        }

        if (segmentAuditionPaused)
            resumeSegmentAudition();
        else
            pauseSegmentAudition();
    }

    void pauseSegmentAudition()
    {
        if (! segmentAuditionActive || segmentAuditionPaused)
            return;

        segmentAuditionRemainingMs = juce::jmax (1.0, segmentAuditionEndMs - juce::Time::getMillisecondCounterHiRes());
        segmentAuditionPaused = true;
        stopTimer();

        if (onPauseAudition)
            onPauseAudition (true);

        updateAuditionButtonText();
        syncSegmentControls();
    }

    void resumeSegmentAudition()
    {
        if (! segmentAuditionActive || ! segmentAuditionPaused)
            return;

        segmentAuditionPaused = false;
        segmentAuditionEndMs = juce::Time::getMillisecondCounterHiRes() + juce::jmax (1.0, segmentAuditionRemainingMs);
        startTimerHz (20);

        if (onPauseAudition)
            onPauseAudition (false);

        updateAuditionButtonText();
        syncSegmentControls();
    }

    void clearSegmentAuditionUiState()
    {
        segmentAuditionActive = false;
        segmentAuditionPaused = false;
        segmentAuditionFileIndex = -1;
        segmentAuditionSegment = -1;
        segmentAuditionEndMs = 0.0;
        segmentAuditionRemainingMs = 0.0;
        stopTimer();
        updateAuditionButtonText();
        syncSegmentControls();
    }

    void updateAuditionButtonText()
    {
        const bool sameSegment = segmentAuditionActive
                              && segmentAuditionFileIndex == previewFileIndex
                              && segmentAuditionSegment == selectedSegment;

        if (sameSegment && segmentAuditionPaused)
            playSegment.setButtonText ("Resume seg");
        else if (sameSegment)
            playSegment.setButtonText ("Pause seg");
        else
            playSegment.setButtonText ("Play seg");
    }

    void saveManualSegmentsForCurrentFile()
    {
        if (previewFileIndex >= 0)
            setManualSegmentsForInput (rules, previewFileIndex, previewSegments);
    }

    void deleteSelectedSegment()
    {
        if (selectedSegment < 0 || selectedSegment >= (int) previewSegments.size())
            return;

        captureUndoState();
        stopAuditionNow();
        previewSegments[(size_t) selectedSegment].enabled = false;
        saveManualSegmentsForCurrentFile();
        const int old = selectedSegment;
        selectedSegment = firstEnabledSegment();
        if (selectedSegment < 0 && old + 1 < (int) previewSegments.size())
            selectedSegment = old + 1;
        refreshProcessedPreviewOnly();
        maybeAutoPlaySelectedSegment();
    }

    void stopAuditionNow()
    {
        if (onStopAudition)
            onStopAudition();

        clearSegmentAuditionUiState();
    }

    void refreshProcessedPreviewOnly()
    {
        juce::AudioBuffer<float> processed;
        if (isSegmentationMode() && ! isCurrentInputDisabled())
            processed = concatenateRanges (previewOriginal, previewSegments, previewSampleRate, rules);
        else if (! isCurrentInputDisabled())
            processed = processBufferByRules (previewOriginal, previewSampleRate, rules);

        waveform.setBuffers (previewOriginal, std::move (processed), previewSegments, previewSampleRate, isSegmentationMode(), previewStatus);
        syncSegmentControls();
        updateControlEnablement();
    }

    void refreshPreview()
    {
        updateControlEnablement();
        syncSourceSelector();

        if (files.empty() || previewFileIndex < 0)
        {
            previewOriginal = {};
            previewSampleRate = 0.0;
            previewSegments.clear();
            waveform.setBuffers (juce::AudioBuffer<float>(), juce::AudioBuffer<float>(), {}, 0.0, isSegmentationMode(),
                                 "No input file was passed to the preview.");
            syncSegmentControls();
            return;
        }

        juce::String error;
        auto previewRules = rules;
        const double maxPreviewSeconds = isSegmentationMode() ? 0.0 : previewRules.previewSeconds;
        const auto data = readAudioFile (files[(size_t) previewFileIndex], previewRules.outputChannels <= 0 ? 2 : previewRules.outputChannels,
                                         previewRules.outputSampleRate, maxPreviewSeconds, error);
        if (! data.has_value())
        {
            previewOriginal = {};
            previewSampleRate = 0.0;
            previewSegments.clear();
            waveform.setBuffers (juce::AudioBuffer<float>(), juce::AudioBuffer<float>(), {}, 0.0, isSegmentationMode(),
                                 error.isNotEmpty() ? error : juce::String ("Could not read preview audio."));
            syncSegmentControls();
            return;
        }

        previewOriginal = data->buffer;
        previewSampleRate = data->sampleRate;
        previewSegments.clear();

        juce::AudioBuffer<float> processed;
        const bool removed = isCurrentInputDisabled();

        if (isSegmentationMode())
        {
            previewSegments = segmentsForInput (rules, previewFileIndex, previewOriginal, previewSampleRate);
            if (! removed)
                processed = concatenateRanges (previewOriginal, previewSegments, previewSampleRate, rules);
        }
        else if (! removed)
        {
            processed = processBufferByRules (previewOriginal, previewSampleRate, rules);
            if (previewRules.stripInternalSilence || previewRules.trimEdges)
                previewSegments = detectSegmentsBySilence (previewOriginal, previewSampleRate, previewRules);
        }

        previewStatus.clear();
        if (previewSampleRate > 0.0)
            previewStatus << files[(size_t) previewFileIndex].getFileName() << " | " << juce::String ((double) previewOriginal.getNumSamples() / previewSampleRate, 2) << "s";
        else
            previewStatus << files[(size_t) previewFileIndex].getFileName();

        if (removed)
            previewStatus << " | removed from import";

        if (isSegmentationMode())
        {
            previewStatus << " | " << enabledSegmentCount() << " kept / " << (int) previewSegments.size() << " total"
                          << za::text::utf8 (" | silence≤") << juce::String (previewRules.silenceThresholdDb, 1) << " dBFS"
                          << za::text::utf8 (" | gap≥") << juce::String (previewRules.minSilenceMs, 0) << " ms"
                          << za::text::utf8 (" | minLen≥") << juce::String (previewRules.minSegmentMs, 0) << " ms";
            if (previewRules.removeLowRms)
                previewStatus << " | reject<" << juce::String (previewRules.minRmsDb, 1) << " dB RMS";
        }

        if (selectedSegment < 0 || selectedSegment >= (int) previewSegments.size() || ! previewSegments[(size_t) selectedSegment].enabled)
            selectedSegment = firstEnabledSegment();

        waveform.setBuffers (previewOriginal, std::move (processed), previewSegments, previewSampleRate, isSegmentationMode(), previewStatus);
        syncSegmentControls();
        updateControlEnablement();
    }

    std::vector<juce::File> files;
    ImportAction action;
    ImportRules rules;
    ImportRules defaultRules;
    ApplyCallback onApply;
    AuditionCallback onAudition;
    StopAuditionCallback onStopAudition;
    PauseAuditionCallback onPauseAudition;

    juce::Label title;
    juce::Label sourceLabel;
    juce::ComboBox sourceSelector;
    ResettableSlider silenceDb, threshold, minSilence, minSegment, preRoll, postRoll, fade, rmsReject;
    ResettableSlider segmentStart, segmentEnd;
    juce::ToggleButton relativeToggle, stripToggle, trimToggle, rejectToggle, normalizeToggle;
    juce::Label segmentLabel;
    juce::TextButton prevSegment, nextSegment, playSegment, stopAudition, deleteSegment, restoreSegments, removeCurrentFile;
    juce::ToggleButton autoPlaySegment;
    WaveformPreview waveform;
    juce::TextButton apply, cancel, resetDefaults;

    bool updatingUi = false;
    bool restoringUndoState = false;
    bool sliderEditUndoCaptured = false;
    bool boundaryEditUndoCaptured = false;
    int boundaryEditSegment = -1;
    bool segmentAutoPlay = false;
    bool segmentAuditionActive = false;
    bool segmentAuditionPaused = false;
    int segmentAuditionFileIndex = -1;
    int segmentAuditionSegment = -1;
    double segmentAuditionEndMs = 0.0;
    double segmentAuditionRemainingMs = 0.0;
    std::vector<UndoState> undoStack;
    std::vector<UndoState> redoStack;
    int previewFileIndex = 0;
    int selectedSegment = -1;
    juce::AudioBuffer<float> previewOriginal;
    double previewSampleRate = 0.0;
    std::vector<SegmentRegion> previewSegments;
    juce::String previewStatus;
};

static inline juce::Rectangle<int> importPreviewUsableDisplayAreaFor (juce::Component& parent)
{
    const auto& displays = juce::Desktop::getInstance().getDisplays();
    if (auto* display = displays.getDisplayForRect (parent.getScreenBounds(), false))
        return display->userArea;

    if (auto* display = displays.getPrimaryDisplay())
        return display->userArea;

    return { 0, 0, 1280, 800 };
}

static inline juce::Rectangle<int> importPreviewDialogBoundsFor (juce::Component& parent, int desiredW, int desiredH)
{
    constexpr int edgeMargin = 18;
    auto userArea = importPreviewUsableDisplayAreaFor (parent).reduced (edgeMargin);
    if (userArea.isEmpty())
        userArea = { edgeMargin, edgeMargin, 1280 - edgeMargin * 2, 800 - edgeMargin * 2 };

    const int minW = juce::jmin (720, userArea.getWidth());
    const int minH = juce::jmin (520, userArea.getHeight());
    const int w = juce::jlimit (minW, userArea.getWidth(), desiredW);
    const int h = juce::jlimit (minH, userArea.getHeight(), desiredH);

    auto centre = parent.getScreenBounds().getCentre();
    if (! userArea.contains (centre))
        centre = userArea.getCentre();

    return juce::Rectangle<int> (w, h).withCentre (centre).constrainedWithin (userArea);
}

static inline void showImportPreviewDialog (juce::Component& parent,
                                           std::vector<juce::File> files,
                                           ImportAction action,
                                           ImportRules rules,
                                           ImportPreviewComponent::ApplyCallback onApply,
                                           ImportPreviewComponent::AuditionCallback onAudition = {},
                                           ImportPreviewComponent::StopAuditionCallback onStopAudition = {},
                                           ImportPreviewComponent::PauseAuditionCallback onPauseAudition = {})
{
    const bool segmentation = action == ImportAction::SegmentLongFile || action == ImportAction::SegmentThenMegaTexture;
    const auto bounds = importPreviewDialogBoundsFor (parent, segmentation ? 1180 : 1040, segmentation ? 760 : 700);
    auto* content = new ImportPreviewComponent (std::move (files), action, rules, std::move (onApply), std::move (onAudition), std::move (onStopAudition), std::move (onPauseAudition));
    content->setSize (bounds.getWidth(), bounds.getHeight());

    juce::DialogWindow::LaunchOptions opts;
    opts.dialogTitle = segmentation ? "Segmentation Preview" : "Import / Preprocess";
    opts.dialogBackgroundColour = juce::Colour (0xff20272d);
    opts.escapeKeyTriggersCloseButton = true;
    opts.useNativeTitleBar = true;
    opts.resizable = true;
    opts.content.setOwned (content);

    if (auto* window = opts.launchAsync())
    {
        window->setResizable (true, true);
        window->setResizeLimits (juce::jmin (720, bounds.getWidth()), juce::jmin (520, bounds.getHeight()), bounds.getWidth(), bounds.getHeight());
        window->setBounds (bounds);
    }
}

} // namespace za::fileimport

