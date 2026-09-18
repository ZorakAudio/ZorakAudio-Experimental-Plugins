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

#include <atomic>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
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
#include "ZAEventExtractor.h"

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
    // Automatic proposals are replaceable. User edits, confirms and rejections
    // are authoritative constraints, including disabled/deleted regions.
    bool locked = false;
    bool example = false;
    float eventScore = 0.8f, startBoundaryScore = 0.8f, endBoundaryScore = 0.8f;
    extractor::Reason reason = extractor::Reason::Quiet;

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
    int version = 2;

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
    // when editing the import again. Input indices refer to the immutable
    // sourceBindings order, including missing/disabled placeholders. Version 2
    // stores cuts in native-source samples with an explicit coordinate rate.
    std::vector<int> disabledInputIndices;
    std::vector<std::vector<SegmentRegion>> manualSegmentsByInput;

    // Source-native coordinate system, bound to fingerprints rather than a
    // filtered file-list position. Legacy recipes migrate when first reopened.
    std::vector<SourceFingerprint> sourceBindings;
    std::vector<double> manualSegmentSampleRates;
    std::vector<uint64_t> snapshotRevisions;
    uint64_t segmentationRevision = 1;
    bool assistedExtraction = false; // legacy recipes retain their old detector
    bool adaptiveBackground = true;
    double cutSensitivity = 0.5, wholeGestures = 0.65, tailPreservation = 0.65;
    bool useExamples = false;
    bool matchesOnly = false;
    int segmentationOutput = -1; // -1 legacy action; 0 separate samples; 1 texture
    double matchThreshold = 0.70;
    extractor::Profile exampleProfile;

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

static inline juce::ValueTree extractionProfileToValueTree (const extractor::Profile& profile)
{
    juce::ValueTree t ("ZA_EXTRACTION_PROFILE");
    t.setProperty ("featureVersion", profile.version, nullptr);
    t.setProperty ("name", juce::String::fromUTF8 (profile.name.c_str()), nullptr);
    for (const auto& e : profile.examples)
    {
        juce::ValueTree node ("EXAMPLE");
        node.setProperty ("name", juce::String::fromUTF8 (e.name.c_str()), nullptr);
        node.setProperty ("sourceId", juce::String::fromUTF8 (e.sourceId.c_str()), nullptr);
        node.setProperty ("start", e.sourceStart, nullptr);
        node.setProperty ("end", e.sourceEnd, nullptr);
        node.setProperty ("sourceRate", e.sourceRate, nullptr);
        node.setProperty ("duration", e.duration, nullptr);
        node.setProperty ("negative", e.negative, nullptr);
        node.setProperty ("lead", e.leadFraction, nullptr);
        node.setProperty ("tail", e.tailFraction, nullptr);
        for (const auto& f : e.frames)
        {
            juce::ValueTree frame ("FRAME");
            juce::StringArray values;
            for (float v : f.bands) values.add (juce::String (v, 9));
            frame.setProperty ("bands", values.joinIntoString (","), nullptr);
            frame.setProperty ("energy", (double) f.energy, nullptr);
            frame.setProperty ("onset", (double) f.onset, nullptr);
            frame.setProperty ("zcr", (double) f.zcr, nullptr);
            node.addChild (frame, -1, nullptr);
        }
        t.addChild (node, -1, nullptr);
    }
    return t;
}

static inline bool extractionProfileFromValueTree (const juce::ValueTree& t, extractor::Profile& profile)
{
    if (! t.hasType ("ZA_EXTRACTION_PROFILE") || (int) t.getProperty ("featureVersion", -1) != extractor::featureVersion
        || t.getNumChildren() > extractor::maxExamples)
        return false;
    extractor::Profile parsed;
    parsed.name = t.getProperty ("name").toString().toStdString();
    auto finite = [] (double v) { return std::isfinite (v); };
    for (const auto& node : t)
    {
        if (! node.hasType ("EXAMPLE") || node.getNumChildren() < 4 || node.getNumChildren() > extractor::maxExampleFrames) return false;
        extractor::Example e;
        e.name = node.getProperty ("name").toString().toStdString();
        e.sourceId = node.getProperty ("sourceId").toString().toStdString();
        e.sourceStart = (int) node.getProperty ("start", 0);
        e.sourceEnd = (int) node.getProperty ("end", 0);
        e.sourceRate = (double) node.getProperty ("sourceRate", 0.0);
        e.duration = (double) node.getProperty ("duration", 0.0);
        e.negative = (bool) node.getProperty ("negative", false);
        e.leadFraction = (float) (double) node.getProperty ("lead", 0.0);
        e.tailFraction = (float) (double) node.getProperty ("tail", 0.0);
        if (! finite (e.duration) || e.duration <= 0.0 || e.duration > 60.0 || ! finite (e.sourceRate)
            || ! finite (e.leadFraction) || ! finite (e.tailFraction)
            || e.leadFraction < 0.0f || e.leadFraction > 1.0f || e.tailFraction < 0.0f || e.tailFraction > 1.0f) return false;
        for (const auto& frame : node)
        {
            extractor::TemplateFrame f;
            const auto values = juce::StringArray::fromTokens (frame.getProperty ("bands").toString(), ",", "");
            if (! frame.hasType ("FRAME") || values.size() != extractor::bandCount) return false;
            for (int k = 0; k < extractor::bandCount; ++k)
            {
                const double v = values[k].getDoubleValue();
                if (! finite (v) || v < 0.0 || v > 1.001) return false;
                f.bands[(size_t) k] = (float) v;
            }
            // Stored spectra are unit-length (or all-zero for silence).
            // Normalise small serialization rounding error, reject other data.
            float norm = 0.0f;
            for (float value : f.bands) norm += value * value;
            if (norm > 1.0e-12f)
            {
                if (std::abs (norm - 1.0f) > 0.02f) return false;
                for (auto& value : f.bands) value /= std::sqrt (norm);
            }
            f.energy = (float) (double) frame.getProperty ("energy", 0.0);
            f.onset = (float) (double) frame.getProperty ("onset", 0.0);
            f.zcr = (float) (double) frame.getProperty ("zcr", 0.0);
            if (! finite (f.energy) || ! finite (f.onset) || ! finite (f.zcr)
                || f.energy < -1.001f || f.energy > 0.001f || f.onset < 0 || f.onset > 1 || f.zcr < 0 || f.zcr > 1) return false;
            e.frames.push_back (f);
        }
        parsed.examples.push_back (std::move (e));
    }
    profile = std::move (parsed);
    return true;
}

static inline void writeSourceFingerprint (juce::ValueTree& t, const SourceFingerprint& fp)
{
    t.setProperty ("path", fp.path, nullptr);
    t.setProperty ("sizeBytes", (juce::int64) fp.sizeBytes, nullptr);
    t.setProperty ("modifiedUtcMs", (juce::int64) fp.modifiedUtcMs, nullptr);
    t.setProperty ("quickHash", (juce::int64) fp.quickHash, nullptr);
}

static inline SourceFingerprint readSourceFingerprint (const juce::ValueTree& t)
{
    return { t.getProperty ("path").toString(), (juce::int64) t.getProperty ("sizeBytes", (juce::int64) 0),
             (juce::int64) t.getProperty ("modifiedUtcMs", (juce::int64) 0),
             (uint64_t) (juce::int64) t.getProperty ("quickHash", (juce::int64) 0) };
}

static inline bool sourceFingerprintMatches (const SourceFingerprint& expected, const SourceFingerprint& actual)
{
    // An unanalysed input has only its path bound. Once decoded, all saved
    // fingerprint fields must agree; changed files are never silently trusted.
    return expected.path == actual.path && (expected.sizeBytes <= 0
        || (expected.sizeBytes == actual.sizeBytes && expected.modifiedUtcMs == actual.modifiedUtcMs && expected.quickHash == actual.quickHash));
}

static inline juce::ValueTree rulesToValueTree (const ImportRules& r)
{
    juce::ValueTree t ("RULES");
    t.setProperty ("version", juce::jmax (2, r.version), nullptr);
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
    t.setProperty ("previewSeconds", r.previewSeconds, nullptr);
    t.setProperty ("assistedExtraction", r.assistedExtraction, nullptr);
    t.setProperty ("adaptiveBackground", r.adaptiveBackground, nullptr);
    t.setProperty ("cutSensitivity", r.cutSensitivity, nullptr);
    t.setProperty ("wholeGestures", r.wholeGestures, nullptr);
    t.setProperty ("tailPreservation", r.tailPreservation, nullptr);
    t.setProperty ("useExamples", r.useExamples, nullptr);
    t.setProperty ("matchesOnly", r.matchesOnly, nullptr);
    t.setProperty ("segmentationOutput", r.segmentationOutput, nullptr);
    t.setProperty ("matchThreshold", r.matchThreshold, nullptr);
    t.setProperty ("segmentationRevision", (juce::int64) r.segmentationRevision, nullptr);
    if (! r.exampleProfile.examples.empty()) t.addChild (extractionProfileToValueTree (r.exampleProfile), -1, nullptr);
    juce::ValueTree bindings ("SOURCE_BINDINGS");
    for (size_t i = 0; i < r.sourceBindings.size(); ++i)
    {
        juce::ValueTree source ("SOURCE");
        writeSourceFingerprint (source, r.sourceBindings[i]);
        source.setProperty ("sampleRate", i < r.manualSegmentSampleRates.size() ? r.manualSegmentSampleRates[i] : 0.0, nullptr);
        source.setProperty ("revision", (juce::int64) (i < r.snapshotRevisions.size() ? r.snapshotRevisions[i] : 0), nullptr);
        bindings.addChild (source, -1, nullptr);
    }
    if (bindings.getNumChildren() > 0) t.addChild (bindings, -1, nullptr);


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
                seg.setProperty ("locked", segment.locked, nullptr);
                seg.setProperty ("example", segment.example, nullptr);
                seg.setProperty ("eventScore", (double) segment.eventScore, nullptr);
                seg.setProperty ("startScore", (double) segment.startBoundaryScore, nullptr);
                seg.setProperty ("endScore", (double) segment.endBoundaryScore, nullptr);
                seg.setProperty ("reason", (int) segment.reason, nullptr);
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
    const double previewSeconds = (double) t.getProperty ("previewSeconds", r.previewSeconds);
    if (std::isfinite (previewSeconds) && previewSeconds > 0.0) r.previewSeconds = previewSeconds;
    r.assistedExtraction = (bool) t.getProperty ("assistedExtraction", false);
    r.adaptiveBackground = (bool) t.getProperty ("adaptiveBackground", true);
    auto readUnit = [&] (const char* name, double fallback)
    {
        const double value = (double) t.getProperty (name, fallback);
        return std::isfinite (value) ? juce::jlimit (0.0, 1.0, value) : fallback;
    };
    r.cutSensitivity = readUnit ("cutSensitivity", 0.5);
    r.wholeGestures = readUnit ("wholeGestures", 0.65);
    r.tailPreservation = readUnit ("tailPreservation", 0.65);
    r.matchThreshold = juce::jlimit (0.4, 0.95, readUnit ("matchThreshold", 0.70));
    r.useExamples = (bool) t.getProperty ("useExamples", false);
    r.matchesOnly = (bool) t.getProperty ("matchesOnly", false);
    r.segmentationOutput = juce::jlimit (-1, 1, (int) t.getProperty ("segmentationOutput", -1));
    r.segmentationRevision = (uint64_t) (juce::int64) t.getProperty ("segmentationRevision", (juce::int64) 1);
    if (! extractionProfileFromValueTree (t.getChildWithName ("ZA_EXTRACTION_PROFILE"), r.exampleProfile)) r.useExamples = false;
    if (auto bindings = t.getChildWithName ("SOURCE_BINDINGS"); bindings.isValid())
        for (const auto& source : bindings)
        {
            if (r.sourceBindings.size() >= 16384) break;
            r.sourceBindings.push_back (readSourceFingerprint (source));
            const double rate = (double) source.getProperty ("sampleRate", 0.0);
            r.manualSegmentSampleRates.push_back (std::isfinite (rate) && rate >= 0.0 ? rate : 0.0);
            r.snapshotRevisions.push_back ((uint64_t) (juce::int64) source.getProperty ("revision", (juce::int64) 0));
        }


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
            if (fileIndex < 0 || fileIndex >= 16384)
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
                // Old manual-only snapshots represent deliberate user edits.
                segment.locked = (bool) segNode.getProperty ("locked", true);
                segment.example = (bool) segNode.getProperty ("example", false);
                auto score = [&] (const char* name) { const double v = (double) segNode.getProperty (name, 0.8); return (float) (std::isfinite (v) ? juce::jlimit (0.0, 1.0, v) : 0.0); };
                segment.eventScore = score ("eventScore");
                segment.startBoundaryScore = score ("startScore");
                segment.endBoundaryScore = score ("endScore");
                segment.reason = (extractor::Reason) juce::jlimit (0, 6, (int) segNode.getProperty ("reason", 5));
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

    if (recipe.rules.sourceBindings.empty())
    {
        recipe.rules.sourceBindings = recipe.inputs;
        recipe.rules.manualSegmentSampleRates.resize (recipe.inputs.size(), recipe.rules.outputSampleRate);
        recipe.rules.snapshotRevisions.resize (recipe.inputs.size(), 0);
        for (size_t i = 0; i < recipe.rules.manualSegmentsByInput.size() && i < recipe.inputs.size(); ++i)
            if (! recipe.rules.manualSegmentsByInput[i].empty()) recipe.rules.snapshotRevisions[i] = recipe.rules.segmentationRevision;
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

static inline extractor::AudioView audioView (const juce::AudioBuffer<float>& b, double rate)
{
    return { b.getArrayOfReadPointers(), b.getNumChannels(), b.getNumSamples(), rate };
}

static inline extractor::Settings extractionSettings (const ImportRules& rules)
{
    extractor::Settings s;
    s.adaptive = rules.adaptiveBackground; s.silenceDb = (float) rules.silenceThresholdDb;
    s.sensitivity = (float) rules.cutSensitivity; s.gesture = (float) rules.wholeGestures; s.tails = (float) rules.tailPreservation;
    s.quietMs = rules.minSilenceMs; s.minimumMs = rules.minSegmentMs; s.maximumMs = rules.maxSegmentMs;
    s.preMs = rules.preRollMs; s.postMs = rules.postRollMs;
    return s;
}

static inline void storeSegmentSnapshot (ImportRules& rules, int index, const std::vector<SegmentRegion>& segments, double rate)
{
    if (index < 0) return;
    const size_t size = (size_t) index + 1;
    setManualSegmentsForInput (rules, index, segments);
    if (rules.manualSegmentSampleRates.size() < size) rules.manualSegmentSampleRates.resize (size, 0.0);
    if (rules.snapshotRevisions.size() < size) rules.snapshotRevisions.resize (size, 0);
    rules.manualSegmentSampleRates[(size_t) index] = rate;
    rules.snapshotRevisions[(size_t) index] = rules.segmentationRevision;
}

static inline std::vector<SegmentRegion> snapshotAtRate (const ImportRules& rules, int index, int samples, double rate)
{
    if (index < 0 || index >= (int) rules.manualSegmentsByInput.size()) return {};
    auto segments = rules.manualSegmentsByInput[(size_t) index];
    double savedRate = index < (int) rules.manualSegmentSampleRates.size() ? rules.manualSegmentSampleRates[(size_t) index] : rules.outputSampleRate;
    if (savedRate <= 0.0) savedRate = rate;
    for (auto& segment : segments)
    {
        if (std::abs (savedRate - rate) > 1.0e-6)
        {
            segment.startSample = (int) std::clamp (std::llround (segment.startSample * rate / savedRate), 0ll, (long long) samples);
            segment.endSample = (int) std::clamp (std::llround (segment.endSample * rate / savedRate), 0ll, (long long) samples);
        }
        segment.startSample = juce::jlimit (0, samples, segment.startSample);
        segment.endSample = juce::jlimit (segment.startSample, samples, segment.endSample);
        if (segment.length() <= 0) segment.enabled = false;
    }
    return segments;
}

static inline std::vector<SegmentRegion> preserveProtectedSegments (std::vector<SegmentRegion> automatic,
                                                                   const std::vector<SegmentRegion>& existing)
{
    return extractor::preserveConstraints (std::move (automatic), existing);
}

static inline std::vector<SegmentRegion> segmentsForInput (const ImportRules& rules,
                                                           int inputIndex,
                                                           const juce::AudioBuffer<float>& b,
                                                           double sr,
                                                           const extractor::Analysis* cached = nullptr,
                                                           const extractor::Cancel& cancel = {},
                                                           std::vector<extractor::Region>* reviewSpans = nullptr,
                                                           bool forceReanalyse = false)
{
    auto existing = snapshotAtRate (rules, inputIndex, b.getNumSamples(), sr);
    const bool hasSnapshot = inputIndex >= 0 && inputIndex < (int) rules.snapshotRevisions.size()
                          && rules.snapshotRevisions[(size_t) inputIndex] == rules.segmentationRevision;
    if (! forceReanalyse && hasSnapshot)
        return existing; // Empty is a valid reviewed result too; never resurrect it.
    if (! forceReanalyse && ! existing.empty() && rules.snapshotRevisions.empty()) return existing;

    extractor::Analysis analysed;
    const bool useProfile = rules.useExamples && rules.exampleProfile.version == extractor::featureVersion && rules.exampleProfile.positives() > 0;
    if ((rules.assistedExtraction || useProfile) && cached == nullptr)
    {
        analysed = extractor::analyse (audioView (b, sr), cancel);
        cached = &analysed;
    }
    if (extractor::cancelled (cancel)) return {};
    std::vector<SegmentRegion> proposals;
    auto makeRegion = [&] (const extractor::Region& r)
    {
        SegmentRegion s;
        s.startSample = r.start; s.endSample = r.end;
        s.eventScore = r.event; s.startBoundaryScore = r.startQuality; s.endBoundaryScore = r.endQuality; s.reason = r.reason;
        return s;
    };
    if (rules.assistedExtraction && cached != nullptr)
    {
        auto detection = extractor::detect (*cached, extractionSettings (rules), cancel);
        for (const auto& r : detection.regions) proposals.push_back (makeRegion (r));
        if (reviewSpans != nullptr) *reviewSpans = std::move (detection.reviewSpans);
    }
    else proposals = detectSegmentsBySilence (b, sr, rules);

    if (useProfile && cached != nullptr)
    {
        auto matches = extractor::findMatches (*cached, rules.exampleProfile, (float) rules.matchThreshold, cancel);
        for (auto& candidate : proposals)
        {
            if (extractor::cancelled (cancel)) return {};
            const auto features = extractor::makeExample (*cached, candidate.startSample, candidate.endSample);
            float positive = 0.0f, negative = 0.0f;
            for (const auto& example : rules.exampleProfile.examples)
                if (example.negative) negative = std::max (negative, extractor::similarity (example, features));
                else positive = std::max (positive, extractor::similarity (example, features));
            candidate.eventScore = extractor::contrastiveScore (positive, negative);
            if (rules.matchesOnly && candidate.eventScore < rules.matchThreshold) candidate.enabled = false;
        }
        // A complete example can span several preliminary cuts. Its proposal
        // replaces only overlapping *automatic* regions; locks are applied last.
        proposals.erase (std::remove_if (proposals.begin(), proposals.end(), [&] (const SegmentRegion& candidate)
        {
            for (const auto& match : matches)
                if (candidate.startSample < match.end && candidate.endSample > match.start) return true;
            return false;
        }), proposals.end());
        // Add as a batch: legitimate overlapping tails must not cause a later
        // accepted match to erase an earlier accepted event.
        for (const auto& match : matches) proposals.push_back (makeRegion (match));
    }
    for (auto& candidate : proposals)
    {
        if (extractor::cancelled (cancel)) return {};
        if (rules.assistedExtraction)
        {
            const int radius = juce::jmax (1, (int) std::llround (sr * 0.002));
            candidate.startSample = extractor::snapBoundary (audioView (b, sr), candidate.startSample, radius);
            candidate.endSample = extractor::snapBoundary (audioView (b, sr), candidate.endSample, radius);
        }
        candidate.endSample = juce::jlimit (candidate.startSample, b.getNumSamples(), candidate.endSample);
        if (cached != nullptr)
        {
            candidate.startBoundaryScore = extractor::boundaryQuality (*cached, candidate.startSample, true);
            candidate.endBoundaryScore = extractor::boundaryQuality (*cached, candidate.endSample, false);
        }
    }
    if (reviewSpans != nullptr)
        reviewSpans->erase (std::remove_if (reviewSpans->begin(), reviewSpans->end(), [&] (const extractor::Region& span)
        {
            for (const auto& s : existing) if (s.locked && span.start < s.endSample && span.end > s.startSample) return true;
            for (const auto& s : proposals) if (span.start < s.endSample && span.end > s.startSample) return true;
            return false;
        }), reviewSpans->end());
    auto result = preserveProtectedSegments (std::move (proposals), existing);
    for (auto& candidate : result)
    {
        if (extractor::cancelled (cancel)) return {};
        if (candidate.locked) continue;
        candidate.rmsDb = linearToDb (computeRmsLinear (b, candidate.startSample, candidate.length()));
        candidate.peakDb = linearToDb (computePeakLinear (b, candidate.startSample, candidate.length()));
        if (candidate.length() <= 0 || (rules.removeLowRms && candidate.rmsDb < rules.minRmsDb)) candidate.enabled = false;
    }
    return result;
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
            const float gIn = (float) i / (float) (fade - 1);
            const float gOut = gIn; // reverse index: the final sample must be zero
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
    if (inN <= 0 || in.getNumChannels() <= 0) return {};
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

static inline std::optional<AudioFileData> readAudioFile (const juce::File& file, int targetChannels, double targetRate, double maxSeconds, juce::String& error, const extractor::Cancel& cancel = {})
{
    juce::AudioFormatManager fm;
    fm.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor (file));
    if (reader == nullptr)
    {
        error = "Could not create audio reader for: " + file.getFullPathName();
        return std::nullopt;
    }

    if (! std::isfinite (reader->sampleRate) || reader->sampleRate <= 0.0 || reader->numChannels == 0)
    {
        error = "Invalid audio format metadata: " + file.getFileName();
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

    bool ok = true;
    for (int offset = 0; offset < readLen && ok; offset += 65536)
    {
        if (extractor::cancelled (cancel)) { error = "Analysis cancelled."; return std::nullopt; }
        ok = reader->read (&buffer, offset, juce::jmin (65536, readLen - offset), offset, true, true);
    }
    if (! ok)
    {
        error = "Read failed: " + file.getFileName();
        return std::nullopt;
    }

    const double sourceRate = reader->sampleRate > 0.0 ? reader->sampleRate : 44100.0;
    const double outRate = targetRate > 0.0 ? targetRate : sourceRate;
    if (targetChannels > 0 && targetChannels != buffer.getNumChannels()) buffer = convertChannels (buffer, targetChannels);
    if (std::abs (sourceRate - outRate) > 1.0e-6) buffer = resampleLinear (buffer, sourceRate, outRate);

    AudioFileData data;
    data.buffer = std::move (buffer);
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
    rules.assistedExtraction = rules.segmentBySilence;
    rules.trimEdges = true;
    rules.rejectNearDuplicates = (action == ImportAction::BuildMegaTexture);
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

static inline RenderResult renderImportActionImpl (const std::vector<juce::File>& inputFiles, ImportAction action, ImportRules rules)
{
    RenderResult result;
    if ((action == ImportAction::SegmentLongFile || action == ImportAction::SegmentThenMegaTexture) && rules.segmentationOutput >= 0)
        action = rules.segmentationOutput == 1 ? ImportAction::SegmentThenMegaTexture : ImportAction::SegmentLongFile;
    // Keep recipe placeholders in their original order, including missing files.
    auto files = rules.sourceBindings.empty() ? filterSupportedExistingFiles (inputFiles) : inputFiles;
    if (files.empty())
    {
        result.message = "No supported audio files were provided.";
        return result;
    }

    if (! rules.sourceBindings.empty() && rules.sourceBindings.size() != files.size())
    {
        result.message = "Recipe source order no longer matches the saved edits. Reopen the original recipe inputs.";
        return result;
    }
    if (rules.sourceBindings.empty()) rules.sourceBindings.resize (files.size());
    for (size_t i = 0; i < files.size(); ++i)
    {
        auto& binding = rules.sourceBindings[i];
        if (binding.path.isEmpty()) binding.path = files[i].getFullPathName();
        if (binding.path != files[i].getFullPathName())
        {
            result.message = "Recipe source order changed: " + files[i].getFileName();
            return result;
        }
        if (isInputIndexDisabled (rules, (int) i)) continue;
        if (! files[i].existsAsFile())
        {
            result.message = "Missing recipe source: " + binding.path + ". Restore the file or remove that source in the extractor.";
            return result;
        }
        const auto current = fingerprintForFile (files[i]);
        if (! sourceFingerprintMatches (binding, current))
        {
            result.message = "Source changed since its cuts were saved: " + files[i].getFileName() + ". Start a fresh import rather than applying old cuts.";
            return result;
        }
        binding = current;
    }
    if (rules.randomSeed == 0)
        rules.randomSeed = deterministicSeedForImport (files, action);

    result.recipe.action = action;
    result.recipe.rules = rules;
    result.recipe.seed = rules.randomSeed;
    result.recipe.displayName = "File Import Recipe";
    result.recipe.inputs = rules.sourceBindings;

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
            auto data = readAudioFile (f, 0, 0.0, 0.0, error);
            if (! data.has_value()) { result.message = error; result.renderedAudio.clear(); return result; }

            auto segments = segmentsForInput (rules, fileIndex, data->buffer, data->sampleRate);
            storeSegmentSnapshot (result.recipe.rules, fileIndex, segments, data->sampleRate);
            for (const auto& s : segments)
            {
                if (! s.enabled || s.length() <= 0)
                    continue;

                auto part = copyRange (data->buffer, s.startSample, s.endSample);
                applyEdgeFades (part, data->sampleRate, rules.edgeFadeMs);
                if (rules.normalizeClipsRms)
                {
                    const double rms = computeRmsLinear (part);
                    if (rms > 1.0e-9) part.applyGain ((float) (dbToLinear (rules.clipTargetRmsDb) / rms));
                }
                if (rules.outputChannels > 0 && part.getNumChannels() != rules.outputChannels) part = convertChannels (part, rules.outputChannels);
                const double partRate = rules.outputSampleRate > 0.0 ? rules.outputSampleRate : data->sampleRate;
                if (std::abs (partRate - data->sampleRate) > 1.0e-6) part = resampleLinear (part, data->sampleRate, partRate);
                result.renderedAudio.push_back (makeRenderedAudioData (std::move (part), partRate,
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
            auto data = readAudioFile (f, 0, 0.0, 0.0, error);
            if (! data.has_value()) { result.message = error; result.renderedAudio.clear(); return result; }

            auto segments = segmentsForInput (rules, fileIndex, data->buffer, data->sampleRate);
            storeSegmentSnapshot (result.recipe.rules, fileIndex, segments, data->sampleRate);
            int localPart = 1;
            for (const auto& s : segments)
            {
                if (! s.enabled || s.length() <= 0)
                    continue;

                auto part = copyRange (data->buffer, s.startSample, s.endSample);
                applyEdgeFades (part, data->sampleRate, rules.edgeFadeMs);
                if (rules.normalizeClipsRms)
                {
                    const double rms = computeRmsLinear (part);
                    if (rms > 1.0e-9) part.applyGain ((float) (dbToLinear (rules.clipTargetRmsDb) / rms));
                }
                if (rules.outputChannels > 0 && part.getNumChannels() != rules.outputChannels) part = convertChannels (part, rules.outputChannels);
                const double partRate = rules.outputSampleRate > 0.0 ? rules.outputSampleRate : data->sampleRate;
                if (std::abs (partRate - data->sampleRate) > 1.0e-6) part = resampleLinear (part, data->sampleRate, partRate);
                auto features = analyseAudioFeatures (part, partRate);
                // A confirmed kept segment is not silently pruned at export.
                if (! s.locked && ! clipPassesRules (features, clips, rules))
                    continue;

                clips.push_back ({ std::move (part), partRate,
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
        {
            if (std::abs (c.sampleRate - sr) > 1.0e-6)
            {
                auto converted = resampleLinear (c.buffer, c.sampleRate, sr);
                appendBuffer (mega, converted, sr, rules);
            }
            else appendBuffer (mega, c.buffer, sr, rules);
        }

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


// Do not let an allocation/decoder/analysis exception escape an import worker
// and terminate the host. All callers receive the existing result/error API.
static inline RenderResult renderImportAction (const std::vector<juce::File>& files, ImportAction action, ImportRules rules)
{
    try { return renderImportActionImpl (files, action, std::move (rules)); }
    catch (const std::exception& error)
    {
        RenderResult result; result.message = "Import failed: " + juce::String (error.what()); return result;
    }
    catch (...)
    {
        RenderResult result; result.message = "Import failed with an unexpected decoder or analysis error."; return result;
    }
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
        const auto rows = rowBounds();
        int row = 0;
        int distance = std::numeric_limits<int>::max();
        for (int i = 0; i < 4; ++i)
        {
            const int d = std::abs (rows[(size_t) i].getCentreY() - p.y);
            if (rows[(size_t) i].contains (p)) { row = i; break; }
            if (d < distance) { distance = d; row = i; }
        }
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

        const auto rows = rowBounds();

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
            auto r = rows[(size_t) i];
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
    std::array<juce::Rectangle<int>, 4> rowBounds() const
    {
        auto area = getLocalBounds().reduced (juce::jlimit (12, 44, getWidth() / 20));
        const int gap = 10;
        const int height = juce::jmax (1, (area.getHeight() - gap * 3) / 4);
        std::array<juce::Rectangle<int>, 4> rows;
        for (auto& row : rows) { row = area.removeFromTop (height); area.removeFromTop (gap); }
        return rows;
    }
    juce::Point<int> hoverPoint { -10000, -10000 };
};

} // namespace za::fileimport

#include "ZAImportPreview.h"
