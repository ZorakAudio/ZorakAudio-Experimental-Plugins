#include "JsfxGfxInput.h"
#include "JsfxGfxMenus.h"
#include "JsfxSliderDeclarations.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cerrno>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <type_traits>
#include <optional>

#if JUCE_WINDOWS
 // Keep Win32's min/max macros out of JUCE/std headers.
 // Without this, including the native file-dialog headers before PluginMarkdownHelp.h
 // breaks expressions such as std::min(...) / std::max(...) on MSVC.
 #ifndef NOMINMAX
  #define NOMINMAX 1
 #endif
 #include <windows.h>
 #include <shobjidl.h>
 #ifdef min
  #undef min
 #endif
 #ifdef max
  #undef max
 #endif
#endif

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "JSFXDSP.h"
#include "JsfxSharedCells.h"
#include "JsfxCompiledProgram.h"
#include "JsfxStateVariables.h"
#include "JsfxHeapMemory.h"
#include "JsfxProcessingSafety.h"
#include "JsfxLegacyAtomics.h"
#include "JsfxTasks.h"
#define JSFX_FAUST_IMPLEMENTATION
#include "JsfxFaust.h"
#undef JSFX_FAUST_IMPLEMENTATION
#if !defined(DSPJSFX_RUNTIME_STATE_ABI) || !((DSPJSFX_RUNTIME_STATE_ABI >= 3 && DSPJSFX_RUNTIME_STATE_ABI <= 6) || (DSPJSFX_RUNTIME_STATE_ABI >= 13 && DSPJSFX_RUNTIME_STATE_ABI <= 17))
 #error "Regenerate JSFXDSP.h and its AOT object with the patched dsp_jsfx_aot.py (runtime state ABI 3 through 6, 13 through 16, or shared native ABI 17)."
#endif
#include "PluginMarkdownHelp.h"
#include "ZAUnicodeText.h"
#include "ZAAudioImportRecipe.h"
#include "DspJsfxRuntime.h"
#include "DspJsfxSamplePool.h"
#include "JsfxHostEnvironment.h"
#include "JsfxGfxFramePool.h"
#include "JsfxGfxMemorySync.h"
#include "DspJsfxAudioFilePreflight.h"

// Back-compat: older generated headers may not export bus inference macros.
// Back-compat: accept newer AOT macro names too
#ifndef DSPJSFX_NUM_INPUTS
  #ifdef DSPJSFX_INPUT_CHANNELS
    #define DSPJSFX_NUM_INPUTS DSPJSFX_INPUT_CHANNELS
  #endif
#endif

#ifndef DSPJSFX_NUM_OUTPUTS
  #ifdef DSPJSFX_OUTPUT_CHANNELS
    #define DSPJSFX_NUM_OUTPUTS DSPJSFX_OUTPUT_CHANNELS
  #endif
#endif

#ifndef DSPJSFX_NUM_SIDECHAIN_INPUTS
  #ifdef DSPJSFX_INPUT_CHANNELS
    #ifdef DSPJSFX_OUTPUT_CHANNELS
      #define DSPJSFX_NUM_SIDECHAIN_INPUTS ((DSPJSFX_INPUT_CHANNELS) > (DSPJSFX_OUTPUT_CHANNELS) ? ((DSPJSFX_INPUT_CHANNELS)-(DSPJSFX_OUTPUT_CHANNELS)) : 0)
    #endif
  #endif
#endif

#ifndef DSPJSFX_USES_MIDI
  #define DSPJSFX_USES_MIDI 0
#endif

#ifndef DSPJSFX_ACCEPTS_MIDI_INPUT
  #define DSPJSFX_ACCEPTS_MIDI_INPUT 0
#endif

#ifndef DSPJSFX_PRODUCES_MIDI_OUTPUT
  #define DSPJSFX_PRODUCES_MIDI_OUTPUT 0
#endif

#ifndef DSPJSFX_PLUGIN_KIND
  #define DSPJSFX_PLUGIN_KIND "audio_effect"
#endif

#ifndef DSPJSFX_HAS_SAMPLE_SECTION
  #define DSPJSFX_HAS_SAMPLE_SECTION 1
#endif

#ifndef DSPJSFX_USES_SAMPLE_POOL
  #define DSPJSFX_USES_SAMPLE_POOL 0
#endif

#ifndef DSPJSFX_USES_LEGACY_FILE_IO
  #define DSPJSFX_USES_LEGACY_FILE_IO 0
#endif

#ifndef DSPJSFX_GFX_VAR_FLAG_TO_GFX
  #define DSPJSFX_GFX_VAR_FLAG_TO_GFX 1u
#endif

#ifndef DSPJSFX_GFX_VAR_FLAG_FROM_GFX
  #define DSPJSFX_GFX_VAR_FLAG_FROM_GFX 2u
#endif

#ifndef DSPJSFX_GFX_VAR_FLAGS_COUNT
  #define DSPJSFX_GFX_VAR_FLAGS_COUNT 0
  static const uint8_t DSPJSFX_GFX_VAR_FLAGS[1] = { 0 };
#endif

#ifndef DSPJSFX_STRING_LITERALS_COUNT
  typedef struct DSPJSFX_StringLiteralDesc { int64_t handle; int32_t length; const uint8_t* data; } DSPJSFX_StringLiteralDesc;
  #define DSPJSFX_STRING_LITERALS_COUNT 0
  static const DSPJSFX_StringLiteralDesc DSPJSFX_STRING_LITERALS[1] = { { 0, 0, nullptr } };
#endif

// Statically linked program; the same entrypoint contract is used by JIT.
static constexpr auto productionProgram=[] {
    za::jsfx::CompiledProgram<DSPJSFX_State> p{};
    p.init=&jsfx_init;p.slider=&jsfx_slider;p.process=&jsfx_process_block;
#if !DSPJSFX_HAS_FAUST
    p.block=&jsfx_block;p.sample=&jsfx_sample;
#endif
    return p;
}();

// ---- You must provide this symbol for AOT (declared in the generated header)
extern "C" void jsfx_ensure_mem (DSPJSFX_State* st, int64_t needed);


// JSFX source is generated into the build dir by build.py as JSFXSource.h.
// If it's missing, we fall back to empty text (no slider params).
#if defined(__has_include)
  #if __has_include("JSFXSource.h")
    #include "JSFXSource.h"
  #else
    static const char* kJsfxSourceText = R"JSFX()JSFX";
  #endif
#else
  static const char* kJsfxSourceText = R"JSFX()JSFX";
#endif

// ------------------------------
// Optional JSFX @gfx interpreter (WDL/YSFX EEL2)
//
// This is included as a single translation-unit chunk to keep the project
// monolithic. Keep the JsfxGfx* helper headers and WDL directory alongside it.
// ------------------------------
#include "YSFXGfxInterpreter.h"
#ifndef DSPJSFX_HAS_NATIVE_GFX
 #define DSPJSFX_HAS_NATIVE_GFX 0
#endif
#if DSPJSFX_HAS_NATIVE_GFX
 #include "NativeGfxPrototype.h"
#endif
#include "JsfxGfxResources.h"
#include "YSFXGfxCommCompat.h"
#include "WDL/fft.h"



class JSFXJuceProcessor;

// Host ownership and heap tracking live in DSPJSFX_State (no global hot-path registry).
namespace
{
static std::mutex gPersistentFileUiStateMutex;
#if ZA_SAMPLE_GFX_TEST_RUNNER
static std::atomic<uint64_t> sampleBaselineFrames { 0 };
#endif

static bool copyTrackPropertyName (const juce::String& name, juce::String& out)
{
    if (name.isEmpty())
        return false;

    out = name;
    return true;
}

static bool copyTrackPropertyName (const std::optional<juce::String>& name, juce::String& out)
{
    if (! name.has_value() || name->isEmpty())
        return false;

    out = *name;
    return true;
}

static inline int64_t getTrackedJsfxMemUsed (DSPJSFX_State* st) noexcept
{
    if (st == nullptr)
        return 0;

    const int64_t tracked = st->memUsed;

    // The DSP heap is intentionally retained across state resets/reloads. If a
    // previous run grew memN enough to cover @gfx history buffers, subsequent
    // writes inside that already-allocated range do not call jsfx_ensure_mem(),
    // so the high-water tracker can remain zero and the @gfx VM receives no
    // mirrored history. Treat an empty tracker as "mirror the current allocation"
    // rather than "mirror nothing"; buildGfxMirrorRanges() still bounds the copy.
    const int64_t effective = tracked > 0 ? tracked : st->memN;
    return std::max<int64_t> ((int64_t) 0, std::min<int64_t> (effective, st->memN));
}

static inline void noteTrackedJsfxMemUsed (DSPJSFX_State* st, int64_t usedEndExclusive) noexcept
{
    if (st == nullptr)
        return;

    const int64_t clamped = std::max<int64_t> ((int64_t) 0,
                                               std::min<int64_t> (usedEndExclusive, st->memN));

    auto& tracked = st->memUsed;
    if (clamped > tracked)
        tracked = clamped;
}

static inline int64_t parseJsfxDeclaredMaxMem (const char* jsfxText) noexcept
{
    if (jsfxText == nullptr)
        return 0;

    std::string text (jsfxText);
    size_t start = 0;

    const std::regex reOptions (R"(^\s*options\s*:\s*(.*)$)", std::regex::ECMAScript | std::regex::icase);
    const std::regex reMaxMem  (R"((?:^|[\s,])maxmem\s*=\s*([0-9]+(?:\.[0-9]+)?))", std::regex::ECMAScript | std::regex::icase);

    while (start < text.size())
    {
        size_t end = text.find_first_of ("\r\n", start);
        if (end == std::string::npos)
            end = text.size();

        std::string line = text.substr (start, end - start);

        size_t next = end;
        while (next < text.size() && (text[next] == '\r' || text[next] == '\n'))
            ++next;
        start = next;

        std::smatch m;
        if (! std::regex_match (line, m, reOptions))
            continue;

        const std::string options = m[1].str();
        std::smatch mm;
        if (! std::regex_search (options, mm, reMaxMem))
            continue;

        const double parsed = std::strtod (mm[1].str().c_str(), nullptr);
        if (parsed <= 0.0)
            return 0;

        return (int64_t) std::floor (parsed + 1.0e-9);
    }

    return 0;
}

static inline int64_t getGfxLogicalJsfxMemN (DSPJSFX_State* st, int64_t declaredMaxMem) noexcept
{
    const int64_t tracked = getTrackedJsfxMemUsed (st);

    if (st == nullptr)
        return tracked;

    // ZA-GFX-MEM-SYNC:
    // memUsed can be positive but stale. The AOT core only calls
    // jsfx_ensure_mem() when an access crosses the current allocation; later
    // JSFX writes inside that already-grown heap may not advance the high-water
    // tracker. Do not let such a stale positive value truncate the normal low
    // @gfx shared prefix, where meters/scopes/analyzer histories usually live.
    const int64_t allocatedLowPrefix =
        std::min<int64_t> (st->memN, (int64_t) kGfxSharedPrefixDoubles);

    int64_t logical = std::max<int64_t> (tracked, allocatedLowPrefix);

    if (declaredMaxMem > 0)
    {
        logical = std::max<int64_t> (logical,
                                     std::min<int64_t> (declaredMaxMem, st->memN));
    }

    return std::max<int64_t> ((int64_t) 0,
                              std::min<int64_t> (logical, st->memN));
}

static inline uint8_t getJsfxGfxVarFlags (int index) noexcept
{
    if (index < 0)
        return (uint8_t) (DSPJSFX_GFX_VAR_FLAG_TO_GFX | DSPJSFX_GFX_VAR_FLAG_FROM_GFX);

    if (index < (int) DSPJSFX_GFX_VAR_FLAGS_COUNT)
        return DSPJSFX_GFX_VAR_FLAGS[index];

    return (uint8_t) (DSPJSFX_GFX_VAR_FLAG_TO_GFX | DSPJSFX_GFX_VAR_FLAG_FROM_GFX);
}

// -----------------------------------------------------------------------------
// Bus inference
//
// The AOT header (JSFXDSP.h) exports macros computed from splN usage:
//   DSPJSFX_NUM_INPUTS, DSPJSFX_NUM_OUTPUTS, DSPJSFX_NUM_SIDECHAIN_INPUTS
//
// We expose these as JUCE buses:
//   - Main input bus channels == output channels (audible path)
//   - Optional sidechain input bus for any extra input channels
//
// Channel order presented to the DSP core is:
//   spl0..spl(out-1)   = main input
//   spl(out)..         = sidechain / extra inputs
// -----------------------------------------------------------------------------

static bool parseJsfxExplicitGfxSyncPolicy (const char* jsfxText)
{
    if (jsfxText == nullptr) return false;
    const std::regex re (R"(^\s*//\s*@za:gfx_sync_policy\s*:?\s*(EXPLICIT|AUTO)\s*(?://.*)?$)",
                         std::regex::ECMAScript | std::regex::icase);
    std::istringstream lines (jsfxText);
    std::string line;
    bool explicitOnly = false;
    while (std::getline (lines, line))
    {
        std::smatch m;
        if (std::regex_match (line, m, re))
        {
            auto token = m[1].str();
            for (auto& c : token) c = (char) std::toupper ((unsigned char) c);
            explicitOnly = token == "EXPLICIT";
        }
    }
    return explicitOnly;
}

static std::vector<GfxSyncMemRange> parseJsfxGfxSyncMemRanges (const char* jsfxText)
{
    std::vector<GfxSyncMemRange> out;

    if (jsfxText == nullptr)
        return out;

    // Optional source/runtime metadata, for scripts that keep display data
    // outside the automatic low-prefix/high-suffix mirror:
    //   // @za:gfx_sync_mem 131200 4096 DSP_TO_GFX
    //   // @za:gfx_sync_mem 135360 4096 DSP_TO_GFX
    //   // @za:gfx_sync_mem 139520 4096 DSP_TO_GFX
    // Direction defaults to DSP_TO_GFX. Supported direction tokens:
    // DSP_TO_GFX, TO_GFX, FROM_GFX, GFX_TO_DSP, BIDIR, BIDIRECTIONAL, BOTH.
    const std::regex reSync (
        R"(^\s*//\s*@za:gfx_sync_mem\s*:?\s*([0-9]+)\s*(?:,|\s)\s*([0-9]+)(?:\s*(?:,|\s)\s*([A-Za-z0-9_\-]+))?.*$)",
        std::regex::ECMAScript | std::regex::icase);

    std::string text (jsfxText);
    size_t start = 0;

    while (start < text.size())
    {
        size_t end = text.find_first_of ("\r\n", start);

        if (end == std::string::npos)
            end = text.size();

        const std::string line = text.substr (start, end - start);

        size_t next = end;
        while (next < text.size() && (text[next] == '\r' || text[next] == '\n'))
            ++next;

        start = next;

        std::smatch m;

        if (! std::regex_match (line, m, reSync))
            continue;

        const int64_t base = (int64_t) std::strtoll (m[1].str().c_str(), nullptr, 10);
        const int64_t count = (int64_t) std::strtoll (m[2].str().c_str(), nullptr, 10);

        if (base < 0 || count <= 0)
            continue;

        const uint8_t flags = m[3].matched
            ? parseGfxSyncMemDirectionToken (trimAscii (m[3].str()))
            : kGfxSyncToGfx;

        out.push_back (GfxSyncMemRange { base, count, flags });
    }

    return out;
}



} // namespace

// ---- JSFX FFT runtime helpers ----------------------------------------------
// JSFX complex FFTs operate on st->mem as an interleaved complex buffer:
//   mem[base + 2*i + 0] = real
//   mem[base + 2*i + 1] = imag
//
// JSFX real FFTs (fft_real()/ifft_real()) operate on size real values in
// place, exposing size/2 packed complex bins in the same region, matching the
// WDL/JSFX convention where the first output pair stores DC and Nyquist.
//
// We back these helpers with Cockos/WDL FFT, which is already compiled into
// the plugin build. In strict mode, semantics match REAPER/JSFX:
//   - fft()/ifft()/fft_real()/ifft_real() operate on WDL's permuted order
//   - fft_permute() converts FFT output to natural order
//   - fft_ipermute() converts natural-order bins back to the order ifft() expects
//   - convolve_c() multiplies complex bins in-place
//
// For back-compat with older AOT builds that assumed in-order fft()/ifft() and
// no-op permute helpers, define ZA_JSFX_FFT_LEGACY_IN_ORDER=1.
#include "JsfxNumericBuiltins.h"

#include "JsfxMemoryBuiltins.h"

#include "JsfxMidiBuiltins.h"
#include "JsfxMidiHost.h"
#include "JsfxIdleRuntime.h"
#include "JsfxSliderBuiltins.h"

#include "JSFXCorrectnessCheck.h"

class JSFXJuceEditor;


#include "JsfxFileRuntime.h"
#include "JsfxStateReset.h"
#include "JsfxParameterState.h"
#include "JsfxAudioHost.h"

class JSFXJuceProcessor final : public juce::AudioProcessor,
                                public za::jsfx::FileRuntime,
                                public za::jsfx::MidiHostRuntime,
                                public za::jsfx::IdleRuntime,
                                public za::jsfx::AudioHostRuntime,
                                public za::jsfx::ParameterState,
                                private juce::AsyncUpdater,
                                private juce::AudioProcessorValueTreeState::Listener
{
public:
    using SliderMask = jsfx_gfx::SliderMask;

    static const char* kOversamplingParamId() noexcept { return "ZA_INTERNAL_OVERSAMPLING"; }

    // Per-slider runtime metadata so we can map JUCE parameters back to the JSFX
    // numeric slider values (especially important for step{A,B,C} enums).
    struct SliderParamInfo
    {
        juce::String pid;
        float min  = 0.0f;
        float max  = 1.0f;
        float rangeStart = 0.0f;
        float rangeEnd   = 1.0f;
        float step = 1.0f;
        bool reversed = false;
        bool isChoice = false;
        float shapeModifier = 0.0f;


        JsfxSliderDecl::Shape shape = JsfxSliderDecl::Shape::Linear;
    };

    // ---- File I/O runtime -------------------------------------------------
    //
    // We support a subset of REAPER JSFX's file_*() APIs by routing them to a
    // background loader thread and exposing the decoded contents to the DSP
    // thread as an in-memory array of doubles.
    //
    // Design goals:
    //   - No disk I/O on the audio thread.
    //   - File loads are promoted at block boundaries (stable view per block).
    //   - API-compatible surface: file_open/file_mem/file_var/file_avail/etc.
    // ----------------------------------------------------------------------

    struct FileSelectionState
    {
        std::vector<juce::String> paths;
        FileLoadMode loadMode = FileLoadMode::SeparateEntries;

        // Optional serialized import recipe. Source paths plus deterministic rules are
        // saved; processed recipe output is rendered directly into the in-memory
        // file-slot cache for replay without temp files.
        juce::String importRecipeXml;
    };

    struct FavoriteFileSelection
    {
        juce::String id;
        juce::String name;
        juce::String category;
        FileSelectionState selection;
        int64_t addedUtcMs = 0;
    };

    struct ImportPreviewAuditionClip
    {
        juce::AudioBuffer<float> buffer;
        int position = 0;
    };

    static za::jsfx::FileRuntime* taskOwner(DSPJSFX_State* state)noexcept {return state?static_cast<JSFXJuceProcessor*>(state->hostOwner):nullptr;}

    JSFXJuceProcessor()
        : juce::AudioProcessor (
              []()
              {
                  auto setForCh = [] (int ch) -> juce::AudioChannelSet
                  {
                      if (ch <= 1) return juce::AudioChannelSet::mono();
                      if (ch == 2) return juce::AudioChannelSet::stereo();
                      return juce::AudioChannelSet::discreteChannels (ch);
                  };

                  const int outCh = juce::jlimit (0, 64, (int) DSPJSFX_NUM_OUTPUTS);
                  const int inCh  = juce::jlimit (0, 64, (int) DSPJSFX_NUM_INPUTS);
                  const int mainInCh = juce::jmin (inCh, juce::jmax (0, outCh));
                  const int scCh  = juce::jmax (0, inCh - mainInCh);

                  auto bp = juce::AudioProcessor::BusesProperties();
                  if (mainInCh > 0)
                      bp = bp.withInput ("Input", setForCh (mainInCh), true);
                  if (outCh > 0)
                      bp = bp.withOutput ("Output", setForCh (outCh), true);
                  if (scCh > 0)
                      bp = bp.withInput ("Sidechain", setForCh (scCh), false);

                  return bp;
              }())

    {
        // Parse slider declarations + optional UI metadata from the JSFX source.
        // Tooltips:  // #TOOLTIP: ... (immediately above a slider line)
        // Help:      // #HELP: ...    (deprecated; the embedded README.md now drives the ? panel)
        sliderDecls = parseJsfxSliderDecls (kJsfxSourceText, nullptr);
        sliderStringUsed.fill (false);
        for (auto& text : stringSliderTexts)
            text = {};
        stringSliderAppliedSeq.fill (0);
        for (auto& seq : stringSliderSeq)
            seq.store (1u, std::memory_order_release);
        for (const auto& s : sliderDecls)
        {
            if (s.isString && s.index0 >= 0 && s.index0 < DSPJSFX_MAX_SLIDERS)
            {
                sliderStringUsed[(size_t) s.index0] = true;
                stringSliderTexts[(size_t) s.index0] = s.stringDefault;
            }
        }

        embeddedReadmeMarkdown = za::pluginui::getEmbeddedPluginReadmeMarkdown();
        if (embeddedReadmeMarkdown.isEmpty())
            embeddedReadmeMarkdown = za::pluginui::fallbackReadmeMarkdown (getName());

        // Parse filename declarations (REAPER JSFX 'filename:N,...').
        // These become DSP-JSFX file slots that can be bound to user-selected files at runtime.
        fileDecls = parseJsfxFilenameDecls (kJsfxSourceText);
        initFileRuntime();
        initSmartIdleConfig();

        // Build slider-alias -> state var index map.
        // JSFX syntax supports: sliderN:someVar=default<...>Label
        // In REAPER JSFX, someVar reflects the slider value.
        sliderAliasVarIndex.fill (-1);

        std::unordered_map<std::string, int> varIndexByName;
        varIndexByName.reserve ((size_t) DSPJSFX_VARS_COUNT);

        for (int vi = 0; vi < DSPJSFX_VARS_COUNT; ++vi)
            if (DSPJSFX_VARS[vi].name != nullptr)
                varIndexByName.emplace (juce::String (DSPJSFX_VARS[vi].name).toLowerCase().toStdString(),
                                        DSPJSFX_VARS[vi].index);

        for (const auto& s : sliderDecls)
        {
            if (s.index0 < 0 || s.index0 >= DSPJSFX_MAX_SLIDERS)
                continue;

            if (s.varName.isEmpty())
                continue;

            if (auto it = varIndexByName.find (s.varName.toLowerCase().toStdString()); it != varIndexByName.end())
                sliderAliasVarIndex[(size_t) s.index0] = it->second;
        }

        // Reset runtime mapping tables (we only create parameters for declared sliders)
        sliderParamUsed.fill (false);

        juce::AudioProcessorValueTreeState::ParameterLayout layout;
        for (const auto& s : sliderDecls)
        {
            if (s.isString)
                continue;

            const auto pid = sanitizeId (s.id);

            // Record how this slider maps back into JSFX runtime values.
            // For AudioParameterChoice we store the original JSFX numeric min/max/step so we can
            // convert a choice index -> (min + index*step) at runtime.
            {
                SliderParamInfo info;
                info.pid      = pid;
                info.min        = s.min;
                info.max        = s.max;
                info.rangeStart = s.rangeStart;
                info.rangeEnd   = s.rangeEnd;
                info.step       = (s.step > 0.0f ? s.step : 1.0f);
                info.reversed   = s.reversed;
                info.isChoice   = (s.isChoice && s.choices.size() > 0);
                info.shape = s.shape;
                info.shapeModifier = s.shapeModifier;
                sliderParamInfo[(size_t) s.index0] = info;
                sliderParamUsed[(size_t) s.index0] = true;
            }

            if (s.isChoice && s.choices.size() > 0)
            {
                // JSFX syntax: step{A,B,C} means *discrete* values, but the JSFX slider variable
                // is still numeric in the original range.
                //
                // We expose it as a CHOICE so hosts show the textual options.
                // Then at runtime we convert choice-index -> JSFX numeric value.
                const float step = (s.step > 0.0f ? s.step : 1.0f);
                const float signedStep = s.reversed ? -step : step;
                int defIdx = (int) std::llround ((s.def - s.rangeStart) / signedStep);
                defIdx = juce::jlimit (0, s.choices.size() - 1, defIdx);

                layout.add (std::make_unique<juce::AudioParameterChoice> (pid, s.name, s.choices, defIdx));
            }
            else
            {
                // JUCE requires an ascending numeric domain, but JSFX range
                // declaration order controls the slider direction. Keep the
                // actual parameter values in [min,max] while custom normalising
                // against the original endpoints. Thus <0,-340,...> maps
                // normalised 0 -> 0 and 1 -> -340, exactly like REAPER.
                auto toValue = [sh = s.shape, mod = s.shapeModifier,
                                declaredStart = s.rangeStart, declaredEnd = s.rangeEnd]
                               (float /*start*/, float /*end*/, float t) -> float
                {
                    t = clamp01f (t);

                    if (sh == JsfxSliderDecl::Shape::Sqr)
                    {
                        const float exp = (mod > 0.0f ? mod : 2.0f);
                        return curveFrom01_sqr (t, declaredStart, declaredEnd, exp);
                    }

                    if (sh == JsfxSliderDecl::Shape::Log)
                        return curveFrom01_log (t, declaredStart, declaredEnd, mod);

                    return declaredStart + t * (declaredEnd - declaredStart);
                };

                auto toNorm = [sh = s.shape, mod = s.shapeModifier,
                               declaredStart = s.rangeStart, declaredEnd = s.rangeEnd]
                              (float /*start*/, float /*end*/, float v) -> float
                {
                    if (sh == JsfxSliderDecl::Shape::Sqr)
                    {
                        const float exp = (mod > 0.0f ? mod : 2.0f);
                        return curveTo01_sqr (v, declaredStart, declaredEnd, exp);
                    }

                    if (sh == JsfxSliderDecl::Shape::Log)
                        return curveTo01_log (v, declaredStart, declaredEnd, mod);

                    if (declaredEnd == declaredStart) return 0.0f;
                    return clamp01f ((v - declaredStart) / (declaredEnd - declaredStart));
                };

                const float signedStep = s.reversed ? -s.step : s.step;
                auto snap = [st = signedStep, origin = s.rangeStart, lo = s.min, hi = s.max]
                            (float /*start*/, float /*end*/, float v) -> float
                {
                    v = juce::jlimit (lo, hi, v);
                    if (st == 0.0f) return v;

                    const float q = std::round ((v - origin) / st);
                    const float snapped = origin + q * st;
                    return juce::jlimit (lo, hi, snapped);
                };

                juce::NormalisableRange<float> range (s.min, s.max, toValue, toNorm, snap);
                auto stringFromValue = [st = s.step] (float v, int /*maxLen*/) -> juce::String
                {
                    // derive decimals from step: 1 -> 0 dp, 0.1 -> 1 dp, 0.01 -> 2 dp, etc
                    int decimals = 0;
                    if (st > 0.0f)
                    {
                        float t = st;
                        // cap at 6 decimals to avoid insanity
                        while (decimals < 6 && t < 1.0f)
                        {
                            t *= 10.0f;
                            ++decimals;
                            // stop early if step is now effectively integer
                            if (std::abs(t - std::round(t)) < 1.0e-6f) break;
                        }
                    }

                    // trim trailing zeros behavior is already good enough with fixed decimals
                    return juce::String (v, decimals);
                };

                auto valueFromString = [] (const juce::String& text) -> float
                {
                    return text.getFloatValue();
                };

                layout.add (std::make_unique<juce::AudioParameterFloat>(
                    pid, s.name, range, s.def, juce::String(),
                    juce::AudioProcessorParameter::genericParameter,
                    stringFromValue, valueFromString
                ));
            }
        }

        layout.add (std::make_unique<juce::AudioParameterChoice> (
            kOversamplingParamId(),
            "Oversampling",
            juce::StringArray { "Off", "2x", "4x", "8x" },
            0));

        apvts = std::make_unique<juce::AudioProcessorValueTreeState> (*this, nullptr, "PARAMS", std::move (layout));

        paramAtomics.fill (nullptr);
        for (size_t i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! sliderParamUsed[i])
                continue;
            const auto& info = sliderParamInfo[i];
            paramAtomics[i] = apvts->getRawParameterValue (info.pid);
        }

        oversamplingParamAtomic = apvts->getRawParameterValue (kOversamplingParamId());
        registerParameterWakeListeners();

        jsfxDeclaredMaxMem = parseJsfxDeclaredMaxMem (kJsfxSourceText);
        gfxSyncMemRanges = parseJsfxGfxSyncMemRanges (kJsfxSourceText);
        gfxSyncExplicitOnly = parseJsfxExplicitGfxSyncPolicy (kJsfxSourceText);
        gfxProfileEnabled = juce::SystemStats::getEnvironmentVariable ("ZA_GFX_PROFILE", "0").getIntValue() != 0;
        if (gfxSyncMemRanges.size() > (size_t) kMaxGfxSyncDeclarations)
            juce::Logger::writeToLog ("GFX: too many gfx_sync_mem declarations (maximum 30); memory transfer disabled.");

        initStateMemory();
        jsfxRuntime.attachToState (&st);
        st.hostOwner = this;
#if DSPJSFX_HAS_TASKS
        st.taskContext = taskRuntime.get();
        taskHeap.state=&st;taskHeap.capacity=DSPJSFX_MAX_MEM_CELLS;
#if DSPJSFX_NATIVE_GFX_LEGACY
        taskHeap.lifecycleMutex=&legacyLifecycleMutex;
        taskHeap.rebind=[this](DSPJSFX_State* state){if(legacyFrame)legacyFrame->bindLegacy(*state,DSPJSFX_NATIVE_GFX_WIDTH,DSPJSFX_NATIVE_GFX_HEIGHT);};
#endif
        za::jsfx::TaskRuntimeHooks<&JSFXJuceProcessor::taskOwner>::bind(*taskRuntime);
#endif
        initGfxSnapshots();
        resetGfxSliderPreviewState();

       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        correctnessRuntime = std::make_unique<jsfx_correctness::Runtime> (this);
       #endif
    }

    ~JSFXJuceProcessor() override
    {
        processorShuttingDown.store (true, std::memory_order_release);
        cancelPendingUpdate();
        unregisterParameterWakeListeners();

        // Drain the shadow VM's atexit while file/sample callback targets still live.
       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        correctnessRuntime.reset();
       #endif

        // Host lifecycle must have stopped processing before processor destruction.
        st.hostOwner = nullptr;
#if DSPJSFX_HAS_TASKS
        taskRuntime.reset(); // Join workers before releasing instance resources.
        st.taskContext = nullptr;
#endif

        shutdownFileRuntime();
        jsfxRuntime.detachFromState();


        za::jsfx::releaseHeap(st);
    }

    juce::AudioProcessorValueTreeState& getAPVTS() noexcept { return *apvts; }
#if DSPJSFX_HAS_TASKS
    void* taskContext() noexcept { return taskRuntime.get(); }
#endif

    void registerParameterWakeListeners()
    {
        if (apvts == nullptr)
            return;

        for (size_t i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
            if (sliderParamUsed[i] && sliderParamInfo[i].pid.isNotEmpty())
                apvts->addParameterListener (sliderParamInfo[i].pid, this);

        apvts->addParameterListener (kOversamplingParamId(), this);
        parameterWakeListenersRegistered = true;
    }

    void unregisterParameterWakeListeners()
    {
        if (apvts == nullptr || ! parameterWakeListenersRegistered)
            return;

        for (size_t i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
            if (sliderParamUsed[i] && sliderParamInfo[i].pid.isNotEmpty())
                apvts->removeParameterListener (sliderParamInfo[i].pid, this);

        apvts->removeParameterListener (kOversamplingParamId(), this);
        parameterWakeListenersRegistered = false;
    }

    // Called by the UI thread (@gfx) to push persistent VM state back to the DSP VM.
    void enqueueGfxVarWrite (int index, double value) noexcept { enqueueGfxStateWrite (GfxStateWrite::Kind::Var, index, value); }
    void enqueueGfxMemWrite (int index, double value) noexcept { enqueueGfxStateWrite (GfxStateWrite::Kind::Mem, index, value); }

    // Bulk GFX->DSP memory lane. The producer is the dedicated GFX worker and
    // the consumer is the audio thread. Ring-slot vector storage persists, so
    // the audio thread performs only memcpy/validation and never allocates or
    // destroys the transferred payload. force=true is reserved for host
    // operations such as GFX file_mem(), which are legitimate JSFX memory
    // writes even when the range was not known to the static mirror policy.
    bool enqueueGfxMemSpanWrite (int64_t base, const double* values, int count, bool force = false) noexcept
    {
        if (base < 0 || values == nullptr || count <= 0)
            return false;

        const uint32_t head = gfxMemSpanHead.load (std::memory_order_relaxed);
        const uint32_t next = (head + 1u) & gfxMemSpanQueueMask;
        if (next == gfxMemSpanTail.load (std::memory_order_acquire))
            return false;

        auto& slot = gfxMemSpanQueue[head];
        try
        {
            slot.data.resize ((size_t) count);
        }
        catch (...)
        {
            return false;
        }

        std::memcpy (slot.data.data(), values, (size_t) count * sizeof (double));
        slot.base = base;
        slot.count = count;
        slot.force = force;
        gfxMemSpanHead.store (next, std::memory_order_release);
        return true;
    }

    // Stage low-latency @gfx slider writes for the audio thread immediately.
    //
    // Host parameter notification still goes through applyGfxSliderChanges() on
    // the message thread, but this shadow lane lets the audio thread pick up the
    // latest UI-authored value on the very next block instead of waiting for the
    // host/APVTS round-trip to complete.
    void stageGfxSliderPreview (const double* newSliders, int count,
                                const SliderMask& changeMask,
                                const SliderMask& automateMask,
                                const SliderMask& automateEndMask) noexcept
    {
        if (newSliders == nullptr)
            return;

        const int n = juce::jlimit (0, DSPJSFX_MAX_SLIDERS, count);
        const auto applyMask = changeMask | automateMask | automateEndMask;
        if (! applyMask.any())
            return;

        bool stagedAny = false;
        for (int i = 0; i < n; ++i)
        {
            if (! applyMask.test (i))
                continue;
            if (! sliderParamUsed[(size_t) i] || ! std::isfinite (newSliders[(size_t) i]))
                continue;

            gfxSliderPreviewValues[(size_t) i].store (newSliders[(size_t) i], std::memory_order_release);
            (void) gfxSliderPreviewSeq[(size_t) i].fetch_add (1u, std::memory_order_acq_rel);
            stagedAny = true;
        }

        if (stagedAny)
        {
            gfxSnapshotForcePublish.store (true, std::memory_order_release);
            requestExternalWakeEvent (true);
        }
    }

    void syncGfxSliderAliasVarsToSliderValues (double* sliders,
                                               int sliderCount,
                                               const double* varsBefore,
                                               const double* varsAfter,
                                               int varsCount,
                                               const double* effectiveSliders,
                                               const SliderMask& notifyMask) const noexcept
    {
        if (sliders == nullptr || varsAfter == nullptr || effectiveSliders == nullptr || ! notifyMask.any())
            return;

        const int n = juce::jlimit (0, DSPJSFX_MAX_SLIDERS, sliderCount);

        for (int i = 0; i < n; ++i)
        {
            if (! notifyMask.test (i))
                continue;

            if (! sliderParamUsed[(size_t) i] || sliderStringUsed[(size_t) i])
                continue;

            const int vIdx = sliderAliasVarIndex[(size_t) i];
            if (vIdx < 0 || vIdx >= varsCount)
                continue;

            const auto sameDouble = [] (double a, double b) noexcept
            {
                if (a == b)
                    return true;
                if (std::isnan (a) && std::isnan (b))
                    return true;
                return std::abs (a - b) <= 1.0e-12;
            };

            // Direct slider()/sliderN writes already update the VM's slider array.
            // Only synthesize a slider-array update when the slider itself stayed
            // at the incoming snapshot value but its alias variable changed.
            if (! sameDouble (sliders[i], effectiveSliders[i]))
                continue;

            if (varsBefore != nullptr && vIdx < varsCount && sameDouble (varsBefore[vIdx], varsAfter[vIdx]))
                continue;

            double v = varsAfter[vIdx];
            if (! std::isfinite (v))
                continue;

            const auto& info = sliderParamInfo[(size_t) i];
            v = juce::jlimit<double> ((double) info.min, (double) info.max, v);

            const double step = (double) (info.step > 0.0f ? info.step : (info.isChoice ? 1.0f : 0.0f));
            if (step > 0.0)
            {
                const double signedStep = info.reversed ? -step : step;
                const double q = std::llround ((v - (double) info.rangeStart) / signedStep);
                v = (double) info.rangeStart + q * signedStep;
                v = juce::jlimit<double> ((double) info.min, (double) info.max, v);
            }

            sliders[i] = v;
        }
    }

    void registerGfxSnapshotClient() noexcept
    {
        gfxSnapshotUsers.fetch_add (1, std::memory_order_acq_rel);
        gfxSnapshotForcePublish.store (true, std::memory_order_release);
    }

    void unregisterGfxSnapshotClient() noexcept
    {
        const int prev = gfxSnapshotUsers.fetch_sub (1, std::memory_order_acq_rel);
        if (prev <= 1)
            gfxSnapshotUsers.store (0, std::memory_order_release);
    }

    const juce::String getName() const override
    {
    #if defined(ZA_PLUGIN_NAME)
        return ZA_PLUGIN_NAME;   // human name (can contain spaces/parentheses)
    #else
        return JucePlugin_Name;  // fallback
    #endif
    }
    bool acceptsMidi() const override { return DSPJSFX_ACCEPTS_MIDI_INPUT != 0; }
    bool producesMidi() const override { return DSPJSFX_PRODUCES_MIDI_OUTPUT != 0; }
    bool isMidiEffect() const override
    {
        return (DSPJSFX_USES_MIDI != 0) && (DSPJSFX_NUM_INPUTS == 0) && (DSPJSFX_NUM_OUTPUTS == 0);
    }
    double getTailLengthSeconds() const override { return smartIdleTailLengthSeconds; }

    static int oversamplingChoiceIndexToFactor (int choiceIndex) noexcept
    {
        switch (choiceIndex)
        {
            case 1:  return 2;
            case 2:  return 4;
            case 3:  return 8;
            case 0:
            default: return 1;
        }
    }

    static int oversamplingFactorToChoiceIndex (int factor) noexcept
    {
        switch (factor)
        {
            case 2:  return 1;
            case 4:  return 2;
            case 8:  return 3;
            case 1:
            default: return 0;
        }
    }

    static int safeOversampledBlockSize (int hostSamples, int factor) noexcept
    {
        return za::jsfx::AudioHostRuntime::safeBlockSize(hostSamples,factor);
    }

    juce::String getOversamplingParameterIdForUi() const { return kOversamplingParamId(); }

    int getRequestedOversamplingFactor() const noexcept
    {
        if (oversamplingParamAtomic == nullptr)
            return 1;

        const int choice = juce::jlimit (0, 3, (int) std::llround (oversamplingParamAtomic->load (std::memory_order_acquire)));
        return oversamplingChoiceIndexToFactor (choice);
    }

    int getActiveOversamplingFactor() const noexcept
    {
        return juce::jlimit (1, 8, oversamplingActiveFactor.load (std::memory_order_acquire));
    }

    double getEffectiveEngineSampleRate() const noexcept
    {
        const double hostRate = oversamplingHostSampleRate.load (std::memory_order_acquire);
        const int factor = getActiveOversamplingFactor();
        const double fallback = st.srate > 1000.0 ? st.srate : 44100.0;

        if (std::isfinite (hostRate) && hostRate > 1000.0)
            return hostRate * (double) factor;

        return fallback;
    }

    double getCurrentFileCacheTargetSampleRate() const noexcept
    {
        // File/cache rebuilds are async, so target the user-requested factor rather
        // than the currently-active factor. The DSP engine applies the same request
        // at the next block boundary; this prevents stopped hosts/state restores from
        // rebuilding caches at a stale rate while waiting for audio to run again.
        const int factor = getRequestedOversamplingFactor();
        if (factor <= 1)
            return 0.0; // native-rate file caches when oversampling is off

        const double hostRate = oversamplingHostSampleRate.load (std::memory_order_acquire);
        if (std::isfinite (hostRate) && hostRate > 1000.0)
            return hostRate * (double) factor;

        const double rate = oversamplingEngineSampleRate.load (std::memory_order_acquire);
        return (std::isfinite (rate) && rate > 1000.0) ? rate : getEffectiveEngineSampleRate();
    }

    void requestUiWakeAsync()
    {
        if (processorShuttingDown.load (std::memory_order_acquire))
            return;

        if (! uiAsyncWakePending.exchange (true, std::memory_order_acq_rel))
            triggerAsyncUpdate();
    }

    void requestExternalWakeEvent (bool mayTriggerAsyncUi = true)
    {
        // This wakes DSP-JSFX's own Smart Idle lifecycle. It does not ask the
        // CLAP host to start/continue processing and therefore cannot defeat
        // host-level CPU idling for otherwise inactive plug-ins.
        pendingExternalWakeEvent.store (true, std::memory_order_release);
        pendingEventWakeCount.fetch_add (1u, std::memory_order_acq_rel);
        gfxSnapshotForcePublish.store (true, std::memory_order_release);

        if (mayTriggerAsyncUi)
            requestUiWakeAsync();
    }

    void setGfxAnalysisVisible(bool visible)
    {
        if (gfxAnalysisVisible.exchange(visible, std::memory_order_acq_rel) != visible)
            requestExternalWakeEvent(true);
    }

    void parameterChanged (const juce::String& parameterID, float) override
    {
        pendingParameterWakeEvent.store (true, std::memory_order_release);
        requestExternalWakeEvent (true);

        if (parameterID == kOversamplingParamId())
        {
            pendingOversamplingParameterChange.store (true, std::memory_order_release);
            pendingFileReloadForEngineRate.store (true, std::memory_order_release);
            pendingSamplePoolRecommitForEngineRate.store (true, std::memory_order_release);
        }
    }

    void handleAsyncUpdate() override
    {
        uiAsyncWakePending.store (false, std::memory_order_release);

        if (processorShuttingDown.load (std::memory_order_acquire))
            return;

        if (pendingFileReloadForEngineRate.exchange (false, std::memory_order_acq_rel))
            reloadCurrentFileSlotsForEngineRate();

        if (pendingSamplePoolRecommitForEngineRate.exchange (false, std::memory_order_acq_rel))
            recommitAllSamplePoolsForEngineRate();

        gfxSnapshotForcePublish.store (true, std::memory_order_release);
        updateHostDisplay();

        if (auto* editor = getActiveEditor())
            editor->repaint();
    }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void prepareToPlay (double sampleRate, int samplesPerBlockExpected) override
    {
#if DSPJSFX_NATIVE_GFX_LEGACY
        const std::lock_guard<std::mutex> legacyLock (legacyLifecycleMutex);
#endif
       #if DSPJSFX_HAS_NATIVE_GFX
        nativeGfxCommandEpoch.fetch_add (1, std::memory_order_acq_rel);
        ++nativeRuntimeEpoch;
       #endif
        const int factor = getRequestedOversamplingFactor();
        const int engineBlockExpected = safeOversampledBlockSize (samplesPerBlockExpected, factor);
        const double engineSampleRate = sampleRate > 1000.0 ? sampleRate * (double) factor : sampleRate;

        oversamplingHostSampleRate.store (sampleRate, std::memory_order_release);
        oversamplingEngineSampleRate.store (engineSampleRate, std::memory_order_release);
        oversamplingHostBlockSizeExpected.store (samplesPerBlockExpected, std::memory_order_release);
        oversamplingActiveFactor.store (factor, std::memory_order_release);
        pendingOversamplingParameterChange.store (false, std::memory_order_release);

        resetStateStructOnly();
        st.srate = engineSampleRate;
        st.currentSampleRate = engineSampleRate;
        prepareMidiRuntime (engineBlockExpected);
        requestEmergencyMidiCleanup();
        updateSmartIdleSampleRate (engineSampleRate);
        resetSmartIdleRuntime();

        // Pre-size scratch buffers to avoid allocation in the audio callback.
        if (samplesPerBlockExpected > 0)
        {
            zeroIn.assign ((size_t) juce::jmax (samplesPerBlockExpected, engineBlockExpected), 0.0f);

            const int outCh = juce::jlimit (0, 64, getTotalNumOutputChannels());
            const int inCh  = juce::jlimit (0, 64, getTotalNumInputChannels());
            const int requiredCh = juce::jlimit (0, 64, juce::jmax ((int) DSPJSFX_NUM_INPUTS, (int) DSPJSFX_NUM_OUTPUTS));
            const int numCh = juce::jmin (64, juce::jmax (requiredCh, juce::jmax (inCh, outCh)));
            const int scratchCh = juce::jmax (0, numCh - outCh);

            if (factor > 1 && numCh > 0 && engineBlockExpected > 0)
            {
                oversampledInput.setSize (numCh, engineBlockExpected, false, false, true);
                oversampledOutput.setSize (numCh, engineBlockExpected, false, false, true);
                oversampledInput.clear();
                oversampledOutput.clear();
            }
            else
            {
                oversampledInput.setSize (0, 0);
                oversampledOutput.setSize (0, 0);
            }

            if (scratchCh > 0)
            {
                scratchOut.setSize (scratchCh, samplesPerBlockExpected, false, false, true);
                scratchOut.clear();
            }
            else
            {
                scratchOut.setSize (0, 0);
            }
        }

        gfxSnapPeriodSamples = (int64_t) juce::jmax (128.0, engineSampleRate / 30.0);
        gfxSnapCountdown = 0;

        // Push params BEFORE @init, matching REAPER JSFX behaviour (sliders are valid in @init).
        lastSlidersValid = false;
        internalSliderPendingMask.clear();
        dspGestureActive.fill (false);
        resetGfxSliderPreviewState();
        (void) pushParamsToStateSliders();
        (void) applyStringSlidersToState (true);

#if DSPJSFX_HAS_FAUST
        st.faustContext=faustEngine.get();
        faustEngine->prepare(double(st.srate),safeOversampledBlockSize(samplesPerBlockExpected,8),sampleRate);
#endif
        productionProgram.initialiseAndPrime(st,[&] {
        // Re-apply slider aliases after init (scripts sometimes touch vars in @init).
        {
            const int varsCap = (int) za::jsfx::variableCount(st);
            for (size_t i = 0; i < sliderAliasVarIndex.size(); ++i)
            {
                const int vIdx = sliderAliasVarIndex[i];
                if (vIdx >= 0 && vIdx < varsCap)
                    st.vars[vIdx] = st.sliders[i];
            }
        }

        });
        syncJsfxLatency();

       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        if (correctnessRuntime != nullptr)
        {
            correctnessRuntime->setRingSampleRate (engineSampleRate);
            correctnessRuntime->resetAndPrime (st);
            if (correctnessRuntime->isReady())
            {
                correctnessRuntime->compareSliderAndVarState (st, -1, -1, "@prepare");
                correctnessRuntime->compareMemoryPages (st, -1, -1, "@prepare");
            }
        }
       #endif

        // Seed an initial GFX snapshot so the UI has something immediately,
        // even before any editor is opened.
        gfxSnapCountdown = 0;
        gfxSnapshotForcePublish.store (true, std::memory_order_release);
        updateGfxSnapshotIfNeeded ((int) gfxSnapPeriodSamples);

        pendingFileReloadForEngineRate.store (true, std::memory_order_release);
        pendingSamplePoolRecommitForEngineRate.store (true, std::memory_order_release);
        requestUiWakeAsync();
    }

    void releaseResources() override
    {
        requestEmergencyMidiCleanup();
        endAllDspGestures();
        internalSliderPendingMask.clear();
        resetGfxSliderPreviewState();
        resetSmartIdleRuntime();
        jsfxRuntime.reset();
        midiRuntime.beginBlock();
        st.midiInCount = 0;
        st.midiInReadIndex = 0;
        st.midiOutCount = 0;
    }

    void reset() override
    {
        requestEmergencyMidiCleanup();
        endAllDspGestures();
        internalSliderPendingMask.clear();
        resetGfxSliderPreviewState();
        resetSmartIdleRuntime();
        jsfxRuntime.reset();
        midiRuntime.beginBlock();
        st.midiInCount = 0;
        st.midiInReadIndex = 0;
        st.midiOutCount = 0;
    }

    void updateTrackProperties (const TrackProperties& properties) override
    {
        juce::String suppliedName;

        // Some hosts send partial updates. Preserve the last known name when
        // the current update does not include a usable host-supplied track name.
        if (! copyTrackPropertyName (properties.name, suppliedName))
            return;

        jsfxRuntime.setHostTrackName (std::string (suppliedName.toRawUTF8()));
        requestExternalWakeEvent (true);
    }

    juce::String getHostTrackName() const
    {
        const auto name = jsfxRuntime.hostTrackName();
        return juce::String::fromUTF8 (name.c_str(), (int) name.size());
    }

    std::pair<juce::String, std::uint64_t> getHostTrackNameSnapshot() const
    {
        const auto snapshot = jsfxRuntime.hostTrackNameSnapshot();
        return { juce::String::fromUTF8 (snapshot.first.c_str(), (int) snapshot.first.size()), snapshot.second };
    }

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        const auto busSizeOrZero = [] (const juce::AudioChannelSet& set) -> int
        {
            return set.isDisabled() ? 0 : set.size();
        };

        const int mainOut = (layouts.outputBuses.size() > 0) ? busSizeOrZero (layouts.getMainOutputChannelSet()) : 0;
        const int mainIn  = (layouts.inputBuses.size()  > 0) ? busSizeOrZero (layouts.getMainInputChannelSet())  : 0;

        if (mainOut < 0 || mainOut > 64 || mainIn < 0 || mainIn > 64)
            return false;

        if (mainIn > 0 && mainOut > 0 && mainIn < mainOut)
            return false;

        int totalIn = 0;
        for (int i = 0; i < layouts.inputBuses.size(); ++i)
            totalIn += busSizeOrZero (layouts.inputBuses[i]);

        int totalOut = 0;
        for (int i = 0; i < layouts.outputBuses.size(); ++i)
            totalOut += busSizeOrZero (layouts.outputBuses[i]);

        if (totalIn < 0 || totalIn > 64 || totalOut < 0 || totalOut > 64)
            return false;

        if (totalIn < (int) DSPJSFX_NUM_INPUTS)
            return false;
        if (totalOut < (int) DSPJSFX_NUM_OUTPUTS)
            return false;

        if (totalIn == 0 && totalOut == 0)
            return DSPJSFX_USES_MIDI != 0;

        return true;
    }

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
    {
        za::jsfx::SilenceOnMemoryFault silenceOnMemoryFault{st,buffer,&midiMessages};

        juce::ScopedNoDenormals _;
        ScopedSamplePoolReads poolReads(*this);

        const int numSamples = buffer.getNumSamples();
        applyOversamplingFactorChangeIfNeeded (numSamples);
        const int oversamplingFactor = getActiveOversamplingFactor();
        const int processSamples = safeOversampledBlockSize (numSamples, oversamplingFactor);
        int activityInputChannels = 0;
        int activityOutputChannels = 0;

        const auto audioTopology=prepareHostAudio(*this,buffer,(int)DSPJSFX_NUM_INPUTS,(int)DSPJSFX_NUM_OUTPUTS,oversamplingFactor);
        const int numCh=audioTopology.channels;
        activityInputChannels=audioTopology.inputs;
        activityOutputChannels=audioTopology.outputs;

        syncJsfxHostTransport(numSamples);
        syncJsfxGfxActivity();
        const bool fileLoadsPromoted = promotePendingFileLoads();

        const bool slidersChanged = pushParamsToStateSliders();
        const bool stringSlidersChanged = applyStringSlidersToState (false);

       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        auto* shadow = (correctnessRuntime != nullptr && correctnessRuntime->isReady())
                         ? correctnessRuntime->getShadow()
                         : nullptr;
       #endif

        if (slidersChanged || stringSlidersChanged)
        {
            productionProgram.parametersChanged(st);

           #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
            if (shadow != nullptr)
            {
                shadow->syncHostSlidersAndAliases (st.sliders, DSPJSFX_MAX_SLIDERS);
                shadow->runSlider();
            }
           #endif
        }

        syncJsfxLatency();
        bool nativeCommandsChanged = false;
   #if DSPJSFX_HAS_NATIVE_GFX
        const auto commandEpoch = nativeGfxCommandEpoch.load (std::memory_order_acquire);
        const bool commandActive = gfxAnalysisVisible.load (std::memory_order_acquire) && nativeGfxInputActive.load (std::memory_order_acquire);
        for (int i = 0; i < DSPJSFX_NATIVE_GFX_COMMAND_COUNT; ++i)
        {
            const double value = commandActive && nativeGfxCommandTags[(size_t) i].load (std::memory_order_acquire) == commandEpoch
                ? nativeGfxCommands[(size_t) i].load (std::memory_order_relaxed) : 0;
            const int index = DSPJSFX_NATIVE_GFX_COMMAND_INDICES[i];
            nativeCommandsChanged = nativeCommandsChanged || st.vars[index] != value;
            writeExternalJsfxVar (index, value); // Keep the optional WDL correctness oracle in step.
        }
   #endif
        const bool gfxWritesApplied = nativeCommandsChanged | applyQueuedGfxStateWrites() | applyQueuedGfxMemSpanWrites();

        st.currentBlockSize = processSamples;
        st.currentSampleRate = st.srate;
        st.samplesblock = (double) processSamples;
        jsfxRuntime.beginBlock (st, processSamples);
        const bool hasIncomingMidi = ! midiMessages.isEmpty();
        const bool hasIncomingJsfxMessages = jsfxRuntime.hasReadyMessages() || jsfxRuntime.hasPendingForThisInstance();
        const bool parameterWakeEvent = pendingParameterWakeEvent.exchange (false, std::memory_order_acq_rel);
        const bool explicitWakeEvent = pendingEventWakeCount.exchange (0u, std::memory_order_acq_rel) != 0u;
        const bool emergencyMidiCleanupRequested = pendingEmergencyMidiCleanup.exchange (false, std::memory_order_acq_rel)
                                               || st.pendingNoteCleanup != 0;
        const bool transportStatusChangedForMidiCleanup = detectMidiTransportStatusChangedForCleanup();
        const bool transportInputCleanupRequested = transportStatusChangedForMidiCleanup
                                                && midiInputNoteTracker.hasAnyUnsafeState();
        const bool transportOutputCleanupRequested = transportStatusChangedForMidiCleanup
                                                 && midiOutputNoteTracker.hasAnyUnsafeState();
        const bool prependInputMidiCleanup = emergencyMidiCleanupRequested || transportInputCleanupRequested;
        const bool appendOutputMidiCleanup = emergencyMidiCleanupRequested || transportOutputCleanupRequested;
        const bool needsMidiCleanup = prependInputMidiCleanup || appendOutputMidiCleanup;
        juce::ignoreUnused (needsMidiCleanup);

        importMidiToState (midiMessages, numSamples, processSamples, oversamplingFactor, prependInputMidiCleanup, emergencyMidiCleanupRequested);
        midiMessages.clear();

        if (appendOutputMidiCleanup)
            appendTrackedOutputMidiCleanupToBuffer (midiMessages, 0, emergencyMidiCleanupRequested);

        if (emergencyMidiCleanupRequested)
            st.pendingNoteCleanup = 0;

       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        if (shadow != nullptr)
        {
            const int blockIndex = correctnessRuntime->getNextBlockIndex();
            shadow->beginBlock (st, processSamples);

            if (slidersChanged || stringSlidersChanged)
                correctnessRuntime->compareSliderAndVarState (st, blockIndex, -1, "pre-@block/@slider");

            correctnessRuntime->compareSliderAndVarState (st, blockIndex, -1, "pre-@block");

            st.samplesblock = (double) processSamples;
            st.currentBlockSize = processSamples;
            st.currentSampleRate = st.srate;

           #if DSPJSFX_HAS_SAMPLE_SECTION
            productionProgram.beginSection(st);
           #else
            productionProgram.processAudio (st,
                                numCh > 0 ? inPtrs.data() : nullptr,
                                numCh > 0 ? outPtrs.data() : nullptr,
                                numCh,
                                processSamples);
           #endif
            shadow->runBlock();

            correctnessRuntime->compareSliderAndVarState (st, blockIndex, -1, "@block");

            const auto shadowMasksAfterBlock = shadow->peekPendingSliderMasks();
            correctnessRuntime->comparePendingSliderMasks (st, shadowMasksAfterBlock, blockIndex, "@block");

            const bool compiledNeedsSlider = anySliderMaskWords (st.pendingSliderChangeMask)
                                          || anySliderMaskWords (st.pendingSliderAutomateMask)
                                          || anySliderMaskWords (st.pendingSliderAutomateEndMask);
            const bool shadowNeedsSlider = shadowMasksAfterBlock.change.any()
                                        || shadowMasksAfterBlock.automate.any()
                                        || shadowMasksAfterBlock.automateEnd.any();

            if (compiledNeedsSlider)
                productionProgram.parametersChanged(st);
            if (shadowNeedsSlider)
                shadow->runSlider();

            if (compiledNeedsSlider || shadowNeedsSlider)
            {
                correctnessRuntime->compareSliderAndVarState (st, blockIndex, -1, "@slider");
                correctnessRuntime->comparePendingSliderMasks (st, shadow->peekPendingSliderMasks(), blockIndex, "@slider");
            }

           #if DSPJSFX_HAS_SAMPLE_SECTION
            if (processSamples > 0)
            {
                const float* const* inData = numCh > 0 ? inPtrs.data() : nullptr;
                for (int sample = 0; sample < processSamples; ++sample)
                {
                    for (int ch = 0; ch < numCh; ++ch)
                        st.spl[ch] = (double) (inData != nullptr && inData[ch] != nullptr ? inData[ch][sample] : 0.0f);

                    shadow->setInputSample (inData, numCh, sample);

                    productionProgram.sampleSection(st);
                    shadow->runSample();

                    correctnessRuntime->observeAudioFrame (st, blockIndex, sample, numCh);
                    correctnessRuntime->compareSliderAndVarState (st, blockIndex, sample, "@sample");

                    for (int ch = 0; ch < numCh; ++ch)
                    {
                        const double compiled = (double) (float) st.spl[ch];
                        const double ref = shadow->getOutputValue (ch);
                        outPtrs[(size_t) ch][sample] = correctnessRuntime->selectOutputSample (ch, compiled, ref, numCh);
                    }
                }
            }
           #endif

            correctnessRuntime->compareMidiOutput (st, blockIndex, "post-block");
            correctnessRuntime->compareMemoryPages (st, blockIndex, processSamples > 0 ? (processSamples - 1) : -1, "post-block");
            correctnessRuntime->noteBlockCompared();
            syncJsfxLatency(); // @block can complete a deferred latency transition

#if ! ((DSPJSFX_NUM_INPUTS == 0) && (DSPJSFX_NUM_OUTPUTS == 0))
            if (oversamplingFactor > 1)
                downsampleOversampledOutputToHost (activityOutputChannels, numSamples, processSamples, oversamplingFactor);

            mixQueuedImportPreviewAudition (hostOutPtrs.empty() ? nullptr : hostOutPtrs.data(), activityOutputChannels, numSamples);
#endif

            consumeDspSliderChanges();
            (void) flushMidiFromState (midiMessages, oversamplingFactor, numSamples);
            jsfxRuntime.endBlock (st);
            publishSmartIdleUiRuntimeSnapshot (resolveSmartIdleModeForCurrentState(), false, 0.0f, 0.0f, 0);
            updateGfxSnapshotIfNeeded (processSamples);
            return;
        }
       #endif

        const bool externalWakeEvent=pendingExternalWakeEvent.exchange(false,std::memory_order_acq_rel);
        const auto idleBlock=beginIdleBlock(numSamples,numCh>0?hostInPtrs.data():nullptr,activityInputChannels,
            hostTransportDiscontinuity || fileLoadsPromoted || externalWakeEvent || hasDeferredSamplePoolAdoptionPending() || parameterWakeEvent || explicitWakeEvent || slidersChanged || stringSlidersChanged || gfxWritesApplied || hasIncomingMidi || hasIncomingJsfxMessages);
        const SmartIdleMode preIdleMode=idleBlock.mode;
        const float inputPeak=idleBlock.inputPeak;
        if(idleBlock.skip)
        {
            clearWritableChannelPointers (numCh > 0 ? hostOutPtrs.data() : nullptr, numCh, numSamples);
            smartIdleRuntime.lastOutputPeak = 0.0f;
            consumeDspSliderChanges();
            (void) flushMidiFromState (midiMessages, oversamplingFactor, numSamples);
            jsfxRuntime.endBlock (st);
            publishSmartIdleUiRuntimeSnapshot (preIdleMode,
                                               true,
                                               inputPeak,
                                               0.0f,
                                               smartIdleRuntime.quietSamples);
            updateGfxSnapshotIfNeeded (processSamples);
            return;
        }

        productionProgram.processAudio (st, numCh > 0 ? inPtrs.data() : nullptr, numCh > 0 ? outPtrs.data() : nullptr, numCh, processSamples);
        syncJsfxLatency();

#if ! ((DSPJSFX_NUM_INPUTS == 0) && (DSPJSFX_NUM_OUTPUTS == 0))
        if (oversamplingFactor > 1)
            downsampleOversampledOutputToHost (activityOutputChannels, numSamples, processSamples, oversamplingFactor);

        mixQueuedImportPreviewAudition (numCh > 0 ? hostOutPtrs.data() : nullptr, activityOutputChannels, numSamples);
#endif

        const bool dspSliderActivity = anySliderMaskWords (st.pendingSliderChangeMask)
                                    || anySliderMaskWords (st.pendingSliderAutomateMask)
                                    || anySliderMaskWords (st.pendingSliderAutomateEndMask);
        consumeDspSliderChanges();

        const bool hadOutgoingMidi=flushMidiFromState(midiMessages,oversamplingFactor,numSamples);
        finishIdleBlock(idleBlock,numCh>0?hostOutPtrs.data():nullptr,activityOutputChannels,numSamples,processSamples,hadOutgoingMidi,dspSliderActivity);
        updateGfxSnapshotIfNeeded (processSamples);
        jsfxRuntime.endBlock (st);
    }

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    void getStateInformation (juce::MemoryBlock& destData) override
    {
        auto tree = apvts->copyState();

        // Persist file slot paths (non-parameter state). Legacy single-file
        // slots keep the old fN property format; multi-file selections are
        // stored as SLOT/PATH child nodes.
        if (! fileDecls.empty())
        {
            juce::ValueTree files ("FILES");
            {
                std::lock_guard<std::mutex> lk (filePathMutex);
                for (int i = 0; i < (int) fileSlots.size(); ++i)
                {
                    const auto& slot = fileSlots[(size_t) i];
                    const auto& paths = slot.currentPaths;
                    const auto loadMode = slot.loadMode;
                    const auto recipeXml = slot.currentRecipeXml;
                    if (paths.empty())
                        continue;

                    if (paths.size() == 1 && loadMode == FileLoadMode::SeparateEntries && recipeXml.isEmpty())
                    {
                        files.setProperty ("f" + juce::String (i), paths.front(), nullptr);
                        continue;
                    }

                    juce::ValueTree slotTree ("SLOT");
                    slotTree.setProperty ("index", i, nullptr);

                    if (recipeXml.isNotEmpty())
                        slotTree.setProperty ("importRecipeXml", recipeXml, nullptr);

                    if (loadMode == FileLoadMode::AppendAsSingleFile)
                        slotTree.setProperty ("mode", "appendEach", nullptr);

                    for (const auto& path : paths)
                    {
                        if (path.isEmpty())
                            continue;

                        juce::ValueTree pathTree ("PATH");
                        pathTree.setProperty ("path", path, nullptr);
                        slotTree.addChild (pathTree, -1, nullptr);
                    }

                    if (slotTree.getNumChildren() > 0)
                        files.addChild (slotTree, -1, nullptr);
                }
            }
            tree.addChild (files, -1, nullptr);
        }

        {
            juce::String lastDir;
            std::vector<FileSelectionState> recentSelections;

            {
                std::lock_guard<std::mutex> lk (filePathMutex);
                lastDir = lastFileDialogDirectory;
                recentSelections = recentFileSelections;
            }

            if (lastDir.isNotEmpty() || ! recentSelections.empty())
            {
                juce::ValueTree fileUi ("FILE_UI");

                if (lastDir.isNotEmpty())
                    fileUi.setProperty ("lastDir", lastDir, nullptr);

                for (const auto& selection : recentSelections)
                {
                    if (selection.paths.empty())
                        continue;

                    juce::ValueTree recent ("RECENT");
                    writeSelectionToValueTree (selection, recent);
                    fileUi.addChild (recent, -1, nullptr);
                }

                tree.addChild (fileUi, -1, nullptr);
            }
        }

        {
            auto stringState = makeStringSliderStateValueTree();
            if (stringState.isValid() && stringState.getNumChildren() > 0)
                tree.addChild (stringState, -1, nullptr);
        }

        {
            const SmartIdleMode userOverride = storedSmartIdleModeToEnum (smartIdleUserOverrideMode.load (std::memory_order_acquire));
            if (userOverride != SmartIdleMode::Auto)
            {
                juce::ValueTree smartIdleState ("SMART_IDLE");
                smartIdleState.setProperty ("userOverrideMode", (int) userOverride, nullptr);
                tree.addChild (smartIdleState, -1, nullptr);
            }
        }

        std::unique_ptr<juce::XmlElement> xml (tree.createXml());
        copyXmlToBinary (*xml, destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        if (auto xml = getXmlFromBinary (data, sizeInBytes))
        {
            auto tree = juce::ValueTree::fromXml (*xml);

            // Pull out file slot state before handing the tree to APVTS.
            auto files = tree.getChildWithName ("FILES");
            if (files.isValid())
                tree.removeChild (files, nullptr);

            auto fileUi = tree.getChildWithName ("FILE_UI");
            if (fileUi.isValid())
                tree.removeChild (fileUi, nullptr);

            auto stringSliders = tree.getChildWithName ("STRING_SLIDERS");
            if (stringSliders.isValid())
                tree.removeChild (stringSliders, nullptr);

            auto smartIdleState = tree.getChildWithName ("SMART_IDLE");
            if (smartIdleState.isValid())
                tree.removeChild (smartIdleState, nullptr);

            apvts->replaceState (tree);
            requestEmergencyMidiCleanup();

            loadPersistentFileUiState();

            if (files.isValid() && ! fileDecls.empty())
            {
                for (const auto& d : fileDecls)
                {
                    const auto savedSelection = getSavedFileSlotSelection (files, d.index0);
                    if (savedSelection.paths.empty())
                    {
                        clearFileSlot (d.index0);
                        continue;
                    }

                    std::vector<juce::File> savedFiles;
                    savedFiles.reserve (savedSelection.paths.size());

                    for (const auto& path : savedSelection.paths)
                    {
                        if (path.isEmpty())
                            continue;

                        savedFiles.emplace_back (path);
                    }

                    if (savedFiles.empty())
                        clearFileSlot (d.index0);
                    else
                        setFileSlotPathsWithMode (d.index0, savedFiles, savedSelection.loadMode, true, savedSelection.importRecipeXml);
                }
            }

            if (fileUi.isValid())
                restoreFileUiState (fileUi, true, true);

            if (stringSliders.isValid())
                restoreStringSliderState (stringSliders);

            SmartIdleMode restoredUserOverride = SmartIdleMode::Auto;
            if (smartIdleState.isValid())
            {
                const auto raw = smartIdleState.getProperty ("userOverrideMode", 0);
                if (raw.isString())
                    restoredUserOverride = parseSmartIdleModeToken (raw.toString().toStdString());
                else
                    restoredUserOverride = storedSmartIdleModeToEnum ((int) raw);
            }
            setSmartIdleUserOverrideModeForUi ((int) restoredUserOverride);

            pendingOversamplingParameterChange.store (true, std::memory_order_release);
            pendingFileReloadForEngineRate.store (true, std::memory_order_release);
            pendingSamplePoolRecommitForEngineRate.store (true, std::memory_order_release);
            requestEmergencyMidiCleanup();
            requestExternalWakeEvent (true);
        }
    }

    // UI metadata accessors (used by the custom editor)
    const std::vector<JsfxSliderDecl>& getJsfxSliderDecls() const noexcept { return sliderDecls; }
    const juce::String& getEmbeddedReadmeMarkdown() const noexcept { return embeddedReadmeMarkdown; }
    const std::vector<JsfxFileDecl>& getJsfxFileDecls() const noexcept { return fileDecls; }

    // Resolve a native JSFX file_open() target for the dedicated @gfx worker.
    // If a numeric filename slot has an enhanced user selection, expose those
    // paths first. Otherwise resolve the declaration/string token using the
    // same practical search roots as packaged GFX resources: source tree while
    // developing, then module/Resources fallbacks for deployed plug-ins.
    std::vector<juce::File> resolveGfxFileCandidates (int filenameIndex,
                                                       const juce::String& token) const
    {
        std::vector<juce::File> out;

        if (filenameIndex >= 0 && filenameIndex < (int) fileSlots.size())
        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            const auto& paths = fileSlots[(size_t) filenameIndex].currentPaths;
            out.reserve (paths.size());
            for (const auto& path : paths)
                if (path.isNotEmpty())
                    out.emplace_back (path);
        }

        if (! out.empty() || token.isEmpty())
            return out;

        if (juce::File::isAbsolutePath (token))
        {
            out.emplace_back (token);
            return out;
        }

        std::vector<juce::File> candidates;
        candidates.emplace_back (juce::File::getCurrentWorkingDirectory().getChildFile (token));

        if (jsfx_gfx_resources::kSourceDir != nullptr && *jsfx_gfx_resources::kSourceDir != 0)
        {
            const juce::File sourceDir (juce::String::fromUTF8 (jsfx_gfx_resources::kSourceDir));
            candidates.emplace_back (sourceDir.getChildFile (token));
            candidates.emplace_back (sourceDir.getParentDirectory().getChildFile ("Data").getChildFile (token));
            candidates.emplace_back (sourceDir.getParentDirectory().getChildFile ("Resources").getChildFile (
                juce::File (token).getFileName()));
        }

        const auto moduleDir = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory();
        candidates.emplace_back (moduleDir.getChildFile (token));
        candidates.emplace_back (moduleDir.getChildFile ("Data").getChildFile (token));
        candidates.emplace_back (moduleDir.getChildFile ("Resources").getChildFile (token));
        candidates.emplace_back (moduleDir.getChildFile ("Resources").getChildFile ("Data").getChildFile (token));
        candidates.emplace_back (moduleDir.getParentDirectory().getChildFile ("Resources").getChildFile (token));

        for (const auto& candidate : candidates)
        {
            if (candidate.existsAsFile())
            {
                out.push_back (candidate);
                return out;
            }
        }

        // Preserve a deterministic failed candidate so file_open() can simply
        // report -1 without having to duplicate the search policy.
        if (! candidates.empty())
            out.push_back (candidates.front());

        return out;
    }

    juce::AudioProcessorValueTreeState& getApvts() noexcept { return *apvts; }
    const juce::AudioProcessorValueTreeState& getApvts() const noexcept { return *apvts; }

    void auditionImportPreviewBuffer (juce::AudioBuffer<float> buffer, double sourceSampleRate)
    {
        if (buffer.getNumSamples() <= 0 || buffer.getNumChannels() <= 0)
            return;

        const double targetRate = getSampleRate() > 1000.0 ? getSampleRate() : (st.srate > 1000.0 ? st.srate : sourceSampleRate);
        if (sourceSampleRate > 1000.0 && targetRate > 1000.0 && std::abs (sourceSampleRate - targetRate) > 1.0)
        {
            const int srcN = buffer.getNumSamples();
            const int srcCh = buffer.getNumChannels();
            const int dstN = juce::jmax (1, (int) std::llround ((double) srcN * targetRate / sourceSampleRate));
            juce::AudioBuffer<float> resampled (srcCh, dstN);

            for (int ch = 0; ch < srcCh; ++ch)
            {
                const auto* src = buffer.getReadPointer (ch);
                auto* dst = resampled.getWritePointer (ch);
                for (int i = 0; i < dstN; ++i)
                {
                    const double srcPos = (double) i * sourceSampleRate / targetRate;
                    const int i0 = juce::jlimit (0, srcN - 1, (int) std::floor (srcPos));
                    const int i1 = juce::jmin (i0 + 1, srcN - 1);
                    const float frac = (float) (srcPos - (double) i0);
                    dst[i] = src[i0] + (src[i1] - src[i0]) * frac;
                }
            }

            buffer = std::move (resampled);
        }

        importPreviewAuditionStopRequested.store (false, std::memory_order_release);
        importPreviewAuditionPaused.store (false, std::memory_order_release);

        auto clip = std::make_unique<ImportPreviewAuditionClip>();
        clip->buffer = std::move (buffer);
        clip->position = 0;

        {
            std::lock_guard<std::mutex> lk (importPreviewAuditionMutex);
            pendingImportPreviewAudition = std::move (clip);
            importPreviewAuditionPending.store (true, std::memory_order_release);
        }

        requestExternalWakeEvent (true);
    }

    void pauseImportPreviewAudition (bool shouldPause)
    {
        // The audio thread retains the clip and its position while paused.
        importPreviewAuditionPaused.store (shouldPause, std::memory_order_release);
        requestExternalWakeEvent (true);
    }

    void stopImportPreviewAudition()
    {
        importPreviewAuditionPaused.store (false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk (importPreviewAuditionMutex);
            pendingImportPreviewAudition.reset();
            importPreviewAuditionPending.store (false, std::memory_order_release);
        }

        importPreviewAuditionStopRequested.store (true, std::memory_order_release);
        requestExternalWakeEvent (true);
    }

    juce::String getStringSliderText (int index0) const
    {
        if (index0 < 0 || index0 >= DSPJSFX_MAX_SLIDERS || ! sliderStringUsed[(size_t) index0])
            return {};
        std::lock_guard<std::mutex> lk (stringSliderMutex);
        return stringSliderTexts[(size_t) index0];
    }

    void setStringSliderText (int index0, juce::String text)
    {
        if (index0 < 0 || index0 >= DSPJSFX_MAX_SLIDERS || ! sliderStringUsed[(size_t) index0])
            return;

        text = text.substring (0, 1024);
        {
            std::lock_guard<std::mutex> lk (stringSliderMutex);
            if (stringSliderTexts[(size_t) index0] == text)
                return;
            stringSliderTexts[(size_t) index0] = text;
        }
        stringSliderSeq[(size_t) index0].fetch_add (1u, std::memory_order_acq_rel);
    }

   #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
    bool hasCorrectnessMonitor() const noexcept { return correctnessRuntime != nullptr; }
    juce::String getCorrectnessStatusText() const
    {
        return correctnessRuntime != nullptr ? correctnessRuntime->getStatusText()
                                           : juce::String ("Correctness monitor unavailable");
    }
    int getCorrectnessMonitorMode() const noexcept
    {
        return correctnessRuntime != nullptr ? (int) correctnessRuntime->getMonitorMode()
                                            : (int) jsfx_correctness::MonitorAudioMode::Compiled;
    }
    void setCorrectnessMonitorMode (int mode) noexcept
    {
        if (correctnessRuntime == nullptr)
            return;

        switch (mode)
        {
            case (int) jsfx_correctness::MonitorAudioMode::Shadow:
                correctnessRuntime->setMonitorMode (jsfx_correctness::MonitorAudioMode::Shadow);
                break;
            case (int) jsfx_correctness::MonitorAudioMode::Delta:
                correctnessRuntime->setMonitorMode (jsfx_correctness::MonitorAudioMode::Delta);
                break;
            case (int) jsfx_correctness::MonitorAudioMode::Compiled:
            default:
                correctnessRuntime->setMonitorMode (jsfx_correctness::MonitorAudioMode::Compiled);
                break;
        }
    }
    bool getCorrectnessFreezeOnFirstMismatch() const noexcept
    {
        return correctnessRuntime != nullptr ? correctnessRuntime->getFreezeOnFirstMismatch() : true;
    }
    void setCorrectnessFreezeOnFirstMismatch (bool shouldFreeze) noexcept
    {
        if (correctnessRuntime != nullptr)
            correctnessRuntime->setFreezeOnFirstMismatch (shouldFreeze);
    }
    void clearCorrectnessMonitor()
    {
        if (correctnessRuntime != nullptr)
            correctnessRuntime->clear();
    }
    juce::String exportCorrectnessArtifacts() const
    {
        return correctnessRuntime != nullptr ? correctnessRuntime->exportBundle (getName())
                                            : juce::String ("Correctness monitor unavailable");
    }
   #else
    bool hasCorrectnessMonitor() const noexcept { return false; }
    juce::String getCorrectnessStatusText() const { return "Correctness monitor disabled at build time"; }
    int getCorrectnessMonitorMode() const noexcept { return 0; }
    void setCorrectnessMonitorMode (int) noexcept {}
    bool getCorrectnessFreezeOnFirstMismatch() const noexcept { return true; }
    void setCorrectnessFreezeOnFirstMismatch (bool) noexcept {}
    void clearCorrectnessMonitor() {}
    juce::String exportCorrectnessArtifacts() const { return "Correctness monitor disabled at build time"; }
   #endif


    struct SmartIdleIndicatorStatus
    {
        bool sleeping = false;
        bool sleepEligible = false;
        juce::String stateText;
        juce::String modeText;
        juce::String tooltip;
    };

    SmartIdleIndicatorStatus getSmartIdleIndicatorStatus() const
    {
        SmartIdleIndicatorStatus status;

        const bool sleeping = smartIdleUiSleeping.load (std::memory_order_relaxed);
        const SmartIdleMode effectiveMode = storedSmartIdleModeToEnum (smartIdleUiEffectiveMode.load (std::memory_order_relaxed));
        const SmartIdleMode optionMode = storedSmartIdleModeToEnum (smartIdleUiOptionMode.load (std::memory_order_relaxed));
        const SmartIdleMode inferredMode = storedSmartIdleModeToEnum (smartIdleUiInferredMode.load (std::memory_order_relaxed));
        const SmartIdleMode userOverrideMode = storedSmartIdleModeToEnum (smartIdleUiUserOverrideMode.load (std::memory_order_relaxed));
        const float inputPeak = smartIdleUiInputPeak.load (std::memory_order_relaxed);
        const float outputPeak = smartIdleUiOutputPeak.load (std::memory_order_relaxed);
        const int64_t quietSamples = smartIdleUiQuietSamples.load (std::memory_order_relaxed);
        const double sampleRate = smartIdleUiSampleRate.load (std::memory_order_relaxed);
        const double holdMs = smartIdleUiHoldMs.load (std::memory_order_relaxed);
        const double tailMs = smartIdleUiTailMs.load (std::memory_order_relaxed);

        const SmartIdleMode configuredMode = optionMode == SmartIdleMode::Auto ? inferredMode : optionMode;
        const bool userOverrideActive = userOverrideMode != SmartIdleMode::Auto;
        const bool runtimeOverride = effectiveMode != configuredMode;

        status.sleeping = sleeping;
        status.sleepEligible = isSmartIdleSleepEligible (effectiveMode);
        status.stateText = sleeping ? "SLEEPING" : "ACTIVE";
        status.modeText = smartIdleModeShortName (effectiveMode);

        juce::String tooltip;
        tooltip << "Smart Idle: " << (sleeping ? "Sleeping" : "Active")
                << "\nEffective mode: " << smartIdleModeDisplayName (effectiveMode);

        if (userOverrideActive)
            tooltip << " (user override)";
        else if (runtimeOverride)
            tooltip << " (runtime override)";

        tooltip << '\n';

        if (optionMode == SmartIdleMode::Auto)
            tooltip << "Configured: Auto -> " << smartIdleModeDisplayName (inferredMode) << '\n';
        else
            tooltip << "Configured: " << smartIdleModeDisplayName (optionMode) << '\n';

        if (userOverrideActive)
            tooltip << "User override: " << smartIdleModeDisplayName (userOverrideMode) << '\n';

        if (status.sleepEligible)
        {
            const double quietMs = sampleRate > 0.0 ? (double) quietSamples * 1000.0 / sampleRate : 0.0;
            const double requiredMs = juce::jmax (holdMs, tailMs);
            tooltip << "Quiet window: " << juce::String (quietMs, 0)
                    << " / " << juce::String (requiredMs, 0) << " ms\n";
        }
        else
        {
            tooltip << "Sleep eligibility: Off in this mode\n";
        }

        tooltip << "Input peak: " << formatSmartIdlePeakDbText (inputPeak)
                << "\nOutput peak: " << formatSmartIdlePeakDbText (outputPeak);

        status.tooltip = tooltip.trimEnd();
        return status;
    }

    int getSmartIdleUserOverrideModeForUi() const noexcept
    {
        return (int) storedSmartIdleModeToEnum (smartIdleUserOverrideMode.load (std::memory_order_acquire));
    }

    void setSmartIdleUserOverrideModeForUi (int storedMode) noexcept
    {
        const SmartIdleMode mode = storedSmartIdleModeToEnum (storedMode);
        smartIdleUserOverrideMode.store ((int) mode, std::memory_order_release);
        smartIdleUiUserOverrideMode.store ((int) mode, std::memory_order_release);
        requestExternalWakeEvent (true);
    }

    bool isSmartIdleSleepingForTest() const noexcept { return smartIdleUiSleeping.load(std::memory_order_relaxed); }

    static juce::String summariseSelectedFilePaths (const std::vector<juce::String>& paths,
                                                    FileLoadMode loadMode = FileLoadMode::SeparateEntries)
    {
        if (paths.empty())
            return "(none)";

        auto lead = juce::File (paths.front()).getFileName();
        if (lead.isEmpty())
            lead = paths.front();

        if (paths.size() == 1)
            return lead;

        const auto extraCount = juce::String ((int) paths.size() - 1);
        if (loadMode == FileLoadMode::AppendAsSingleFile)
            return "Appended: " + lead + " (+" + extraCount + ")";

        return lead + " (+" + extraCount + ")";
    }

    static juce::String buildSelectedFileTooltip (const std::vector<juce::String>& paths,
                                                  FileLoadMode loadMode = FileLoadMode::SeparateEntries,
                                                  int maxLines = 10)
    {
        if (paths.empty())
            return {};

        juce::String out;

        if (loadMode == FileLoadMode::AppendAsSingleFile && paths.size() > 1)
            out << "Import Multiple (Append Each) -> single runtime file\n";

        const int limit = juce::jmin ((int) paths.size(), maxLines);

        for (int i = 0; i < limit; ++i)
        {
            if (i > 0)
                out << "\n";

            out << paths[(size_t) i];
        }

        if ((int) paths.size() > limit)
            out << za::text::utf8 ("\n… +") << juce::String ((int) paths.size() - limit) << " more";

        return out;
    }

    FileLoadMode getFileSlotLoadMode (int slotIndex0) const
    {
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return FileLoadMode::SeparateEntries;

        std::lock_guard<std::mutex> lk (filePathMutex);
        return fileSlots[(size_t) slotIndex0].loadMode;
    }

    FileSelectionState getFileSlotSelectionState (int slotIndex0) const
    {
        FileSelectionState selection;
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return selection;

        std::lock_guard<std::mutex> lk (filePathMutex);
        selection.paths = fileSlots[(size_t) slotIndex0].currentPaths;
        selection.loadMode = fileSlots[(size_t) slotIndex0].loadMode;
        selection.importRecipeXml = fileSlots[(size_t) slotIndex0].currentRecipeXml;
        return selection;
    }

    juce::String getFileSlotDisplayText (int slotIndex0) const
    {
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return {};

        const auto selection = getFileSlotSelectionState (slotIndex0);
        const auto summary = summariseSelectedFilePaths (selection.paths, selection.loadMode);
        const auto stt = fileSlots[(size_t) slotIndex0].state.load();

        switch (stt)
        {
            case FileState::Unassigned:   return "(none)";
            case FileState::Loading:      return za::text::utf8 ("Loading… ") + summary;
            case FileState::ReadyActive:
            case FileState::PendingDirect: return summary;
            case FileState::ReadyPending: return za::text::utf8 ("Ready… ") + summary;
            case FileState::PendingClear: return "(clearing)";
            case FileState::Error:        return "Error: " + summary;
            default:                      break;
        }

        return summary;
    }

    std::vector<juce::String> getFileSlotPaths (int slotIndex0) const
    {
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return {};

        std::lock_guard<std::mutex> lk (filePathMutex);
        return fileSlots[(size_t) slotIndex0].currentPaths;
    }

    juce::String getFileSlotTooltip (int slotIndex0) const
    {
        const auto selection = getFileSlotSelectionState (slotIndex0);
        return buildSelectedFileTooltip (selection.paths, selection.loadMode);
    }

    bool hasFileSlotSelection (int slotIndex0) const
    {
        return ! getFileSlotPaths (slotIndex0).empty();
    }

    bool fileSlotMatchesSelection (int slotIndex0, const FileSelectionState& selection) const
    {
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return false;

        const auto currentSelection = getFileSlotSelectionState (slotIndex0);
        return selectionsEqualIgnoreCase (currentSelection, selection);
    }

    juce::File getPreferredFileChooserStartDirectory (int slotIndex0, bool preferLastDialogDirectory = false) const
    {
        std::vector<juce::String> currentPaths;
        juce::String lastDir;

        {
            std::lock_guard<std::mutex> lk (filePathMutex);

            if (slotIndex0 >= 0 && slotIndex0 < (int) fileSlots.size())
                currentPaths = fileSlots[(size_t) slotIndex0].currentPaths;

            lastDir = lastFileDialogDirectory;
        }

        auto resolveLastDialogDirectory = [&lastDir]() -> juce::File
        {
            if (lastDir.isEmpty())
                return {};

            const auto dir = juce::File (lastDir);
            return dir.isDirectory() ? dir : juce::File();
        };

        auto resolveSlotDirectory = [&currentPaths]() -> juce::File
        {
            for (auto it = currentPaths.rbegin(); it != currentPaths.rend(); ++it)
            {
                const auto dir = juce::File (*it).getParentDirectory();
                if (dir.isDirectory())
                    return dir;
            }

            return {};
        };

        if (preferLastDialogDirectory)
        {
            if (const auto dir = resolveLastDialogDirectory(); dir.isDirectory())
                return dir;
        }

        if (const auto dir = resolveSlotDirectory(); dir.isDirectory())
            return dir;

        if (const auto dir = resolveLastDialogDirectory(); dir.isDirectory())
            return dir;

        return juce::File::getSpecialLocation (juce::File::userDocumentsDirectory);
    }

    std::vector<FileSelectionState> getRecentFileSelections() const
    {
        std::lock_guard<std::mutex> lk (filePathMutex);
        return recentFileSelections;
    }

    std::vector<FavoriteFileSelection> getFavoriteFileSelections() const
    {
        std::lock_guard<std::mutex> lk (filePathMutex);
        return favoriteFileSelections;
    }

    std::vector<juce::String> getFavoriteCategories() const
    {
        std::lock_guard<std::mutex> lk (filePathMutex);
        auto categories = favoriteCategories;

        for (const auto& favorite : favoriteFileSelections)
            addFavoriteCategoryToList (categories, favorite.category);

        sortFavoriteCategories (categories);
        return categories;
    }

    juce::String getSuggestedFavoriteNameForSelection (const FileSelectionState& selection) const
    {
        FileSelectionState normalised = selection;
        normaliseSelectionPaths (normalised);

        if (normalised.paths.empty())
            return "Favorite";

        const auto fallbackName = makeDefaultFavoriteName (normalised);

        std::lock_guard<std::mutex> lk (filePathMutex);
        for (const auto& favorite : favoriteFileSelections)
            if (selectionsEqualIgnoreCase (favorite.selection, normalised))
                return sanitiseFavoriteName (favorite.name, normalised);

        return fallbackName;
    }

    juce::String getSuggestedFavoriteCategoryForSelection (const FileSelectionState& selection) const
    {
        FileSelectionState normalised = selection;
        normaliseSelectionPaths (normalised);

        if (normalised.paths.empty())
            return {};

        std::lock_guard<std::mutex> lk (filePathMutex);
        for (const auto& favorite : favoriteFileSelections)
            if (selectionsEqualIgnoreCase (favorite.selection, normalised))
                return sanitiseFavoriteCategory (favorite.category);

        // Do not infer a new category from filenames/directories. Categories are
        // user-created labels now; a new favorite defaults to Uncategorized.
        return {};
    }

    bool addFavoriteSelection (const FileSelectionState& selection,
                               const juce::String& requestedName = {},
                               const juce::String& requestedCategory = {})
    {
        bool changed = false;

        {
            std::lock_guard<std::mutex> lk (filePathMutex);

            FileSelectionState normalised = selection;
            normaliseSelectionPaths (normalised);
            if (normalised.paths.empty())
                return false;

            const auto resolvedName = sanitiseFavoriteName (requestedName, normalised);
            const auto resolvedCategory = sanitiseFavoriteCategory (requestedCategory);

            if (resolvedCategory.isNotEmpty())
            {
                changed = addFavoriteCategoryToList (favoriteCategories, resolvedCategory) || changed;
                rememberFavoriteCategoryUpsertLocked (resolvedCategory);
            }

            auto it = std::find_if (favoriteFileSelections.begin(), favoriteFileSelections.end(),
                                    [&normalised] (const FavoriteFileSelection& existing)
                                    {
                                        return selectionsEqualIgnoreCase (existing.selection, normalised);
                                    });

            if (it != favoriteFileSelections.end())
            {
                if (it->id.isEmpty())
                {
                    it->id = juce::Uuid().toString();
                    changed = true;
                }

                if (it->name != resolvedName)
                {
                    it->name = resolvedName;
                    changed = true;
                }

                if (it->category != resolvedCategory)
                {
                    it->category = resolvedCategory;
                    changed = true;
                }

                if (changed)
                    rememberFavoriteUpsertLocked (*it);
            }
            else
            {
                FavoriteFileSelection favorite;
                favorite.id = juce::Uuid().toString();
                favorite.name = resolvedName;
                favorite.category = resolvedCategory;
                favorite.selection = normalised;
                favorite.addedUtcMs = juce::Time::getCurrentTime().toMilliseconds();
                favoriteFileSelections.push_back (std::move (favorite));
                rememberFavoriteUpsertLocked (favoriteFileSelections.back());
                sortFavoritesNewestFirst (favoriteFileSelections);
                changed = true;
            }
        }

        if (changed)
            savePersistentFileUiState();

        return changed;
    }

    bool updateFavorite (const juce::String& favoriteId,
                         const juce::String& requestedName,
                         const juce::String& requestedCategory)
    {
        bool changed = false;

        {
            std::lock_guard<std::mutex> lk (filePathMutex);

            for (auto& favorite : favoriteFileSelections)
            {
                if (! favoriteIdsEqualIgnoreCase (favorite.id, favoriteId))
                    continue;

                const auto resolvedName = sanitiseFavoriteName (requestedName, favorite.selection);
                const auto resolvedCategory = sanitiseFavoriteCategory (requestedCategory);

                if (resolvedCategory.isNotEmpty())
                {
                    changed = addFavoriteCategoryToList (favoriteCategories, resolvedCategory) || changed;
                    rememberFavoriteCategoryUpsertLocked (resolvedCategory);
                }

                if (favorite.name != resolvedName)
                {
                    favorite.name = resolvedName;
                    changed = true;
                }

                if (favorite.category != resolvedCategory)
                {
                    favorite.category = resolvedCategory;
                    changed = true;
                }

                if (changed)
                    rememberFavoriteUpsertLocked (favorite);

                break;
            }
        }

        if (changed)
            savePersistentFileUiState();

        return changed;
    }

    bool removeFavorite (const juce::String& favoriteId)
    {
        bool changed = false;

        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            auto newEnd = std::remove_if (favoriteFileSelections.begin(), favoriteFileSelections.end(),
                                          [&favoriteId] (const FavoriteFileSelection& favorite)
                                          {
                                              return favoriteIdsEqualIgnoreCase (favorite.id, favoriteId);
                                          });
            changed = (newEnd != favoriteFileSelections.end());
            favoriteFileSelections.erase (newEnd, favoriteFileSelections.end());
            if (changed)
                rememberFavoriteDeleteLocked (favoriteId);
        }

        if (changed)
            savePersistentFileUiState();

        return changed;
    }

    bool addFavoriteCategory (const juce::String& requestedCategory)
    {
        bool changed = false;

        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            changed = addFavoriteCategoryToList (favoriteCategories, requestedCategory);
            if (changed)
                rememberFavoriteCategoryUpsertLocked (requestedCategory);
        }

        if (changed)
            savePersistentFileUiState();

        return changed;
    }

    bool addFavoriteSubCategory (const juce::String& requestedParentCategory,
                                 const juce::String& requestedChildCategory)
    {
        const auto parent = sanitiseFavoriteCategory (requestedParentCategory);
        const auto child = sanitiseFavoriteCategory (requestedChildCategory);

        if (child.isEmpty())
            return false;

        juce::String fullCategory;
        if (parent.isNotEmpty())
            fullCategory = parent + "/" + child;
        else
            fullCategory = child;

        return addFavoriteCategory (fullCategory);
    }

    bool renameFavoriteCategory (const juce::String& requestedOldCategory,
                                 const juce::String& requestedNewCategory)
    {
        const auto oldCategory = sanitiseFavoriteCategory (requestedOldCategory);
        const auto newCategory = sanitiseFavoriteCategory (requestedNewCategory);

        if (oldCategory.isEmpty() || newCategory.isEmpty() || stringEqualsIgnoreCase (oldCategory, newCategory))
            return false;

        bool changed = false;

        {
            std::lock_guard<std::mutex> lk (filePathMutex);

            std::vector<juce::String> remappedCategories;
            remappedCategories.reserve (favoriteCategories.size() + 1);

            for (const auto& category : favoriteCategories)
            {
                const auto remapped = remapFavoriteCategoryPath (category, oldCategory, newCategory);
                changed = (! stringEqualsIgnoreCase (category, remapped)) || changed;
                addFavoriteCategoryToList (remappedCategories, remapped);
            }

            for (auto& favorite : favoriteFileSelections)
            {
                const auto remapped = remapFavoriteCategoryPath (favorite.category, oldCategory, newCategory);
                if (! stringEqualsIgnoreCase (favorite.category, remapped))
                {
                    favorite.category = remapped;
                    changed = true;
                }

                addFavoriteCategoryToList (remappedCategories, favorite.category);
            }

            addFavoriteCategoryToList (remappedCategories, newCategory);
            sortFavoriteCategories (remappedCategories);
            favoriteCategories = std::move (remappedCategories);

            if (changed)
                rememberFavoriteCategoryRenameLocked (oldCategory, newCategory);
        }

        if (changed)
            savePersistentFileUiState();

        return changed;
    }

    bool removeFavoriteCategory (const juce::String& requestedCategory)
    {
        const auto category = sanitiseFavoriteCategory (requestedCategory);
        if (category.isEmpty())
            return false;

        bool changed = false;

        {
            std::lock_guard<std::mutex> lk (filePathMutex);

            auto newEnd = std::remove_if (favoriteCategories.begin(), favoriteCategories.end(),
                                          [&category] (const juce::String& existing)
                                          {
                                              return favoriteCategoryMatchesOrIsChild (existing, category);
                                          });

            changed = (newEnd != favoriteCategories.end());
            favoriteCategories.erase (newEnd, favoriteCategories.end());

            for (auto& favorite : favoriteFileSelections)
            {
                if (favoriteCategoryMatchesOrIsChild (favorite.category, category))
                {
                    favorite.category.clear();
                    changed = true;
                }
            }

            if (changed)
                rememberFavoriteCategoryDeleteLocked (category);
        }

        if (changed)
            savePersistentFileUiState();

        return changed;
    }


    void setFileSlotPath (int slotIndex0, const juce::File& file, bool shouldRememberForUi = true)
    {
        setFileSlotPathsWithMode (slotIndex0,
                                  std::vector<juce::File> { file },
                                  FileLoadMode::SeparateEntries,
                                  shouldRememberForUi);
    }

    void setFileSlotSelection (int slotIndex0,
                               const FileSelectionState& selection,
                               bool shouldRememberForUi = true)
    {
        std::vector<juce::File> files;
        files.reserve (selection.paths.size());

        for (const auto& path : selection.paths)
        {
            if (path.isEmpty())
                continue;

            files.emplace_back (path);
        }

        setFileSlotPathsWithMode (slotIndex0, files, selection.loadMode, shouldRememberForUi, selection.importRecipeXml);
    }

    // ------------------------------------------------------------
    // DSP runtime hooks for JSFX file_*() builtins
    //
    // DSP-JSFX host extension for multi-file slots:
    //   h  = file_open_multi(slot[, mode]);
    //   n  = file_multi_count(h);
    //   ok = file_multi_select(h, zeroBasedIndex);
    //
    // file_open() remains backward-compatible and simply starts on entry 0.
    // file_multi_select() changes which selected file the handle exposes to
    // file_avail/file_var/file_mem/file_riff/file_text/file_seek/file_rewind.
    // Selecting a new entry resets the item cursor to 0. Invalid indices
    // return 0 and leave the previous selection unchanged.
    // ------------------------------------------------------------

    // ------------------------------------------------------------
    // Host <- @gfx / DSP-internal slider edits
    //
    // JSFX scripts can modify sliders from @gfx or from DSP sections
    // (@block/@sample) and call sliderchange()/slider_automate() to notify the
    // host. The @gfx path is out-of-band and the DSP path is realtime, so both
    // end up funneled through the same conversion helpers here.
    // ------------------------------------------------------------
    static constexpr int kGfxSliderPreviewGraceMs = 100;

    int getGfxSliderPreviewGraceSamples() const noexcept
    {
        if (st.srate > 0.0)
            return juce::jmax (128, (int) std::llround (st.srate * ((double) kGfxSliderPreviewGraceMs / 1000.0)));

        return 4096;
    }

    void resetGfxSliderPreviewState() noexcept
    {
        for (size_t i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            gfxSliderPreviewValues[i].store (0.0, std::memory_order_relaxed);
            gfxSliderPreviewSeq[i].store (0u, std::memory_order_relaxed);
            gfxSliderPreviewSeenSeq[i] = 0u;
            gfxSliderPreviewAckSeq[i] = 0u;
            gfxSliderPreviewGraceSamplesRemaining[i] = 0;
        }
    }

    bool jsfxSliderValueToParameterNormalised (size_t sliderIndex, double jsfxValue, float& outNorm) const
    {
        if (! std::isfinite (jsfxValue))
            return false;

        if (apvts == nullptr || sliderIndex >= sliderParamUsed.size() || ! sliderParamUsed[sliderIndex])
            return false;

        const auto& info = sliderParamInfo[sliderIndex];
        auto* p = apvts->getParameter (info.pid);
        if (p == nullptr)
            return false;

        float rawForParam = 0.0f;
        if (info.isChoice)
        {
            const double step = (double) (info.step > 0.0f ? info.step : 1.0f);
            const double signedStep = info.reversed ? -step : step;
            int idx = (int) std::llround ((jsfxValue - (double) info.rangeStart) / signedStep);

            if (auto* c = dynamic_cast<juce::AudioParameterChoice*> (p))
            {
                const int maxIdx = juce::jmax (0, c->choices.size() - 1);
                idx = juce::jlimit (0, maxIdx, idx);
            }
            else
            {
                const auto r = p->getNormalisableRange();
                idx = juce::jlimit ((int) std::llround (r.start), (int) std::llround (r.end), idx);
            }

            rawForParam = (float) idx;
        }
        else
        {
            double v = juce::jlimit<double> ((double) info.min, (double) info.max, jsfxValue);
            const double step = (double) (info.step > 0.0f ? info.step : 0.0f);
            if (step > 0.0)
            {
                const double signedStep = info.reversed ? -step : step;
                const double q = std::llround ((v - (double) info.rangeStart) / signedStep);
                v = (double) info.rangeStart + q * signedStep;
                v = juce::jlimit<double> ((double) info.min, (double) info.max, v);
            }
            rawForParam = (float) v;
        }

        outNorm = p->convertTo0to1 (rawForParam);
        return true;
    }

    double hostParameterToJsfxSliderValue (size_t sliderIndex) const
    {
        if (sliderIndex >= sliderParamUsed.size() || ! sliderParamUsed[sliderIndex])
            return st.sliders[sliderIndex];

        const auto& info = sliderParamInfo[sliderIndex];
        double newVal = st.sliders[sliderIndex];

        if (auto* v = paramAtomics[sliderIndex])
        {
            newVal = (double) v->load();
        }
        else if (apvts != nullptr)
        {
            if (auto* p = apvts->getParameter (info.pid))
            {
                const float norm = p->getValue();
                newVal = (double) p->convertFrom0to1 (norm);
            }
        }

        return za::jsfx::hostSliderValue(info,newVal);
    }

    static bool sliderValuesEquivalent (const SliderParamInfo& info, double a, double b) noexcept
    {
        return za::jsfx::sliderValuesEquivalent(info,a,b);
    }

    SliderMask actualGfxSliderChanges(const SliderMask& requested, const double* before, const double* after) const {
        return za::jsfx::actualSliderChanges(requested,before,after,[&](int i){return sliderParamUsed[(size_t)i];},[&](int i,double a,double b){return sliderValuesEquivalent(sliderParamInfo[(size_t)i],a,b);});
    }
#if ZA_NATIVE_GFX_TEST_RUNNER
    void testGfxButtonChange(int index,double value) {
        std::array<double,DSPJSFX_MAX_SLIDERS> before{},after{};SliderMask all;
        for(int i=0;i<DSPJSFX_MAX_SLIDERS;++i){before[i]=after[i]=hostParameterToJsfxSliderValue((size_t)i);all.set(i);}
        after[(size_t)index]=value;
        const auto changed=actualGfxSliderChanges(all,before.data(),after.data());
        applyGfxSliderChanges(after.data(),DSPJSFX_MAX_SLIDERS,changed,changed,changed);
    }
#endif

    void applyGfxSliderChanges (const double* newSliders, int count,
                                const SliderMask& changeMask,
                                const SliderMask& automateMask,
                                const SliderMask& automateEndMask)
    {
        if (newSliders == nullptr || apvts == nullptr)
            return;

        const int n = juce::jlimit (0, DSPJSFX_MAX_SLIDERS, count);
        const auto applyMask = changeMask | automateMask | automateEndMask;

        for (int i = 0; i < n; ++i)
        {
            if (! applyMask.test (i))
                continue;

            if (! sliderParamUsed[(size_t)i] || ! std::isfinite (newSliders[i]))
                continue;

            const auto& info = sliderParamInfo[(size_t)i];
            auto* p = apvts->getParameter (info.pid);
            if (p == nullptr)
                continue;
            float norm = 0.0f;
            if (! jsfxSliderValueToParameterNormalised ((size_t) i, newSliders[i], norm))
                continue;

            const bool needsHostSet = ! sliderValuesEquivalent (info, hostParameterToJsfxSliderValue ((size_t) i), newSliders[i]);

            if (automateMask.test (i))
            {
                if (! gfxGestureActive[(size_t)i])
                {
                    p->beginChangeGesture();
                    gfxGestureActive[(size_t)i] = true;
                }
            }

            if (needsHostSet)
                p->setValueNotifyingHost (norm);

            if (automateEndMask.test (i))
            {
                if (gfxGestureActive[(size_t)i])
                {
                    p->endChangeGesture();
                    gfxGestureActive[(size_t)i] = false;
                }
                else
                {
                    p->endChangeGesture();
                }
            }
        }
    }

    void consumeDspSliderChanges()
    {
        const SliderMask changeMask = loadSliderMaskWords (st.pendingSliderChangeMask);
        const SliderMask automateMask = loadSliderMaskWords (st.pendingSliderAutomateMask);
        const SliderMask automateEndMask = loadSliderMaskWords (st.pendingSliderAutomateEndMask);
        const SliderMask applyMask = changeMask | automateMask | automateEndMask;

        if (! applyMask.any() || apvts == nullptr)
        {
            clearSliderMaskWords (st.pendingSliderChangeMask);
            clearSliderMaskWords (st.pendingSliderAutomateMask);
            clearSliderMaskWords (st.pendingSliderAutomateEndMask);
            return;
        }

        for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! applyMask.test (i))
                continue;

            if (! sliderParamUsed[(size_t) i])
                continue;

            const auto& info = sliderParamInfo[(size_t) i];
            auto* p = apvts->getParameter (info.pid);
            if (p == nullptr)
                continue;

            float norm = 0.0f;
            if (! jsfxSliderValueToParameterNormalised ((size_t) i, st.sliders[i], norm))
                continue;

            const bool needsHostSet = ! sliderValuesEquivalent (info, hostParameterToJsfxSliderValue ((size_t) i), st.sliders[i]);

            if (automateMask.test (i))
            {
                if (! dspGestureActive[(size_t) i])
                {
                    p->beginChangeGesture();
                    dspGestureActive[(size_t) i] = true;
                }
            }

            if (needsHostSet)
                p->setValueNotifyingHost (norm);

            if (automateEndMask.test (i))
            {
                if (dspGestureActive[(size_t) i])
                {
                    p->endChangeGesture();
                    dspGestureActive[(size_t) i] = false;
                }
                else
                {
                    p->endChangeGesture();
                }
            }

            const uint32_t previewSeq = gfxSliderPreviewSeq[(size_t) i].load (std::memory_order_acquire);
            gfxSliderPreviewSeenSeq[(size_t) i] = previewSeq;
            gfxSliderPreviewAckSeq[(size_t) i] = previewSeq;
            gfxSliderPreviewGraceSamplesRemaining[(size_t) i] = 0;

            retainScriptSlider(i,st.sliders[i]);
        }

        gfxSnapshotForcePublish.store (true, std::memory_order_release);
        lastSlidersValid = true;
        clearSliderMaskWords (st.pendingSliderChangeMask);
        clearSliderMaskWords (st.pendingSliderAutomateMask);
        clearSliderMaskWords (st.pendingSliderAutomateEndMask);
    }

    void endAllGfxGestures()
    {
        if (apvts == nullptr)
            return;

        for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! gfxGestureActive[(size_t)i])
                continue;

            if (! sliderParamUsed[(size_t)i])
                continue;

            const auto& info = sliderParamInfo[(size_t)i];
            if (auto* p = apvts->getParameter (info.pid))
                p->endChangeGesture();

            gfxGestureActive[(size_t)i] = false;
        }
    }


    void endAllDspGestures()
    {
        if (apvts == nullptr)
        {
            dspGestureActive.fill (false);
            return;
        }

        for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! dspGestureActive[(size_t)i])
                continue;

            if (! sliderParamUsed[(size_t)i])
                continue;

            const auto& info = sliderParamInfo[(size_t)i];
            if (auto* p = apvts->getParameter (info.pid))
                p->endChangeGesture();

            dspGestureActive[(size_t)i] = false;
        }
    }




    // ------------------------------------------------------------
    // GFX snapshot access (audio thread -> UI thread)
    //
    // We keep a small triple-buffered copy of the JSFX runtime state
    // (sliders / vars / mem) so the @gfx interpreter can render meters
    // without ever touching the realtime DSP state directly.
    // ------------------------------------------------------------
    struct GfxSnapshot
    {
        struct MemSpan
        {
            int64_t base = 0;
            int count = 0;
            std::vector<double> data;
        };

       #if DSPJSFX_HAS_NATIVE_GFX
        jsfx_native_gfx::Publication nativePublication;
        std::array<uint32_t, DSPJSFX_MAX_SLIDERS> nativeSliderAcks {};
        uint64_t nativeRuntimeEpoch = 0;
       #endif
        std::array<double, DSPJSFX_MAX_SLIDERS> sliders {};
        std::vector<double> vars;
        std::array<MemSpan, kMaxGfxMemSpans> memSpans {};
        int memSpanCount = 0;
        std::array<GfxMirrorRange, kMaxGfxMemSpans> writableMemRanges {};
        int writableMemRangeCount = 0;
        uint64_t sequence = 0;
        double copyMilliseconds = 0.0;
        int64_t logicalMemN = 0;
        int varsCount = 0;
        double srate = 0.0;
        double samplesblock = 0.0;
    };

   #if DSPJSFX_HAS_NATIVE_GFX
#if DSPJSFX_NATIVE_GFX_LEGACY
    std::mutex& legacyGfxMutex() noexcept { return legacyLifecycleMutex; }
    DSPJSFX_State& legacyScriptState() noexcept { return st; }
    std::shared_ptr<jsfx_native_gfx::Frame> legacyGraphics() noexcept { return legacyFrame; }
    uint64_t legacyEpoch() const noexcept { return nativeRuntimeEpoch; } // under lifecycle mutex
#endif
    // Latest-value scalar mailbox. Critical release cannot be dropped by a full queue.
   #if ZA_SAMPLE_GFX_TEST_RUNNER
    void captureNativeGfxUi (const jsfx_native_gfx::Frame& frame)
    {
        const std::lock_guard<std::mutex> lock (nativeGfxDiagnosticMutex);
        jsfxCopyCells (nativeGfxDiagnosticVars.data(), frame.scriptState().vars, DSPJSFX_VARS_COUNT);
        nativeGfxDiagnosticFault = frame.memoryFault || frame.state.memoryFault;
    }
    double nativeGfxUiValue (const char* name)
    {
        const std::lock_guard<std::mutex> lock (nativeGfxDiagnosticMutex);
        const int index = jsfx_native_gfx::Frame::findIndex (name);
        return index >= 0 ? nativeGfxDiagnosticVars[(size_t) index] : 0;
    }
    bool nativeGfxFaulted()
    {
        const std::lock_guard<std::mutex> lock (nativeGfxDiagnosticMutex);
        return nativeGfxDiagnosticFault;
    }
   #endif
    uint32_t nativeGfxPreviewSequence (int slider) const noexcept
    {
        return gfxSliderPreviewSeq[(size_t) slider].load (std::memory_order_acquire);
    }
    uint64_t nativeGfxCommandEpochValue() const noexcept
    {
        return nativeGfxCommandEpoch.load (std::memory_order_acquire);
    }
    void setNativeGfxInputActive (bool active) noexcept
    {
        // Invalidate old frames before enabling input, so focus regain cannot
        // re-arm a command posted by a frame that started before focus loss.
        nativeGfxCommandEpoch.fetch_add (1, std::memory_order_acq_rel);
        nativeGfxInputActive.store (active, std::memory_order_release);
        requestExternalWakeEvent (false);
    }
    void setNativeGfxCommand (int command, double value, uint64_t epoch) noexcept
    {
        if (command < 0 || command >= DSPJSFX_NATIVE_GFX_COMMAND_COUNT) return;
        const double next = std::isfinite (value) ? value : 0;
        const double previous = nativeGfxCommands[(size_t) command].exchange (next, std::memory_order_relaxed);
        const auto previousEpoch = nativeGfxCommandTags[(size_t) command].exchange (epoch, std::memory_order_release);
        if (previous != next || previousEpoch != epoch) requestExternalWakeEvent (false);
    }
    void restoreNativeGfxUi (jsfx_native_gfx::Frame& frame)
    {
        for (int i = 0; i < DSPJSFX_NATIVE_GFX_PERSIST_COUNT; ++i)
            frame.state.vars[DSPJSFX_NATIVE_GFX_PERSIST_INDICES[i]] = nativeGfxUiState[(size_t) i];
    }
    void saveNativeGfxUi (const jsfx_native_gfx::Frame& frame)
    {
        for (int i = 0; i < DSPJSFX_NATIVE_GFX_PERSIST_COUNT; ++i)
            nativeGfxUiState[(size_t) i] = frame.state.vars[DSPJSFX_NATIVE_GFX_PERSIST_INDICES[i]];
    }
   #endif
   #if ZA_SAMPLE_GFX_TEST_RUNNER
    void sampleCopyStats (uint64_t& count, double& total, double& maximum) const noexcept
    {
        count = sampleCopyCount.load (std::memory_order_relaxed);
        total = sampleCopyTotal.load (std::memory_order_relaxed);
        maximum = sampleCopyMax.load (std::memory_order_relaxed);
    }
   #endif
    bool usesExplicitGfxSync() const noexcept { return gfxSyncExplicitOnly; }
    bool isGfxProfilingEnabled() const noexcept { return gfxProfileEnabled; }

    // A reader pin and a writer claim compete on the same slot atomic.
    // No reader can enter after a writer has selected a slot, even if its
    // front-index load was delayed across several publications.
    const GfxSnapshot* beginGfxSnapshotRead (int& outIndex) noexcept
    {
        outIndex = -1;
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            const int idx = gfxSnapFront.load (std::memory_order_acquire);
            auto& owners = gfxSnapOwners[(size_t) idx];
            int count = owners.load (std::memory_order_relaxed);
            if (count < 0 || count == std::numeric_limits<int>::max())
                continue;
            if (! owners.compare_exchange_strong (count, count + 1,
                                                   std::memory_order_acquire,
                                                   std::memory_order_relaxed))
                continue;
            if (gfxSnapFront.load (std::memory_order_acquire) == idx)
            {
                outIndex = idx;
                return &gfxSnaps[(size_t) idx];
            }
            owners.fetch_sub (1, std::memory_order_release);
        }
        return nullptr; // Skip this frame rather than block either thread.
    }

    void endGfxSnapshotRead (int index) noexcept
    {
        if (index >= 0 && index < (int) gfxSnaps.size())
            gfxSnapOwners[(size_t) index].fetch_sub (1, std::memory_order_release);
    }


    void setFileSlotRenderedAudioData (int slotIndex0,
                                       const std::vector<juce::File>& sourceFiles,
                                       std::vector<za::fileimport::AudioFileData> renderedAudio,
                                       bool shouldRememberForUi = true,
                                       const juce::String& importRecipeXml = {})
    {
        if (slotIndex0 < 0 || slotIndex0 >= (int) fileSlots.size())
            return;

        if (renderedAudio.empty())
        {
            clearFileSlot (slotIndex0);
            return;
        }

        auto samplePoolSources = makeSamplePoolSourcesFromImportAudio (renderedAudio);

        auto cached = std::make_unique<CachedFileSet>();
        cached->entries.reserve (renderedAudio.size());

        for (auto& audio : renderedAudio)
        {
            auto data = makeCachedFileDataFromImportAudio (audio);
            if (! data)
                continue;

            CachedFileEntry entry;
            entry.path = audio.sourceName.isNotEmpty() ? audio.sourceName : juce::String ("memory://za-import/rendered");
            entry.data = std::move (data);
            cached->entries.push_back (std::move (entry));
        }

        auto& slot = fileSlots[(size_t) slotIndex0];
        std::unique_lock<std::mutex> handoffLock (slot.handoffMutex);

        if (cached->entries.empty())
        {
            slot.errorCode.store (2);
            slot.state.store (FileState::Error);
            return;
        }

        std::vector<juce::File> normalisedFiles;
        std::vector<juce::String> fullPaths;
        normalisedFiles.reserve (sourceFiles.size());
        fullPaths.reserve (sourceFiles.size());

        for (const auto& file : sourceFiles)
        {
            const auto fullPath = file.getFullPathName();
            if (fullPath.isEmpty())
                continue;

            normalisedFiles.push_back (file);
            fullPaths.push_back (fullPath);
        }

        if (fullPaths.empty())
        {
            for (const auto& entry : cached->entries)
                fullPaths.push_back (entry.path);
        }

        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            slot.currentPaths = fullPaths;
            slot.loadMode = FileLoadMode::SeparateEntries;
            slot.currentRecipeXml = importRecipeXml;
            slot.currentSamplePoolMemorySources = std::move (samplePoolSources);

            if (shouldRememberForUi && ! normalisedFiles.empty())
                rememberFileSelectionLocked (normalisedFiles, FileLoadMode::SeparateEntries, importRecipeXml);
        }

        if (auto* oldPending = slot.pending.exchange (nullptr))
            delete oldPending;

        slot.errorCode.store (0);
        const uint64_t gen = slot.generation.fetch_add (1) + 1;
        slot.pendingGeneration.store (gen);

        auto* raw = cached.release();
        if (auto* old = slot.pending.exchange (raw))
            delete old;

        slot.state.store (FileState::ReadyPending);
        handoffLock.unlock();
        if (shouldRememberForUi)
            savePersistentFileUiState();
        notifyFileSlotChangedForUiAndSamplePools (slotIndex0);
    }

private:
    void mixQueuedImportPreviewAudition (float* const* outputChannels, int numOutputChannels, int numSamples)
    {
        if (importPreviewAuditionStopRequested.exchange (false, std::memory_order_acq_rel))
            activeImportPreviewAudition.reset();

        if (numSamples <= 0 || outputChannels == nullptr || numOutputChannels <= 0)
            return;

        if (importPreviewAuditionPending.load (std::memory_order_acquire))
        {
            if (importPreviewAuditionMutex.try_lock())
            {
                if (pendingImportPreviewAudition != nullptr)
                    activeImportPreviewAudition = std::move (pendingImportPreviewAudition);

                importPreviewAuditionPending.store (false, std::memory_order_release);
                importPreviewAuditionMutex.unlock();
            }
        }

        if (activeImportPreviewAudition == nullptr || importPreviewAuditionPaused.load (std::memory_order_acquire))
            return;

        auto& clip = *activeImportPreviewAudition;
        const int sourceSamples = clip.buffer.getNumSamples();
        const int sourceChannels = clip.buffer.getNumChannels();
        if (sourceSamples <= 0 || sourceChannels <= 0 || clip.position >= sourceSamples)
        {
            activeImportPreviewAudition.reset();
            return;
        }

        const int todo = juce::jmin (numSamples, sourceSamples - clip.position);
        constexpr float auditionGain = 0.85f;

        for (int outCh = 0; outCh < numOutputChannels; ++outCh)
        {
            auto* dst = outputChannels[outCh];
            if (dst == nullptr)
                continue;

            const int srcCh = sourceChannels == 1 ? 0 : juce::jmin (outCh, sourceChannels - 1);
            const auto* src = clip.buffer.getReadPointer (srcCh, clip.position);
            for (int i = 0; i < todo; ++i)
                dst[i] += src[i] * auditionGain;
        }

        clip.position += todo;
        if (clip.position >= sourceSamples)
            activeImportPreviewAudition.reset();
    }

    static constexpr int kMaxRecentFiles = 10;

    using SavedFileSlotSelection = FileSelectionState;
    using RecentFileSelection = FileSelectionState;
    using FavoriteSelection = FavoriteFileSelection;

    static int findGeneratedStateVarIndexIgnoreCase (const char* wantedName) noexcept
    {
        if (wantedName == nullptr || *wantedName == '\0')
            return -1;

        const juce::String wanted (wantedName);
        for (int i = 0; i < DSPJSFX_VARS_COUNT; ++i)
        {
            if (DSPJSFX_VARS[i].name == nullptr)
                continue;

            if (juce::String (DSPJSFX_VARS[i].name).equalsIgnoreCase (wanted))
                return DSPJSFX_VARS[i].index;
        }

        return -1;
    }

    void initSmartIdleConfig() {
        initialiseIdle(st,kJsfxSourceText,{DSPJSFX_NUM_INPUTS,DSPJSFX_NUM_OUTPUTS,DSPJSFX_ACCEPTS_MIDI_INPUT!=0,DSPJSFX_PRODUCES_MIDI_OUTPUT!=0,!fileDecls.empty()},[](const char* name){return findGeneratedStateVarIndexIgnoreCase(name);});
    }
    bool idleIsNonRealtime() const noexcept override {return isNonRealtime();}

    static bool stringEqualsIgnoreCase (const juce::String& a, const juce::String& b)
    {
        return a.compareIgnoreCase (b) == 0;
    }

    static bool pathsEqualIgnoreCase (const std::vector<juce::String>& a, const std::vector<juce::String>& b)
    {
        if (a.size() != b.size())
            return false;

        for (size_t i = 0; i < a.size(); ++i)
            if (! stringEqualsIgnoreCase (a[i], b[i]))
                return false;

        return true;
    }

    static bool selectionsEqualIgnoreCase (const FileSelectionState& a, const FileSelectionState& b)
    {
        return a.loadMode == b.loadMode
            && a.importRecipeXml == b.importRecipeXml
            && pathsEqualIgnoreCase (a.paths, b.paths);
    }

    static FileLoadMode fileLoadModeFromToken (const juce::String& modeToken)
    {
        if (modeToken.equalsIgnoreCase ("appendEach") || modeToken.equalsIgnoreCase ("append"))
            return FileLoadMode::AppendAsSingleFile;

        return FileLoadMode::SeparateEntries;
    }

    static juce::String fileLoadModeToToken (FileLoadMode loadMode)
    {
        return loadMode == FileLoadMode::AppendAsSingleFile ? juce::String ("appendEach")
                                                            : juce::String ("separate");
    }

    static void normaliseSelectionPaths (FileSelectionState& selection)
    {
        selection.paths.erase (std::remove_if (selection.paths.begin(), selection.paths.end(),
                                               [] (const juce::String& path)
                                               {
                                                   return path.isEmpty();
                                               }),
                               selection.paths.end());
    }

    static FileSelectionState getSelectionFromValueTree (const juce::ValueTree& tree)
    {
        FileSelectionState out;
        out.loadMode = fileLoadModeFromToken (tree.getProperty ("mode").toString());
        out.importRecipeXml = tree.getProperty ("importRecipeXml").toString();
        if (out.importRecipeXml.isEmpty())
        {
            const auto recipeTree = tree.getChildWithName ("ZA_IMPORT_RECIPE");
            if (recipeTree.isValid())
                if (auto xml = recipeTree.createXml())
                    out.importRecipeXml = xml->toString();
        }

        const auto singlePath = tree.getProperty ("path").toString();
        if (singlePath.isNotEmpty())
            out.paths.push_back (singlePath);

        for (int pathIndex = 0; pathIndex < tree.getNumChildren(); ++pathIndex)
        {
            const auto pathTree = tree.getChild (pathIndex);
            if (! pathTree.hasType ("PATH"))
                continue;

            const auto path = pathTree.getProperty ("path").toString();
            if (path.isEmpty())
                continue;

            out.paths.push_back (path);
        }

        normaliseSelectionPaths (out);
        return out;
    }

    static void writeSelectionToValueTree (const FileSelectionState& selection, juce::ValueTree& tree)
    {
        FileSelectionState normalised = selection;
        normaliseSelectionPaths (normalised);

        if (normalised.paths.empty())
            return;

        if (normalised.importRecipeXml.isNotEmpty())
            tree.setProperty ("importRecipeXml", normalised.importRecipeXml, nullptr);

        if (normalised.loadMode != FileLoadMode::SeparateEntries || normalised.paths.size() > 1)
            tree.setProperty ("mode", fileLoadModeToToken (normalised.loadMode), nullptr);

        if (normalised.paths.size() == 1)
        {
            tree.setProperty ("path", normalised.paths.front(), nullptr);
            return;
        }

        for (const auto& path : normalised.paths)
        {
            juce::ValueTree pathTree ("PATH");
            pathTree.setProperty ("path", path, nullptr);
            tree.addChild (pathTree, -1, nullptr);
        }
    }


    static bool favoriteIdsEqualIgnoreCase (const juce::String& a, const juce::String& b)
    {
        return a.isNotEmpty() && b.isNotEmpty() && stringEqualsIgnoreCase (a, b);
    }

    static bool favoriteEntriesEqualIgnoreCase (const FavoriteFileSelection& a, const FavoriteFileSelection& b)
    {
        if (favoriteIdsEqualIgnoreCase (a.id, b.id))
            return true;

        return selectionsEqualIgnoreCase (a.selection, b.selection);
    }

    static juce::String normaliseFavoriteUserText (juce::String text)
    {
        text = text.replaceCharacters ("\r\n\t", "   ");
        return text.trim();
    }

    static std::vector<juce::String> splitFavoriteCategoryPath (juce::String text)
    {
        text = normaliseFavoriteUserText (text).replaceCharacter ('\\', '/');

        std::vector<juce::String> parts;
        juce::String current;

        for (int i = 0; i < text.length(); ++i)
        {
            const auto c = text[i];
            if (c == '/')
            {
                auto part = normaliseFavoriteUserText (current);
                if (part.isNotEmpty())
                    parts.push_back (part);

                current.clear();
                continue;
            }

            current << juce::String::charToString (c);
        }

        auto part = normaliseFavoriteUserText (current);
        if (part.isNotEmpty())
            parts.push_back (part);

        return parts;
    }

    static juce::String normaliseFavoriteCategoryPath (const juce::String& requestedCategory)
    {
        const auto parts = splitFavoriteCategoryPath (requestedCategory);
        juce::StringArray out;

        for (const auto& part : parts)
            if (part.isNotEmpty())
                out.add (part);

        return out.joinIntoString ("/");
    }

    static bool categoryExistsInList (const std::vector<juce::String>& categories,
                                      const juce::String& category)
    {
        const auto normalised = normaliseFavoriteCategoryPath (category);
        if (normalised.isEmpty())
            return false;

        return std::any_of (categories.begin(), categories.end(),
                            [&normalised] (const juce::String& existing)
                            {
                                return stringEqualsIgnoreCase (normaliseFavoriteCategoryPath (existing), normalised);
                            });
    }

    static void sortFavoriteCategories (std::vector<juce::String>& categories)
    {
        for (auto& category : categories)
            category = normaliseFavoriteCategoryPath (category);

        categories.erase (std::remove_if (categories.begin(), categories.end(),
                                          [] (const juce::String& category)
                                          {
                                              return category.trim().isEmpty();
                                          }),
                          categories.end());

        std::stable_sort (categories.begin(), categories.end(),
                          [] (const juce::String& a, const juce::String& b)
                          {
                              const auto aa = normaliseFavoriteCategoryPath (a);
                              const auto bb = normaliseFavoriteCategoryPath (b);

                              const bool aIsParentOfB = bb.startsWithIgnoreCase (aa + "/");
                              const bool bIsParentOfA = aa.startsWithIgnoreCase (bb + "/");

                              if (aIsParentOfB != bIsParentOfA)
                                  return aIsParentOfB;

                              return aa.compareIgnoreCase (bb) < 0;
                          });

        categories.erase (std::unique (categories.begin(), categories.end(),
                                       [] (const juce::String& a, const juce::String& b)
                                       {
                                           return stringEqualsIgnoreCase (normaliseFavoriteCategoryPath (a),
                                                                         normaliseFavoriteCategoryPath (b));
                                       }),
                          categories.end());
    }

    static bool addFavoriteCategoryToList (std::vector<juce::String>& categories,
                                           const juce::String& requestedCategory)
    {
        const auto parts = splitFavoriteCategoryPath (requestedCategory);
        if (parts.empty())
            return false;

        bool changed = false;
        juce::String current;

        for (const auto& part : parts)
        {
            if (current.isNotEmpty())
                current << "/";

            current << part;

            if (! categoryExistsInList (categories, current))
            {
                categories.push_back (current);
                changed = true;
            }
        }

        if (changed)
            sortFavoriteCategories (categories);

        return changed;
    }

    static bool favoriteCategoryMatchesOrIsChild (const juce::String& requestedCategory,
                                                  const juce::String& requestedParentCategory)
    {
        const auto category = normaliseFavoriteCategoryPath (requestedCategory);
        const auto parent = normaliseFavoriteCategoryPath (requestedParentCategory);

        if (category.isEmpty() || parent.isEmpty())
            return false;

        if (stringEqualsIgnoreCase (category, parent))
            return true;

        return category.startsWithIgnoreCase (parent + "/");
    }

    static juce::String remapFavoriteCategoryPath (const juce::String& requestedCategory,
                                                   const juce::String& requestedOldCategory,
                                                   const juce::String& requestedNewCategory)
    {
        const auto category = normaliseFavoriteCategoryPath (requestedCategory);
        const auto oldCategory = normaliseFavoriteCategoryPath (requestedOldCategory);
        const auto newCategory = normaliseFavoriteCategoryPath (requestedNewCategory);

        if (category.isEmpty() || oldCategory.isEmpty() || newCategory.isEmpty())
            return category;

        if (stringEqualsIgnoreCase (category, oldCategory))
            return newCategory;

        if (category.startsWithIgnoreCase (oldCategory + "/"))
            return normaliseFavoriteCategoryPath (newCategory + category.substring (oldCategory.length()));

        return category;
    }

    static juce::String trimFavoriteNameEdges (juce::String text)
    {
        text = normaliseFavoriteUserText (text);

        while (text.isNotEmpty())
        {
            const auto c = text[text.length() - 1];
            if (juce::CharacterFunctions::isLetterOrDigit (c))
                break;

            text = text.dropLastCharacters (1).trim();
        }

        while (text.isNotEmpty())
        {
            const auto c = text[0];
            if (juce::CharacterFunctions::isLetterOrDigit (c))
                break;

            text = text.substring (1).trim();
        }

        return text.trim();
    }

    static juce::String trimFavoriteCategoryEdges (juce::String text)
    {
        return trimFavoriteNameEdges (std::move (text));
    }

    static juce::String findCommonFileStemPrefix (const std::vector<juce::String>& paths)
    {
        if (paths.empty())
            return {};

        juce::String prefix = juce::File (paths.front()).getFileNameWithoutExtension();
        if (prefix.isEmpty())
            prefix = juce::File (paths.front()).getFileName();

        if (prefix.isEmpty())
            prefix = paths.front();

        for (size_t i = 1; i < paths.size() && prefix.isNotEmpty(); ++i)
        {
            juce::String other = juce::File (paths[i]).getFileNameWithoutExtension();
            if (other.isEmpty())
                other = juce::File (paths[i]).getFileName();
            if (other.isEmpty())
                other = paths[i];

            const int maxLen = juce::jmin (prefix.length(), other.length());
            int sharedLen = 0;

            while (sharedLen < maxLen)
            {
                if (juce::CharacterFunctions::toLowerCase (prefix[sharedLen])
                      != juce::CharacterFunctions::toLowerCase (other[sharedLen]))
                    break;

                ++sharedLen;
            }

            prefix = prefix.substring (0, sharedLen);
        }

        prefix = trimFavoriteNameEdges (prefix);
        if (prefix.length() < 3)
            return {};

        return prefix;
    }

    static juce::String makeDefaultFavoriteName (const FileSelectionState& selection)
    {
        FileSelectionState normalised = selection;
        normaliseSelectionPaths (normalised);

        if (normalised.paths.empty())
            return "Favorite";

        if (const auto prefix = findCommonFileStemPrefix (normalised.paths); prefix.isNotEmpty())
            return prefix;

        const juce::File leadFile (normalised.paths.front());
        auto leadStem = trimFavoriteNameEdges (leadFile.getFileNameWithoutExtension());
        if (leadStem.isEmpty())
            leadStem = trimFavoriteNameEdges (leadFile.getFileName());
        if (leadStem.isEmpty())
            leadStem = trimFavoriteNameEdges (normalised.paths.front());

        if (normalised.paths.size() == 1 && leadStem.isNotEmpty())
            return leadStem;

        const auto firstParent = leadFile.getParentDirectory();
        if (firstParent.getFullPathName().isNotEmpty())
        {
            const auto allSameParent = std::all_of (normalised.paths.begin(), normalised.paths.end(),
                                                    [&firstParent] (const juce::String& path)
                                                    {
                                                        return stringEqualsIgnoreCase (juce::File (path).getParentDirectory().getFullPathName(),
                                                                                       firstParent.getFullPathName());
                                                    });

            if (allSameParent)
            {
                auto parentName = trimFavoriteNameEdges (firstParent.getFileName());
                if (parentName.isEmpty())
                    parentName = trimFavoriteNameEdges (firstParent.getFullPathName());

                if (parentName.isNotEmpty())
                    return parentName;
            }
        }

        if (leadStem.isNotEmpty())
            return leadStem + " +" + juce::String ((int) normalised.paths.size() - 1);

        return "Favorite";
    }

    static juce::String makeDefaultFavoriteCategory (const FileSelectionState& selection)
    {
        FileSelectionState normalised = selection;
        normaliseSelectionPaths (normalised);

        if (normalised.paths.empty())
            return {};

        const juce::File leadFile (normalised.paths.front());
        const auto firstParent = leadFile.getParentDirectory();
        if (firstParent.getFullPathName().isEmpty())
            return {};

        const auto allSameParent = std::all_of (normalised.paths.begin(), normalised.paths.end(),
                                                [&firstParent] (const juce::String& path)
                                                {
                                                    return stringEqualsIgnoreCase (juce::File (path).getParentDirectory().getFullPathName(),
                                                                                   firstParent.getFullPathName());
                                                });

        if (! allSameParent)
            return {};

        auto category = trimFavoriteCategoryEdges (firstParent.getFileName());
        if (category.isEmpty())
            category = trimFavoriteCategoryEdges (firstParent.getFullPathName());

        return category;
    }

    static juce::String sanitiseFavoriteName (const juce::String& requestedName,
                                              const FileSelectionState& selection)
    {
        auto cleaned = normaliseFavoriteUserText (requestedName);
        if (cleaned.isEmpty())
            cleaned = makeDefaultFavoriteName (selection);
        else
            cleaned = normaliseFavoriteUserText (cleaned);

        if (cleaned.isEmpty())
            cleaned = "Favorite";

        return cleaned;
    }

    static juce::String sanitiseFavoriteCategory (const juce::String& requestedCategory)
    {
        return normaliseFavoriteCategoryPath (requestedCategory);
    }

    static FavoriteFileSelection getFavoriteFromValueTree (const juce::ValueTree& tree)
    {
        FavoriteFileSelection out;
        out.id = tree.getProperty ("id").toString();
        out.name = tree.getProperty ("name").toString();
        out.category = tree.getProperty ("category").toString();
        out.addedUtcMs = (juce::int64) tree.getProperty ("addedUtcMs", (juce::int64) 0);
        out.selection = getSelectionFromValueTree (tree);

        if (out.selection.paths.empty())
            return out;

        if (out.id.isEmpty())
            out.id = juce::Uuid().toString();

        out.name = sanitiseFavoriteName (out.name, out.selection);
        out.category = sanitiseFavoriteCategory (out.category);
        return out;
    }

    static void writeFavoriteToValueTree (const FavoriteFileSelection& favorite, juce::ValueTree& tree)
    {
        FavoriteFileSelection normalised = favorite;
        normaliseSelectionPaths (normalised.selection);

        if (normalised.selection.paths.empty())
            return;

        if (normalised.id.isNotEmpty())
            tree.setProperty ("id", normalised.id, nullptr);

        const auto name = sanitiseFavoriteName (normalised.name, normalised.selection);
        if (name.isNotEmpty())
            tree.setProperty ("name", name, nullptr);

        const auto category = sanitiseFavoriteCategory (normalised.category);
        if (category.isNotEmpty())
            tree.setProperty ("category", category, nullptr);

        if (normalised.addedUtcMs > 0)
            tree.setProperty ("addedUtcMs", (juce::int64) normalised.addedUtcMs, nullptr);

        writeSelectionToValueTree (normalised.selection, tree);
    }

    static void sortFavoritesNewestFirst (std::vector<FavoriteFileSelection>& favorites)
    {
        std::stable_sort (favorites.begin(), favorites.end(),
                          [] (const FavoriteFileSelection& a, const FavoriteFileSelection& b)
                          {
                              return a.addedUtcMs > b.addedUtcMs;
                          });
    }

    static void pushRecentSelectionToFront (std::vector<RecentFileSelection>& recentSelections,
                                            const RecentFileSelection& selection)
    {
        FileSelectionState normalised = selection;
        normaliseSelectionPaths (normalised);

        if (normalised.paths.empty())
            return;

        recentSelections.erase (std::remove_if (recentSelections.begin(), recentSelections.end(),
                                                [&normalised] (const RecentFileSelection& existing)
                                                {
                                                    return selectionsEqualIgnoreCase (existing, normalised);
                                                }),
                                recentSelections.end());

        recentSelections.insert (recentSelections.begin(), normalised);

        if ((int) recentSelections.size() > kMaxRecentFiles)
            recentSelections.resize ((size_t) kMaxRecentFiles);
    }

    static SavedFileSlotSelection getSavedFileSlotSelection (const juce::ValueTree& files, int slotIndex0)
    {
        SavedFileSlotSelection out;

        for (int i = 0; i < files.getNumChildren(); ++i)
        {
            const auto slotTree = files.getChild (i);
            if (! slotTree.hasType ("SLOT"))
                continue;

            if ((int) slotTree.getProperty ("index", -1) != slotIndex0)
                continue;

            out = getSelectionFromValueTree (slotTree);
            break;
        }

        if (! out.paths.empty())
            return out;

        const juce::Identifier key ("f" + juce::String (slotIndex0));
        if (files.hasProperty (key))
        {
            const auto path = files.getProperty (key).toString();
            if (path.isNotEmpty())
                out.paths.push_back (path);
        }

        return out;
    }

    static juce::String getPersistentFileUiStateBaseName (const juce::String& pluginName)
    {
        auto baseName = juce::File::createLegalFileName (pluginName);
        if (baseName.isEmpty())
            baseName = "JSFX";

        return baseName;
    }

    juce::File getPersistentFileUiStateFile() const
    {
        return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                    .getChildFile ("ZorakAudio")
                    .getChildFile ("JSFXFileUiState")
                    .getChildFile (getPersistentFileUiStateBaseName (getName()) + "_recent.xml");
    }

    juce::String getPersistentFileUiStateLockName() const
    {
        return "ZA_JSFX_FILE_UI_" + getPersistentFileUiStateBaseName (getName());
    }

    void clearRememberedFileUiState()
    {
        std::lock_guard<std::mutex> lk (filePathMutex);
        lastFileDialogDirectory.clear();
        recentFileSelections.clear();
        favoriteFileSelections.clear();
        favoriteCategories.clear();
        pendingFavoriteUpserts.clear();
        pendingFavoriteDeleteIds.clear();
        pendingFavoriteCategoryUpserts.clear();
        pendingFavoriteCategoryRenames.clear();
        pendingFavoriteCategoryDeletePaths.clear();
    }

    void rememberFileSelectionLocked (const std::vector<juce::File>& files,
                                      FileLoadMode loadMode,
                                      const juce::String& importRecipeXml = {})
    {
        if (files.empty())
            return;

        for (const auto& file : files)
        {
            const auto parent = file.getParentDirectory();
            if (parent.isDirectory())
            {
                lastFileDialogDirectory = parent.getFullPathName();
                break;
            }
        }

        RecentFileSelection selection;
        selection.loadMode = loadMode;
        selection.importRecipeXml = importRecipeXml;
        selection.paths.reserve (files.size());

        for (const auto& file : files)
        {
            const auto fullPath = file.getFullPathName();
            if (fullPath.isEmpty())
                continue;

            selection.paths.push_back (fullPath);
        }

        pushRecentSelectionToFront (recentFileSelections, selection);
    }

    struct PersistentFileUiSnapshot
    {
        juce::String lastDir;
        std::vector<RecentFileSelection> recent;
        std::vector<FavoriteFileSelection> favorites;
        std::vector<juce::String> categories;

        // Per-instance mutation intent since the last successful persistent save.
        // The transaction applies these to the newest on-disk snapshot instead of
        // serializing this instance's possibly stale favorites/categories vector.
        std::vector<FavoriteFileSelection> favoriteUpserts;
        std::vector<juce::String> favoriteDeleteIds;
        std::vector<juce::String> categoryUpserts;
        std::vector<std::pair<juce::String, juce::String>> categoryRenames;
        std::vector<juce::String> categoryDeletePaths;
    };

    PersistentFileUiSnapshot capturePersistentFileUiSnapshot (bool includePendingMutations = true) const
    {
        std::lock_guard<std::mutex> lk (filePathMutex);

        PersistentFileUiSnapshot snapshot;
        snapshot.lastDir = lastFileDialogDirectory;
        snapshot.recent = recentFileSelections;
        snapshot.favorites = favoriteFileSelections;
        snapshot.categories = favoriteCategories;

        if (includePendingMutations)
        {
            snapshot.favoriteUpserts = pendingFavoriteUpserts;
            snapshot.favoriteDeleteIds = pendingFavoriteDeleteIds;
            snapshot.categoryUpserts = pendingFavoriteCategoryUpserts;
            snapshot.categoryRenames = pendingFavoriteCategoryRenames;
            snapshot.categoryDeletePaths = pendingFavoriteCategoryDeletePaths;
        }

        return snapshot;
    }

    void applyPersistentFileUiSnapshot (const PersistentFileUiSnapshot& snapshot)
    {
        std::lock_guard<std::mutex> lk (filePathMutex);
        lastFileDialogDirectory = snapshot.lastDir;
        recentFileSelections = snapshot.recent;
        favoriteFileSelections = snapshot.favorites;
        favoriteCategories = snapshot.categories;
    }

    static bool favoriteUpsertEntriesEqual (const FavoriteFileSelection& a,
                                            const FavoriteFileSelection& b)
    {
        return favoriteEntriesEqualIgnoreCase (a, b)
            && stringEqualsIgnoreCase (a.name, b.name)
            && stringEqualsIgnoreCase (a.category, b.category)
            && a.addedUtcMs == b.addedUtcMs;
    }

    void clearAppliedPendingPersistentFileUiMutations (const PersistentFileUiSnapshot& applied)
    {
        std::lock_guard<std::mutex> lk (filePathMutex);

        pendingFavoriteUpserts.erase (std::remove_if (pendingFavoriteUpserts.begin(), pendingFavoriteUpserts.end(), [&applied] (const FavoriteFileSelection& pending)
        {
            return std::any_of (applied.favoriteUpserts.begin(), applied.favoriteUpserts.end(), [&pending] (const FavoriteFileSelection& done)
            {
                return favoriteUpsertEntriesEqual (pending, done);
            });
        }), pendingFavoriteUpserts.end());

        pendingFavoriteDeleteIds.erase (std::remove_if (pendingFavoriteDeleteIds.begin(), pendingFavoriteDeleteIds.end(), [&applied] (const juce::String& pending)
        {
            return std::any_of (applied.favoriteDeleteIds.begin(), applied.favoriteDeleteIds.end(), [&pending] (const juce::String& done)
            {
                return favoriteIdsEqualIgnoreCase (pending, done);
            });
        }), pendingFavoriteDeleteIds.end());

        pendingFavoriteCategoryUpserts.erase (std::remove_if (pendingFavoriteCategoryUpserts.begin(), pendingFavoriteCategoryUpserts.end(), [&applied] (const juce::String& pending)
        {
            return std::any_of (applied.categoryUpserts.begin(), applied.categoryUpserts.end(), [&pending] (const juce::String& done)
            {
                return stringEqualsIgnoreCase (pending, done);
            });
        }), pendingFavoriteCategoryUpserts.end());

        pendingFavoriteCategoryRenames.erase (std::remove_if (pendingFavoriteCategoryRenames.begin(), pendingFavoriteCategoryRenames.end(), [&applied] (const std::pair<juce::String, juce::String>& pending)
        {
            return std::any_of (applied.categoryRenames.begin(), applied.categoryRenames.end(), [&pending] (const std::pair<juce::String, juce::String>& done)
            {
                return stringEqualsIgnoreCase (pending.first, done.first)
                    && stringEqualsIgnoreCase (pending.second, done.second);
            });
        }), pendingFavoriteCategoryRenames.end());

        pendingFavoriteCategoryDeletePaths.erase (std::remove_if (pendingFavoriteCategoryDeletePaths.begin(), pendingFavoriteCategoryDeletePaths.end(), [&applied] (const juce::String& pending)
        {
            return std::any_of (applied.categoryDeletePaths.begin(), applied.categoryDeletePaths.end(), [&pending] (const juce::String& done)
            {
                return stringEqualsIgnoreCase (pending, done);
            });
        }), pendingFavoriteCategoryDeletePaths.end());
    }

    static void appendCaseInsensitiveUnique (std::vector<juce::String>& items, const juce::String& value)
    {
        const auto trimmed = value.trim();
        if (trimmed.isEmpty())
            return;

        const auto exists = std::any_of (items.begin(), items.end(), [&trimmed] (const juce::String& existing)
        {
            return stringEqualsIgnoreCase (existing, trimmed);
        });

        if (! exists)
            items.push_back (trimmed);
    }

    void rememberFavoriteCategoryUpsertLocked (const juce::String& requestedCategory)
    {
        const auto category = sanitiseFavoriteCategory (requestedCategory);
        if (category.isEmpty())
            return;

        appendCaseInsensitiveUnique (pendingFavoriteCategoryUpserts, category);
    }

    void rememberFavoriteUpsertLocked (FavoriteFileSelection favorite)
    {
        normaliseSelectionPaths (favorite.selection);
        if (favorite.selection.paths.empty())
            return;

        if (favorite.id.isEmpty())
            favorite.id = juce::Uuid().toString();

        if (favorite.name.trim().isEmpty())
            favorite.name = sanitiseFavoriteName ({}, favorite.selection);

        favorite.category = sanitiseFavoriteCategory (favorite.category);

        if (favorite.addedUtcMs <= 0)
            favorite.addedUtcMs = juce::Time::getCurrentTime().toMilliseconds();

        if (favorite.category.isNotEmpty())
            rememberFavoriteCategoryUpsertLocked (favorite.category);

        pendingFavoriteDeleteIds.erase (std::remove_if (pendingFavoriteDeleteIds.begin(), pendingFavoriteDeleteIds.end(), [&favorite] (const juce::String& id)
        {
            return favoriteIdsEqualIgnoreCase (id, favorite.id);
        }), pendingFavoriteDeleteIds.end());

        auto it = std::find_if (pendingFavoriteUpserts.begin(), pendingFavoriteUpserts.end(), [&favorite] (const FavoriteFileSelection& existing)
        {
            return favoriteIdsEqualIgnoreCase (existing.id, favorite.id)
                || selectionsEqualIgnoreCase (existing.selection, favorite.selection);
        });

        if (it == pendingFavoriteUpserts.end())
            pendingFavoriteUpserts.push_back (std::move (favorite));
        else
            *it = std::move (favorite);
    }

    void rememberFavoriteDeleteLocked (const juce::String& favoriteId)
    {
        const auto id = favoriteId.trim();
        if (id.isEmpty())
            return;

        appendCaseInsensitiveUnique (pendingFavoriteDeleteIds, id);

        pendingFavoriteUpserts.erase (std::remove_if (pendingFavoriteUpserts.begin(), pendingFavoriteUpserts.end(), [&id] (const FavoriteFileSelection& favorite)
        {
            return favoriteIdsEqualIgnoreCase (favorite.id, id);
        }), pendingFavoriteUpserts.end());
    }

    void rememberFavoriteCategoryRenameLocked (const juce::String& requestedOldCategory,
                                               const juce::String& requestedNewCategory)
    {
        const auto oldCategory = sanitiseFavoriteCategory (requestedOldCategory);
        const auto newCategory = sanitiseFavoriteCategory (requestedNewCategory);
        if (oldCategory.isEmpty() || newCategory.isEmpty() || stringEqualsIgnoreCase (oldCategory, newCategory))
            return;

        pendingFavoriteCategoryDeletePaths.erase (std::remove_if (pendingFavoriteCategoryDeletePaths.begin(), pendingFavoriteCategoryDeletePaths.end(), [&oldCategory] (const juce::String& existing)
        {
            return stringEqualsIgnoreCase (existing, oldCategory);
        }), pendingFavoriteCategoryDeletePaths.end());

        pendingFavoriteCategoryRenames.push_back ({ oldCategory, newCategory });
        rememberFavoriteCategoryUpsertLocked (newCategory);
    }

    void rememberFavoriteCategoryDeleteLocked (const juce::String& requestedCategory)
    {
        const auto category = sanitiseFavoriteCategory (requestedCategory);
        if (category.isEmpty())
            return;

        appendCaseInsensitiveUnique (pendingFavoriteCategoryDeletePaths, category);

        pendingFavoriteCategoryUpserts.erase (std::remove_if (pendingFavoriteCategoryUpserts.begin(), pendingFavoriteCategoryUpserts.end(), [&category] (const juce::String& existing)
        {
            return favoriteCategoryMatchesOrIsChild (existing, category);
        }), pendingFavoriteCategoryUpserts.end());
    }

    static bool writeTextFileTransactionally (const juce::File& target, const juce::String& text)
    {
        const auto dir = target.getParentDirectory();
        if (dir.createDirectory().failed())
            return false;

        const auto temp = dir.getNonexistentChildFile (target.getFileNameWithoutExtension() + "." + juce::Uuid().toString(),
                                                       target.getFileExtension() + ".tmp",
                                                       false);

        if (! temp.replaceWithText (text, false, false, "\n"))
        {
            temp.deleteFile();
            return false;
        }

        const bool replaced = target.existsAsFile() ? temp.replaceFileIn (target)
                                                    : temp.moveFileTo (target);
        if (! replaced)
        {
            temp.deleteFile();
            return false;
        }

        return true;
    }

    static void normalisePersistentSnapshotForSave (PersistentFileUiSnapshot& snapshot)
    {
        std::vector<RecentFileSelection> recent;
        recent.reserve (snapshot.recent.size());

        for (auto selection : snapshot.recent)
        {
            normaliseSelectionPaths (selection);
            if (selection.paths.empty())
                continue;

            const bool alreadySeen = std::any_of (recent.begin(), recent.end(), [&selection] (const RecentFileSelection& existing)
            {
                return selectionsEqualIgnoreCase (existing, selection);
            });

            if (alreadySeen)
                continue;

            recent.push_back (std::move (selection));
            if ((int) recent.size() >= kMaxRecentFiles)
                break;
        }

        snapshot.recent = std::move (recent);

        std::vector<FavoriteFileSelection> favorites;
        favorites.reserve (snapshot.favorites.size());

        for (auto favorite : snapshot.favorites)
        {
            normaliseSelectionPaths (favorite.selection);
            if (favorite.selection.paths.empty())
                continue;

            if (favorite.id.isEmpty())
                favorite.id = juce::Uuid().toString();

            favorite.name = sanitiseFavoriteName (favorite.name, favorite.selection);
            favorite.category = sanitiseFavoriteCategory (favorite.category);

            if (favorite.addedUtcMs <= 0)
                favorite.addedUtcMs = juce::Time::getCurrentTime().toMilliseconds();

            auto it = std::find_if (favorites.begin(), favorites.end(), [&favorite] (const FavoriteFileSelection& existing)
            {
                return favoriteEntriesEqualIgnoreCase (existing, favorite);
            });

            if (it == favorites.end())
                favorites.push_back (std::move (favorite));
            else
                *it = std::move (favorite);
        }

        snapshot.favorites = std::move (favorites);

        std::vector<juce::String> categories;
        categories.reserve (snapshot.categories.size() + snapshot.favorites.size());

        for (const auto& category : snapshot.categories)
            addFavoriteCategoryToList (categories, category);

        for (const auto& favorite : snapshot.favorites)
            addFavoriteCategoryToList (categories, favorite.category);

        sortFavoritesNewestFirst (snapshot.favorites);
        sortFavoriteCategories (categories);
        snapshot.categories = std::move (categories);
    }

    PersistentFileUiSnapshot snapshotFromPersistentFileUiTree (const juce::ValueTree& fileUi) const
    {
        PersistentFileUiSnapshot snapshot;
        if (! fileUi.isValid())
            return snapshot;

        snapshot.lastDir = fileUi.getProperty ("lastDir").toString();
        snapshot.recent.reserve ((size_t) juce::jmax (0, fileUi.getNumChildren()));
        snapshot.favorites.reserve ((size_t) juce::jmax (0, fileUi.getNumChildren()));
        snapshot.categories.reserve ((size_t) juce::jmax (0, fileUi.getNumChildren()));

        for (int i = 0; i < fileUi.getNumChildren(); ++i)
        {
            const auto child = fileUi.getChild (i);

            if (child.hasType ("RECENT"))
            {
                if ((int) snapshot.recent.size() >= kMaxRecentFiles)
                    continue;

                auto selection = getSelectionFromValueTree (child);
                if (selection.paths.empty())
                    continue;

                const bool alreadySeen = std::any_of (snapshot.recent.begin(), snapshot.recent.end(), [&selection] (const RecentFileSelection& existing)
                {
                    return selectionsEqualIgnoreCase (existing, selection);
                });

                if (! alreadySeen)
                    snapshot.recent.push_back (std::move (selection));
            }
            else if (child.hasType ("FAVORITE"))
            {
                auto favorite = getFavoriteFromValueTree (child);
                if (favorite.selection.paths.empty())
                    continue;

                addFavoriteCategoryToList (snapshot.categories, favorite.category);

                const bool alreadySeen = std::any_of (snapshot.favorites.begin(), snapshot.favorites.end(), [&favorite] (const FavoriteFileSelection& existing)
                {
                    return favoriteEntriesEqualIgnoreCase (existing, favorite);
                });

                if (! alreadySeen)
                    snapshot.favorites.push_back (std::move (favorite));
            }
            else if (child.hasType ("CATEGORY"))
            {
                addFavoriteCategoryToList (snapshot.categories, child.getProperty ("name").toString());
            }
        }

        if (snapshot.lastDir.isEmpty())
        {
            for (const auto& selection : snapshot.recent)
            {
                if (selection.paths.empty())
                    continue;

                const auto dir = juce::File (selection.paths.front()).getParentDirectory();
                if (dir.isDirectory())
                {
                    snapshot.lastDir = dir.getFullPathName();
                    break;
                }
            }

            if (snapshot.lastDir.isEmpty())
            {
                for (const auto& favorite : snapshot.favorites)
                {
                    if (favorite.selection.paths.empty())
                        continue;

                    const auto dir = juce::File (favorite.selection.paths.front()).getParentDirectory();
                    if (dir.isDirectory())
                    {
                        snapshot.lastDir = dir.getFullPathName();
                        break;
                    }
                }
            }
        }

        normalisePersistentSnapshotForSave (snapshot);
        return snapshot;
    }

    PersistentFileUiSnapshot readPersistentFileUiSnapshotFromDiskOnly() const
    {
        const auto stateFile = getPersistentFileUiStateFile();
        if (! stateFile.existsAsFile())
            return {};

        std::unique_ptr<juce::XmlElement> xml (juce::parseXML (stateFile));
        if (xml == nullptr)
            return {};

        const auto tree = juce::ValueTree::fromXml (*xml);
        if (! tree.isValid())
            return {};

        return snapshotFromPersistentFileUiTree (tree);
    }

    static juce::ValueTree makePersistentFileUiValueTree (PersistentFileUiSnapshot snapshot)
    {
        normalisePersistentSnapshotForSave (snapshot);

        juce::ValueTree fileUi ("FILE_UI");
        if (snapshot.lastDir.isNotEmpty())
            fileUi.setProperty ("lastDir", snapshot.lastDir, nullptr);

        for (const auto& selection : snapshot.recent)
        {
            if (selection.paths.empty())
                continue;

            juce::ValueTree recent ("RECENT");
            writeSelectionToValueTree (selection, recent);
            fileUi.addChild (recent, -1, nullptr);
        }

        for (const auto& category : snapshot.categories)
        {
            const auto name = sanitiseFavoriteCategory (category);
            if (name.isEmpty())
                continue;

            juce::ValueTree categoryTree ("CATEGORY");
            categoryTree.setProperty ("name", name, nullptr);
            fileUi.addChild (categoryTree, -1, nullptr);
        }

        for (const auto& favorite : snapshot.favorites)
        {
            if (favorite.selection.paths.empty())
                continue;

            juce::ValueTree favoriteTree ("FAVORITE");
            writeFavoriteToValueTree (favorite, favoriteTree);
            fileUi.addChild (favoriteTree, -1, nullptr);
        }

        return fileUi;
    }

    void mergeFavoriteIntoSnapshot (PersistentFileUiSnapshot& target, FavoriteFileSelection incoming) const
    {
        normaliseSelectionPaths (incoming.selection);
        if (incoming.selection.paths.empty())
            return;

        if (incoming.id.isEmpty())
            incoming.id = juce::Uuid().toString();

        incoming.name = sanitiseFavoriteName (incoming.name, incoming.selection);
        incoming.category = sanitiseFavoriteCategory (incoming.category);

        if (incoming.addedUtcMs <= 0)
            incoming.addedUtcMs = juce::Time::getCurrentTime().toMilliseconds();

        auto it = std::find_if (target.favorites.begin(), target.favorites.end(), [&incoming] (const FavoriteFileSelection& existing)
        {
            return favoriteIdsEqualIgnoreCase (existing.id, incoming.id);
        });

        if (it == target.favorites.end())
        {
            it = std::find_if (target.favorites.begin(), target.favorites.end(), [&incoming] (const FavoriteFileSelection& existing)
            {
                return selectionsEqualIgnoreCase (existing.selection, incoming.selection);
            });
        }

        if (it == target.favorites.end())
            target.favorites.push_back (std::move (incoming));
        else
            *it = std::move (incoming);
    }

    PersistentFileUiSnapshot mergePersistentFileUiSnapshots (PersistentFileUiSnapshot disk,
                                                             PersistentFileUiSnapshot local) const
    {
        normalisePersistentSnapshotForSave (disk);
        normalisePersistentSnapshotForSave (local);

        if (local.lastDir.isNotEmpty())
            disk.lastDir = local.lastDir;

        if (! local.recent.empty())
        {
            for (auto it = local.recent.rbegin(); it != local.recent.rend(); ++it)
                pushRecentSelectionToFront (disk.recent, *it);
        }

        for (const auto& rename : local.categoryRenames)
        {
            std::vector<juce::String> remappedCategories;
            remappedCategories.reserve (disk.categories.size() + 1);

            for (const auto& category : disk.categories)
                addFavoriteCategoryToList (remappedCategories, remapFavoriteCategoryPath (category, rename.first, rename.second));

            for (auto& favorite : disk.favorites)
            {
                favorite.category = remapFavoriteCategoryPath (favorite.category, rename.first, rename.second);
                addFavoriteCategoryToList (remappedCategories, favorite.category);
            }

            addFavoriteCategoryToList (remappedCategories, rename.second);
            sortFavoriteCategories (remappedCategories);
            disk.categories = std::move (remappedCategories);
        }

        for (const auto& category : local.categoryDeletePaths)
        {
            disk.categories.erase (std::remove_if (disk.categories.begin(), disk.categories.end(), [&category] (const juce::String& existing)
            {
                return favoriteCategoryMatchesOrIsChild (existing, category);
            }), disk.categories.end());

            for (auto& favorite : disk.favorites)
                if (favoriteCategoryMatchesOrIsChild (favorite.category, category))
                    favorite.category.clear();
        }

        for (const auto& id : local.favoriteDeleteIds)
        {
            disk.favorites.erase (std::remove_if (disk.favorites.begin(), disk.favorites.end(), [&id] (const FavoriteFileSelection& favorite)
            {
                return favoriteIdsEqualIgnoreCase (favorite.id, id);
            }), disk.favorites.end());
        }

        for (const auto& category : local.categoryUpserts)
            addFavoriteCategoryToList (disk.categories, category);

        for (const auto& favorite : local.favoriteUpserts)
            mergeFavoriteIntoSnapshot (disk, favorite);

        // First-run/project-load fallback: when there is no persisted favorite
        // state yet and this instance has an in-memory favorite list, seed disk
        // with that list without ever replacing a non-empty disk snapshot.
        if (disk.favorites.empty() && local.favoriteUpserts.empty())
        {
            for (const auto& favorite : local.favorites)
                mergeFavoriteIntoSnapshot (disk, favorite);
        }

        if (disk.categories.empty() && local.categoryUpserts.empty())
        {
            for (const auto& category : local.categories)
                addFavoriteCategoryToList (disk.categories, category);
        }

        normalisePersistentSnapshotForSave (disk);
        return disk;
    }

    bool writePersistentFileUiStateSnapshotUnlocked (const PersistentFileUiSnapshot& snapshot) const
    {
        const auto stateFile = getPersistentFileUiStateFile();
        const auto fileUi = makePersistentFileUiValueTree (snapshot);
        std::unique_ptr<juce::XmlElement> xml (fileUi.createXml());
        if (xml == nullptr)
            return false;

        return writeTextFileTransactionally (stateFile, xml->toString());
    }

    void savePersistentFileUiState()
    {
        const auto local = capturePersistentFileUiSnapshot (true);

        std::lock_guard<std::mutex> sameProcessGuard (gPersistentFileUiStateMutex);

        juce::InterProcessLock fileLock (getPersistentFileUiStateLockName());
        if (! fileLock.enter (10000))
        {
            DBG ("Could not acquire persistent FILE_UI lock; skipping save to avoid clobbering favorites.");
            return;
        }

        struct ScopedFileUiLockExit
        {
            juce::InterProcessLock& lock;
            ~ScopedFileUiLockExit() { lock.exit(); }
        } releaseLock { fileLock };

        const auto merged = mergePersistentFileUiSnapshots (readPersistentFileUiSnapshotFromDiskOnly(), local);

        if (writePersistentFileUiStateSnapshotUnlocked (merged))
        {
            applyPersistentFileUiSnapshot (merged);
            clearAppliedPendingPersistentFileUiMutations (local);
        }
    }

    void restoreFileUiState (const juce::ValueTree& fileUi,
                             bool mergeWithExisting = false,
                             bool saveAsPersistent = false)
    {
        juce::String restoredLastDir = fileUi.getProperty ("lastDir").toString();
        std::vector<RecentFileSelection> restoredRecent;
        std::vector<FavoriteFileSelection> restoredFavorites;
        std::vector<juce::String> restoredCategories;
        restoredRecent.reserve ((size_t) juce::jmax (0, fileUi.getNumChildren()));
        restoredFavorites.reserve ((size_t) juce::jmax (0, fileUi.getNumChildren()));
        restoredCategories.reserve ((size_t) juce::jmax (0, fileUi.getNumChildren()));

        for (int i = 0; i < fileUi.getNumChildren(); ++i)
        {
            const auto child = fileUi.getChild (i);

            if (child.hasType ("RECENT"))
            {
                if ((int) restoredRecent.size() >= kMaxRecentFiles)
                    continue;

                auto selection = getSelectionFromValueTree (child);
                if (selection.paths.empty())
                    continue;

                const bool alreadySeen = std::any_of (restoredRecent.begin(), restoredRecent.end(),
                                                      [&selection] (const RecentFileSelection& existing)
                                                      {
                                                          return selectionsEqualIgnoreCase (existing, selection);
                                                      });
                if (alreadySeen)
                    continue;

                restoredRecent.push_back (std::move (selection));

                if ((int) restoredRecent.size() >= kMaxRecentFiles)
                    continue;
            }
            else if (child.hasType ("FAVORITE"))
            {
                auto favorite = getFavoriteFromValueTree (child);
                if (favorite.selection.paths.empty())
                    continue;

                addFavoriteCategoryToList (restoredCategories, favorite.category);

                const bool alreadySeen = std::any_of (restoredFavorites.begin(), restoredFavorites.end(),
                                                      [&favorite] (const FavoriteFileSelection& existing)
                                                      {
                                                          return favoriteEntriesEqualIgnoreCase (existing, favorite);
                                                      });
                if (alreadySeen)
                    continue;

                restoredFavorites.push_back (std::move (favorite));
            }
            else if (child.hasType ("CATEGORY"))
            {
                addFavoriteCategoryToList (restoredCategories, child.getProperty ("name").toString());
            }
        }

        sortFavoritesNewestFirst (restoredFavorites);
        sortFavoriteCategories (restoredCategories);

        const auto persistentRestoreFavoriteUpserts = restoredFavorites;
        const auto persistentRestoreCategoryUpserts = restoredCategories;

        if (restoredLastDir.isEmpty())
        {
            for (const auto& selection : restoredRecent)
            {
                if (selection.paths.empty())
                    continue;

                const auto dir = juce::File (selection.paths.front()).getParentDirectory();
                if (dir.isDirectory())
                {
                    restoredLastDir = dir.getFullPathName();
                    break;
                }
            }

            if (restoredLastDir.isEmpty())
            {
                for (const auto& favorite : restoredFavorites)
                {
                    if (favorite.selection.paths.empty())
                        continue;

                    const auto dir = juce::File (favorite.selection.paths.front()).getParentDirectory();
                    if (dir.isDirectory())
                    {
                        restoredLastDir = dir.getFullPathName();
                        break;
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lk (filePathMutex);

            if (! mergeWithExisting)
            {
                lastFileDialogDirectory = restoredLastDir;
                recentFileSelections = std::move (restoredRecent);
                favoriteFileSelections = std::move (restoredFavorites);
                favoriteCategories = std::move (restoredCategories);
            }
            else
            {
                if (lastFileDialogDirectory.isEmpty())
                    lastFileDialogDirectory = restoredLastDir;

                if (recentFileSelections.empty())
                {
                    recentFileSelections = std::move (restoredRecent);
                }
                else
                {
                    for (const auto& selection : restoredRecent)
                    {
                        const bool alreadySeen = std::any_of (recentFileSelections.begin(), recentFileSelections.end(),
                                                              [&selection] (const RecentFileSelection& existing)
                                                              {
                                                                  return selectionsEqualIgnoreCase (existing, selection);
                                                              });
                        if (alreadySeen)
                            continue;

                        recentFileSelections.push_back (selection);
                        if ((int) recentFileSelections.size() >= kMaxRecentFiles)
                            break;
                    }
                }

                if (favoriteFileSelections.empty())
                {
                    favoriteFileSelections = std::move (restoredFavorites);
                }
                else
                {
                    for (const auto& favorite : restoredFavorites)
                    {
                        auto it = std::find_if (favoriteFileSelections.begin(), favoriteFileSelections.end(),
                                                [&favorite] (const FavoriteFileSelection& existing)
                                                {
                                                    return favoriteEntriesEqualIgnoreCase (existing, favorite);
                                                });

                        if (it == favoriteFileSelections.end())
                        {
                            favoriteFileSelections.push_back (favorite);
                            continue;
                        }

                        if (it->id.isEmpty())
                            it->id = favorite.id;

                        if (it->name.isEmpty())
                            it->name = favorite.name;

                        if (it->category.isEmpty() && favorite.category.isNotEmpty())
                            it->category = favorite.category;

                        if (it->addedUtcMs <= 0 && favorite.addedUtcMs > 0)
                            it->addedUtcMs = favorite.addedUtcMs;
                    }
                }

                for (const auto& category : restoredCategories)
                    addFavoriteCategoryToList (favoriteCategories, category);

                for (const auto& favorite : favoriteFileSelections)
                    addFavoriteCategoryToList (favoriteCategories, favorite.category);

                sortFavoritesNewestFirst (favoriteFileSelections);
                sortFavoriteCategories (favoriteCategories);

                if (lastFileDialogDirectory.isEmpty() && ! recentFileSelections.empty())
                {
                    const auto dir = juce::File (recentFileSelections.front().paths.front()).getParentDirectory();
                    if (dir.isDirectory())
                        lastFileDialogDirectory = dir.getFullPathName();
                }

                if (lastFileDialogDirectory.isEmpty() && ! favoriteFileSelections.empty())
                {
                    const auto dir = juce::File (favoriteFileSelections.front().selection.paths.front()).getParentDirectory();
                    if (dir.isDirectory())
                        lastFileDialogDirectory = dir.getFullPathName();
                }
            }
        }

        if (saveAsPersistent)
        {
            {
                std::lock_guard<std::mutex> lk (filePathMutex);

                for (const auto& category : persistentRestoreCategoryUpserts)
                    rememberFavoriteCategoryUpsertLocked (category);

                for (const auto& favorite : persistentRestoreFavoriteUpserts)
                    rememberFavoriteUpsertLocked (favorite);
            }

            savePersistentFileUiState();
        }
    }

    void loadPersistentFileUiState()
    {
        const auto stateFile = getPersistentFileUiStateFile();
        if (! stateFile.existsAsFile())
            return;

        juce::InterProcessLock fileLock (getPersistentFileUiStateLockName());
        if (! fileLock.enter (1500))
            return;

        std::unique_ptr<juce::XmlElement> xml (juce::parseXML (stateFile));
        fileLock.exit();

        if (xml == nullptr)
            return;

        const auto tree = juce::ValueTree::fromXml (*xml);
        if (! tree.isValid())
            return;

        restoreFileUiState (tree, false, false);
    }

    void initFileRuntime()
    {
        initialiseFileSlots(fileDecls);

        loadPersistentFileUiState();

        // Auto-load defaults if the filename token looks like a path and resolves.
        for (const auto& d : fileDecls)
        {
            if (d.defaultPath.isEmpty())
                continue;

            const auto f = resolveFileToken (d.defaultPath);
            if (f.existsAsFile())
                setFileSlotPath (d.index0, f, false);
        }
    }

    void reinitialiseJsfxForCurrentEngineRate (int hostBlockSamples)
    {
        const int factor = getActiveOversamplingFactor();
        const int engineBlock = safeOversampledBlockSize (hostBlockSamples, factor);
        const double engineRate = getEffectiveEngineSampleRate();

        endAllDspGestures();
        resetStateStructOnly();
        st.srate = engineRate;
        st.currentSampleRate = engineRate;
        prepareMidiRuntime (engineBlock);
        requestEmergencyMidiCleanup();
        updateSmartIdleSampleRate (engineRate);
        resetSmartIdleRuntime();

        gfxSnapPeriodSamples = (int64_t) juce::jmax (128.0, engineRate / 30.0);
        gfxSnapCountdown = 0;

        lastSlidersValid = false;
        internalSliderPendingMask.clear();
        dspGestureActive.fill (false);
        resetGfxSliderPreviewState();
        (void) pushParamsToStateSliders();
        (void) applyStringSlidersToState (true);

#if DSPJSFX_HAS_FAUST
        st.faustContext=faustEngine.get();
        faustEngine->reset(double(st.srate));
#endif
        productionProgram.initialiseAndPrime(st,[&] {
        const int varsCap = (int) za::jsfx::variableCount(st);
        for (size_t i = 0; i < sliderAliasVarIndex.size(); ++i)
        {
            const int vIdx = sliderAliasVarIndex[i];
            if (vIdx >= 0 && vIdx < varsCap)
                st.vars[vIdx] = st.sliders[i];
        }

        });
        syncJsfxLatency();

       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        if (correctnessRuntime != nullptr)
        {
            correctnessRuntime->setRingSampleRate (engineRate);
            correctnessRuntime->resetAndPrime (st);
        }
       #endif

        gfxSnapshotForcePublish.store (true, std::memory_order_release);
        updateGfxSnapshotIfNeeded ((int) gfxSnapPeriodSamples);
        requestExternalWakeEvent (false);
    }

    void applyOversamplingFactorChangeIfNeeded (int hostBlockSamples)
    {
        const int requestedFactor = getRequestedOversamplingFactor();
        const int currentFactor = getActiveOversamplingFactor();
        const double hostRate = getSampleRate() > 1000.0 ? getSampleRate() : oversamplingHostSampleRate.load (std::memory_order_acquire);

        if (hostRate > 1000.0)
            oversamplingHostSampleRate.store (hostRate, std::memory_order_release);

        const double currentEngineRate = (hostRate > 1000.0 ? hostRate * (double) requestedFactor : st.srate);
        const bool factorChanged = requestedFactor != currentFactor;
        const bool rateChanged = currentEngineRate > 1000.0 && std::abs (currentEngineRate - oversamplingEngineSampleRate.load (std::memory_order_acquire)) > 1.0;

        if (! factorChanged && ! rateChanged)
        {
            pendingOversamplingParameterChange.store (false, std::memory_order_release);
            return;
        }

#if DSPJSFX_NATIVE_GFX_LEGACY
        // Engine re-init may clear scalar storage. Never wait for GFX on audio.
        std::unique_lock<std::mutex> legacyLock (legacyLifecycleMutex, std::try_to_lock);
        if (!legacyLock.owns_lock()) return;
#endif
        oversamplingActiveFactor.store (requestedFactor, std::memory_order_release);
        oversamplingEngineSampleRate.store (currentEngineRate, std::memory_order_release);
        pendingOversamplingParameterChange.store (false, std::memory_order_release);

        reinitialiseJsfxForCurrentEngineRate (hostBlockSamples);

        pendingFileReloadForEngineRate.store (true, std::memory_order_release);
        pendingSamplePoolRecommitForEngineRate.store (true, std::memory_order_release);
        requestUiWakeAsync();
    }

    void reloadCurrentFileSlotsForEngineRate()
    {
        struct ReloadJob
        {
            int slot = -1;
            std::vector<juce::File> files;
            FileLoadMode mode = FileLoadMode::SeparateEntries;
            juce::String recipeXml;
        };

        std::vector<ReloadJob> jobs;
        {
            std::lock_guard<std::mutex> lk (filePathMutex);
            for (int i = 0; i < (int) fileSlots.size(); ++i)
            {
                const auto& slot = fileSlots[(size_t) i];
                if (slot.currentPaths.empty())
                    continue;

                ReloadJob job;
                job.slot = i;
                job.mode = slot.loadMode;
                job.recipeXml = slot.currentRecipeXml;

                for (const auto& path : slot.currentPaths)
                    if (path.isNotEmpty())
                        job.files.emplace_back (path);

                if (! job.files.empty())
                    jobs.push_back (std::move (job));
            }
        }

        for (auto& job : jobs)
            setFileSlotPathsWithMode (job.slot, job.files, job.mode, false, job.recipeXml);
    }

    void initialiseJsfxHostDefaults()
    {
        za::jsfx::initialiseHostDefaults(DSPJSFX_PROCESS_CHANNELS,[this](const char* name,double value){writeExternalJsfxVar(findGeneratedStateVarIndexIgnoreCase(name),value);});
    }

    void initStateMemory()
    {
        std::memset (&st, 0, sizeof (st));
        stateVariables.bind(st,DSPJSFX_VARS_COUNT);
        initialiseJsfxHostDefaults();
        const int64_t initialN = DSPJSFX_NATIVE_GFX_LEGACY ? DSPJSFX_MAX_MEM_CELLS : 65536;
        za::jsfx::allocateHeap(st,initialN);
        prepareMidiRuntime (0);

        st.hostOwner = this;
#if DSPJSFX_NATIVE_GFX_LEGACY
        st.atomicContext = &legacyAtomicMutex;
        if(!legacyFrame)legacyFrame=std::make_shared<jsfx_native_gfx::Frame>();
        legacyFrame->resetDrawing();
        legacyFrame->bindLegacy(st,DSPJSFX_NATIVE_GFX_WIDTH,DSPJSFX_NATIVE_GFX_HEIGHT);
        legacyFrame->frameRecording=false;
        legacyFrame->set("gfx_texth",8);
        legacyFrame->imageLoader=[](const juce::String& name){return jsfx_gfx_resources::loadImage(name);};
        legacyFrame->fileResolver=[this](int index,const juce::String& token){return resolveGfxFileCandidates(index,token);};
        legacyFrame->configureResources(kJsfxSourceText);
        midiRuntime.nativeStrings.graphics=legacyFrame.get();
        st.memoryFault = st.mem == nullptr ? 1 : 0;
        for (const char* name : {"gfx_r", "gfx_g", "gfx_b", "gfx_a", "gfx_a2"}) {
            const int i = jsfx_native_gfx::Frame::findIndex(name);
            if (i >= 0) st.vars[i] = 1;
        }
#endif
        st.memUsed = st.memN;
    }

    void resetStateStructOnly()
    {
#if DSPJSFX_HAS_TASKS
        taskRuntime->reset();
#endif
        za::jsfx::resetGuestStatePreservingHeap(st,stateVariables,DSPJSFX_VARS_COUNT);
#if DSPJSFX_HAS_FAUST
        st.faustContext=faustEngine.get();
#endif
#if DSPJSFX_HAS_TASKS
        st.taskContext = taskRuntime.get();
#endif
        initialiseJsfxHostDefaults();

        jsfxRuntime.reset();
        midiRuntime.resetAll();
        bindMidiRuntimeBuffers();

        st.hostOwner = this;
#if DSPJSFX_NATIVE_GFX_LEGACY
        st.atomicContext = &legacyAtomicMutex;
        if(!legacyFrame)legacyFrame=std::make_shared<jsfx_native_gfx::Frame>();
        legacyFrame->resetDrawing();
        legacyFrame->bindLegacy(st,DSPJSFX_NATIVE_GFX_WIDTH,DSPJSFX_NATIVE_GFX_HEIGHT);
        legacyFrame->frameRecording=false;
        legacyFrame->set("gfx_texth",8);
        legacyFrame->imageLoader=[](const juce::String& name){return jsfx_gfx_resources::loadImage(name);};
        legacyFrame->fileResolver=[this](int index,const juce::String& token){return resolveGfxFileCandidates(index,token);};
        legacyFrame->configureResources(kJsfxSourceText);
        midiRuntime.nativeStrings.graphics=legacyFrame.get();
        st.memoryFault = st.mem == nullptr ? 1 : 0;
        for (const char* name : {"gfx_r", "gfx_g", "gfx_b", "gfx_a", "gfx_a2"}) {
            const int i = jsfx_native_gfx::Frame::findIndex(name);
            if (i >= 0) st.vars[i] = 1;
        }
#endif
        st.memUsed = st.memN;
    }

    void bindMidiRuntimeBuffers() { za::jsfx::MidiHostRuntime::bindMidiRuntimeBuffers(st); }

    void prepareMidiRuntime (int samplesPerBlockExpected) {
        za::jsfx::MidiHostRuntime::prepareMidiRuntime(st,samplesPerBlockExpected);
        hostTransportTracker.reset();hostTransportDiscontinuity=false;
    }

    // All externally injected DSP variables must also reach the WDL oracle.
    void writeExternalJsfxVar(int index, double value)
    {
        if (index < 0)
            return;
       #if DSPJSFX_NATIVE_GFX_LEGACY
        const int alias = DSPJSFX_LEGACY_SLIDER_ALIASES[index];
        if (alias >= 0) st.sliders[alias] = value;
        else st.vars[index] = value;
       #else
        st.vars[index] = value;
       #endif
       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        if (correctnessRuntime != nullptr)
            correctnessRuntime->applyExternalVarWrite(index, value);
       #endif
    }

    void syncJsfxGfxActivity()
    {
        za::jsfx::syncGfxActivity(gfxAnalysisVisible.load(std::memory_order_acquire),isNonRealtime(),[this](const char* name,double value){writeExternalJsfxVar(findGeneratedStateVarIndexIgnoreCase(name),value);});
    }

    void syncJsfxHostTransport(int hostSamples)
    {
        hostTransportDiscontinuity=za::jsfx::syncHostTransport(za::jsfx::collectTransport(getPlayHead()),hostTransportTracker,hostSamples,getSampleRate(),std::max(getTotalNumInputChannels(),getTotalNumOutputChannels()),[this](const char* name,double value){writeExternalJsfxVar(findGeneratedStateVarIndexIgnoreCase(name),value);});
    }

    void syncJsfxLatency()
    {
        static const int idx = findGeneratedStateVarIndexIgnoreCase ("pdc_delay");
        const double raw = idx >= 0 ? st.vars[idx] : 0.0;
        const int samples = za::jsfx::latencyFromGuest(raw,getActiveOversamplingFactor());
        if (getLatencySamples() != samples)
            setLatencySamples(samples);
    }

    bool queryMidiTransportPlayingForCleanup (bool& isPlaying)
    {
        isPlaying = false;

        auto* playHead = getPlayHead();
        if (playHead == nullptr)
            return false;

       #if JUCE_MAJOR_VERSION >= 7
        const auto position = playHead->getPosition();
        if (! position)
            return false;

        isPlaying = position->getIsPlaying();
        return true;
       #else
        juce::AudioPlayHead::CurrentPositionInfo position;
        if (! playHead->getCurrentPosition (position))
            return false;

        isPlaying = position.isPlaying;
        return true;
       #endif
    }

    bool detectMidiTransportStatusChangedForCleanup()
    {
        bool playing = false;
        if (! queryMidiTransportPlayingForCleanup (playing))
        {
            resetMidiTransportStatusForCleanup();
            return false;
        }

        return observeMidiTransport(true,playing);
    }

    void requestEmergencyMidiCleanup()
    {
        pendingEmergencyMidiCleanup.store (true, std::memory_order_release);
        // st.pendingNoteCleanup is read/cleared exclusively by the audio thread.
    }

    void importMidiToState(const juce::MidiBuffer& midi,int hostSamples,int engineSamples,int scale,bool prepend,bool allNotes) {
        za::jsfx::MidiHostRuntime::importMidiToState(st,midi,hostSamples,engineSamples,scale,prepend,allNotes);
    }
    bool flushMidiFromState(juce::MidiBuffer& midi,int divisor=1,int samples=-1) {
        return za::jsfx::MidiHostRuntime::flushMidiFromState(st,midi,divisor,samples);
    }

    bool applyStringSliderIndexToState (size_t i, const juce::String& text) noexcept
    {
        if (i >= DSPJSFX_MAX_SLIDERS || ! sliderStringUsed[i])
            return false;

        double* slot = &st.sliders[i];
        const auto utf8 = text.toRawUTF8();
        const int len = (int) std::strlen (utf8);
        if (jsfx_string_assign_utf8 (&st, slot, utf8, len) <= 0 && len > 0)
            return false;

        const int varsCap = (int) za::jsfx::variableCount(st);
        const int vIdx = sliderAliasVarIndex[i];
        if (vIdx >= 0 && vIdx < varsCap)
            st.vars[vIdx] = jsfxCellLoad(slot);

        return true;
    }

    bool applyStringSlidersToState (bool force)
    {
        bool changed = false;
        for (size_t i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! sliderStringUsed[i])
                continue;

            const uint32_t seq = stringSliderSeq[i].load (std::memory_order_acquire);
            if (! force && seq == stringSliderAppliedSeq[i])
                continue;

            juce::String text;
            {
                std::lock_guard<std::mutex> lk (stringSliderMutex);
                text = stringSliderTexts[i];
            }

            if (applyStringSliderIndexToState (i, text) || force)
            {
                stringSliderAppliedSeq[i] = seq;
                changed = true;
            }
        }
        return changed;
    }

    juce::ValueTree makeStringSliderStateValueTree() const
    {
        juce::ValueTree root ("STRING_SLIDERS");
        std::lock_guard<std::mutex> lk (stringSliderMutex);
        for (size_t i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! sliderStringUsed[i])
                continue;
            juce::ValueTree item ("SLIDER");
            item.setProperty ("index", (int) i, nullptr);
            item.setProperty ("text", stringSliderTexts[i], nullptr);
            root.addChild (item, -1, nullptr);
        }
        return root;
    }

    void restoreStringSliderState (const juce::ValueTree& root)
    {
        for (int ci = 0; ci < root.getNumChildren(); ++ci)
        {
            auto item = root.getChild (ci);
            if (! item.hasType ("SLIDER"))
                continue;
            const int idx = (int) item.getProperty ("index", -1);
            if (idx < 0 || idx >= DSPJSFX_MAX_SLIDERS || ! sliderStringUsed[(size_t) idx])
                continue;
            setStringSliderText (idx, item.getProperty ("text", {}).toString());
        }
    }

    bool pushParamsToStateSliders()
    {
        bool changed = false;
        const int varsCap = (int) za::jsfx::variableCount(st);
        const int previewGraceSamples = getGfxSliderPreviewGraceSamples();
        const int previewGraceDecrement = juce::jmax (1, st.currentBlockSize > 0 ? st.currentBlockSize : 1);

        for (size_t i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! sliderParamUsed[i])
                continue;
            const auto& info = sliderParamInfo[i];
            const double hostVal = hostParameterToJsfxSliderValue (i);
            double newVal = hostVal;

            if (! internalSliderPendingMask.test ((int) i))
            {
                const uint32_t previewSeq = gfxSliderPreviewSeq[i].load (std::memory_order_acquire);
                if (previewSeq != gfxSliderPreviewAckSeq[i])
                {
                    if (previewSeq != gfxSliderPreviewSeenSeq[i])
                    {
                        gfxSliderPreviewSeenSeq[i] = previewSeq;
                        gfxSliderPreviewGraceSamplesRemaining[i] = previewGraceSamples;
                    }

                    const double previewVal = gfxSliderPreviewValues[i].load (std::memory_order_acquire);
                    if (sliderValuesEquivalent (info, hostVal, previewVal))
                    {
                        gfxSliderPreviewAckSeq[i] = previewSeq;
                        gfxSliderPreviewGraceSamplesRemaining[i] = 0;
                    }
                    else if (gfxSliderPreviewGraceSamplesRemaining[i] > 0)
                    {
                        newVal = previewVal;
                        gfxSliderPreviewGraceSamplesRemaining[i] = juce::jmax (0, gfxSliderPreviewGraceSamplesRemaining[i] - previewGraceDecrement);
                    }
                    else
                    {
                        gfxSliderPreviewAckSeq[i] = previewSeq;
                    }
                }
            }

            newVal=resolveHostSlider((int)i,info,hostVal,newVal);

            changed |= applyHostSlider(st,(int)i,newVal,[&](double value){
                const int vIdx=sliderAliasVarIndex[i];
                if(vIdx>=0 && vIdx<varsCap)st.vars[vIdx]=value;
            });
        }

        lastSlidersValid = true;
        return changed;
    }

    struct GfxStateWrite
    {
        enum class Kind : uint8_t { Var = 0, Mem = 1 };
        Kind kind {};
        int32_t index = 0;
        double value = 0.0;
    };

    static constexpr uint32_t gfxWriteQueueSize = 8192u; // power of two
    static constexpr uint32_t gfxWriteQueueMask = gfxWriteQueueSize - 1u;
    static_assert ((gfxWriteQueueSize & (gfxWriteQueueSize - 1u)) == 0u, "gfxWriteQueueSize must be power of two");

    std::array<GfxStateWrite, gfxWriteQueueSize> gfxWriteQueue {};
    std::atomic<uint32_t> gfxWriteHead { 0 }; // UI thread producer
    std::atomic<uint32_t> gfxWriteTail { 0 }; // audio thread consumer

    bool enqueueGfxStateWrite (GfxStateWrite::Kind kind, int index, double value) noexcept
    {
        if (index < 0 || ! std::isfinite (value))
            return false;

        const uint32_t head = gfxWriteHead.load (std::memory_order_relaxed);
        const uint32_t next = (head + 1u) & gfxWriteQueueMask;

        if (next == gfxWriteTail.load (std::memory_order_acquire))
            return false; // full, drop

        gfxWriteQueue[head] = GfxStateWrite { kind, (int32_t) index, value };
        gfxWriteHead.store (next, std::memory_order_release);
        return true;
    }

    bool applyQueuedGfxStateWrites() noexcept
    {
        uint32_t tail = gfxWriteTail.load (std::memory_order_relaxed);
        const uint32_t head = gfxWriteHead.load (std::memory_order_acquire);
        bool changed = false;

        const int varsCap = (int) za::jsfx::variableCount(st);

        while (tail != head)
        {
            const auto w = gfxWriteQueue[tail];
            changed = true;

            if (w.kind == GfxStateWrite::Kind::Var)
            {
                if (w.index >= 0 && w.index < varsCap
                    && (getJsfxGfxVarFlags (w.index) & DSPJSFX_GFX_VAR_FLAG_FROM_GFX) != 0u)
                {
                    st.vars[(size_t) w.index] = w.value;
                   #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
                    if (correctnessRuntime != nullptr)
                        correctnessRuntime->applyExternalVarWrite (w.index, w.value);
                   #endif
                }
            }
            else // Mem
            {
                const int64_t mi = (int64_t) w.index;
                const int64_t logicalMemN = getGfxLogicalJsfxMemN (&st, jsfxDeclaredMaxMem);

                const bool writable = isDirectionalGfxMemIndex (
                    mi, logicalMemN, st.memN, gfxSyncExplicitOnly, gfxSyncMemRanges, kGfxSyncFromGfx);

                if (st.mem != nullptr
                    && mi >= 0
                    && mi < st.memN
                    && writable)
                {
                    st.mem[(size_t) mi] = w.value;
                   #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
                    if (correctnessRuntime != nullptr)
                        correctnessRuntime->applyExternalMemWrite ((int) mi, w.value);
                   #endif
                }
            }

            tail = (tail + 1u) & gfxWriteQueueMask;
        }

        gfxWriteTail.store (tail, std::memory_order_release);
        return changed;
    }

    struct GfxMemSpanWrite
    {
        int64_t base = 0;
        int count = 0;
        bool force = false;
        std::vector<double> data;
    };

    static constexpr uint32_t gfxMemSpanQueueSize = 256u; // power of two
    static constexpr uint32_t gfxMemSpanQueueMask = gfxMemSpanQueueSize - 1u;
    static_assert ((gfxMemSpanQueueSize & (gfxMemSpanQueueSize - 1u)) == 0u,
                   "gfxMemSpanQueueSize must be power of two");

    std::array<GfxMemSpanWrite, gfxMemSpanQueueSize> gfxMemSpanQueue {};
    std::atomic<uint32_t> gfxMemSpanHead { 0 }; // GFX worker producer
    std::atomic<uint32_t> gfxMemSpanTail { 0 }; // audio thread consumer

    // A native GFX file_mem() writes into the same JSFX mem[] namespace that
    // @sample/@block use in REAPER. Once a host-forced GFX span reaches the
    // DSP VM, keep that region bidirectionally mirrored on subsequent
    // snapshots so later DSP edits are visible to @gfx as well. This list is
    // owned exclusively by the audio thread (both mutation and snapshot use).
    std::array<GfxMirrorRange, kMaxGfxMemSpans> gfxDynamicSharedMemRanges {};
    int gfxDynamicSharedMemRangeCount = 0;

    void noteDynamicGfxSharedMemRange (int64_t base, int64_t count) noexcept
    {
        if (base < 0 || count <= 0 || st.memN <= 0)
            return;

        int n = appendGfxMirrorRange (gfxDynamicSharedMemRanges,
                                      gfxDynamicSharedMemRangeCount,
                                      base, count, st.memN);
        n = sortAndMergeGfxMirrorRanges (gfxDynamicSharedMemRanges, n);
        gfxDynamicSharedMemRangeCount = n;
    }

    bool applyQueuedGfxMemSpanWrites() noexcept
    {
        uint32_t tail = gfxMemSpanTail.load (std::memory_order_relaxed);
        const uint32_t head = gfxMemSpanHead.load (std::memory_order_acquire);
        bool changed = false;

        while (tail != head)
        {
            auto& w = gfxMemSpanQueue[tail];
            if (w.base >= 0 && w.count > 0 && (int) w.data.size() >= w.count)
            {
                int64_t base = w.base;
                int64_t count = w.count;
                const int64_t maxEnd = std::numeric_limits<int64_t>::max() - base;
                if (count > maxEnd) count = maxEnd;

                if (w.force && count > 0 && base + count > st.memN)
                    jsfx_ensure_mem (&st, base + count);

                if (st.mem != nullptr && base < st.memN && count > 0)
                {
                    count = std::min<int64_t> (count, st.memN - base);
                    bool writable = w.force;
                    if (! writable && count > 0)
                    {
                        const int64_t logicalMemN = getGfxLogicalJsfxMemN (&st, jsfxDeclaredMaxMem);
                        writable = isDirectionalGfxMemIndex (
                            base, logicalMemN, st.memN, gfxSyncExplicitOnly, gfxSyncMemRanges, kGfxSyncFromGfx)
                            && isDirectionalGfxMemIndex (
                                base + count - 1, logicalMemN, st.memN, gfxSyncExplicitOnly, gfxSyncMemRanges, kGfxSyncFromGfx);
                    }

                    if (writable && count > 0)
                    {
                        jsfxWriteCells (st.mem + base, w.data.data(), (size_t) count);
                        noteTrackedJsfxMemUsed (&st, base + count);
                        if (w.force)
                            noteDynamicGfxSharedMemRange (base, count);
                        changed = true;

                       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
                        if (correctnessRuntime != nullptr)
                            for (int64_t i = 0; i < count; ++i)
                                correctnessRuntime->applyExternalMemWrite ((int) (base + i), w.data[(size_t) i]);
                       #endif
                    }
                }
            }

            tail = (tail + 1u) & gfxMemSpanQueueMask;
        }

        gfxMemSpanTail.store (tail, std::memory_order_release);
        return changed;
    }

    za::jsfx::StateVariables stateVariables;
    DSPJSFX_State st {};
#if DSPJSFX_HAS_FAUST
    std::unique_ptr<jsfx_faust::Engine> faustEngine { std::make_unique<jsfx_faust::Engine>() };
#endif
#if DSPJSFX_HAS_TASKS
    std::unique_ptr<jsfx_tasks::Runtime> taskRuntime { std::make_unique<jsfx_tasks::Runtime>() };
#endif
    za::jsfx::DspJsfxRuntime jsfxRuntime;
    za::jsfx::HostTransportTracker hostTransportTracker;
    bool hostTransportDiscontinuity = false;
    std::atomic<bool> gfxAnalysisVisible { false };
    std::atomic<bool> pendingEmergencyMidiCleanup { false };
    std::unique_ptr<juce::AudioProcessorValueTreeState> apvts;

   #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
    std::unique_ptr<jsfx_correctness::Runtime> correctnessRuntime;
   #endif

// ---- GFX snapshot triple-buffer ----
void initGfxSnapshots()
{
#if DSPJSFX_NATIVE_GFX_LEGACY
    return; // No mirror buffers or publications in live shared-state mode.
#endif
    for (auto& owners : gfxSnapOwners)
        owners.store (0, std::memory_order_relaxed);
    const int varCount = (int) za::jsfx::variableCount(st);

    for (auto& b : gfxSnaps)
    {
        b.varsCount = varCount;
        b.vars.resize ((size_t) varCount, 0.0);
        b.memSpanCount = 0;
        b.logicalMemN = getGfxLogicalJsfxMemN (&st, jsfxDeclaredMaxMem);

       #if DSPJSFX_HAS_NATIVE_GFX
        b.nativePublication.allocate(); // Once, before audio starts; no heap mirror.
        continue;
       #endif

        if (gfxSyncExplicitOnly)
        {
            std::array<GfxMirrorRange, kMaxGfxMemSpans> ranges {};
            const int64_t capacity = std::max<int64_t> (st.memN, jsfxDeclaredMaxMem);
            const int n = buildDirectionalGfxRanges (capacity, capacity, true,
                                                     gfxSyncMemRanges, kGfxSyncToGfx, ranges);
            for (int i = 0; i < n; ++i)
                b.memSpans[(size_t) i].data.reserve ((size_t) ranges[(size_t) i].count);
        }
        else
        {
            b.memSpans[0].data.reserve ((size_t) kGfxSharedPrefixDoubles);
            b.memSpans[1].data.reserve ((size_t) kGfxSharedSuffixDoubles);
        }
    }
}

void updateGfxSnapshotIfNeeded (int numSamples)
{
#if DSPJSFX_NATIVE_GFX_LEGACY
    juce::ignoreUnused(numSamples);
    return;
#endif
    if (st.srate <= 0.0 || numSamples <= 0)
        return;

    const bool forcePublish = gfxSnapshotForcePublish.exchange (false, std::memory_order_acq_rel);
    if (! forcePublish && gfxSnapshotUsers.load (std::memory_order_acquire) <= 0)
        return;

    if (gfxSnapPeriodSamples <= 0)
        gfxSnapPeriodSamples = (int64_t) juce::jmax (128.0, st.srate / 30.0); // ~30 Hz snapshots

    if (! forcePublish)
    {
        gfxSnapCountdown -= (int64_t) numSamples;
        if (gfxSnapCountdown > 0)
            return;

        gfxSnapCountdown += gfxSnapPeriodSamples;
    }
    else
    {
        gfxSnapCountdown = gfxSnapPeriodSamples;
    }

    const int front = gfxSnapFront.load (std::memory_order_acquire);
    int writeIdx = -1;
    for (int candidate = 0; candidate < (int) gfxSnaps.size(); ++candidate)
    {
        if (candidate == front)
            continue;
        int expected = 0;
        if (gfxSnapOwners[(size_t) candidate].compare_exchange_strong (
                expected, -1, std::memory_order_acquire, std::memory_order_relaxed))
        {
            writeIdx = candidate;
            break;
        }
    }
    if (writeIdx < 0)
    {
        gfxSnapshotForcePublish.store (true, std::memory_order_release);
        return;
    }
    struct ReleaseWriter
    {
        std::atomic<int>& owners;
        ~ReleaseWriter() { owners.store (0, std::memory_order_release); }
    } releaseWriter { gfxSnapOwners[(size_t) writeIdx] };

    try
    {
    auto& b = gfxSnaps[(size_t) writeIdx];
    const double copyStart = gfxProfileEnabled ? juce::Time::getMillisecondCounterHiRes() : 0.0;

    // Ensure buffers are large enough (rare; can allocate on audio thread).
    const int varCount = (int) za::jsfx::variableCount(st);
    if ((int) b.vars.size() != varCount)
        b.vars.resize ((size_t) varCount, 0.0);

    // Copy state. Sliders + vars are tiny; mem is mirrored as bounded windows
    // (low prefix + high suffix) plus optional explicit sparse sync ranges, so
    // @gfx can see meters/scopes/analyzer summaries without dragging the entire
    // DSP heap through the UI bridge every frame.
    jsfxCopyCells (b.sliders.data(), st.sliders, DSPJSFX_MAX_SLIDERS);

    if ((int) DSPJSFX_GFX_VAR_FLAGS_COUNT <= 0)
    {
        jsfxCopyCells (b.vars.data(), st.vars, (size_t) varCount);
    }
    else
    {
        for (int i = 0; i < varCount; ++i)
        {
            if ((getJsfxGfxVarFlags (i) & DSPJSFX_GFX_VAR_FLAG_TO_GFX) != 0u)
                b.vars[(size_t) i] = st.vars[i];
        }
    }

   #if DSPJSFX_HAS_NATIVE_GFX
    b.nativePublication.publish (st);
    b.nativeSliderAcks = gfxSliderPreviewAckSeq;
    b.nativeRuntimeEpoch = nativeRuntimeEpoch;
   #if ZA_SAMPLE_GFX_TEST_RUNNER
    for (const auto* name : {"color_tilt_db", "output_harmonics_amt", "posteq_solo_target"})
    {
        const int i = jsfx_native_gfx::Frame::findIndex (name);
        if (i >= 0) b.vars[(size_t) i] = st.vars[i]; // Diagnostics only, never imported by the UI.
    }
   #endif
   #endif
    b.memSpanCount = 0;
    b.writableMemRangeCount = 0;

    const int64_t automaticLogicalMemN = getGfxLogicalJsfxMemN (&st, jsfxDeclaredMaxMem);
    b.logicalMemN = automaticLogicalMemN;

    if (! DSPJSFX_HAS_NATIVE_GFX && st.mem != nullptr && st.memN > 0)
    {
        std::array<GfxMirrorRange, kMaxGfxMemSpans> ranges {};
        for (const auto& r : gfxSyncMemRanges)
            if (r.base >= 0 && r.count > 0)
                b.logicalMemN = std::max<int64_t> (b.logicalMemN,
                    std::min<int64_t> (st.memN, gfxSafeEndExclusive (r.base, r.count)));

        int rangeCount = buildDirectionalGfxRanges (
            automaticLogicalMemN, st.memN, gfxSyncExplicitOnly,
            gfxSyncMemRanges, kGfxSyncToGfx, ranges);
        b.writableMemRangeCount = buildDirectionalGfxRanges (
            automaticLogicalMemN, st.memN, gfxSyncExplicitOnly,
            gfxSyncMemRanges, kGfxSyncFromGfx, b.writableMemRanges);

        // Ranges introduced dynamically by native @gfx file_mem() have true
        // shared-memory semantics too. Add them to both directions after the
        // static policy is built; sort/merge prevents duplicate copies when a
        // script also declared @za:gfx_sync_mem for the same buffer.
        for (int i = 0; i < gfxDynamicSharedMemRangeCount; ++i)
        {
            const auto& r = gfxDynamicSharedMemRanges[(size_t) i];
            rangeCount = appendGfxMirrorRange (ranges, rangeCount,
                                               r.base, r.count, st.memN);
            b.writableMemRangeCount = appendGfxMirrorRange (b.writableMemRanges,
                                                            b.writableMemRangeCount,
                                                            r.base, r.count, st.memN);
            b.logicalMemN = std::max<int64_t> (b.logicalMemN,
                std::min<int64_t> (st.memN, gfxSafeEndExclusive (r.base, r.count)));
        }
        rangeCount = sortAndMergeGfxMirrorRanges (ranges, rangeCount);
        b.writableMemRangeCount = sortAndMergeGfxMirrorRanges (
            b.writableMemRanges, b.writableMemRangeCount);

        for (int i = 0; i < rangeCount && b.memSpanCount < (int) b.memSpans.size(); ++i)
        {
            const auto& r = ranges[(size_t) i];

            if (r.count <= 0 || r.base < 0 || r.base >= st.memN)
                continue;

            const int count = (int) std::min<int64_t> ((int64_t) r.count,
                                                       st.memN - r.base);

            if (count <= 0)
                continue;

            auto& span = b.memSpans[(size_t) b.memSpanCount];
            span.base = r.base;
            span.count = count;

            if (span.data.capacity() < (size_t) count)
                span.data.reserve ((size_t) count);

            if ((int) span.data.size() != count)
                span.data.resize ((size_t) count, 0.0);

            jsfxCopyCells (span.data.data(), st.mem + r.base, (size_t) count);
            ++b.memSpanCount;
        }
    }

    for (int i = b.memSpanCount; i < (int) b.memSpans.size(); ++i)
    {
        auto& span = b.memSpans[(size_t) i];
        span.base = 0;
        span.count = 0;
        span.data.clear();
    }

    b.varsCount = varCount;
    b.srate = st.srate;
    b.samplesblock = st.samplesblock;

    b.sequence = ++gfxSnapshotSequence;
    b.copyMilliseconds = gfxProfileEnabled ? juce::Time::getMillisecondCounterHiRes() - copyStart : 0.0;
   #if ZA_SAMPLE_GFX_TEST_RUNNER
    sampleCopyTotal.store (sampleCopyTotal.load (std::memory_order_relaxed) + b.copyMilliseconds, std::memory_order_relaxed);
    sampleCopyMax.store (std::max (sampleCopyMax.load (std::memory_order_relaxed), b.copyMilliseconds), std::memory_order_relaxed);
    sampleCopyCount.fetch_add (1, std::memory_order_relaxed);
   #endif
    gfxSnapFront.store (writeIdx, std::memory_order_release);
    }
    catch (const std::bad_alloc&)
    {
        // Keep the previous front frame; never leak writer ownership on failure.
        gfxSnapshotForcePublish.store (true, std::memory_order_release);
    }
}

// Snapshot buffers
#if DSPJSFX_NATIVE_GFX_LEGACY
std::mutex legacyLifecycleMutex, legacyAtomicMutex;
std::shared_ptr<jsfx_native_gfx::Frame> legacyFrame;
#endif
#if DSPJSFX_HAS_NATIVE_GFX
#if ZA_SAMPLE_GFX_TEST_RUNNER
std::mutex nativeGfxDiagnosticMutex;
std::array<double, DSPJSFX_VARS_COUNT> nativeGfxDiagnosticVars {};
bool nativeGfxDiagnosticFault = false;
#endif
uint64_t nativeRuntimeEpoch = 0; // Audio/prepare owned; published with the snapshot.
std::array<std::atomic<double>, DSPJSFX_NATIVE_GFX_COMMAND_COUNT> nativeGfxCommands {};
std::array<std::atomic<uint64_t>, DSPJSFX_NATIVE_GFX_COMMAND_COUNT> nativeGfxCommandTags {};
std::atomic<uint64_t> nativeGfxCommandEpoch { 0 };
std::atomic<bool> nativeGfxInputActive { false };
std::array<double, DSPJSFX_NATIVE_GFX_PERSIST_COUNT> nativeGfxUiState {};
#endif
#if ZA_SAMPLE_GFX_TEST_RUNNER
std::atomic<uint64_t> sampleCopyCount { 0 };
std::atomic<double> sampleCopyTotal { 0 }, sampleCopyMax { 0 };
#endif
std::array<GfxSnapshot, 3> gfxSnaps {};
    std::atomic<int> gfxSnapFront { 0 };
    std::array<std::atomic<int>, 3> gfxSnapOwners {}; // -1: writer; >=0: reader pins
    std::atomic<int> gfxSnapshotUsers { 0 };
    std::atomic<bool> gfxSnapshotForcePublish { false };
    std::mutex importPreviewAuditionMutex;
    std::unique_ptr<ImportPreviewAuditionClip> pendingImportPreviewAudition;
    std::unique_ptr<ImportPreviewAuditionClip> activeImportPreviewAudition;
    std::atomic<bool> importPreviewAuditionPending { false };
    std::atomic<bool> importPreviewAuditionStopRequested { false };
    std::atomic<bool> importPreviewAuditionPaused { false };
    std::atomic<bool> pendingExternalWakeEvent { false };
    std::atomic<std::uint32_t> pendingEventWakeCount { 0 };
    std::atomic<bool> pendingParameterWakeEvent { false };
    std::atomic<bool> uiAsyncWakePending { false };
    std::atomic<bool> processorShuttingDown { false };
    std::atomic<bool> pendingOversamplingParameterChange { false };
    std::atomic<bool> pendingFileReloadForEngineRate { false };
    std::atomic<bool> pendingSamplePoolRecommitForEngineRate { false };
    std::atomic<int> oversamplingActiveFactor { 1 };
    std::atomic<double> oversamplingHostSampleRate { 44100.0 };
    std::atomic<double> oversamplingEngineSampleRate { 44100.0 };
    std::atomic<int> oversamplingHostBlockSizeExpected { 0 };
    int64_t gfxSnapPeriodSamples = 0;
    int64_t gfxSnapCountdown = 0;
    int64_t jsfxDeclaredMaxMem = 0;
    std::vector<GfxSyncMemRange> gfxSyncMemRanges;
    bool gfxSyncExplicitOnly = false; // Immutable after construction.
    bool gfxProfileEnabled = false;
    uint64_t gfxSnapshotSequence = 0; // Producer-owned, unlike the recycled slot index.

    std::array<std::atomic<float>*, DSPJSFX_MAX_SLIDERS> paramAtomics {};
    std::atomic<float>* oversamplingParamAtomic = nullptr;
    bool parameterWakeListenersRegistered = false;

    // Low-latency @gfx -> audio slider shadow lane.
    //
    // The worker thread stages the latest UI-authored value here immediately so
    // the audio thread can consume it on the next block without waiting for the
    // host/APVTS round-trip on the message thread.
    std::array<std::atomic<double>, DSPJSFX_MAX_SLIDERS> gfxSliderPreviewValues {};
    std::array<std::atomic<uint32_t>, DSPJSFX_MAX_SLIDERS> gfxSliderPreviewSeq {};
    std::array<uint32_t, DSPJSFX_MAX_SLIDERS> gfxSliderPreviewSeenSeq {};
    std::array<uint32_t, DSPJSFX_MAX_SLIDERS> gfxSliderPreviewAckSeq {};
    std::array<int, DSPJSFX_MAX_SLIDERS> gfxSliderPreviewGraceSamplesRemaining {};

    // Tracks touch automation sessions initiated from @gfx (slider_automate).
    // This state lives on the UI thread only.
    std::array<bool, DSPJSFX_MAX_SLIDERS> gfxGestureActive {};

    std::vector<JsfxSliderDecl> sliderDecls;
    juce::String embeddedReadmeMarkdown;

    // ---- File slot declarations (from JSFX 'filename:' lines) ----

    // ---- File I/O runtime state ----

    // Worker-set aggregate hint. False costs one atomic load per audio block
    // only in builds that actually use sample_pool_*; true triggers a try-lock
    // scan until every deferred generation has been explicitly adopted.

    juce::String lastFileDialogDirectory;
    std::vector<RecentFileSelection> recentFileSelections;
    std::vector<FavoriteFileSelection> favoriteFileSelections;
    std::vector<juce::String> favoriteCategories;
    std::vector<FavoriteFileSelection> pendingFavoriteUpserts;
    std::vector<juce::String> pendingFavoriteDeleteIds;
    std::vector<juce::String> pendingFavoriteCategoryUpserts;
    std::vector<std::pair<juce::String, juce::String>> pendingFavoriteCategoryRenames;
    std::vector<juce::String> pendingFavoriteCategoryDeletePaths;



    std::array<SliderParamInfo, DSPJSFX_MAX_SLIDERS> sliderParamInfo {};
    std::array<int, DSPJSFX_MAX_SLIDERS> sliderAliasVarIndex {};

    std::array<bool, DSPJSFX_MAX_SLIDERS> sliderStringUsed {};
    std::array<juce::String, DSPJSFX_MAX_SLIDERS> stringSliderTexts {};
    mutable std::mutex             stringSliderMutex;
    std::array<std::atomic<uint32_t>, DSPJSFX_MAX_SLIDERS> stringSliderSeq {};
    std::array<uint32_t, DSPJSFX_MAX_SLIDERS> stringSliderAppliedSeq {};

    std::array<bool, DSPJSFX_MAX_SLIDERS> sliderParamUsed {};

    std::array<bool, DSPJSFX_MAX_SLIDERS> dspGestureActive {};
};

class FilteredPanel final : public juce::Component, private juce::Timer
{
public:
    class FileChooseButton final : public juce::TextButton
    {
    public:
        using juce::TextButton::TextButton;

        std::function<void()> onPopupClick;

        void mouseDown (const juce::MouseEvent& e) override
        {
            if (e.mods.isPopupMenu())
            {
                if (onPopupClick)
                    onPopupClick();
                return;
            }

            juce::TextButton::mouseDown (e);
        }

        void mouseUp (const juce::MouseEvent& e) override
        {
            if (e.mods.isPopupMenu())
                return;

            juce::TextButton::mouseUp (e);
        }
    };

    explicit FilteredPanel (JSFXJuceProcessor& p)
        : proc (p)
    {
        // --- File slots (from JSFX `filename:` declarations) ---
        for (const auto& d : proc.getJsfxFileDecls())
        {
            // Plain filename:N declarations are native JSFX resources. Only
            // // #FILE: opted-in declarations get DSP-JSFX's enhanced
            // importer UI and host-level drag/drop ownership.
            if (! d.enhancedImport)
                continue;

            Row row;
            row.isFile = true;
            row.fileIndex0 = d.index0;

            row.label = std::make_unique<juce::Label>();
            row.label->setText (d.name, juce::dontSendNotification);
            row.label->setJustificationType (juce::Justification::centredLeft);

            if (d.description.isNotEmpty())
                row.label->setTooltip (d.description);

            row.filePath = std::make_unique<juce::Label>();
            row.filePath->setText (proc.getFileSlotDisplayText (d.index0), juce::dontSendNotification);
            row.filePath->setJustificationType (juce::Justification::centredLeft);
            row.filePath->setColour (juce::Label::textColourId, juce::Colours::white);
            row.filePath->setTooltip (proc.getFileSlotTooltip (d.index0));

            row.choose = std::make_unique<FileChooseButton> ("Open...");
            row.clear = std::make_unique<juce::TextButton> ("Clear");

            row.choose->setTooltip ("Left-click to open one file. Right-click for Open Multiple, Import Multiple (Append Each), favorites, and recent files.");
            row.clear->setTooltip ("Clear the current file selection");

            const int slot = d.index0;
            auto* chooseButton = row.choose.get();

            row.choose->onClick = [this, slot] { browseForFileSlot (slot, FileBrowseMode::OpenSingle); };
            row.choose->onPopupClick = [this, slot, chooseButton]
            {
                if (chooseButton != nullptr)
                    showRecentMenuForFileSlot (slot, *chooseButton);
            };

            row.clear->onClick = [this, slot] { proc.clearFileSlot (slot); };

            addAndMakeVisible (*row.label);
            addAndMakeVisible (*row.filePath);
            addAndMakeVisible (*row.choose);
            addAndMakeVisible (*row.clear);

            rows.push_back (std::move (row));
            hasFileRows = true;
        }

        // --- Sliders ---
        for (const auto& sd : proc.getJsfxSliderDecls())
        {
            if (sd.hidden)
                continue;

            Row row;
            row.isFile = false;

            row.label = std::make_unique<juce::Label>();
            row.label->setText (sd.name, juce::dontSendNotification);
            row.label->setJustificationType (juce::Justification::centredLeft);

            if (sd.tooltip.isNotEmpty())
                row.label->setTooltip (sd.tooltip);

            if (sd.isString)
            {
                row.text = std::make_unique<juce::TextEditor>();
                row.text->setMultiLine (false);
                row.text->setReturnKeyStartsNewLine (false);
                row.text->setSelectAllWhenFocused (true);
                row.text->setText (proc.getStringSliderText (sd.index0), false);
                if (sd.tooltip.isNotEmpty())
                    row.text->setTooltip (sd.tooltip);

                const int idx = sd.index0;
                auto* editor = row.text.get();
                row.text->onTextChange = [this, idx, editor]
                {
                    if (editor != nullptr)
                        proc.setStringSliderText (idx, editor->getText());
                };
                addAndMakeVisible (*row.text);
            }
            else
            {
                const bool isEnum = (sd.isChoice && sd.choices.size() > 0);
                if (isEnum)
                {
                    row.combo = std::make_unique<juce::ComboBox>();
                    row.combo->setEditableText (false);
                    row.combo->setJustificationType (juce::Justification::centredLeft);
                    row.combo->addItemList (sd.choices, 1);

                    row.comboAttach = std::make_unique<ComboAttach> (proc.getApvts(), sanitizeId ("slider" + juce::String (sd.index0 + 1)), *row.combo);
                    addAndMakeVisible (*row.combo);
                }
                else
                {
                    row.slider = std::make_unique<juce::Slider>();
                    row.slider->setSliderStyle (juce::Slider::LinearHorizontal);
                    row.slider->setTextBoxStyle (juce::Slider::TextBoxRight, false, 80, 20);

                    row.sliderAttach = std::make_unique<SliderAttach> (proc.getApvts(), sanitizeId ("slider" + juce::String (sd.index0 + 1)), *row.slider);
                    addAndMakeVisible (*row.slider);
                }
            }

            addAndMakeVisible (*row.label);
            rows.push_back (std::move (row));
        }

        favoriteNamePrompt = std::make_unique<FavoriteNamePromptOverlay>();
        favoriteNamePrompt->setVisible (false);

        if (hasFileRows)
            startTimerHz (10);
    }

    int preferredWidth() const noexcept
    {
        const int labelW = getPreferredLabelWidth();
        const int minControlW = hasFileRows ? 360 : 420;
        return juce::jmax (560, labelW + minControlW + kOuterPadding * 2 + kColumnGap);
    }

    int preferredHeight() const noexcept
    {
        if (rows.empty())
            return kOuterPadding * 2;

        return kOuterPadding * 2 + (int) rows.size() * kRowPitch - kRowGap;
    }

    bool hasAnyVisibleControls() const noexcept { return ! rows.empty(); }

    bool hasBlockingPromptOpen() const noexcept
    {
        return favoriteNamePrompt != nullptr && favoriteNamePrompt->isPromptOpen();
    }

    bool handleBlockingPromptKeyPress (const juce::KeyPress& key)
    {
        if (favoriteNamePrompt != nullptr && favoriteNamePrompt->isPromptOpen())
        {
            favoriteNamePrompt->handleKeyPressFromHost (key);
            return true;
        }
        return false;
    }

    bool handleBlockingPromptKeyState (bool isKeyDown)
    {
        if (favoriteNamePrompt != nullptr && favoriteNamePrompt->isPromptOpen())
        {
            favoriteNamePrompt->handleKeyStateFromHost (isKeyDown);
            return true;
        }
        return false;
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colours::darkgrey);
    }

    void resized() override
    {
        auto content = getLocalBounds().reduced (kOuterPadding, kOuterPadding);
        const int maxContentW = juce::jmin (kMaxContentWidth, juce::jmax (preferredWidth(), content.getWidth()));

        if (content.getWidth() > maxContentW)
            content = content.withSizeKeepingCentre (maxContentW, content.getHeight());

        const int labelW = juce::jlimit (kMinLabelWidth,
                                         juce::jmin (kMaxLabelWidth, juce::jmax (kMinLabelWidth, content.getWidth() / 3)),
                                         getPreferredLabelWidth());

        for (auto& row : rows)
        {
            auto rowR = content.removeFromTop (kRowHeight);
            content.removeFromTop (kRowGap);

            auto labR = rowR.removeFromLeft (labelW);
            rowR.removeFromLeft (kColumnGap);
            row.label->setBounds (labR);

            if (row.isFile)
            {
                const int clearW = 68;
                const int chooseW = 96;

                auto clearR = rowR.removeFromRight (clearW).reduced (1);
                auto chooseR = rowR.removeFromRight (chooseW).reduced (1);

                row.clear->setBounds (clearR);
                row.choose->setBounds (chooseR);
                row.filePath->setBounds (rowR.reduced (1));
            }
            else if (row.slider)
            {
                row.slider->setBounds (rowR);
            }
            else if (row.combo)
            {
                row.combo->setBounds (rowR.reduced (0, 1));
            }
            else if (row.text)
            {
                row.text->setBounds (rowR.reduced (0, 1));
            }
        }

        if (favoriteNamePrompt != nullptr && favoriteNamePrompt->getParentComponent() == this)
            favoriteNamePrompt->setBounds (getLocalBounds());
    }

private:
    enum class FileBrowseMode
    {
        OpenSingle,
        OpenMultipleReplace,
        ImportMultipleAppendEach,
    };

    enum class FileBrowseFilter
    {
        CommonMedia,
        Wave,
        Flac,
        Mp3,
        AllFiles,
    };

    struct FileBrowseFilterSpec
    {
        FileBrowseFilter filter = FileBrowseFilter::CommonMedia;
        juce::String label;
        juce::String patterns;
    };

    class FavoriteNamePromptOverlay final : public juce::Component
    {
    public:
        class PromptTextEditor final : public juce::TextEditor
        {
        public:
            bool keyPressed (const juce::KeyPress& key) override
            {
                if (juce::TextEditor::keyPressed (key))
                    return true;

                const auto mods = key.getModifiers();
                if (mods.isCommandDown() || (mods.isCtrlDown() && ! mods.isAltDown()))
                    return false;

                const auto ch = key.getTextCharacter();
                if (ch >= 32 && ch != 127)
                {
                    insertTextAtCaret (juce::String::charToString (ch));
                    return true;
                }

                return false;
            }

            bool keyStateChanged (bool isKeyDown) override
            {
                if (juce::TextEditor::keyStateChanged (isKeyDown))
                    return true;

                return hasKeyboardFocus (true);
            }
        };

        static juce::String normalisePromptUserText (juce::String text)
        {
            text = text.replaceCharacters ("\r\n\t", "   ");
            return text.trim();
        }

        FavoriteNamePromptOverlay()
        {
            setVisible (false);
            setInterceptsMouseClicks (true, true);
            setWantsKeyboardFocus (true);
            setMouseClickGrabsKeyboardFocus (true);
            setFocusContainerType (juce::Component::FocusContainerType::keyboardFocusContainer);
            setAlwaysOnTop (true);

            nameLabel.setText ("Name", juce::dontSendNotification);
            nameLabel.setJustificationType (juce::Justification::centredLeft);
            nameLabel.setColour (juce::Label::textColourId, juce::Colours::white.withAlpha (0.90f));

            categoryLabel.setText ("Category", juce::dontSendNotification);
            categoryLabel.setJustificationType (juce::Justification::centredLeft);
            categoryLabel.setColour (juce::Label::textColourId, juce::Colours::white.withAlpha (0.90f));

            nameEditor.setMultiLine (false);
            nameEditor.setReturnKeyStartsNewLine (false);
            nameEditor.setWantsKeyboardFocus (true);
            nameEditor.setMouseClickGrabsKeyboardFocus (true);
            nameEditor.onReturnKey = [this] { dismissPrompt (true); };
            nameEditor.onEscapeKey = [this] { dismissPrompt (false); };

            categoryCombo.setEditableText (false);
            categoryCombo.setJustificationType (juce::Justification::centredLeft);
            categoryCombo.setTextWhenNoChoicesAvailable ("Uncategorized");

            okButton.onClick = [this] { dismissPrompt (true); };
            cancelButton.onClick = [this] { dismissPrompt (false); };

            addAndMakeVisible (nameLabel);
            addAndMakeVisible (nameEditor);
            addAndMakeVisible (categoryLabel);
            addAndMakeVisible (categoryCombo);
            addAndMakeVisible (okButton);
            addAndMakeVisible (cancelButton);
        }

        ~FavoriteNamePromptOverlay() override
        {
            if (isCurrentlyModal())
                exitModalState (0);

            if (isOnDesktop())
                removeFromDesktop();
        }

        void showPrompt (const juce::String& titleToUse,
                         const juce::String& messageToUse,
                         const juce::String& nameLabelToUse,
                         const juce::String& initialName,
                         const juce::String& fallbackNameToUse,
                         const juce::String& initialCategory,
                         const std::vector<juce::String>& categoryChoices,
                         bool shouldShowCategorySelector,
                         juce::Component* anchorComponent,
                         std::function<void(juce::String, juce::String)> onConfirmCallback)
        {
            title = titleToUse;
            message = messageToUse;
            fallbackName = fallbackNameToUse;
            onConfirm = std::move (onConfirmCallback);
            showCategorySelector = shouldShowCategorySelector;

            nameLabel.setText (nameLabelToUse.isNotEmpty() ? nameLabelToUse : "Name", juce::dontSendNotification);
            rebuildCategoryCombo (categoryChoices, initialCategory);

            nameEditor.setText (initialName, juce::dontSendNotification);
            okButton.setButtonText (showCategorySelector ? "Save" : "OK");

            categoryLabel.setVisible (showCategorySelector);
            categoryCombo.setVisible (showCategorySelector);

            if (isCurrentlyModal())
                exitModalState (0);

            if (getParentComponent() != nullptr)
                getParentComponent()->removeChildComponent (this);

            const auto size = getDialogSize();
            setSize (size.x, size.y);

            if (! isOnDesktop())
                addToDesktop (juce::ComponentPeer::windowIsTemporary);

            setBounds (calculateDesktopBounds (anchorComponent, size));
            setVisible (true);

            enterModalState (true);
            toFront (true);
            grabKeyboardFocus();
            resized();

            juce::MessageManager::callAsync ([safe = juce::Component::SafePointer<FavoriteNamePromptOverlay> (this)]
            {
                if (safe == nullptr || ! safe->isVisible())
                    return;

                safe->toFront (true);
                safe->nameEditor.grabKeyboardFocus();
                safe->nameEditor.selectAll();
            });
        }

        bool isPromptOpen() const noexcept
        {
            return isVisible();
        }

        bool handleKeyPressFromHost (const juce::KeyPress& key)
        {
            if (! isVisible())
                return false;

            if (key == juce::KeyPress::escapeKey)
            {
                dismissPrompt (false);
                return true;
            }

            if (key == juce::KeyPress::returnKey)
            {
                dismissPrompt (true);
                return true;
            }

            if (auto* editor = getFocusedPromptEditor())
                if (editor->keyPressed (key))
                    return true;

            const auto ch = key.getTextCharacter();
            if (ch >= 32 && ch != 127)
            {
                auto* editor = getFocusedPromptEditor();
                if (editor == nullptr)
                    editor = &nameEditor;

                editor->grabKeyboardFocus();
                editor->insertTextAtCaret (juce::String::charToString (ch));
                return true;
            }

            return true;
        }

        bool handleKeyStateFromHost (bool /*isKeyDown*/)
        {
            return isVisible();
        }

        bool keyPressed (const juce::KeyPress& key) override
        {
            return handleKeyPressFromHost (key);
        }

        bool keyStateChanged (bool isKeyDown) override
        {
            return handleKeyStateFromHost (isKeyDown);
        }

        void dismissPrompt (bool shouldConfirm)
        {
            if (! isVisible())
                return;

            auto callback = std::move (onConfirm);
            auto chosenName = normalisePromptUserText (nameEditor.getText());
            const auto chosenCategory = showCategorySelector ? getSelectedCategory() : juce::String();

            if (chosenName.isEmpty())
                chosenName = normalisePromptUserText (fallbackName);

            setVisible (false);

            if (isCurrentlyModal())
                exitModalState (0);

            if (isOnDesktop())
                removeFromDesktop();

            onConfirm = {};

            if (shouldConfirm && callback)
                callback (chosenName, chosenCategory);
        }

        void paint (juce::Graphics& g) override
        {
            if (! isVisible())
                return;

            const auto dialogBounds = getLocalBounds().toFloat().reduced (0.5f);

            g.setColour (juce::Colours::darkgrey.brighter (0.30f));
            g.fillRoundedRectangle (dialogBounds, 8.0f);

            g.setColour (juce::Colours::black.withAlpha (0.70f));
            g.drawRoundedRectangle (dialogBounds, 8.0f, 1.0f);

            auto textArea = getLocalBounds().reduced (16, 12);
            const auto titleArea = textArea.removeFromTop (22);
            const auto messageArea = textArea.removeFromTop (showCategorySelector ? 40 : 34);

            g.setColour (juce::Colours::white);
            g.setFont (15.0f);
            g.drawText (title, titleArea, juce::Justification::centredLeft, true);

            g.setColour (juce::Colours::white.withAlpha (0.85f));
            g.setFont (13.0f);
            g.drawFittedText (message, messageArea, juce::Justification::topLeft, 2, 0.85f);
        }

        void resized() override
        {
            auto content = getLocalBounds().reduced (16, 12);
            content.removeFromTop (22);
            content.removeFromTop (showCategorySelector ? 40 : 34);

            nameLabel.setBounds (content.removeFromTop (18));
            nameEditor.setBounds (content.removeFromTop (28));

            if (showCategorySelector)
            {
                content.removeFromTop (8);
                categoryLabel.setBounds (content.removeFromTop (18));
                categoryCombo.setBounds (content.removeFromTop (28));
            }

            auto buttons = content.removeFromBottom (28);
            const int buttonW = 88;

            cancelButton.setBounds (buttons.removeFromRight (buttonW));
            buttons.removeFromRight (8);
            okButton.setBounds (buttons.removeFromRight (buttonW));
        }

        void inputAttemptWhenModal() override
        {
            toFront (true);
            nameEditor.grabKeyboardFocus();
        }

    private:
        PromptTextEditor* getFocusedPromptEditor() noexcept
        {
            if (auto* focused = juce::Component::getCurrentlyFocusedComponent())
            {
                if (focused == &nameEditor)
                    return &nameEditor;

                if (auto* editor = focused->findParentComponentOfClass<PromptTextEditor>())
                {
                    if (editor == &nameEditor)
                        return editor;
                }
            }

            if (nameEditor.hasKeyboardFocus (true))
                return &nameEditor;

            return nullptr;
        }

        static bool promptStringEqualsIgnoreCase (const juce::String& a,
                                                    const juce::String& b)
        {
            return a.compareIgnoreCase (b) == 0;
        }

        static std::vector<juce::String> splitPromptCategoryPath (juce::String text)
        {
            text = normalisePromptUserText (text).replaceCharacter ('\\', '/');

            std::vector<juce::String> parts;
            juce::String current;

            for (int i = 0; i < text.length(); ++i)
            {
                const auto c = text[i];
                if (c == '/')
                {
                    auto part = normalisePromptUserText (current);
                    if (part.isNotEmpty())
                        parts.push_back (part);

                    current.clear();
                    continue;
                }

                current << juce::String::charToString (c);
            }

            auto part = normalisePromptUserText (current);
            if (part.isNotEmpty())
                parts.push_back (part);

            return parts;
        }

        static juce::String normalisePromptCategoryPath (const juce::String& requestedCategory)
        {
            const auto parts = splitPromptCategoryPath (requestedCategory);
            juce::StringArray out;

            for (const auto& part : parts)
                if (part.isNotEmpty())
                    out.add (part);

            return out.joinIntoString ("/");
        }

        static void sortPromptCategories (std::vector<juce::String>& categories)
        {
            for (auto& category : categories)
                category = normalisePromptCategoryPath (category);

            categories.erase (std::remove_if (categories.begin(), categories.end(),
                                              [] (const juce::String& category)
                                              {
                                                  return category.trim().isEmpty();
                                              }),
                              categories.end());

            std::stable_sort (categories.begin(), categories.end(),
                              [] (const juce::String& a, const juce::String& b)
                              {
                                  const auto aa = normalisePromptCategoryPath (a);
                                  const auto bb = normalisePromptCategoryPath (b);

                                  const bool aIsParentOfB = bb.startsWithIgnoreCase (aa + "/");
                                  const bool bIsParentOfA = aa.startsWithIgnoreCase (bb + "/");

                                  if (aIsParentOfB != bIsParentOfA)
                                      return aIsParentOfB;

                                  return aa.compareIgnoreCase (bb) < 0;
                              });

            categories.erase (std::unique (categories.begin(), categories.end(),
                                           [] (const juce::String& a, const juce::String& b)
                                           {
                                               return promptStringEqualsIgnoreCase (normalisePromptCategoryPath (a),
                                                                                   normalisePromptCategoryPath (b));
                                           }),
                              categories.end());
        }

        juce::String getSelectedCategory() const
        {
            const int index = categoryCombo.getSelectedId() - 1;
            if (index >= 0 && index < (int) categoryItems.size())
                return categoryItems[(size_t) index];

            return {};
        }

        void rebuildCategoryCombo (const std::vector<juce::String>& categoryChoices,
                                   const juce::String& initialCategory)
        {
            categoryItems.clear();
            categoryCombo.clear (juce::dontSendNotification);

            auto addChoice = [this] (const juce::String& category)
            {
                const auto normalised = normalisePromptCategoryPath (category);

                if (normalised.isEmpty())
                    return;

                const bool exists = std::any_of (categoryItems.begin(), categoryItems.end(),
                                                 [&normalised] (const juce::String& existing)
                                                 {
                                                     return promptStringEqualsIgnoreCase (existing, normalised);
                                                 });

                if (! exists)
                    categoryItems.push_back (normalised);
            };

            for (const auto& category : categoryChoices)
                addChoice (category);

            addChoice (initialCategory);

            sortPromptCategories (categoryItems);

            categoryCombo.addItem ("Uncategorized", 1);
            std::vector<juce::String> withUncategorised;
            withUncategorised.push_back ({});

            int itemId = 2;
            for (const auto& category : categoryItems)
            {
                categoryCombo.addItem (category, itemId++);
                withUncategorised.push_back (category);
            }

            categoryItems = std::move (withUncategorised);

            const auto normalisedInitial = normalisePromptCategoryPath (initialCategory);
            int selectedId = 1;

            for (int i = 0; i < (int) categoryItems.size(); ++i)
            {
                if (promptStringEqualsIgnoreCase (categoryItems[(size_t) i], normalisedInitial))
                {
                    selectedId = i + 1;
                    break;
                }
            }

            categoryCombo.setSelectedId (selectedId, juce::dontSendNotification);
        }

        juce::Point<int> getDialogSize() const noexcept
        {
            return { 480, showCategorySelector ? 238 : 182 };
        }

        juce::Rectangle<int> calculateDesktopBounds (juce::Component* anchorComponent,
                                                     juce::Point<int> size) const
        {
            juce::Rectangle<int> anchor;

            if (anchorComponent != nullptr)
                anchor = anchorComponent->localAreaToGlobal (anchorComponent->getLocalBounds());

            if (anchor.isEmpty())
                anchor = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->userArea;

            auto bounds = juce::Rectangle<int> (size.x, size.y).withCentre (anchor.getCentre());

            if (const auto* display = juce::Desktop::getInstance().getDisplays().getDisplayForRect (bounds))
                bounds = bounds.constrainedWithin (display->userArea);

            return bounds;
        }

        juce::String title;
        juce::String message;
        juce::String fallbackName;
        std::function<void(juce::String, juce::String)> onConfirm;
        juce::Label nameLabel;
        juce::Label categoryLabel;
        PromptTextEditor nameEditor;
        juce::ComboBox categoryCombo;
        juce::TextButton okButton { "Save" };
        juce::TextButton cancelButton { "Cancel" };
        std::vector<juce::String> categoryItems;
        bool showCategorySelector = true;
    };

    static juce::String shortenPathForMenu (const juce::String& path, int maxChars = 64)
    {
        if (path.length() <= maxChars)
            return path;

        const int head = juce::jmax (10, (maxChars - 3) / 2);
        const int tail = juce::jmax (10, maxChars - 3 - head);
        return path.substring (0, head) + "..." + path.substring (path.length() - tail);
    }

    static juce::String formatRecentMenuLabel (const JSFXJuceProcessor::FileSelectionState& selection)
    {
        if (selection.paths.empty())
            return {};

        const juce::File leadFile (selection.paths.front());
        auto label = leadFile.getFileName();
        if (label.isEmpty())
            label = selection.paths.front();

        if (selection.importRecipeXml.isNotEmpty())
            label = "Recipe: " + label;

        if (selection.paths.size() > 1)
        {
            const auto extraCount = juce::String ((int) selection.paths.size() - 1);
            if (selection.loadMode == JSFXJuceProcessor::FileLoadMode::AppendAsSingleFile)
                label = "Appended: " + label + " (+" + extraCount + ")";
            else
                label = "Multiple: " + label + " (+" + extraCount + ")";
        }

        const auto dir = leadFile.getParentDirectory().getFullPathName();
        if (dir.isNotEmpty())
            label << "  [" << shortenPathForMenu (dir) << "]";

        return label;
    }

    static juce::String formatFavoriteMenuLabel (const JSFXJuceProcessor::FavoriteFileSelection& favorite)
    {
        auto label = favorite.name.trim();
        if (label.isEmpty())
            label = JSFXJuceProcessor::summariseSelectedFilePaths (favorite.selection.paths, favorite.selection.loadMode);

        return label;
    }

    static juce::String formatFavoriteCategoryMenuLabel (const juce::String& category, int itemCount)
    {
        auto label = category.trim();
        if (label.isEmpty())
            label = "Uncategorized";

        if (itemCount > 1)
            label << " (" << juce::String (itemCount) << ")";

        return label;
    }

    static juce::String formatFavoritePathMenuLabel (const juce::String& path)
    {
        const juce::File file (path);

        auto label = file.getFileName();
        if (label.isEmpty())
            label = path;

        const auto dir = file.getParentDirectory().getFullPathName();
        if (dir.isNotEmpty())
            label << "  [" << shortenPathForMenu (dir, 56) << "]";

        if (! file.existsAsFile())
            label << " (missing)";

        return label;
    }

    static bool selectionExistsOnDisk (const JSFXJuceProcessor::FileSelectionState& selection)
    {
        if (selection.paths.empty())
            return false;

        for (const auto& path : selection.paths)
            if (! juce::File (path).existsAsFile())
                return false;

        return true;
    }

    juce::Component* findModalPromptHost() const
    {
        if (auto* editor = findParentComponentOfClass<juce::AudioProcessorEditor>())
            return editor;

        if (auto* top = getTopLevelComponent())
            return top;

        return const_cast<FilteredPanel*> (this);
    }

    template <typename PromptType>
    void attachPromptToHost (PromptType* prompt)
    {
        if (prompt == nullptr)
            return;

        auto* host = findModalPromptHost();
        if (host == nullptr)
            host = this;

        if (prompt->getParentComponent() != host)
        {
            if (auto* oldParent = prompt->getParentComponent())
                oldParent->removeChildComponent (prompt);

            host->addChildComponent (*prompt);
            prompt->setVisible (false);
        }

        prompt->setBounds (host->getLocalBounds());
    }

    void attachFavoritePromptToHost()
    {
        attachPromptToHost (favoriteNamePrompt.get());
    }

    void showFavoriteNamePrompt (const juce::String& title,
                                 const juce::String& message,
                                 const juce::String& initialText,
                                 const juce::String& fallbackName,
                                 const juce::String& initialCategory,
                                 std::function<void(juce::String, juce::String)> onConfirm)
    {
        if (favoriteNamePrompt == nullptr)
            return;

        favoriteNamePrompt->showPrompt (title,
                                        message,
                                        "Name",
                                        initialText,
                                        fallbackName,
                                        initialCategory,
                                        proc.getFavoriteCategories(),
                                        true,
                                        findModalPromptHost(),
                                        std::move (onConfirm));
    }

    void showCategoryNamePrompt (const juce::String& title,
                                 const juce::String& message,
                                 const juce::String& initialText,
                                 const juce::String& fallbackName,
                                 std::function<void(juce::String)> onConfirm)
    {
        if (favoriteNamePrompt == nullptr)
            return;

        favoriteNamePrompt->showPrompt (title,
                                        message,
                                        "Category",
                                        initialText,
                                        fallbackName,
                                        {},
                                        {},
                                        false,
                                        findModalPromptHost(),
                                        [callback = std::move (onConfirm)] (juce::String chosenName, juce::String)
                                        {
                                            if (callback)
                                                callback (chosenName);
                                        });
    }

    static const std::vector<FileBrowseFilterSpec>& getFileBrowseFilterSpecs()
    {
        static const std::vector<FileBrowseFilterSpec> specs
        {
            { FileBrowseFilter::CommonMedia, "Common Media", "*.wav;*.wave;*.flac;*.mp3" },
            { FileBrowseFilter::Wave,        "Wave",         "*.wav;*.wave" },
            { FileBrowseFilter::Flac,        "Flac",         "*.flac" },
            { FileBrowseFilter::Mp3,         "MP3",          "*.mp3" },
            { FileBrowseFilter::AllFiles,    "* (*)",        "*" },
        };

        return specs;
    }

    static FileBrowseFilterSpec getFileBrowseFilterSpec (FileBrowseFilter filter)
    {
        const auto& specs = getFileBrowseFilterSpecs();

        for (const auto& spec : specs)
            if (spec.filter == filter)
                return spec;

        return specs.front();
    }

#if JUCE_WINDOWS
    bool browseForFileSlotWithWindowsNativeDialog (int slot,
                                                   FileBrowseMode mode,
                                                   FileBrowseFilter defaultFilter,
                                                   const juce::String& chooserTitle,
                                                   const juce::File& startDir)
    {
        const bool allowMultiple = (mode != FileBrowseMode::OpenSingle);
        const bool appendEach = (mode == FileBrowseMode::ImportMultipleAppendEach);

        HRESULT coInit = CoInitializeEx (nullptr, COINIT_APARTMENTTHREADED);
        const bool shouldUninitCom = SUCCEEDED (coInit);

        if (FAILED (coInit) && coInit != RPC_E_CHANGED_MODE)
            return false;

        IFileOpenDialog* dialog = nullptr;
        HRESULT hr = CoCreateInstance (CLSID_FileOpenDialog,
                                       nullptr,
                                       CLSCTX_INPROC_SERVER,
                                       IID_PPV_ARGS (&dialog));

        if (FAILED (hr) || dialog == nullptr)
        {
            if (shouldUninitCom)
                CoUninitialize();

            return false;
        }

        auto releaseDialog = [&]
        {
            dialog->Release();

            if (shouldUninitCom)
                CoUninitialize();
        };

        DWORD options = 0;
        if (SUCCEEDED (dialog->GetOptions (&options)))
        {
            options |= FOS_FORCEFILESYSTEM | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST;

            if (allowMultiple)
                options |= FOS_ALLOWMULTISELECT;

            dialog->SetOptions (options);
        }

        std::vector<std::wstring> filterLabels;
        std::vector<std::wstring> filterPatterns;
        std::vector<COMDLG_FILTERSPEC> filterSpecs;

        const auto& specs = getFileBrowseFilterSpecs();
        filterLabels.reserve (specs.size());
        filterPatterns.reserve (specs.size());
        filterSpecs.reserve (specs.size());

        UINT defaultFilterIndex = 1;
        for (size_t i = 0; i < specs.size(); ++i)
        {
            const auto& spec = specs[i];
            auto label = spec.label;

            if (spec.filter == FileBrowseFilter::CommonMedia)
                label << " (*.wav;*.wave;*.flac;*.mp3)";
            else if (spec.filter == FileBrowseFilter::Wave)
                label << " (*.wav;*.wave)";
            else if (spec.filter == FileBrowseFilter::Flac)
                label << " (*.flac)";
            else if (spec.filter == FileBrowseFilter::Mp3)
                label << " (*.mp3)";

            filterLabels.push_back (std::wstring (label.toWideCharPointer()));
            filterPatterns.push_back (std::wstring (spec.patterns.toWideCharPointer()));

            if (spec.filter == defaultFilter)
                defaultFilterIndex = (UINT) i + 1;
        }

        for (size_t i = 0; i < specs.size(); ++i)
            filterSpecs.push_back ({ filterLabels[i].c_str(), filterPatterns[i].c_str() });

        if (! filterSpecs.empty())
        {
            dialog->SetFileTypes ((UINT) filterSpecs.size(), filterSpecs.data());
            dialog->SetFileTypeIndex (defaultFilterIndex);
        }

        const std::wstring title (chooserTitle.toWideCharPointer());
        dialog->SetTitle (title.c_str());

        if (startDir.isDirectory())
        {
            IShellItem* folder = nullptr;
            const std::wstring folderPath (startDir.getFullPathName().toWideCharPointer());

            if (SUCCEEDED (SHCreateItemFromParsingName (folderPath.c_str(), nullptr, IID_PPV_ARGS (&folder)))
                && folder != nullptr)
            {
                dialog->SetFolder (folder);
                folder->Release();
            }
        }

        HWND owner = nullptr;
        if (auto* host = findModalPromptHost())
            if (auto* peer = host->getPeer())
                owner = static_cast<HWND> (peer->getNativeHandle());

        hr = dialog->Show (owner);

        if (hr == HRESULT_FROM_WIN32 (ERROR_CANCELLED))
        {
            releaseDialog();
            return true;
        }

        if (FAILED (hr))
        {
            releaseDialog();
            return false;
        }

        std::vector<juce::File> selectedFiles;

        auto appendShellItem = [&selectedFiles] (IShellItem* item)
        {
            if (item == nullptr)
                return;

            PWSTR path = nullptr;
            if (SUCCEEDED (item->GetDisplayName (SIGDN_FILESYSPATH, &path)) && path != nullptr)
            {
                const juce::File selectedFile { juce::String (path) };
                if (selectedFile.existsAsFile())
                    selectedFiles.push_back (selectedFile);

                CoTaskMemFree (path);
            }
        };

        if (allowMultiple)
        {
            IShellItemArray* results = nullptr;
            if (SUCCEEDED (dialog->GetResults (&results)) && results != nullptr)
            {
                DWORD count = 0;
                if (SUCCEEDED (results->GetCount (&count)))
                {
                    selectedFiles.reserve ((size_t) count);

                    for (DWORD i = 0; i < count; ++i)
                    {
                        IShellItem* item = nullptr;
                        if (SUCCEEDED (results->GetItemAt (i, &item)) && item != nullptr)
                        {
                            appendShellItem (item);
                            item->Release();
                        }
                    }
                }

                results->Release();
            }
        }
        else
        {
            IShellItem* result = nullptr;
            if (SUCCEEDED (dialog->GetResult (&result)) && result != nullptr)
            {
                appendShellItem (result);
                result->Release();
            }
        }

        releaseDialog();

        if (selectedFiles.empty())
            return true;

        if (appendEach)
            proc.setFileSlotPathsWithMode (slot, selectedFiles, JSFXJuceProcessor::FileLoadMode::AppendAsSingleFile);
        else
            proc.setFileSlotPaths (slot, selectedFiles);

        return true;
    }
#endif

    void browseForFileSlot (int slot,
                            FileBrowseMode mode = FileBrowseMode::OpenSingle,
                            FileBrowseFilter filter = FileBrowseFilter::CommonMedia)
    {
        const bool allowMultiple = (mode != FileBrowseMode::OpenSingle);
        const bool appendEach = (mode == FileBrowseMode::ImportMultipleAppendEach);
        const auto startDir = proc.getPreferredFileChooserStartDirectory (slot, appendEach);
        const auto filterSpec = getFileBrowseFilterSpec (filter);

        juce::String chooserTitle = "Select file for slot " + juce::String (slot);
        switch (mode)
        {
            case FileBrowseMode::OpenSingle:
                chooserTitle = "Select file for slot " + juce::String (slot);
                break;
            case FileBrowseMode::OpenMultipleReplace:
                chooserTitle = "Select files for slot " + juce::String (slot);
                break;
            case FileBrowseMode::ImportMultipleAppendEach:
                chooserTitle = "Import files to append into one slot file " + juce::String (slot);
                break;
            default:
                break;
        }

        chooserTitle << " - " << filterSpec.label;

       #if JUCE_WINDOWS
        if (browseForFileSlotWithWindowsNativeDialog (slot, mode, filter, chooserTitle, startDir))
            return;
       #endif

        int flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
        if (allowMultiple)
            flags |= juce::FileBrowserComponent::canSelectMultipleItems;

        activeFileChooser = std::make_unique<juce::FileChooser> (chooserTitle,
                                                                 startDir,
                                                                 filterSpec.patterns,
                                                                 true,
                                                                 false,
                                                                 findModalPromptHost());

        juce::Component::SafePointer<FilteredPanel> safeThis (this);
        activeFileChooser->launchAsync (flags,
                                        [safeThis, slot, appendEach, allowMultiple] (const juce::FileChooser& chooser)
                                        {
                                            if (safeThis == nullptr)
                                                return;

                                            std::vector<juce::File> selectedFiles;

                                            if (allowMultiple)
                                            {
                                                const auto results = chooser.getResults();
                                                selectedFiles.reserve ((size_t) results.size());

                                                for (const auto& file : results)
                                                    if (file.existsAsFile())
                                                        selectedFiles.push_back (file);
                                            }
                                            else
                                            {
                                                const auto file = chooser.getResult();
                                                if (file.existsAsFile())
                                                    selectedFiles.push_back (file);
                                            }

                                            if (! selectedFiles.empty())
                                            {
                                                if (appendEach)
                                                    safeThis->proc.setFileSlotPathsWithMode (slot, selectedFiles, JSFXJuceProcessor::FileLoadMode::AppendAsSingleFile);
                                                else
                                                    safeThis->proc.setFileSlotPaths (slot, selectedFiles);
                                            }

                                            juce::MessageManager::callAsync ([safeThis]
                                            {
                                                if (safeThis != nullptr)
                                                    safeThis->activeFileChooser.reset();
                                            });
                                        });
    }



    // ---- ZA_FILE_SLOT_RECIPE_IMPORT_PATCH: menu-driven recipe import ----
    void browseForImportRecipeSlot (int slot,
                                    za::fileimport::ImportAction action,
                                    FileBrowseFilter filter = FileBrowseFilter::CommonMedia)
    {
        const bool allowMultiple = true; // Extraction also supports multiple source recordings.
        const auto startDir = proc.getPreferredFileChooserStartDirectory (slot, true);
        const auto filterSpec = getFileBrowseFilterSpec (filter);

        juce::String chooserTitle;
        switch (action)
        {
            case za::fileimport::ImportAction::BuildMegaTexture:
                chooserTitle = "Select files to build Mega Texture for slot " + juce::String (slot);
                break;
            case za::fileimport::ImportAction::SegmentLongFile:
                chooserTitle = "Select long file to segment for slot " + juce::String (slot);
                break;
            case za::fileimport::ImportAction::ModifyExisting:
                chooserTitle = "Select files to modify/preprocess for slot " + juce::String (slot);
                break;
            case za::fileimport::ImportAction::SegmentThenMegaTexture:
                chooserTitle = "Select files to segment then build Mega Texture for slot " + juce::String (slot);
                break;
            default:
                chooserTitle = "Select import files for slot " + juce::String (slot);
                break;
        }

        chooserTitle << " - " << filterSpec.label;

        int flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
        if (allowMultiple)
            flags |= juce::FileBrowserComponent::canSelectMultipleItems;

        activeFileChooser = std::make_unique<juce::FileChooser> (chooserTitle,
                                                                 startDir,
                                                                 filterSpec.patterns,
                                                                 true,
                                                                 false,
                                                                 findModalPromptHost());

        juce::Component::SafePointer<FilteredPanel> safeThis (this);
        activeFileChooser->launchAsync (flags,
                                        [safeThis, slot, action, allowMultiple] (const juce::FileChooser& chooser)
                                        {
                                            if (safeThis == nullptr)
                                                return;

                                            std::vector<juce::File> selectedFiles;
                                            if (allowMultiple)
                                            {
                                                const auto results = chooser.getResults();
                                                selectedFiles.reserve ((size_t) results.size());
                                                for (const auto& file : results)
                                                    if (file.existsAsFile())
                                                        selectedFiles.push_back (file);
                                            }
                                            else
                                            {
                                                const auto file = chooser.getResult();
                                                if (file.existsAsFile())
                                                    selectedFiles.push_back (file);
                                            }

                                            if (! selectedFiles.empty())
                                                safeThis->startImportActionForFileSlot (slot, std::move (selectedFiles), action);

                                            juce::MessageManager::callAsync ([safeThis]
                                            {
                                                if (safeThis != nullptr)
                                                    safeThis->activeFileChooser.reset();
                                            });
                                        });
    }

    std::vector<juce::File> sourceFilesFromSelectionOrRecipe (const JSFXJuceProcessor::FileSelectionState& selection) const
    {
        std::vector<juce::File> files;
        std::set<juce::String> seen;

        auto addIfValid = [&] (const juce::String& path)
        {
            if (path.isEmpty())
                return;

            juce::File f (path);
            if (! f.existsAsFile())
                return;

            const auto key = f.getFullPathName().toLowerCase();
            if (seen.insert (key).second)
                files.push_back (f);
        };

        if (selection.importRecipeXml.isNotEmpty())
        {
            if (auto xml = juce::parseXML (selection.importRecipeXml))
            {
                auto tree = juce::ValueTree::fromXml (*xml);
                auto recipe = za::fileimport::recipeFromValueTree (tree);
                if (! recipe.inputs.empty())
                {
                    // Indices refer to this exact recipe order. Missing sources
                    // stay visible placeholders; filtering would retarget cuts.
                    for (const auto& input : recipe.inputs)
                        files.emplace_back (input.path);
                    return files;
                }
            }
        }

        for (const auto& path : selection.paths)
            addIfValid (path);

        return za::fileimport::filterSupportedExistingFiles (files);
    }

    void editCurrentImportRecipeForFileSlot (int slot, JSFXJuceProcessor::FileSelectionState selection)
    {
        if (selection.importRecipeXml.isEmpty())
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::InfoIcon,
                                                    "Edit Import Recipe",
                                                    "The current file slot is not backed by an import recipe.");
            return;
        }

        auto xml = juce::parseXML (selection.importRecipeXml);
        if (xml == nullptr)
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
                                                    "Edit Import Recipe",
                                                    "The saved import recipe XML could not be parsed.");
            return;
        }

        auto recipe = za::fileimport::recipeFromValueTree (juce::ValueTree::fromXml (*xml));
        auto files = sourceFilesFromSelectionOrRecipe (selection);
        if (files.empty())
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
                                                    "Edit Import Recipe",
                                                    "None of the original source files for this recipe could be found.");
            return;
        }

        auto action = recipe.action;
        if (action == za::fileimport::ImportAction::LoadSeparate)
            action = za::fileimport::ImportAction::ModifyExisting;

        auto rules = recipe.rules;
        auto previewFiles = files;

        juce::Component::SafePointer<FilteredPanel> safeThis (this);
        za::fileimport::showImportPreviewDialog (*this, std::move (previewFiles), action, rules,
            [safeThis, slot, filesForApply = std::move (files), action] (za::fileimport::ImportRules acceptedRules) mutable
            {
                if (safeThis != nullptr)
                    safeThis->renderImportActionForFileSlotAsync (slot, std::move (filesForApply), action, acceptedRules);
            },
            [safeThis] (juce::AudioBuffer<float> buffer, double sampleRate) mutable
            {
                if (safeThis != nullptr)
                    safeThis->proc.auditionImportPreviewBuffer (std::move (buffer), sampleRate);
            },
            [safeThis]
            {
                if (safeThis != nullptr)
                    safeThis->proc.stopImportPreviewAudition();
            },
            [safeThis] (bool paused)
            {
                if (safeThis != nullptr)
                    safeThis->proc.pauseImportPreviewAudition (paused);
            }, "File slot " + juce::String (slot + 1));
    }

    void startImportActionForFileSlot (int slot, std::vector<juce::File> files, za::fileimport::ImportAction action)
    {
        files = za::fileimport::filterSupportedExistingFiles (files);
        if (files.empty())
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
                                                    "Import files",
                                                    "No supported audio files were found. Supported: WAV, AIFF, FLAC, OGG, MP3, M4A, CAF, W64.");
            return;
        }

        if (action == za::fileimport::ImportAction::LoadSeparate)
        {
            proc.setFileSlotPathsWithMode (slot, files, JSFXJuceProcessor::FileLoadMode::SeparateEntries, true);
            return;
        }

        if (action == za::fileimport::ImportAction::AppendRawAsSingle)
        {
            auto rules = za::fileimport::makeDefaultRulesForAction (action);
            renderImportActionForFileSlotAsync (slot, std::move (files), action, rules);
            return;
        }

        auto rules = za::fileimport::makeDefaultRulesForAction (action);

        // Keep a stable copy for the preview dialog. Function argument evaluation order
        // can otherwise move `files` into the apply lambda before the preview receives it,
        // leaving the dialog with no waveform input.
        auto previewFiles = files;

        juce::Component::SafePointer<FilteredPanel> safeThis (this);
        za::fileimport::showImportPreviewDialog (*this, std::move (previewFiles), action, rules,
            [safeThis, slot, filesForApply = std::move (files), action] (za::fileimport::ImportRules acceptedRules) mutable
            {
                if (safeThis != nullptr)
                    safeThis->renderImportActionForFileSlotAsync (slot, std::move (filesForApply), action, acceptedRules);
            },
            [safeThis] (juce::AudioBuffer<float> buffer, double sampleRate) mutable
            {
                if (safeThis != nullptr)
                    safeThis->proc.auditionImportPreviewBuffer (std::move (buffer), sampleRate);
            },
            [safeThis]
            {
                if (safeThis != nullptr)
                    safeThis->proc.stopImportPreviewAudition();
            },
            [safeThis] (bool paused)
            {
                if (safeThis != nullptr)
                    safeThis->proc.pauseImportPreviewAudition (paused);
            }, "File slot " + juce::String (slot + 1));
    }

    void renderImportActionForFileSlotAsync (int slot,
                                             std::vector<juce::File> files,
                                             za::fileimport::ImportAction action,
                                             za::fileimport::ImportRules rules)
    {
        juce::Component::SafePointer<FilteredPanel> safeThis (this);
        std::thread ([safeThis, slot, files = std::move (files), action, rules] () mutable
        {
            auto result = za::fileimport::renderImportAction (files, action, rules);
            juce::MessageManager::callAsync ([safeThis, slot, sourceFiles = files, result = std::move (result)] () mutable
            {
                if (safeThis == nullptr)
                    return;

                if (! result.ok)
                {
                    juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
                                                            "Import failed",
                                                            result.message.isNotEmpty() ? result.message : "The import recipe produced no output.");
                    return;
                }

                juce::String recipeXml;
                if (auto xml = za::fileimport::recipeToValueTree (result.recipe).createXml())
                    recipeXml = xml->toString();

                if (! result.renderedAudio.empty())
                {
                    safeThis->proc.setFileSlotRenderedAudioData (slot, result.files.empty() ? sourceFiles : result.files, std::move (result.renderedAudio), true, recipeXml);
                }
                else
                {
                    const auto mode = result.loadMode == za::fileimport::RenderedLoadMode::AppendAsSingleFile
                                        ? JSFXJuceProcessor::FileLoadMode::AppendAsSingleFile
                                        : JSFXJuceProcessor::FileLoadMode::SeparateEntries;
                    safeThis->proc.setFileSlotPathsWithMode (slot, result.files, mode, true, recipeXml);
                }
            });
        }).detach();
    }
    // ---- /ZA_FILE_SLOT_RECIPE_IMPORT_PATCH ----

    void showImportActionMenuForFileSlot (int slot, std::vector<juce::File> files, za::fileimport::ImportAction suggested)
    {
        files = za::fileimport::filterSupportedExistingFiles (files);
        if (files.empty())
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
                                                    "Import files",
                                                    "No supported audio files were found. Supported: WAV, AIFF, FLAC, OGG, MP3, M4A, CAF, W64.");
            return;
        }

        juce::PopupMenu menu;
        enum ImportMenuIds
        {
            kLoadSeparate = 1,
            kAppendRaw = 2,
            kMegaTexture = 3,
            kSegmentLong = 4,
            kModify = 5,
            kSegmentThenMega = 6
        };

        menu.addSectionHeader (files.size() > 1 ? "Import multiple files" : "Import single file");
        menu.addItem (kLoadSeparate, files.size() > 1 ? "Load as separate entries" : "Load directly", true,
                      suggested == za::fileimport::ImportAction::LoadSeparate);
        menu.addItem (kAppendRaw, files.size() > 1 ? "Append raw as one file" : "Load single file as one runtime entry", true,
                      suggested == za::fileimport::ImportAction::AppendRawAsSingle);
        menu.addSeparator();
        menu.addItem (kMegaTexture, "Build Mega Texture...", true,
                      suggested == za::fileimport::ImportAction::BuildMegaTexture);
        menu.addItem (kSegmentLong, "Segment / auto-segment...", true,
                      suggested == za::fileimport::ImportAction::SegmentLongFile);
        menu.addItem (kModify, "Modify / preprocess existing...", true,
                      suggested == za::fileimport::ImportAction::ModifyExisting);
        menu.addItem (kSegmentThenMega, "Segment then build Mega Texture...", true,
                      suggested == za::fileimport::ImportAction::SegmentThenMegaTexture);

        juce::Component::SafePointer<FilteredPanel> safeThis (this);
        menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (this).withMousePosition().withDeletionCheck (*this),
                            [safeThis, slot, files = std::move (files)] (int choice) mutable
        {
            if (safeThis == nullptr || choice == 0)
                return;

            auto action = za::fileimport::ImportAction::LoadSeparate;
            switch (choice)
            {
                case 1: action = za::fileimport::ImportAction::LoadSeparate; break;
                case 2: action = za::fileimport::ImportAction::AppendRawAsSingle; break;
                case 3: action = za::fileimport::ImportAction::BuildMegaTexture; break;
                case 4: action = za::fileimport::ImportAction::SegmentLongFile; break;
                case 5: action = za::fileimport::ImportAction::ModifyExisting; break;
                case 6: action = za::fileimport::ImportAction::SegmentThenMegaTexture; break;
                default: break;
            }

            safeThis->startImportActionForFileSlot (slot, std::move (files), action);
        });
    }

    void pasteFilesFromClipboardForSlot (int slot)
    {
        auto files = za::fileimport::parseFilesFromClipboardText (juce::SystemClipboard::getTextFromClipboard());
        if (files.empty())
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::InfoIcon,
                                                    "Paste Files / URI",
                                                    "Clipboard does not contain supported audio file paths or file:// URIs.");
            return;
        }

        const auto action = files.size() > 1 ? za::fileimport::ImportAction::BuildMegaTexture
                                             : za::fileimport::ImportAction::LoadSeparate;
        showImportActionMenuForFileSlot (slot, std::move (files), action);
    }

    void showRecentMenuForFileSlot (int slot, juce::Component& target)
    {
        enum
        {
            kAddFavoriteMenuId = 1,
            kClearMenuId = 2,
            kAddCategoryMenuId = 3,
            kPasteClipboardMenuId = 4,
            kAutoSegmentCurrentMenuId = 5,
            kEditCurrentRecipeMenuId = 6,
            kModifyCurrentMenuId = 7,
            kFirstDynamicMenuId = 100,
        };

        const auto currentSelection = proc.getFileSlotSelectionState (slot);
        const bool hasSelection = ! currentSelection.paths.empty();
        const bool hasImportRecipe = currentSelection.importRecipeXml.isNotEmpty();
        const auto suggestedFavoriteName = proc.getSuggestedFavoriteNameForSelection (currentSelection);
        const auto suggestedFavoriteCategory = proc.getSuggestedFavoriteCategoryForSelection (currentSelection);
        const auto recentSelections = proc.getRecentFileSelections();
        const auto favoriteSelections = proc.getFavoriteFileSelections();
        const auto categoryChoices = proc.getFavoriteCategories();

        int nextMenuId = kFirstDynamicMenuId;

        std::unordered_map<int, std::pair<FileBrowseMode, FileBrowseFilter>> browseItems;
        std::unordered_map<int, za::fileimport::ImportAction> recipeImportItems;
        std::unordered_map<int, JSFXJuceProcessor::FavoriteFileSelection> favoriteLoadItems;
        std::unordered_map<int, JSFXJuceProcessor::FavoriteFileSelection> favoriteEditItems;
        std::unordered_map<int, JSFXJuceProcessor::FavoriteFileSelection> favoriteRemoveItems;
        std::unordered_map<int, JSFXJuceProcessor::FileSelectionState> recentLoadItems;
        std::unordered_map<int, juce::String> categoryAddSubItems;
        std::unordered_map<int, juce::String> categoryRenameItems;
        std::unordered_map<int, juce::String> categoryRemoveItems;

        auto addBrowseSubMenu = [&] (juce::PopupMenu& parentMenu,
                                     const juce::String& label,
                                     FileBrowseMode mode)
        {
            juce::PopupMenu browseMenu;
            int defaultResult = 0;

            for (const auto& spec : getFileBrowseFilterSpecs())
            {
                const int id = nextMenuId++;
                browseItems.emplace (id, std::make_pair (mode, spec.filter));

                if (spec.filter == FileBrowseFilter::CommonMedia)
                    defaultResult = id;

                auto itemLabel = spec.label;
                if (spec.filter == FileBrowseFilter::CommonMedia)
                    itemLabel << " (default)";

                browseMenu.addItem (id, itemLabel);
            }

            parentMenu.addSubMenu (label, browseMenu, true, juce::Image(), false, defaultResult);
        };

        auto addFavoriteEntryToMenu = [&] (juce::PopupMenu& parentMenu,
                                           const JSFXJuceProcessor::FavoriteFileSelection& favorite)
        {
            const bool exists = selectionExistsOnDisk (favorite.selection);
            const bool isCurrent = proc.fileSlotMatchesSelection (slot, favorite.selection);

            juce::PopupMenu favoriteSub;
            const int loadId = exists ? nextMenuId++ : 0;

            if (exists)
            {
                favoriteLoadItems.emplace (loadId, favorite);
                favoriteSub.addItem (loadId, "Load", true, isCurrent);
                favoriteSub.addSeparator();
            }
            else
            {
                favoriteSub.addItem (nextMenuId++, "Load (missing files)", false);
                favoriteSub.addSeparator();
            }

            if (favorite.selection.loadMode == JSFXJuceProcessor::FileLoadMode::AppendAsSingleFile
                && favorite.selection.paths.size() > 1)
            {
                favoriteSub.addItem (nextMenuId++, "Mode: Append Each -> single runtime file", false);
            }

            const int maxDetailItems = 12;
            const int detailCount = juce::jmin ((int) favorite.selection.paths.size(), maxDetailItems);

            for (int i = 0; i < detailCount; ++i)
                favoriteSub.addItem (nextMenuId++, formatFavoritePathMenuLabel (favorite.selection.paths[(size_t) i]), false);

            if ((int) favorite.selection.paths.size() > detailCount)
                favoriteSub.addItem (nextMenuId++, "... +" + juce::String ((int) favorite.selection.paths.size() - detailCount) + " more", false);

            favoriteSub.addSeparator();

            const int editId = nextMenuId++;
            const int removeId = nextMenuId++;

            favoriteEditItems.emplace (editId, favorite);
            favoriteRemoveItems.emplace (removeId, favorite);

            favoriteSub.addItem (editId, "Edit...");
            favoriteSub.addItem (removeId, "Remove from Favorites");

            auto label = formatFavoriteMenuLabel (favorite);
            if (! exists)
                label << " (missing)";

            parentMenu.addSubMenu (label, favoriteSub, true, juce::Image(), isCurrent, loadId);
        };

        juce::PopupMenu favoritesMenu;

        std::vector<JSFXJuceProcessor::FavoriteFileSelection> uncategorisedFavorites;
        for (const auto& favorite : favoriteSelections)
        {
            if (favorite.category.trim().isEmpty())
                uncategorisedFavorites.push_back (favorite);
        }

        if (favoriteSelections.empty())
        {
            favoritesMenu.addItem (nextMenuId++, "No favorites", false);
        }
        else if (categoryChoices.empty())
        {
            for (const auto& favorite : favoriteSelections)
                addFavoriteEntryToMenu (favoritesMenu, favorite);
        }
        else
        {
            if (! uncategorisedFavorites.empty())
            {
                juce::PopupMenu uncategorisedMenu;
                for (const auto& favorite : uncategorisedFavorites)
                    addFavoriteEntryToMenu (uncategorisedMenu, favorite);

                favoritesMenu.addSubMenu (formatFavoriteCategoryMenuLabel (juce::String(), (int) uncategorisedFavorites.size()),
                                          uncategorisedMenu,
                                          true);
            }

            for (const auto& category : categoryChoices)
            {
                std::vector<JSFXJuceProcessor::FavoriteFileSelection> matchingFavorites;
                for (const auto& favorite : favoriteSelections)
                    if (favorite.category.compareIgnoreCase (category) == 0)
                        matchingFavorites.push_back (favorite);

                if (matchingFavorites.empty())
                    continue;

                juce::PopupMenu categoryMenu;
                for (const auto& favorite : matchingFavorites)
                    addFavoriteEntryToMenu (categoryMenu, favorite);

                favoritesMenu.addSubMenu (formatFavoriteCategoryMenuLabel (category, (int) matchingFavorites.size()),
                                          categoryMenu,
                                          true);
            }
        }

        juce::PopupMenu categoriesMenu;
        categoriesMenu.addItem (kAddCategoryMenuId, "Add Category...");

        if (categoryChoices.empty())
        {
            categoriesMenu.addSeparator();
            categoriesMenu.addItem (nextMenuId++, "No categories yet", false);
        }
        else
        {
            categoriesMenu.addSeparator();

            for (const auto& category : categoryChoices)
            {
                juce::PopupMenu categoryMenu;

                const int addSubId = nextMenuId++;
                const int renameId = nextMenuId++;
                const int removeId = nextMenuId++;

                categoryAddSubItems.emplace (addSubId, category);
                categoryRenameItems.emplace (renameId, category);
                categoryRemoveItems.emplace (removeId, category);

                categoryMenu.addItem (addSubId, "Add Sub-Category...");
                categoryMenu.addSeparator();
                categoryMenu.addItem (renameId, "Rename Category...");
                categoryMenu.addItem (removeId, "Remove Category (uncategorize favorites)");

                categoriesMenu.addSubMenu (category, categoryMenu, true);
            }
        }

        juce::PopupMenu recentMenu;
        if (recentSelections.empty())
        {
            recentMenu.addItem (nextMenuId++, "No recent files", false);
        }
        else
        {
            for (const auto& selection : recentSelections)
            {
                const bool exists = selectionExistsOnDisk (selection);
                const bool isCurrent = proc.fileSlotMatchesSelection (slot, selection);

                const int recentId = nextMenuId++;
                recentLoadItems.emplace (recentId, selection);

                auto label = formatRecentMenuLabel (selection);
                if (! exists)
                    label << " (missing)";

                recentMenu.addItem (recentId, label, exists, isCurrent);
            }
        }

        juce::PopupMenu menu;
        addBrowseSubMenu (menu, "Open...", FileBrowseMode::OpenSingle);
        addBrowseSubMenu (menu, "Open Multiple...", FileBrowseMode::OpenMultipleReplace);
        addBrowseSubMenu (menu, "Import Multiple (Append Each)...", FileBrowseMode::ImportMultipleAppendEach);

        auto addRecipeImportItem = [&] (const juce::String& label, za::fileimport::ImportAction action)
        {
            const int id = nextMenuId++;
            recipeImportItems.emplace (id, action);
            menu.addItem (id, label);
        };

        menu.addSeparator();
        addRecipeImportItem ("Build Mega Texture...", za::fileimport::ImportAction::BuildMegaTexture);
        addRecipeImportItem ("Extract Samples...", za::fileimport::ImportAction::SegmentLongFile);
        addRecipeImportItem ("Modify / Preprocess Existing...", za::fileimport::ImportAction::ModifyExisting);
        addRecipeImportItem ("Segment Then Build Mega Texture...", za::fileimport::ImportAction::SegmentThenMegaTexture);
        menu.addItem (kPasteClipboardMenuId, "Paste Files / URIs from Clipboard...");
        menu.addItem (kAutoSegmentCurrentMenuId, "Auto-Segment Current Selection...", hasSelection);
        menu.addItem (kModifyCurrentMenuId, "Modify Current Selection...", hasSelection);
        menu.addItem (kEditCurrentRecipeMenuId, "Edit Samples / Import Recipe...", hasImportRecipe);

        menu.addSeparator();
        menu.addItem (kAddFavoriteMenuId, "Add to Favorites...", hasSelection);
        menu.addItem (kClearMenuId, "Clear", hasSelection);
        menu.addSeparator();
        menu.addSubMenu ("Categories", categoriesMenu, true);
        menu.addSubMenu ("Favorites", favoritesMenu, true);
        menu.addSubMenu ("Open Recent", recentMenu, true);

        juce::Component::SafePointer<FilteredPanel> safeThis (this);
        menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (target)
                                                     .withMousePosition()
                                                     .withDeletionCheck (*this),
                            [safeThis,
                             slot,
                             currentSelection,
                             suggestedFavoriteName,
                             suggestedFavoriteCategory,
                             browseItems,
                             recipeImportItems,
                             favoriteLoadItems,
                             favoriteEditItems,
                             favoriteRemoveItems,
                             recentLoadItems,
                             categoryAddSubItems,
                             categoryRenameItems,
                             categoryRemoveItems] (int result)
                            {
                                if (safeThis == nullptr || result == 0)
                                    return;

                                if (const auto browseIt = browseItems.find (result); browseIt != browseItems.end())
                                {
                                    safeThis->browseForFileSlot (slot, browseIt->second.first, browseIt->second.second);
                                    return;
                                }

                                if (const auto recipeIt = recipeImportItems.find (result); recipeIt != recipeImportItems.end())
                                {
                                    safeThis->browseForImportRecipeSlot (slot, recipeIt->second);
                                    return;
                                }

                                if (result == kAddFavoriteMenuId)
                                {
                                    safeThis->showFavoriteNamePrompt ("Add to Favorites",
                                                                      "Name this favorite and choose a category from the drop-down. Categories are user-created; new favorites default to Uncategorized.",
                                                                      suggestedFavoriteName,
                                                                      suggestedFavoriteName,
                                                                      suggestedFavoriteCategory,
                                                                      [safeThis, currentSelection] (juce::String chosenName,
                                                                                                    juce::String chosenCategory)
                                                                      {
                                                                          if (safeThis != nullptr)
                                                                              safeThis->proc.addFavoriteSelection (currentSelection,
                                                                                                                  chosenName,
                                                                                                                  chosenCategory);
                                                                      });
                                    return;
                                }

                                if (result == kClearMenuId)
                                {
                                    safeThis->proc.clearFileSlot (slot);
                                    return;
                                }

                                if (result == kPasteClipboardMenuId)
                                {
                                    safeThis->pasteFilesFromClipboardForSlot (slot);
                                    return;
                                }

                                if (result == kAutoSegmentCurrentMenuId)
                                {
                                    auto currentFiles = safeThis->sourceFilesFromSelectionOrRecipe (currentSelection);
                                    safeThis->startImportActionForFileSlot (slot, std::move (currentFiles), za::fileimport::ImportAction::SegmentLongFile);
                                    return;
                                }

                                if (result == kModifyCurrentMenuId)
                                {
                                    auto currentFiles = safeThis->sourceFilesFromSelectionOrRecipe (currentSelection);
                                    safeThis->startImportActionForFileSlot (slot, std::move (currentFiles), za::fileimport::ImportAction::ModifyExisting);
                                    return;
                                }

                                if (result == kEditCurrentRecipeMenuId)
                                {
                                    safeThis->editCurrentImportRecipeForFileSlot (slot, currentSelection);
                                    return;
                                }

                                if (result == kAddCategoryMenuId)
                                {
                                    safeThis->showCategoryNamePrompt ("Add Category",
                                                                      "Create a new Favorites category. Use / in the name if you want to create nested levels directly.",
                                                                      {},
                                                                      {},
                                                                      [safeThis] (juce::String category)
                                                                      {
                                                                          if (safeThis != nullptr)
                                                                              safeThis->proc.addFavoriteCategory (category);
                                                                      });
                                    return;
                                }

                                if (const auto categoryIt = categoryAddSubItems.find (result); categoryIt != categoryAddSubItems.end())
                                {
                                    const auto parentCategory = categoryIt->second;
                                    safeThis->showCategoryNamePrompt ("Add Sub-Category",
                                                                      "Create a sub-category under: " + parentCategory,
                                                                      {},
                                                                      {},
                                                                      [safeThis, parentCategory] (juce::String childCategory)
                                                                      {
                                                                          if (safeThis != nullptr)
                                                                              safeThis->proc.addFavoriteSubCategory (parentCategory, childCategory);
                                                                      });
                                    return;
                                }

                                if (const auto categoryIt = categoryRenameItems.find (result); categoryIt != categoryRenameItems.end())
                                {
                                    const auto category = categoryIt->second;
                                    safeThis->showCategoryNamePrompt ("Rename Category",
                                                                      "Rename this category. Favorites and sub-categories underneath it will be moved with it.",
                                                                      category,
                                                                      category,
                                                                      [safeThis, category] (juce::String newCategory)
                                                                      {
                                                                          if (safeThis != nullptr)
                                                                              safeThis->proc.renameFavoriteCategory (category, newCategory);
                                                                      });
                                    return;
                                }

                                if (const auto categoryIt = categoryRemoveItems.find (result); categoryIt != categoryRemoveItems.end())
                                {
                                    safeThis->proc.removeFavoriteCategory (categoryIt->second);
                                    return;
                                }

                                if (const auto favoriteIt = favoriteLoadItems.find (result); favoriteIt != favoriteLoadItems.end())
                                {
                                    safeThis->proc.setFileSlotSelection (slot, favoriteIt->second.selection);
                                    return;
                                }

                                if (const auto recentIt = recentLoadItems.find (result); recentIt != recentLoadItems.end())
                                {
                                    safeThis->proc.setFileSlotSelection (slot, recentIt->second);
                                    return;
                                }

                                if (const auto editIt = favoriteEditItems.find (result); editIt != favoriteEditItems.end())
                                {
                                    const auto favorite = editIt->second;
                                    const auto fallbackName = favorite.name.isNotEmpty()
                                                                ? favorite.name
                                                                : safeThis->proc.getSuggestedFavoriteNameForSelection (favorite.selection);

                                    safeThis->showFavoriteNamePrompt ("Edit Favorite",
                                                                      "Change the display name and choose a category from the drop-down. Use the Categories menu to add or rename categories.",
                                                                      fallbackName,
                                                                      fallbackName,
                                                                      favorite.category,
                                                                      [safeThis, favoriteId = favorite.id] (juce::String chosenName,
                                                                                                            juce::String chosenCategory)
                                                                      {
                                                                          if (safeThis != nullptr)
                                                                              safeThis->proc.updateFavorite (favoriteId,
                                                                                                            chosenName,
                                                                                                            chosenCategory);
                                                                      });
                                    return;
                                }

                                if (const auto removeIt = favoriteRemoveItems.find (result); removeIt != favoriteRemoveItems.end())
                                {
                                    safeThis->proc.removeFavorite (removeIt->second.id);
                                    return;
                                }
                            });
    }


    static constexpr int kRowHeight = 30;
    static constexpr int kRowGap = 6;
    static constexpr int kRowPitch = kRowHeight + kRowGap;
    static constexpr int kOuterPadding = 8;
    static constexpr int kColumnGap = 12;
    static constexpr int kMinLabelWidth = 170;
    static constexpr int kMaxLabelWidth = 240;
    static constexpr int kMaxContentWidth = 920;

    int getPreferredLabelWidth() const noexcept
    {
        int labelW = kMinLabelWidth;

        for (const auto& row : rows)
        {
            if (row.label)
                labelW = std::max (labelW, row.label->getFont().getStringWidth (row.label->getText()) + 20);
        }

        return juce::jmin (labelW, kMaxLabelWidth);
    }

    void timerCallback() override
    {
        for (auto& row : rows)
        {
            if (! row.isFile || row.filePath == nullptr)
                continue;

            const auto text = proc.getFileSlotDisplayText (row.fileIndex0);
            if (row.filePath->getText() != text)
                row.filePath->setText (text, juce::dontSendNotification);

            row.filePath->setTooltip (proc.getFileSlotTooltip (row.fileIndex0));

            const bool hasPath = proc.hasFileSlotSelection (row.fileIndex0);
            if (row.clear != nullptr)
                row.clear->setEnabled (hasPath);
        }
    }

    JSFXJuceProcessor& proc;

    using SliderAttach = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboAttach  = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    struct Row
    {
        bool isFile = false;
        int fileIndex0 = -1;

        std::unique_ptr<juce::Label> label;

        // Slider/enum rows
        std::unique_ptr<juce::Slider> slider;
        std::unique_ptr<juce::ComboBox> combo;
        std::unique_ptr<juce::TextEditor> text;
        std::unique_ptr<SliderAttach> sliderAttach;
        std::unique_ptr<ComboAttach> comboAttach;

        // File rows
        std::unique_ptr<juce::Label> filePath;
        std::unique_ptr<FileChooseButton> choose;
        std::unique_ptr<juce::TextButton> clear;
    };

    std::vector<Row> rows;

    bool hasFileRows = false;
    std::unique_ptr<juce::FileChooser> activeFileChooser;
    std::unique_ptr<FavoriteNamePromptOverlay> favoriteNamePrompt;
};

// ============================================================
// Custom editor: bounded HELP overlay + row-hover tooltips sourced from JSFX comments.
// ============================================================
class JSFXJuceEditor final : public juce::AudioProcessorEditor, public juce::FileDragAndDropTarget
{
public:
    using SliderMask = jsfx_gfx::SliderMask;

    // -----------------------
    // Unicode-safe LookAndFeel
    // -----------------------
    class UnicodeLNF final : public juce::LookAndFeel_V4
    {
    public:
        static juce::Font pickFont (float height)
        {
           #if JUCE_WINDOWS
            // Good Unicode coverage on Windows.
            juce::Font f ("Segoe UI", height, juce::Font::plain);
            if (f.getTypefaceName().isNotEmpty())
                return f;
            return juce::Font (height);
           #elif JUCE_MAC
            // Let CoreText handle fallback; default font tends to be robust.
            return juce::Font (height);
           #else
            // Linux distros vary wildly; default + fallback is usually best unless you embed fonts.
            // If you ship Noto, swap this to that typeface.
            return juce::Font (height);
           #endif
        }

        juce::Font getLabelFont (juce::Label&) override
        {
            return pickFont (14.0f);
        }

        juce::Font getTextButtonFont (juce::TextButton&, int buttonHeight) override
        {
            return pickFont (juce::jlimit (12.0f, 16.0f, (float) buttonHeight * 0.55f));
        }

        juce::Font getComboBoxFont (juce::ComboBox&) override
        {
            return pickFont (14.0f);
        }

        juce::Font getPopupMenuFont() override
        {
            return pickFont (13.5f);
        }

        void drawTooltip (juce::Graphics& g, const juce::String& text, int width, int height) override
        {
            auto bounds = juce::Rectangle<int> (0, 0, width, height);

            g.setColour (juce::Colours::black.withAlpha (0.82f));
            g.fillRoundedRectangle (bounds.toFloat().reduced (1.0f), 6.0f);

            g.setColour (juce::Colours::white.withAlpha (0.20f));
            g.drawRoundedRectangle (bounds.toFloat().reduced (1.0f), 6.0f, 1.0f);

            g.setColour (juce::Colours::white);
            g.setFont (pickFont (13.5f));

            auto textArea = bounds.reduced (8, 6);
            g.drawFittedText (text, textArea, juce::Justification::centredLeft, 4);
        }
    };

    class SmartIdleBadge final : public juce::Component, public juce::SettableTooltipClient, private juce::Timer
    {
    public:
        explicit SmartIdleBadge (JSFXJuceProcessor& p)
            : processor (p)
        {
            refreshNow();
            startTimerHz (8);
        }

        int preferredWidth() const noexcept { return 196; }

        void paint (juce::Graphics& g) override
        {
            auto bounds = getLocalBounds().toFloat().reduced (0.5f);
            if (bounds.isEmpty())
                return;

            const auto fill = sleeping ? juce::Colour::fromRGBA (76, 86, 98, 224)
                                       : juce::Colour::fromRGBA (44, 130, 90, 228);
            const auto outline = sleeping ? juce::Colour::fromRGBA (208, 214, 220, 96)
                                          : juce::Colour::fromRGBA (214, 255, 231, 112);
            const auto dot = sleeping ? juce::Colour::fromRGB (224, 228, 233)
                                      : juce::Colour::fromRGB (236, 255, 244);

            g.setColour (fill);
            g.fillRoundedRectangle (bounds, 11.0f);

            g.setColour (outline);
            g.drawRoundedRectangle (bounds, 11.0f, 1.0f);

            auto area = getLocalBounds().reduced (10, 0);
            const int dotSize = 8;
            g.setColour (dot);
            g.fillEllipse ((float) area.getX(),
                           (float) (area.getCentreY() - dotSize / 2),
                           (float) dotSize,
                           (float) dotSize);
            area.removeFromLeft (dotSize + 8);

            auto modeArea = area.removeFromRight (58);

            g.setColour (juce::Colours::white.withAlpha (0.96f));
            g.setFont (JSFXJuceEditor::UnicodeLNF::pickFont (13.5f).boldened());
            g.drawText (stateText, area, juce::Justification::centredLeft, true);

            g.setColour (juce::Colours::white.withAlpha (0.76f));
            g.setFont (JSFXJuceEditor::UnicodeLNF::pickFont (11.5f));
            g.drawText (modeText, modeArea, juce::Justification::centredRight, true);
        }

        void mouseDown (const juce::MouseEvent& e) override
        {
            if (! e.mods.isPopupMenu())
            {
                juce::Component::mouseDown (e);
                return;
            }

            const int current = processor.getSmartIdleUserOverrideModeForUi();

            juce::PopupMenu menu;
            menu.addItem (1, "Default / Auto", true, current == 0);
            menu.addSeparator();
            menu.addItem (2, "Never Sleep", true, current == 4);
            menu.addItem (3, "Sleep on Silence", true, current == 1);
            menu.addItem (4, "Sleep on Events", true, current == 2);
            menu.addItem (5, "Free-running", true, current == 3);

            juce::Component::SafePointer<SmartIdleBadge> safeThis (this);
            menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (this),
                                [safeThis] (int result)
                                {
                                    if (safeThis == nullptr || result <= 0)
                                        return;

                                    int mode = 0;
                                    switch (result)
                                    {
                                        case 2:  mode = 4; break;
                                        case 3:  mode = 1; break;
                                        case 4:  mode = 2; break;
                                        case 5:  mode = 3; break;
                                        case 1:
                                        default: mode = 0; break;
                                    }

                                    safeThis->processor.setSmartIdleUserOverrideModeForUi (mode);
                                    safeThis->refreshNow();
                                });
        }

    private:
        void timerCallback() override
        {
            refreshNow();
        }

        void refreshNow()
        {
            const auto status = processor.getSmartIdleIndicatorStatus();
            if (sleeping != status.sleeping
                || stateText != status.stateText
                || modeText != status.modeText
                || tooltipText != status.tooltip)
            {
                sleeping = status.sleeping;
                stateText = status.stateText;
                modeText = status.modeText;
                tooltipText = status.tooltip;
                setTooltip (tooltipText);
                repaint();
            }
        }

        JSFXJuceProcessor& processor;
        bool sleeping = false;
        juce::String stateText { "ACTIVE" };
        juce::String modeText { "INPUT" };
        juce::String tooltipText;
    };

    explicit JSFXJuceEditor (JSFXJuceProcessor& p)
        : juce::AudioProcessorEditor (&p)
        , processor (p)
        , genericEditor (p)
        , gfxView (p)
        , smartIdleBadge (p)
        , tooltipWindow (*this, kTooltipDelayMs)
    {
        setLookAndFeel (&unicodeLnf);
        tooltipWindow.setLookAndFeel (&unicodeLnf);
        setOpaque (true);
        setWantsKeyboardFocus (true);
        setMouseClickGrabsKeyboardFocus (true);
        setFocusContainerType (juce::Component::FocusContainerType::keyboardFocusContainer);
        setColour (juce::ResizableWindow::backgroundColourId, juce::Colour (0xff2f3a41));

        controlsViewport.setViewedComponent (&genericEditor, false);
        controlsViewport.setScrollBarsShown (true, false, true, false);
        controlsViewport.setSingleStepSizes (0, 32);
        addAndMakeVisible (controlsViewport);

        addAndMakeVisible (gfxView);
        gfxView.setVisible (gfxView.hasGfx());

        addAndMakeVisible (smartIdleBadge);

        oversamplingBox.addItem ("OS Off", 1);
        oversamplingBox.addItem ("OS 2x", 2);
        oversamplingBox.addItem ("OS 4x", 3);
        oversamplingBox.addItem ("OS 8x", 4);
        oversamplingBox.setTooltip ("Run JSFX DSP and sample caches at an integer multiple of the host sample rate.");
        addAndMakeVisible (oversamplingBox);
        oversamplingAttach = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (
            processor.getApvts(), processor.getOversamplingParameterIdForUi(), oversamplingBox);

        helpButton.setButtonText ("?");
        helpButton.setTooltip ("Open embedded README");
        helpButton.onClick = [this]
        {
            if (helpOverlay.isVisible())
                hideHelp();
            else
                showHelp();
        };
        addAndMakeVisible (helpButton);

        addChildComponent (importLandingPad);
        importLandingPad.setVisible (false);

       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        debugButton.setButtonText ("DBG");
        debugButton.setTooltip ("Show correctness monitor");
        debugButton.setVisible (processor.hasCorrectnessMonitor());
        debugButton.onClick = [this]
        {
            if (correctnessOverlay.isVisible())
                hideCorrectnessMonitor();
            else
                showCorrectnessMonitor();
        };
        addAndMakeVisible (debugButton);

        addChildComponent (correctnessOverlay);
        correctnessOverlay.setVisible (false);
       #endif

        addChildComponent (helpOverlay);
        helpOverlay.setVisible (false);

        setResizable (true, true);

        genericPrefW = juce::jmax (kMinEditorWidth, genericEditor.preferredWidth());
        genericPrefH = genericEditor.hasAnyVisibleControls() ? genericEditor.preferredHeight() : 0;

        const bool hasGfx      = gfxView.hasGfx();
        const bool hasControls = genericEditor.hasAnyVisibleControls();
        const auto screenBudget = getScreenAwareEditorBudget();

        const auto plan = za::pluginui::planEditorSections (kTopBarH,
                                                            kMinEditorWidth,
                                                            screenBudget.maxWidth,
                                                            screenBudget.maxHeight,
                                                            hasControls,
                                                            genericPrefW,
                                                            genericPrefH,
                                                            hasGfx,
                                                            getDeclaredGfxPreferredWidth(),
                                                            getDeclaredGfxPreferredHeight(),
                                                            kSectionGap,
                                                            kMinimumScaledGfxHeight);

        const auto resizeLimits = computeEditorResizeLimits (plan, screenBudget, hasControls, hasGfx);
        setResizeLimits (resizeLimits.minWidth,
                         resizeLimits.minHeight,
                         resizeLimits.maxWidth,
                         resizeLimits.maxHeight);

        // The JSFX @gfx declaration is the requested opening GFX area. The
        // #GFX_RESIZE policy decides whether that declared size is also a
        // minimum/fixed canvas or only the responsive initial target.
        setSize (juce::jlimit (resizeLimits.minWidth, resizeLimits.maxWidth, plan.initialWidth),
                 juce::jlimit (resizeLimits.minHeight, resizeLimits.maxHeight, plan.initialHeight));
    }

    ~JSFXJuceEditor() override
    {
        tooltipWindow.setLookAndFeel (nullptr);
        setLookAndFeel (nullptr);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (findColour (juce::ResizableWindow::backgroundColourId));

        auto topBar = getLocalBounds().removeFromTop (kTopBarH);
        g.setColour (juce::Colour (0xff273239));
        g.fillRect (topBar);

        g.setColour (juce::Colours::white.withAlpha (0.08f));
        g.drawLine ((float) topBar.getX(),
                    (float) topBar.getBottom() - 0.5f,
                    (float) topBar.getRight(),
                    (float) topBar.getBottom() - 0.5f,
                    1.0f);
    }

    void resized() override
    {
        auto r = getLocalBounds();
        auto top = r.removeFromTop (kTopBarH);

        const int btnSize = 24;
        int right = top.getRight() - 8;

        const int badgeH = 24;
        const int badgeW = juce::jmin (smartIdleBadge.preferredWidth(), juce::jmax (140, top.getWidth() - 230));
        smartIdleBadge.setBounds (8, (kTopBarH - badgeH) / 2, badgeW, badgeH);
        smartIdleBadge.toFront (false);

        const int osW = 96;
        oversamplingBox.setBounds (right - osW, (kTopBarH - btnSize) / 2, osW, btnSize);
        oversamplingBox.toFront (false);
        right -= osW + 6;

        helpButton.setBounds (right - btnSize, (kTopBarH - btnSize) / 2, btnSize, btnSize);
        helpButton.toFront (false);
        right -= btnSize + 6;

       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        if (debugButton.isVisible())
        {
            debugButton.setBounds (right - 44, (kTopBarH - btnSize) / 2, 44, btnSize);
            debugButton.toFront (false);
        }
       #endif

        const bool hasGfx      = gfxView.hasGfx();
        const bool hasControls = genericEditor.hasAnyVisibleControls();

        genericPrefW = juce::jmax (kMinEditorWidth, genericEditor.preferredWidth());
        genericPrefH = hasControls ? genericEditor.preferredHeight() : 0;

        if (hasControls && hasGfx)
        {
            const int targetGfxH = getScaledDeclaredGfxPreferredHeightForWidth (r.getWidth());
            const int minGfxH = juce::jmin (targetGfxH,
                                            juce::jmax (kMinimumScaledGfxHeight, targetGfxH / 2));

            const int availableH = r.getHeight();

            // Mixed generic-controls + @gfx scripts are common in upstream
            // JSFX (including Joep Vanlier's). Treat the controls as a compact
            // scrollable header rather than allowing them to consume their full
            // preferred height and squeeze the actual @gfx UI. Around 30% keeps
            // six-ish REAPER-style sliders visible while making @gfx dominant.
            const int combinedControlsCap = juce::jmax (getMinControlsViewportHeight(),
                                                        (int) std::lround ((double) availableH * 0.30));
            int controlsH = juce::jmin (genericPrefH, combinedControlsCap);

            // The declared @gfx size still gets a protected minimum when the
            // host window is small; excess controls simply scroll vertically.
            controlsH = juce::jmin (controlsH,
                                    juce::jmax (0, availableH - kSectionGap - minGfxH));
            controlsH = juce::jmax (0, controlsH);
            auto controlsArea = r.removeFromTop (juce::jmax (0, controlsH));
            controlsViewport.setVisible (true);
            controlsViewport.setBounds (controlsArea);
            updateControlsViewportContentSize();

            if (! r.isEmpty())
                r.removeFromTop (juce::jmin (kSectionGap, r.getHeight()));

            gfxView.setBounds (r);
        }
        else if (hasControls)
        {
            controlsViewport.setVisible (true);
            controlsViewport.setBounds (r);
            updateControlsViewportContentSize();
            gfxView.setBounds (0, 0, 0, 0);
        }
        else
        {
            controlsViewport.setVisible (false);
            controlsViewport.setBounds (0, 0, 0, 0);
            genericEditor.setSize (0, 0);

            if (hasGfx)
                gfxView.setBounds (r);
            else
                gfxView.setBounds (0, 0, 0, 0);
        }

        importLandingPad.setBounds (getLocalBounds());
        if (importLandingPad.isVisible())
            importLandingPad.toFront (false);

        helpOverlay.setBounds (getLocalBounds());
       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        correctnessOverlay.setBounds (getLocalBounds());
       #endif
    }

    void parentHierarchyChanged() override
    {
        const bool hasGfx = gfxView.hasGfx();
        const bool hasControls = genericEditor.hasAnyVisibleControls();
        const auto screenBudget = getScreenAwareEditorBudget();
        const auto plan = za::pluginui::planEditorSections (kTopBarH,
                                                            kMinEditorWidth,
                                                            screenBudget.maxWidth,
                                                            screenBudget.maxHeight,
                                                            hasControls,
                                                            genericPrefW,
                                                            genericPrefH,
                                                            hasGfx,
                                                            getDeclaredGfxPreferredWidth(),
                                                            getDeclaredGfxPreferredHeight(),
                                                            kSectionGap,
                                                            kMinimumScaledGfxHeight);

        const auto resizeLimits = computeEditorResizeLimits (plan, screenBudget, hasControls, hasGfx);
        setResizeLimits (resizeLimits.minWidth,
                         resizeLimits.minHeight,
                         resizeLimits.maxWidth,
                         resizeLimits.maxHeight);
    }

    bool keyPressed (const juce::KeyPress& key) override
    {
        if (genericEditor.hasBlockingPromptOpen())
            return genericEditor.handleBlockingPromptKeyPress (key);

        if (isPasteImportKey (key))
        {
            pasteFileUriImport();
            return true;
        }

        return juce::AudioProcessorEditor::keyPressed (key);
    }

    bool keyStateChanged (bool isKeyDown) override
    {
        if (genericEditor.hasBlockingPromptOpen())
            return genericEditor.handleBlockingPromptKeyState (isKeyDown);

        return juce::AudioProcessorEditor::keyStateChanged (isKeyDown);
    }

    // ---- ZA_FILE_IMPORT_RECIPE_PATCH: drag/drop + clipboard ingress ----
    //
    // The editor is intentionally the only JUCE FileDragAndDropTarget. If GfxView
    // is another target, JUCE changes targets when the cursor crosses into @gfx and
    // sends fileDragExit() here. That dismissed Corpus' enhanced #FILE landing pad
    // even though Corpus does not consume gfx_getdropfile().
    bool isInterestedInFileDrag (const juce::StringArray& files) override
    {
        const bool enhancedImport = getDefaultImportFileSlot() >= 0
                                 && za::fileimport::containsSupportedFileExtension (files);
        return enhancedImport || gfxView.acceptsNativeFileDrop();
    }

    bool shouldRouteDropToNativeGfx (int x, int y) const noexcept
    {
        return gfxView.acceptsNativeFileDrop() && gfxView.getBounds().contains (x, y);
    }

    bool canUseEnhancedImportFor (const juce::StringArray& files) const
    {
        return getDefaultImportFileSlot() >= 0
            && za::fileimport::containsSupportedFileExtension (files);
    }

    void updateFileDragRouting (const juce::StringArray& files, int x, int y)
    {
        // Native JSFX gfx_getdropfile() owns a drop only inside @gfx and only if
        // the script actually consumes that API. Otherwise #FILE owns the whole
        // editor, including the custom GFX area.
        if (shouldRouteDropToNativeGfx (x, y))
        {
            if (importLandingPad.isVisible())
            {
                importLandingPad.setVisible (false);
                repaint();
            }
            return;
        }

        if (! canUseEnhancedImportFor (files))
        {
            if (importLandingPad.isVisible())
            {
                importLandingPad.setVisible (false);
                repaint();
            }
            return;
        }

        if (! importLandingPad.isVisible())
        {
            importLandingPad.setVisible (true);
            importLandingPad.toFront (true);
        }
        importLandingPad.setHoverPoint ({ x, y });
        repaint();
    }

    void fileDragEnter (const juce::StringArray& files, int x, int y) override
    {
        updateFileDragRouting (files, x, y);
    }

    void fileDragMove (const juce::StringArray& files, int x, int y) override
    {
        updateFileDragRouting (files, x, y);
    }

    void fileDragExit (const juce::StringArray& files) override
    {
        juce::ignoreUnused (files);
        importLandingPad.setVisible (false);
        repaint();
    }

    void filesDropped (const juce::StringArray& paths, int x, int y) override
    {
        if (shouldRouteDropToNativeGfx (x, y))
        {
            importLandingPad.setVisible (false);
            gfxView.enqueueNativeFileDrop (paths, gfxView.getLocalPoint (this, juce::Point<int> { x, y }).toFloat());
            repaint();
            return;
        }

        if (! canUseEnhancedImportFor (paths))
        {
            importLandingPad.setVisible (false);
            repaint();
            return;
        }

        std::vector<juce::File> files;
        files.reserve ((size_t) paths.size());
        for (const auto& path : paths)
            files.emplace_back (path);

        files = za::fileimport::filterSupportedExistingFiles (files);
        const bool multiple = files.size() > 1;
        const auto suggested = importLandingPad.actionForPoint ({ x, y }, multiple);

        importLandingPad.setVisible (false);
        repaint();

        startImportAction (std::move (files), suggested);
    }

    bool isPasteImportKey (const juce::KeyPress& key) const
    {
        const auto mods = key.getModifiers();
        return (key.getTextCharacter() == 'v' || key.getTextCharacter() == 'V')
               && (mods.isCommandDown() || mods.isCtrlDown());
    }

    void pasteFileUriImport()
    {
        auto files = za::fileimport::parseFilesFromClipboardText (juce::SystemClipboard::getTextFromClipboard());
        if (files.empty())
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::InfoIcon,
                                                    "Paste Files / URI",
                                                    "Clipboard does not contain supported audio file paths or file:// URIs.");
            return;
        }

        const auto action = files.size() > 1 ? za::fileimport::ImportAction::BuildMegaTexture
                                             : za::fileimport::ImportAction::LoadSeparate;
        showImportActionMenu (std::move (files), action);
    }

    void showImportActionMenu (std::vector<juce::File> files, za::fileimport::ImportAction suggested)
    {
        if (files.empty())
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
                                                    "Import files",
                                                    "No supported audio files were found. Supported: WAV, AIFF, FLAC, OGG, MP3, M4A, CAF, W64.");
            return;
        }

        juce::PopupMenu menu;
        enum ImportMenuIds
        {
            kLoadSeparate = 1,
            kAppendRaw = 2,
            kMegaTexture = 3,
            kSegmentLong = 4,
            kModify = 5,
            kSegmentThenMega = 6
        };

        menu.addSectionHeader (files.size() > 1 ? "Import multiple files" : "Import single file");
        menu.addItem (kLoadSeparate,
                      files.size() > 1 ? "Load as separate entries" : "Load directly",
                      true,
                      suggested == za::fileimport::ImportAction::LoadSeparate);
        menu.addItem (kAppendRaw,
                      files.size() > 1 ? "Append raw as one file" : "Load single file as one runtime entry",
                      true,
                      suggested == za::fileimport::ImportAction::AppendRawAsSingle);
        menu.addSeparator();
        menu.addItem (kMegaTexture,
                      "Build Mega Texture...",
                      true,
                      suggested == za::fileimport::ImportAction::BuildMegaTexture);
        menu.addItem (kSegmentLong,
                      "Segment / auto-segment...",
                      true,
                      suggested == za::fileimport::ImportAction::SegmentLongFile);
        menu.addItem (kModify,
                      "Modify / preprocess existing...",
                      true,
                      suggested == za::fileimport::ImportAction::ModifyExisting);
        menu.addItem (kSegmentThenMega,
                      "Segment then build Mega Texture...",
                      true,
                      suggested == za::fileimport::ImportAction::SegmentThenMegaTexture);

        juce::Component::SafePointer<JSFXJuceEditor> safeThis (this);
        menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (this).withMousePosition().withDeletionCheck (*this),
                            [safeThis, files = std::move (files)] (int choice) mutable
        {
            if (safeThis == nullptr || choice == 0)
                return;

            auto action = za::fileimport::ImportAction::LoadSeparate;
            switch (choice)
            {
                case 1: action = za::fileimport::ImportAction::LoadSeparate; break;
                case 2: action = za::fileimport::ImportAction::AppendRawAsSingle; break;
                case 3: action = za::fileimport::ImportAction::BuildMegaTexture; break;
                case 4: action = za::fileimport::ImportAction::SegmentLongFile; break;
                case 5: action = za::fileimport::ImportAction::ModifyExisting; break;
                case 6: action = za::fileimport::ImportAction::SegmentThenMegaTexture; break;
                default: break;
            }

            safeThis->startImportAction (std::move (files), action);
        });
    }

    int getDefaultImportFileSlot() const noexcept
    {
        const auto& decls = processor.getJsfxFileDecls();
        for (const auto& d : decls)
            if (d.enhancedImport)
                return d.index0;

        return -1;
    }

    void startImportAction (std::vector<juce::File> files, za::fileimport::ImportAction action)
    {
        const int slot = getDefaultImportFileSlot();
        if (slot < 0)
        {
            juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
                                                    "Import files",
                                                    "This JSFX declares no filename slots, so imported audio cannot be attached.");
            return;
        }

        if (action == za::fileimport::ImportAction::LoadSeparate)
        {
            processor.setFileSlotPathsWithMode (slot, files, JSFXJuceProcessor::FileLoadMode::SeparateEntries, true);
            return;
        }

        if (action == za::fileimport::ImportAction::AppendRawAsSingle)
        {
            auto rules = za::fileimport::makeDefaultRulesForAction (action);
            renderImportActionAsync (slot, std::move (files), action, rules);
            return;
        }

        auto rules = za::fileimport::makeDefaultRulesForAction (action);

        // Keep a stable copy for the preview dialog. Function argument evaluation order
        // can otherwise move `files` into the apply lambda before the preview receives it,
        // leaving the dialog with no waveform input.
        auto previewFiles = files;

        juce::Component::SafePointer<JSFXJuceEditor> safeThis (this);
        za::fileimport::showImportPreviewDialog (*this, std::move (previewFiles), action, rules,
            [safeThis, slot, filesForApply = std::move (files), action] (za::fileimport::ImportRules acceptedRules) mutable
            {
                if (safeThis != nullptr)
                    safeThis->renderImportActionAsync (slot, std::move (filesForApply), action, acceptedRules);
            },
            [safeThis] (juce::AudioBuffer<float> buffer, double sampleRate) mutable
            {
                if (safeThis != nullptr)
                    safeThis->processor.auditionImportPreviewBuffer (std::move (buffer), sampleRate);
            },
            [safeThis]
            {
                if (safeThis != nullptr)
                    safeThis->processor.stopImportPreviewAudition();
            },
            [safeThis] (bool paused)
            {
                if (safeThis != nullptr)
                    safeThis->processor.pauseImportPreviewAudition (paused);
            }, "File slot " + juce::String (slot + 1));
    }

    void renderImportActionAsync (int slot, std::vector<juce::File> files, za::fileimport::ImportAction action, za::fileimport::ImportRules rules)
    {
        juce::Component::SafePointer<JSFXJuceEditor> safeThis (this);
        std::thread ([safeThis, slot, files = std::move (files), action, rules] () mutable
        {
            auto result = za::fileimport::renderImportAction (files, action, rules);
            juce::MessageManager::callAsync ([safeThis, slot, sourceFiles = files, result = std::move (result)] () mutable
            {
                if (safeThis == nullptr)
                    return;

                if (! result.ok)
                {
                    juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
                                                            "Import failed",
                                                            result.message.isNotEmpty() ? result.message : "The import recipe produced no output.");
                    return;
                }

                juce::String recipeXml;
                if (auto xml = za::fileimport::recipeToValueTree (result.recipe).createXml())
                    recipeXml = xml->toString();

                if (! result.renderedAudio.empty())
                {
                    safeThis->processor.setFileSlotRenderedAudioData (slot, result.files.empty() ? sourceFiles : result.files, std::move (result.renderedAudio), true, recipeXml);
                }
                else
                {
                    const auto mode = result.loadMode == za::fileimport::RenderedLoadMode::AppendAsSingleFile
                                        ? JSFXJuceProcessor::FileLoadMode::AppendAsSingleFile
                                        : JSFXJuceProcessor::FileLoadMode::SeparateEntries;
                    safeThis->processor.setFileSlotPathsWithMode (slot, result.files, mode, true, recipeXml);
                }
            });
        }).detach();
    }
    // ---- /ZA_FILE_IMPORT_RECIPE_PATCH ----

private:
    static constexpr int kTopBarH = 40;
    static constexpr int kSectionGap = 6;
    static constexpr int kMinEditorWidth = 520;
    static constexpr int kMaxEditorWidth = 2400;
    static constexpr int kMaxEditorHeight = 2600;
    static constexpr int kFallbackGfxWidth = 640;
    static constexpr int kFallbackGfxHeight = 360;
    static constexpr int kMinimumScaledGfxHeight = 240;
    static constexpr double kMaxScreenWidthFraction = 0.92;
    static constexpr double kMaxScreenHeightFraction = 0.88;

    struct EditorScreenBudget
    {
        int maxWidth = kMaxEditorWidth;
        int maxHeight = kMaxEditorHeight;
    };

    struct EditorResizeLimits
    {
        int minWidth = kMinEditorWidth;
        int minHeight = kTopBarH + 120;
        int maxWidth = kMaxEditorWidth;
        int maxHeight = kMaxEditorHeight;
    };

    EditorResizeLimits computeEditorResizeLimits (const za::pluginui::ScaledSectionLayout& plan,
                                                     const EditorScreenBudget& screenBudget,
                                                     bool hasControls,
                                                     bool hasGfx) const noexcept
    {
        EditorResizeLimits out;
        out.maxWidth = juce::jmax (screenBudget.maxWidth, kMinEditorWidth);
        out.maxHeight = juce::jmax (screenBudget.maxHeight, kTopBarH + 120);

        int desiredMinW = plan.minWidth;
        int desiredMinH = juce::jmax (kTopBarH + 120, plan.minHeight);

        if (hasGfx && gfxView.declaredSizeIsMinimum())
        {
            desiredMinW = juce::jmax (desiredMinW, getDeclaredGfxPreferredWidth());

            const int gap = hasControls ? kSectionGap : 0;
            const int minControlsH = hasControls ? getMinControlsViewportHeight() : 0;
            const int declaredGfxH = getDeclaredGfxPreferredHeight();
            desiredMinH = juce::jmax (desiredMinH, kTopBarH + minControlsH + gap + declaredGfxH);
        }

        out.minWidth = juce::jlimit (kMinEditorWidth, out.maxWidth, desiredMinW);
        out.minHeight = juce::jlimit (kTopBarH + 120, out.maxHeight, desiredMinH);
        out.maxWidth = juce::jmax (out.maxWidth, out.minWidth);
        out.maxHeight = juce::jmax (out.maxHeight, out.minHeight);
        return out;
    }

    EditorScreenBudget getScreenAwareEditorBudget() const
    {
        juce::Rectangle<int> userArea;

        const auto& displays = juce::Desktop::getInstance().getDisplays();
        if (auto* display = displays.getDisplayForRect (getScreenBounds(), false))
            userArea = display->userArea;

        if (userArea.isEmpty())
        {
            if (auto* display = displays.getPrimaryDisplay())
                userArea = display->userArea;
        }

        if (userArea.isEmpty())
            userArea = { 0, 0, kMaxEditorWidth, kMaxEditorHeight };

        return { juce::jlimit (kMinEditorWidth,
                               kMaxEditorWidth,
                               (int) std::lround ((double) userArea.getWidth() * kMaxScreenWidthFraction)),
                 juce::jlimit (kTopBarH + 120,
                               kMaxEditorHeight,
                               (int) std::lround ((double) userArea.getHeight() * kMaxScreenHeightFraction)) };
    }

    int getDeclaredGfxPreferredWidth() const noexcept
    {
        const int pref = gfxView.preferredWidth();
        return pref > 0 ? pref : kFallbackGfxWidth;
    }

    int getDeclaredGfxPreferredHeight() const noexcept
    {
        const int pref = gfxView.preferredHeight();
        return pref > 0 ? pref : kFallbackGfxHeight;
    }

    int getScaledDeclaredGfxPreferredHeightForWidth (int /*availableWidth*/) const noexcept
    {
        // The preferred @gfx height is authoritative for the initial layout.
        // Actual resizing is reported to the JSFX by assigning GfxView the real
        // remaining bounds, not by pre-scaling the declared height here.
        return getDeclaredGfxPreferredHeight();
    }

    int getMinControlsViewportHeight() const noexcept
    {
        if (genericPrefH <= 0)
            return 0;

        return juce::jmin (genericPrefH, 180);
    }

    void updateControlsViewportContentSize()
    {
        if (! controlsViewport.isVisible())
            return;

        const int contentH = juce::jmax (controlsViewport.getMaximumVisibleHeight(), genericPrefH);
        const int pass1W = juce::jmax (1, controlsViewport.getMaximumVisibleWidth());
        genericEditor.setSize (pass1W, contentH);

        const int pass2W = juce::jmax (1, controlsViewport.getMaximumVisibleWidth());
        if (pass2W != genericEditor.getWidth())
            genericEditor.setSize (pass2W, contentH);
    }

    UnicodeLNF unicodeLnf;

    // -----------------------
    // Correctness monitor overlay
    // -----------------------
   #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
    class CorrectnessOverlay final : public juce::Component, private juce::Timer
    {
    public:
        explicit CorrectnessOverlay (JSFXJuceProcessor& p)
            : processor (p)
        {
            title.setText ("CORRECTNESS CHECK", juce::dontSendNotification);
            title.setJustificationType (juce::Justification::centredLeft);
            title.setFont (UnicodeLNF::pickFont (16.0f).boldened());
            addAndMakeVisible (title);

            close.setButtonText ("X");
            close.onClick = [this] { setVisible (false); };
            addAndMakeVisible (close);

            modeLabel.setText ("Monitor audio", juce::dontSendNotification);
            modeLabel.setJustificationType (juce::Justification::centredLeft);
            addAndMakeVisible (modeLabel);

            mode.addItem ("Compiled DSP-JSFX", 1);
            mode.addItem ("Shadow EEL2", 2);
            mode.addItem ("Delta", 3);
            mode.onChange = [this]
            {
                processor.setCorrectnessMonitorMode (juce::jmax (0, mode.getSelectedId() - 1));
                refreshNow();
            };
            addAndMakeVisible (mode);

            freezeToggle.setButtonText ("Freeze on first mismatch");
            freezeToggle.onClick = [this]
            {
                processor.setCorrectnessFreezeOnFirstMismatch (freezeToggle.getToggleState());
                refreshNow();
            };
            addAndMakeVisible (freezeToggle);

            clearButton.setButtonText ("Clear");
            clearButton.onClick = [this]
            {
                processor.clearCorrectnessMonitor();
                refreshNow();
            };
            addAndMakeVisible (clearButton);

            exportButton.setButtonText ("Export 30s");
            exportButton.onClick = [this]
            {
                footer.setText (processor.exportCorrectnessArtifacts(), juce::dontSendNotification);
                refreshNow();
            };
            addAndMakeVisible (exportButton);

            footer.setJustificationType (juce::Justification::centredLeft);
            footer.setFont (UnicodeLNF::pickFont (12.5f));
            addAndMakeVisible (footer);

            status.setMultiLine (true);
            status.setReadOnly (true);
            status.setScrollbarsShown (true);
            status.setCaretVisible (false);
            status.setPopupMenuEnabled (true);
            status.setFont (UnicodeLNF::pickFont (13.5f));
            status.setLineSpacing (1.2f);
            status.setColour (juce::TextEditor::backgroundColourId, juce::Colours::transparentBlack);
            status.setColour (juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
            status.setColour (juce::TextEditor::shadowColourId, juce::Colours::transparentBlack);
            addAndMakeVisible (status);

            setWantsKeyboardFocus (true);
            refreshNow();
            startTimerHz (8);
        }

        void refreshNow()
        {
            const bool enabled = processor.hasCorrectnessMonitor();
            mode.setEnabled (enabled);
            freezeToggle.setEnabled (enabled);
            clearButton.setEnabled (enabled);
            exportButton.setEnabled (enabled);

            mode.setSelectedId (juce::jlimit (1, 3, processor.getCorrectnessMonitorMode() + 1), juce::dontSendNotification);
            freezeToggle.setToggleState (processor.getCorrectnessFreezeOnFirstMismatch(), juce::dontSendNotification);

            const auto statusText = processor.getCorrectnessStatusText();
            if (status.getText() != statusText)
                status.setText (statusText, juce::dontSendNotification);
        }

        void paint (juce::Graphics& g) override
        {
            g.fillAll (juce::Colours::black.withAlpha (0.55f));

            auto panel = panelBounds.toFloat();
            g.setColour (juce::Colours::darkgrey.withAlpha (0.96f));
            g.fillRoundedRectangle (panel, 10.0f);

            g.setColour (juce::Colours::white.withAlpha (0.25f));
            g.drawRoundedRectangle (panel, 10.0f, 1.5f);
        }

        void resized() override
        {
            auto r = getLocalBounds();
            const int margin = juce::jlimit (16, 48, juce::jmin (getWidth(), getHeight()) / 14);
            auto panel = r.reduced (margin);

            panel.setWidth (juce::jmax (620, panel.getWidth()));
            panel.setHeight (juce::jmax (360, panel.getHeight()));
            panel = panel.withCentre (r.getCentre());
            panelBounds = panel;

            auto inner = panel.reduced (16);
            auto header = inner.removeFromTop (30);
            close.setBounds (header.removeFromRight (28));
            title.setBounds (header);

            inner.removeFromTop (10);
            auto controls = inner.removeFromTop (70);

            auto row1 = controls.removeFromTop (28);
            modeLabel.setBounds (row1.removeFromLeft (110));
            mode.setBounds (row1.removeFromLeft (220));

            auto row2 = controls.removeFromTop (32);
            freezeToggle.setBounds (row2.removeFromLeft (220));
            exportButton.setBounds (row2.removeFromRight (110));
            row2.removeFromRight (8);
            clearButton.setBounds (row2.removeFromRight (80));

            inner.removeFromTop (8);
            auto footerArea = inner.removeFromBottom (24);
            footer.setBounds (footerArea);

            status.setBounds (inner);
        }

        void mouseUp (const juce::MouseEvent& e) override
        {
            if (! panelBounds.contains (e.getPosition()))
                setVisible (false);
        }

    private:
        void timerCallback() override
        {
            if (isVisible())
                refreshNow();
        }

        JSFXJuceProcessor& processor;
        juce::Rectangle<int> panelBounds;
        juce::Label title;
        juce::Label modeLabel;
        juce::Label footer;
        juce::TextButton close;
        juce::ComboBox mode;
        juce::ToggleButton freezeToggle;
        juce::TextButton clearButton;
        juce::TextButton exportButton;
        juce::TextEditor status;
    };
   #endif

    // -----------------------
    // Tooltip handling (JUCE TooltipWindow)
    // -----------------------
    class SmartTooltipWindow final : public juce::TooltipWindow
    {
    public:
        explicit SmartTooltipWindow (JSFXJuceEditor& owner, int delayMs)
            : juce::TooltipWindow (&owner, delayMs), editor (owner)
        {
        }

    private:
        juce::String getTipFor (juce::Component& c) override
        {
            // Hide tooltips while the HELP overlay is open or while dragging.
            if (editor.helpOverlay.isVisible())
                return {};

           #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
            if (editor.correctnessOverlay.isVisible())
                return {};
           #endif

            if (juce::ModifierKeys::getCurrentModifiers().isAnyMouseButtonDown())
                return {};

            return juce::TooltipWindow::getTipFor (c);
        }

        JSFXJuceEditor& editor;
    };

// JSFX @gfx view (rendered by YSFXGfxInterpreter)
// -----------------------
class GfxView final : public juce::Component,
                      private juce::AsyncUpdater
{
public:
    enum class ResizePolicy
    {
        fixed,
        responsive,
        scale,
        scaleUp,
        scaleDown
    };

    explicit GfxView (JSFXJuceProcessor& p)
        : processor (p)
    {
        // The canvas may deliberately occupy only part of this component
        // for fixed/scaled modes. Let the editor background show through
        // instead of painting unused area as a giant black GFX surface.
        setOpaque (false);
        setInterceptsMouseClicks (true, true);

        setWantsKeyboardFocus (true);
        setMouseClickGrabsKeyboardFocus (true);

        menuBridge.setCallbacks (
            [this]() { triggerAsyncUpdate(); },
            [this]() { quietInputForMenu(); },
            [this]() { notifyWorker(); });

        menuOverlay.onFinished = [this] (int result)
        {
            menuBridge.completeMenu (result);
            grabKeyboardFocus();
            repaint();
        };

        addAndMakeVisible (menuOverlay);
        menuOverlay.setVisible (false);

       #if DSPJSFX_HAS_NATIVE_GFX
        setName ("Native JSFX graphics");
#if DSPJSFX_NATIVE_GFX_LEGACY
        nativeFrame = processor.legacyGraphics();
#else
        nativeFrame = std::make_shared<jsfx_native_gfx::Frame>();
#endif
        nativeFrame->menuPort = &menuBridge;
        processor.restoreNativeGfxUi (*nativeFrame);
        nativeFileDropConsumer = sourceUsesNativeGfxGetDropFile(kJsfxSourceText);
        nativeFrame->imageLoader=[](const juce::String& name){return jsfx_gfx_resources::loadImage(name);};
        nativeFrame->configureResources(kJsfxSourceText);
        hasGfxFlag = DSPJSFX_NATIVE_GFX_PRESENT != 0;
        gfxCompiledOkFlag = true;
        gfxPrefW = DSPJSFX_NATIVE_GFX_WIDTH;
        gfxPrefH = DSPJSFX_NATIVE_GFX_HEIGHT;
       #else
       #if ZA_SAMPLE_GFX_TEST_RUNNER
        setName ("EEL JSFX graphics");
       #endif
        jsfx_gfx_compat::registerBuiltins();
        nativeFileDropConsumer = sourceUsesNativeGfxGetDropFile (kJsfxSourceText);
        interp = std::make_unique<jsfx_gfx::Interpreter> (
            kJsfxSourceText,
            [](const juce::String& name){return jsfx_gfx_resources::loadImage(name);},
            [this] (int filenameIndex, const juce::String& token)
            {
                return processor.resolveGfxFileCandidates (filenameIndex, token);
            });

        if (interp != nullptr)
        {
            hasGfxFlag = interp->hasGfxSection();
            gfxCompiledOkFlag = interp->gfxCompiledOk();
            gfxLastError = interp->getLastError();
            gfxPrefW = interp->preferredWidth();
            gfxPrefH = interp->preferredHeight();
        }

       #endif
        gfxResizePolicy = parseGfxResizePolicy (kJsfxSourceText);

        if (hasGfxFlag && gfxCompiledOkFlag)
        {
            if (interp != nullptr) interp->setMenuPort (&menuBridge);
            processor.registerGfxSnapshotClient();
            updateRenderTargetSize();
            startWorker();
        }
    }

    ~GfxView() override
    {
        processor.setGfxAnalysisVisible(false);
        stopWorker();
#if DSPJSFX_NATIVE_GFX_LEGACY
        // Deliver a final released-input frame before detaching the UI context.
        // The script releases its own held state; no named guest cells are reset by the host.
        if (nativeFrame && nativeFrame->state.sharedState && legacyHasExecutedFrame.load(std::memory_order_acquire) &&
            (windowVisible.load(std::memory_order_acquire) || legacyReleaseFramePending.exchange(false, std::memory_order_acq_rel)) &&
            !nativeRuntimeFault.load(std::memory_order_acquire)) {
            const std::lock_guard<std::mutex> lock(processor.legacyGfxMutex());
            nativeFrame->focused = nativeFrame->visible = false;
            nativeFrame->menuPort = nullptr;
            nativeFrame->keys.clear(); nativeFrame->keysDown.clear();
            nativeFrame->set("mouse_cap", 0);
            nativeFrame->set("mouse_wheel", 0); nativeFrame->set("mouse_hwheel", 0);
            nativeFrame->run();
        }
#endif
       #if DSPJSFX_HAS_NATIVE_GFX
        processor.setNativeGfxInputActive (false);
        processor.saveNativeGfxUi (*nativeFrame);
       #endif
        cancelPendingUpdate();
        menuBridge.cancelAll();
        menuOverlay.forceCloseSilently();

        if (hasGfxFlag && gfxCompiledOkFlag)
            processor.unregisterGfxSnapshotClient();

        processor.endAllGfxGestures();
        if (interp != nullptr)
            interp->setMenuPort (nullptr);
        interp.reset(); // Run VM atexit before the view's callback targets die.
    }

    void visibilityChanged() override { updateAnalysisVisibility(); }
    void parentHierarchyChanged() override { updateAnalysisVisibility(); }

    void updateAnalysisVisibility()
    {
        windowFocused.store (hasKeyboardFocus (true), std::memory_order_release);
        const bool wasVisible = windowVisible.exchange (isShowing(), std::memory_order_acq_rel);
        if (auto* peer = getPeer())
            displayScale.store (juce::jlimit (1.0, 4.0, (double) peer->getPlatformScaleFactor()), std::memory_order_release);
        processor.setGfxAnalysisVisible(hasGfxFlag && gfxCompiledOkFlag && isShowing()
                                       && getWidth() > 0 && getHeight() > 0);
#if DSPJSFX_NATIVE_GFX_LEGACY
        if (hasGfxFlag && wasVisible && !isShowing()) {
            focusLost(juce::Component::focusChangedDirectly);
            legacyReleaseFramePending.store(true, std::memory_order_release);
            notifyWorker();
        }
#endif
    }

    bool hasGfx() const noexcept { return hasGfxFlag; }
    bool acceptsNativeFileDrop() const noexcept
    {
        return hasGfxFlag && gfxCompiledOkFlag && nativeFileDropConsumer;
    }
    int preferredHeight() const noexcept { return gfxPrefH; }
    int preferredWidth()  const noexcept { return gfxPrefW; }
    ResizePolicy resizePolicy() const noexcept { return gfxResizePolicy; }

    bool declaredSizeIsMinimum() const noexcept
    {
        switch (gfxResizePolicy)
        {
            case ResizePolicy::scale:
            case ResizePolicy::scaleDown:
                return false;

            case ResizePolicy::fixed:
            case ResizePolicy::responsive:
            case ResizePolicy::scaleUp:
            default:
                return true;
        }
    }

    bool usesResponsiveGfxSize() const noexcept
    {
        return gfxResizePolicy == ResizePolicy::responsive;
    }

    void paint (juce::Graphics& g) override
    {
       #if DSPJSFX_HAS_NATIVE_GFX
        if (nativeRuntimeFault.load (std::memory_order_acquire))
        {
            g.fillAll (juce::Colours::black); g.setColour (juce::Colours::red);
            g.drawText ("Native graphics publication contract violated", getLocalBounds(), juce::Justification::centred);
            return;
        }
       #endif
        if (! hasGfxFlag)
        {
            g.fillAll (juce::Colours::black);
            g.setColour (juce::Colours::white.withAlpha (0.5f));
            g.drawText ("(no @gfx section)", getLocalBounds(), juce::Justification::centred);
            return;
        }

        if (! gfxCompiledOkFlag)
        {
            g.fillAll (juce::Colours::black);

            const auto area = getLocalBounds().reduced (6);
            juce::AttributedString text;
            text.setJustification (juce::Justification::topLeft);
            text.append ("@gfx compile error:\n" + (gfxLastError.isNotEmpty() ? gfxLastError : juce::String ("(unknown error)")),
                         juce::Font (14.0f),
                         juce::Colours::red.withAlpha (0.9f));

            juce::TextLayout layout;
            layout.createLayout (text, (float) area.getWidth());
            layout.draw (g, area.toFloat());
            return;
        }

        const juce::Image frame = framePool.front();

        if (frame.isNull())
            return;

        const auto placement = getFramePlacement (getWidth(), getHeight());
        g.saveState();
        const float pixelScale = placement.scale * (float) placement.renderW / (float) juce::jmax (1, frame.getWidth());
        g.addTransform (juce::AffineTransform::scale (pixelScale)
                            .translated (placement.offsetX, placement.offsetY));
        g.drawImageAt (frame, 0, 0);
        g.restoreState();
    }

    void resized() override
    {
        menuOverlay.setBounds (getLocalBounds());
        updateRenderTargetSize();
        updateAnalysisVisibility();

        notifyWorker();
        repaint();
    }

    void mouseMove (const juce::MouseEvent& e) override { updateMouse (e, false, false, false); }
    void mouseDrag (const juce::MouseEvent& e) override { updateMouse (e, true, false, false); }

    void mouseDown (const juce::MouseEvent& e) override
    {
        grabKeyboardFocus();
        updateMouse (e, true, false, true);
    }

    void mouseUp (const juce::MouseEvent& e) override
    {
        // Some JUCE backends report the just-released button as still present
        // in e.mods during mouseUp(). Use the current global modifier state so
        // releases are observed immediately and in the correct order.
        updateMouse (e, true, true, true);
    }

    void mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& d) override
    {
       #if DSPJSFX_HAS_NATIVE_GFX && ! DSPJSFX_NATIVE_GFX_INTERACTIVE
        return; // Display-only contract.
       #endif
        if (! hasGfxFlag || ! gfxCompiledOkFlag)
            return;

        {
            const std::lock_guard<std::mutex> lock (inputMutex);
            updateMouseCapFromModifiers (e.mods);

            const auto logical = physicalToLogical (e.position);

            MouseStateFrame frame;
            frame.mouseX = logical.x;
            frame.mouseY = logical.y;
            frame.mouseCap = mouseCap;
            frame.pendingWheel = (float) (d.deltaY * 120.0);
            frame.pendingHWheel = (float) (d.deltaX * 120.0);
            frame.captureStateWrites = true;
            frame.preserveOrdering = false;

            enqueueMouseFrameLocked (frame);
            sharedInput.captureStateWrites = true;
        }

        notifyWorker();
    }

    bool keyPressed (const juce::KeyPress& key) override
    {
       #if DSPJSFX_HAS_NATIVE_GFX && ! DSPJSFX_NATIVE_GFX_INTERACTIVE
        return false; // Display-only contract.
       #endif
        if (! hasGfxFlag || ! gfxCompiledOkFlag)
            return false;

        const int jsfxCode = juceKeyPressToJsfx (key);
        if (jsfxCode == 0)
            return false;

        {
            const std::lock_guard<std::mutex> lock (inputMutex);
            updateMouseCapFromModifiers (juce::ModifierKeys::getCurrentModifiers());
            sharedInput.mouseCap = mouseCap;
            sharedInput.captureStateWrites = true;
            sharedInput.keyEvents.push_back (KeyEvent { jsfxCode, true, true, key.getKeyCode() });
        }

        trackedKeys[(uint32_t) jsfxCode] = key.getKeyCode();
        notifyWorker();
        return true;
    }

    bool keyStateChanged (bool /*isKeyDown*/) override
    {
       #if DSPJSFX_HAS_NATIVE_GFX && ! DSPJSFX_NATIVE_GFX_INTERACTIVE
        return false; // Display-only contract.
       #endif
        if (! hasGfxFlag || ! gfxCompiledOkFlag)
            return false;

        bool changed = false;
        bool capChanged = false;
        {
            const std::lock_guard<std::mutex> lock (inputMutex);
            updateMouseCapFromModifiers (juce::ModifierKeys::getCurrentModifiers());
            capChanged = sharedInput.mouseCap != mouseCap;
            sharedInput.mouseCap = mouseCap;

            for (auto it = trackedKeys.begin(); it != trackedKeys.end();)
            {
                const int jsfxCode = (int) it->first;
                const int juceKeyCode = it->second;
                const bool downNow = juce::KeyPress::isKeyCurrentlyDown (juceKeyCode);

                if (! downNow)
                {
                    sharedInput.keyEvents.push_back (KeyEvent { jsfxCode, false, false, juceKeyCode });
                    it = trackedKeys.erase (it);
                    changed = true;
                }
                else
                {
                    ++it;
                }
            }

            if (changed || capChanged)
                sharedInput.captureStateWrites = true;
        }

        if (changed || capChanged)
            notifyWorker();

        return changed || capChanged;
    }

    void focusGained (juce::Component::FocusChangeType) override
    {
       #if DSPJSFX_HAS_NATIVE_GFX
        processor.setNativeGfxInputActive (true);
       #endif
        updateAnalysisVisibility();
        notifyWorker();
    }

    // The parent editor routes OS drops here after deciding that native
    // gfx_getdropfile(), rather than the enhanced #FILE importer, owns them.
    void enqueueNativeFileDrop (const juce::StringArray& files, juce::Point<float> position)
    {
        if (! acceptsNativeFileDrop())
            return;

        std::vector<juce::String> batch;
        batch.reserve ((size_t) juce::jmin (files.size(), 1024));
        for (const auto& path : files)
            if (path.isNotEmpty() && batch.size() < 1024)
                batch.push_back (path);

        if (! batch.empty())
        {
            const std::lock_guard<std::mutex> lock (inputMutex);
            size_t queuedDrops = 0;
            for (const auto& pending : sharedInput.mouseFrames)
                if (!pending.droppedFiles.empty()) ++queuedDrops;
            if (queuedDrops >= 64) return;
            // Deliver the files with their hit-test position, after earlier
            // input edges, rather than relying on a preceding mouse move.
            const auto logical = physicalToLogical (position);
            MouseStateFrame frame;
            frame.mouseX = logical.x;
            frame.mouseY = logical.y;
            frame.mouseCap = sharedInput.mouseCap & ~kMouseButtonMask;
            frame.captureStateWrites = frame.preserveOrdering = true;
            frame.droppedFiles = std::move (batch);
            enqueueMouseFrameLocked (frame);
            sharedInput.captureStateWrites = true;
        }
        notifyWorker();
    }

    void focusLost (juce::Component::FocusChangeType) override
    {
       #if DSPJSFX_HAS_NATIVE_GFX
        processor.setNativeGfxInputActive (false);
       #endif
        windowFocused.store (false, std::memory_order_release);
        trackedKeys.clear();

        {
            const std::lock_guard<std::mutex> lock (inputMutex);
            sharedInput.clearKeys = true;
            sharedInput.pendingWheel = 0.0f;
            sharedInput.pendingHWheel = 0.0f;

            const int releasedCap = sharedInput.mouseCap & ~kMouseButtonMask;
            sharedInput.mouseFrames.clear();

            MouseStateFrame frame;
            frame.mouseX = sharedInput.mouseX;
            frame.mouseY = sharedInput.mouseY;
            frame.mouseCap = releasedCap;
            frame.captureStateWrites = true;
            frame.preserveOrdering = true;
            sharedInput.mouseFrames.push_back (frame);

            sharedInput.mouseCap = releasedCap;
            sharedInput.captureStateWrites = true;
            mouseCap = releasedCap;
        }

        notifyWorker();
        processor.endAllGfxGestures();
    }

private:
    struct MemDiffSpan
    {
        int64_t base = 0;
        std::vector<double> before;
        std::vector<double> after;
    };

    struct KeyEvent
    {
        int jsfxCode = 0;
        bool keyDown = false;
        bool enqueueChar = false;
        int rawKeyCode = 0;
    };

    struct MouseStateFrame
    {
        float mouseX = 0.0f;
        float mouseY = 0.0f;
        int mouseCap = 0;
        float pendingWheel = 0.0f;
        float pendingHWheel = 0.0f;
        bool captureStateWrites = false;
        bool preserveOrdering = false; // true for button-edge/release frames; false means latest-state aggregation
        std::vector<juce::String> droppedFiles; // Attached to this input frame.
    };

    struct SharedInputState
    {
        float mouseX = 0.0f;
        float mouseY = 0.0f;
        int mouseCap = 0;
        float pendingWheel = 0.0f;
        float pendingHWheel = 0.0f;
        bool captureStateWrites = false;
        bool clearKeys = false;
        std::deque<KeyEvent> keyEvents;
        std::deque<std::vector<juce::String>> droppedFileBatches;
        std::deque<MouseStateFrame> mouseFrames;
    };

    struct PendingSliderApply
    {
        std::array<double, DSPJSFX_MAX_SLIDERS> sliders {};
        SliderMask changeMask {};
        SliderMask automateMask {};
        SliderMask automateEndMask {};
        bool pending = false;
    };

    using MenuBridge=JsfxGfxMenuBridge;

    // A custom @gfx surface should claim file drops only when the script
    // actually consumes gfx_getdropfile(). Merely having @gfx must not steal
    // drops from an enhanced #FILE importer (Sample/Corpus are examples).
    // Ignore comments/string literals and treat gfx_getdropfile(-1) as the
    // reset/acknowledge form rather than evidence that the script consumes a
    // pathname.
    static bool sourceUsesNativeGfxGetDropFile (const char* jsfxText) noexcept
    {
        if (jsfxText == nullptr)
            return false;

        const std::string text (jsfxText);
        enum class LexState { normal, lineComment, blockComment, stringLiteral };
        LexState state = LexState::normal;
        const std::string needle = "gfx_getdropfile";

        const auto isIdent = [] (char c) noexcept
        {
            const unsigned char u = (unsigned char) c;
            return std::isalnum (u) != 0 || c == '_';
        };

        for (size_t i = 0; i < text.size();)
        {
            const char c = text[i];
            const char n = i + 1 < text.size() ? text[i + 1] : '\0';

            if (state == LexState::lineComment)
            {
                if (c == '\n' || c == '\r') state = LexState::normal;
                ++i;
                continue;
            }
            if (state == LexState::blockComment)
            {
                if (c == '*' && n == '/') { state = LexState::normal; i += 2; }
                else ++i;
                continue;
            }
            if (state == LexState::stringLiteral)
            {
                if (c == '\\' && i + 1 < text.size()) { i += 2; continue; }
                if (c == '"') state = LexState::normal;
                ++i;
                continue;
            }

            if (c == '/' && n == '/') { state = LexState::lineComment; i += 2; continue; }
            if (c == '/' && n == '*') { state = LexState::blockComment; i += 2; continue; }
            if (c == '"') { state = LexState::stringLiteral; ++i; continue; }

            if (i + needle.size() <= text.size()
                && text.compare (i, needle.size(), needle) == 0
                && (i == 0 || ! isIdent (text[i - 1]))
                && (i + needle.size() == text.size() || ! isIdent (text[i + needle.size()])))
            {
                size_t p = i + needle.size();
                while (p < text.size() && std::isspace ((unsigned char) text[p]) != 0) ++p;
                if (p >= text.size() || text[p] != '(') { i += needle.size(); continue; }
                ++p;
                while (p < text.size() && std::isspace ((unsigned char) text[p]) != 0) ++p;

                // gfx_getdropfile(-1) only clears/acknowledges the pending drop.
                // Any other first argument can request a pathname.
                bool resetOnly = false;
                size_t q = p;
                if (q < text.size() && text[q] == '-')
                {
                    ++q;
                    while (q < text.size() && std::isspace ((unsigned char) text[q]) != 0) ++q;
                    if (q < text.size() && text[q] == '1')
                    {
                        ++q;
                        while (q < text.size() && std::isspace ((unsigned char) text[q]) != 0) ++q;
                        resetOnly = q < text.size() && (text[q] == ')' || text[q] == ',');
                    }
                }

                if (! resetOnly)
                    return true;

                i += needle.size();
                continue;
            }

            ++i;
        }

        return false;
    }

    static std::string lowerAsciiLocal (std::string text)
    {
        std::transform (text.begin(), text.end(), text.begin(), [] (unsigned char c)
        {
            return (char) std::tolower (c);
        });
        return text;
    }

    static ResizePolicy parseGfxResizePolicyToken (std::string token) noexcept
    {
        token = lowerAsciiLocal (trimAscii (std::move (token)));
        std::replace (token.begin(), token.end(), '-', '_');

        if (token == "responsive" || token == "resize" || token == "resizable" || token == "native")
            return ResizePolicy::responsive;

        if (token == "scale" || token == "scaled" || token == "scale_both" || token == "scaled_both")
            return ResizePolicy::scale;

        if (token == "scale_up" || token == "up" || token == "grow" || token == "scaleup")
            return ResizePolicy::scaleUp;

        if (token == "scale_down" || token == "down" || token == "shrink" || token == "scaledown")
            return ResizePolicy::scaleDown;

        return ResizePolicy::fixed;
    }

    static ResizePolicy parseGfxResizePolicy (const char* jsfxText)
    {
        if (jsfxText == nullptr)
            return ResizePolicy::responsive;

        std::string text (jsfxText);
        size_t start = 0;

        const std::regex reComment (R"(^\s*//\s*#GFX_RESIZE:\s*([^\s;#]+).*$)",
                                    std::regex::ECMAScript | std::regex::icase);
        const std::regex reOptions (R"(^\s*options\s*:\s*(.*)$)",
                                    std::regex::ECMAScript | std::regex::icase);
        const std::regex reOptionValue (R"((?:^|[\s,])gfx_resize\s*=\s*([^\s,]+))",
                                        std::regex::ECMAScript | std::regex::icase);

        while (start < text.size())
        {
            size_t end = text.find_first_of ("\r\n", start);
            if (end == std::string::npos)
                end = text.size();

            const std::string line = text.substr (start, end - start);

            size_t next = end;
            while (next < text.size() && (text[next] == '\r' || text[next] == '\n'))
                ++next;
            start = next;

            std::smatch m;
            if (std::regex_match (line, m, reComment))
                return parseGfxResizePolicyToken (m[1].str());

            if (std::regex_match (line, m, reOptions))
            {
                const std::string payload = m[1].str();
                std::smatch ov;
                if (std::regex_search (payload, ov, reOptionValue))
                    return parseGfxResizePolicyToken (ov[1].str());
            }
        }

        // Preserve the v6 behavior when the script has no explicit policy: @gfx
        // is the requested/minimum responsive size, and user/host resizing is
        // passed through as live gfx_w/gfx_h. Fixed legacy UIs should declare:
        // // #GFX_RESIZE: fixed
        return ResizePolicy::responsive;
    }

    struct FramePlacement
    {
        float scale = 1.0f;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        int renderW = 1;
        int renderH = 1;
    };

    int declaredCanvasWidth() const noexcept
    {
        return juce::jmax (1, gfxPrefW > 0 ? gfxPrefW : 640);
    }

    int declaredCanvasHeight() const noexcept
    {
        return juce::jmax (1, gfxPrefH > 0 ? gfxPrefH : 360);
    }

    FramePlacement getFramePlacement (int viewW, int viewH) const noexcept
    {
        FramePlacement out;

        if (gfxResizePolicy == ResizePolicy::responsive)
        {
            out.renderW = juce::jmax (1, viewW);
            out.renderH = juce::jmax (1, viewH);
            return out;
        }

        out.renderW = declaredCanvasWidth();
        out.renderH = declaredCanvasHeight();

        const float sx = (float) juce::jmax (1, viewW) / (float) out.renderW;
        const float sy = (float) juce::jmax (1, viewH) / (float) out.renderH;
        const float fit = juce::jmax (0.0001f, juce::jmin (sx, sy));

        switch (gfxResizePolicy)
        {
            case ResizePolicy::scale:
                out.scale = fit;
                break;

            case ResizePolicy::scaleUp:
                out.scale = juce::jmax (1.0f, fit);
                break;

            case ResizePolicy::scaleDown:
                out.scale = juce::jmin (1.0f, fit);
                break;

            case ResizePolicy::fixed:
            case ResizePolicy::responsive:
            default:
                out.scale = 1.0f;
                break;
        }

        const float visualW = (float) out.renderW * out.scale;
        const float visualH = (float) out.renderH * out.scale;

        const bool centeringHelps = gfxResizePolicy == ResizePolicy::scale
                                 || (gfxResizePolicy == ResizePolicy::scaleUp && out.scale > 1.0f)
                                 || (gfxResizePolicy == ResizePolicy::scaleDown && out.scale < 1.0f);

        if (centeringHelps)
        {
            out.offsetX = ((float) viewW - visualW) * 0.5f;
            out.offsetY = ((float) viewH - visualH) * 0.5f;
        }

        return out;
    }

    void updateRenderTargetSize()
    {
        const auto placement = getFramePlacement (getWidth(), getHeight());
        targetWidth.store (placement.renderW, std::memory_order_release);
        targetHeight.store (placement.renderH, std::memory_order_release);
        canvasResetRequested.store (true, std::memory_order_release);
    }

    juce::Point<float> physicalToLogical (juce::Point<float> physical) const noexcept
    {
        const auto placement = getFramePlacement (getWidth(), getHeight());
        const float safeScale = juce::jmax (placement.scale, 0.0001f);
        const float pixels = (float) framebufferPixelScale.load (std::memory_order_acquire);
        return { (physical.x - placement.offsetX) * pixels / safeScale,
                 (physical.y - placement.offsetY) * pixels / safeScale };
    }

    juce::Point<float> logicalToPhysical (juce::Point<float> logical) const noexcept
    {
        const auto placement = getFramePlacement (getWidth(), getHeight());
        const float pixels = (float) framebufferPixelScale.load (std::memory_order_acquire);
        return { placement.offsetX + logical.x * placement.scale / pixels,
                 placement.offsetY + logical.y * placement.scale / pixels };
    }

    void handleAsyncUpdate() override
    {
        updateAnalysisVisibility();
        bool repaintNeeded = repaintPending.exchange (false, std::memory_order_acq_rel);
        {
            int id = 0;
            juce::String name;
            juce::Image custom;
            {
                const std::lock_guard<std::mutex> lock (cursorMutex);
                if (cursorPending) { id = pendingCursorId; name = pendingCursorName; custom = pendingCursorImage; cursorPending = false; }
            }
            if (id != 0)
            {
                if (custom.isValid()) setMouseCursor (juce::MouseCursor (custom, 0, 0));
                else setMouseCursor (standardGfxCursor (id, name));
            }
        }

        if (menuBridge.takePendingCancel())
        {
            if (menuOverlay.isMenuShowing())
                menuOverlay.forceCloseSilently();

            menuBridge.completeMenu (0);
            repaintNeeded = true;
        }

        juce::String menuDesc;
        int menuX = 0;
        int menuY = 0;
        if (menuBridge.takePendingOpen (menuDesc, menuX, menuY))
        {
            const auto menuPos = logicalToPhysical ({ (float) menuX, (float) menuY });
            menuOverlay.openMenu (menuDesc, { (int) std::lround (menuPos.x),
                                              (int) std::lround (menuPos.y) });
            if (! menuOverlay.isMenuShowing())
                menuBridge.completeMenu (0);
            repaintNeeded = true;
        }

        PendingSliderApply sliderApply;
        {
            const std::lock_guard<std::mutex> lock (pendingSliderMutex);
            sliderApply = pendingSliderApply;
            pendingSliderApply.pending = false;
            pendingSliderApply.changeMask.clear();
            pendingSliderApply.automateMask.clear();
            pendingSliderApply.automateEndMask.clear();
        }

        if (sliderApply.pending)
        {
            processor.applyGfxSliderChanges (sliderApply.sliders.data(), DSPJSFX_MAX_SLIDERS,
                                             sliderApply.changeMask,
                                             sliderApply.automateMask,
                                             sliderApply.automateEndMask);
        }

        if (repaintNeeded || sliderApply.pending)
            repaint();
    }

    void startWorker()
    {
        stopWorkerFlag.store (false, std::memory_order_release);
        workerWakeFlag.store (true, std::memory_order_release);
        workerThread = std::thread ([this] { workerLoop(); });
        workerCv.notify_one();
    }

    void stopWorker()
    {
        // gfx_showmenu() now blocks the dedicated worker thread until the menu
        // is dismissed. Tear-down must therefore cancel any outstanding menu
        // wait before joining, or the editor can deadlock on close.
        menuBridge.close();
        menuOverlay.forceCloseSilently();

        stopWorkerFlag.store (true, std::memory_order_release);
        workerWakeFlag.store (true, std::memory_order_release);
        workerCv.notify_all();

        if (workerThread.joinable())
            workerThread.join();
    }

    void notifyWorker()
    {
        workerWakeFlag.store (true, std::memory_order_release);
        workerCv.notify_one();
    }

    void quietInputForMenu()
    {
        const std::lock_guard<std::mutex> lock (inputMutex);

        const int releasedCap = sharedInput.mouseCap & ~kMouseButtonMask;
        const bool hadButtons = (sharedInput.mouseCap & kMouseButtonMask) != 0;

        sharedInput.mouseFrames.clear();
        sharedInput.pendingWheel = 0.0f;
        sharedInput.pendingHWheel = 0.0f;

        if (hadButtons)
        {
            MouseStateFrame frame;
            frame.mouseX = sharedInput.mouseX;
            frame.mouseY = sharedInput.mouseY;
            frame.mouseCap = releasedCap;
            frame.captureStateWrites = true;
            frame.preserveOrdering = true;
            sharedInput.mouseFrames.push_back (frame);
        }

        sharedInput.mouseCap = releasedCap;
        sharedInput.captureStateWrites = hadButtons;
        mouseCap = releasedCap;
    }

    void updateMouse (const juce::MouseEvent& e, bool markDirty, bool useCurrentModifiers, bool preserveOrdering)
    {
       #if DSPJSFX_HAS_NATIVE_GFX && ! DSPJSFX_NATIVE_GFX_INTERACTIVE
        return; // Display-only contract.
       #endif
        if (! hasGfxFlag || ! gfxCompiledOkFlag)
            return;

        {
            const std::lock_guard<std::mutex> lock (inputMutex);
            updateMouseCapFromModifiers (useCurrentModifiers ? juce::ModifierKeys::getCurrentModifiers()
                                                             : e.mods);

            const auto logical = physicalToLogical (e.position);

            MouseStateFrame frame;
            frame.mouseX = logical.x;
            frame.mouseY = logical.y;
            frame.mouseCap = mouseCap;
            frame.captureStateWrites = markDirty;
            frame.preserveOrdering = preserveOrdering;

            enqueueMouseFrameLocked (frame);

            if (markDirty)
                sharedInput.captureStateWrites = true;
        }

        notifyWorker();
    }

    void queueSliderApply (const std::array<double, DSPJSFX_MAX_SLIDERS>& newSliders,
                           const SliderMask& changeMask,
                           const SliderMask& automateMask,
                           const SliderMask& automateEndMask)
    {
        const SliderMask applyMask = changeMask | automateMask | automateEndMask;
        if (! applyMask.any())
            return;

        processor.stageGfxSliderPreview (newSliders.data(), DSPJSFX_MAX_SLIDERS,
                                         changeMask,
                                         automateMask,
                                         automateEndMask);

        {
            const std::lock_guard<std::mutex> lock (pendingSliderMutex);
            for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
            {
                if (! applyMask.test (i))
                    continue;
                pendingSliderApply.sliders[(size_t) i] = newSliders[(size_t) i];
            }

            pendingSliderApply.changeMask.merge (changeMask);
            pendingSliderApply.automateMask.merge (automateMask);
            pendingSliderApply.automateEndMask.merge (automateEndMask);
            pendingSliderApply.pending = true;
        }

        triggerAsyncUpdate();
    }

    void publishCanvas()
    {
        framePool.publish (workerCanvas);
#if DSPJSFX_HAS_NATIVE_GFX && (ZA_NATIVE_GFX_TEST_RUNNER || ZA_SAMPLE_GFX_TEST_RUNNER)
        jsfx_native_gfx::publishedFrames.fetch_add(1,std::memory_order_release);
#endif

        repaintPending.store (true, std::memory_order_release);
        triggerAsyncUpdate();
    }

    void workerLoop()
    {
        while (! stopWorkerFlag.load (std::memory_order_acquire))
        {
            {
                std::unique_lock<std::mutex> lock (workerWaitMutex);
                workerCv.wait_for (lock, std::chrono::milliseconds (33), [this] {
                    return stopWorkerFlag.load (std::memory_order_acquire)
                        || workerWakeFlag.exchange (false, std::memory_order_acq_rel);
                });
            }

            if (stopWorkerFlag.load (std::memory_order_acquire))
                break;

            renderWorkerFrame();
        }
    }

   #if DSPJSFX_HAS_NATIVE_GFX
    void renderNativeWorkerFrame()
    {
#if DSPJSFX_NATIVE_GFX_LEGACY
        if (!windowVisible.load(std::memory_order_acquire) &&
            !legacyReleaseFramePending.exchange(false, std::memory_order_acq_rel)) return;
#endif
        if (nativeRuntimeFault.load (std::memory_order_acquire)) return;
        const int w = juce::jlimit (1, 8192, targetWidth.load (std::memory_order_acquire));
        const int h = juce::jlimit (1, 8192, targetHeight.load (std::memory_order_acquire));
        if (! framePool.acquire (workerCanvas, w, h)) return;
        int index = -1;
#if DSPJSFX_NATIVE_GFX_LEGACY
        std::unique_lock<std::mutex> legacyLock (processor.legacyGfxMutex());
        auto& liveState = processor.legacyScriptState();
        if (liveState.mem == nullptr) return;
        JSFXJuceProcessor::GfxSnapshot live;
        jsfxCopyCells (live.sliders.data(), liveState.sliders, DSPJSFX_MAX_SLIDERS);
        live.nativeRuntimeEpoch = processor.legacyEpoch();
        live.srate = liveState.srate; live.samplesblock = liveState.samplesblock;
        const auto* snapshot = &live;
#else
        const auto* snapshot = processor.beginGfxSnapshotRead (index);
        if (snapshot == nullptr) return;
#endif
        struct ReleaseSnapshot {
            JSFXJuceProcessor& processor; int index;
            ~ReleaseSnapshot() { if (index >= 0) processor.endGfxSnapshotRead (index); }
        } pin { processor, index };
        SharedInputState inputCopy;
        bool hasQueuedMouseFramesRemaining = false;
        {
            const std::lock_guard<std::mutex> lock (inputMutex);

            if (! sharedInput.mouseFrames.empty())
            {
                const auto frame = sharedInput.mouseFrames.front();
                sharedInput.mouseFrames.pop_front();

                inputCopy.mouseX = frame.mouseX;
                inputCopy.mouseY = frame.mouseY;
                inputCopy.mouseCap = frame.mouseCap;
                inputCopy.pendingWheel = frame.pendingWheel;
                inputCopy.pendingHWheel = frame.pendingHWheel;
                inputCopy.captureStateWrites = frame.captureStateWrites;
                if (!frame.droppedFiles.empty())
                    inputCopy.droppedFileBatches.push_back (frame.droppedFiles);
            }
            else
            {
                inputCopy.mouseX = sharedInput.mouseX;
                inputCopy.mouseY = sharedInput.mouseY;
                inputCopy.mouseCap = sharedInput.mouseCap;
                inputCopy.pendingWheel = sharedInput.pendingWheel;
                inputCopy.pendingHWheel = sharedInput.pendingHWheel;
                inputCopy.captureStateWrites = false;
            }

            inputCopy.captureStateWrites = inputCopy.captureStateWrites || sharedInput.captureStateWrites;
            inputCopy.clearKeys = sharedInput.clearKeys;
            inputCopy.keyEvents.swap (sharedInput.keyEvents);

            sharedInput.pendingWheel = 0.0f;
            sharedInput.pendingHWheel = 0.0f;
            sharedInput.captureStateWrites = false;
            sharedInput.clearKeys = false;
            hasQueuedMouseFramesRemaining = ! sharedInput.mouseFrames.empty();
        }


        if (nativeUiRuntimeEpoch != snapshot->nativeRuntimeEpoch)
        {
            uiSliderOverrideMask.clear();
            if (nativeTouchMask.any())
                queueSliderApply (snapshot->sliders, {}, {}, nativeTouchMask);
            nativeTouchMask.clear();
#if !DSPJSFX_NATIVE_GFX_LEGACY
            nativeFrame->set ("ui_graphics_initialized", 0);
#endif
            for (int i = 0; i < DSPJSFX_NATIVE_GFX_COMMAND_COUNT; ++i)
                nativeFrame->state.vars[DSPJSFX_NATIVE_GFX_COMMAND_INDICES[i]] = 0;
            nativeUiRuntimeEpoch = snapshot->nativeRuntimeEpoch;
        }
        std::memcpy (effectiveSliders.data(), snapshot->sliders.data(), sizeof (double) * DSPJSFX_MAX_SLIDERS);
        for (int i = 0; !DSPJSFX_NATIVE_GFX_LEGACY && i < DSPJSFX_MAX_SLIDERS; ++i)
            if (uiSliderOverrideMask.test (i))
            {
                if (snapshot->nativeSliderAcks[(size_t) i] == nativeUiPreviewSeq[(size_t) i])
                    uiSliderOverrideMask.reset (i);
                else
                    effectiveSliders[(size_t) i] = uiSliderOverrideValues[(size_t) i];
            }
#if DSPJSFX_NATIVE_GFX_LEGACY
        nativeFrame->bindLegacy (liveState, w, h);
#else
        nativeFrame->begin (snapshot->vars.data(), snapshot->varsCount, effectiveSliders.data(),
                            snapshot->srate, snapshot->samplesblock, w, h);
        nativeFrame->publication = &snapshot->nativePublication;
#endif
        nativeFrame->focused = windowFocused.load (std::memory_order_acquire);
#if DSPJSFX_HAS_TASKS
        nativeFrame->state.taskContext = processor.taskContext();
#endif
        nativeFrame->visible = windowVisible.load (std::memory_order_acquire);
        nativeFrame->set ("mouse_x", inputCopy.mouseX);
        nativeFrame->set ("mouse_y", inputCopy.mouseY);
        nativeFrame->set ("mouse_cap", inputCopy.mouseCap);
        nativeFrame->set ("mouse_wheel", inputCopy.pendingWheel);
        nativeFrame->set ("mouse_hwheel", inputCopy.pendingHWheel);
        if (inputCopy.clearKeys) { nativeFrame->keys.clear(); nativeFrame->keysDown.clear(); }
        for (const auto& evt : inputCopy.keyEvents)
        {
            if (evt.enqueueChar && nativeFrame->keys.size() < 1024) nativeFrame->keys.push_back (evt.jsfxCode);
            if (evt.keyDown) nativeFrame->keysDown.insert (evt.jsfxCode);
            else nativeFrame->keysDown.erase (evt.jsfxCode);
            if (evt.rawKeyCode > 0 && evt.rawKeyCode < 128)
            {
                const int raw = evt.rawKeyCode;
                const int lower = raw >= 'A' && raw <= 'Z' ? raw + ('a' - 'A') : raw;
                for (int key : { raw, lower })
                    if (evt.keyDown) nativeFrame->keysDown.insert (key); else nativeFrame->keysDown.erase (key);
            }
        }
        const bool reset = canvasResetRequested.exchange(false, std::memory_order_acq_rel)
            || lastCanvasWidth != w || lastCanvasHeight != h;
        lastCanvasWidth=w;lastCanvasHeight=h;
        if(reset)workerCanvas.clear(workerCanvas.getBounds(),juce::Colours::black);
        jsfx_gfx::GfxRenderSession renderer(workerCanvas,[&](bool discard){
            if(!reset && !discard && !jsfx_gfx::copyFramebufferHistory(workerCanvas,framePool.front()))
                workerCanvas.clear(workerCanvas.getBounds(),juce::Colours::black);
        });
        nativeFrame->renderPort=&renderer;
        struct ReleaseRenderPort { jsfx_native_gfx::Frame& frame; ~ReleaseRenderPort(){frame.renderPort=nullptr;} } renderPortBinding{*nativeFrame};
        for(auto& batch:inputCopy.droppedFileBatches)nativeFrame->pendingDroppedFileBatches.push_back(std::move(batch));
        if(nativeFrame->get("gfx_ext_retina")>0)nativeFrame->set("gfx_ext_retina",displayScale.load(std::memory_order_acquire));
        const auto commandEpoch = processor.nativeGfxCommandEpochValue();
        nativeFrame->run(); // The immutable publication remains pinned through all reads.
#if DSPJSFX_NATIVE_GFX_LEGACY
        legacyHasExecutedFrame.store(true, std::memory_order_release);
#endif
        const bool fault = snapshot->nativePublication.invalid || nativeFrame->memoryFault || nativeFrame->state.memoryFault;
        nativeFrame->publication = nullptr;
        if (index >= 0) processor.endGfxSnapshotRead (index); pin.index = -1; // Raster/menu completion never touches DSP memory.
        if (fault)
        {
           #if ZA_SAMPLE_GFX_TEST_RUNNER
            processor.captureNativeGfxUi (*nativeFrame);
           #endif
            juce::Logger::writeToLog ("Native GFX publication fault");
            nativeRuntimeFault.store (true, std::memory_order_release);
            processor.setNativeGfxInputActive (false);
            triggerAsyncUpdate();
            return;
        }
        jsfxCopyCells (vmSliders.data(), nativeFrame->scriptState().sliders, DSPJSFX_MAX_SLIDERS);
        SliderMask changes = processor.actualGfxSliderChanges(nativeFrame->changeMask,effectiveSliders.data(),vmSliders.data());
#if !DSPJSFX_NATIVE_GFX_LEGACY
        for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
            if (std::abs (vmSliders[(size_t) i] - effectiveSliders[(size_t) i]) > 1e-12)
                changes.set (i);
#endif
        auto automate = nativeFrame->automateMask;
        auto endTouch = nativeFrame->automateEndMask;
        // Native controls bracket host gestures, including legacy sliderchange-only controls.
        za::jsfx::bracketGfxGesture(changes,automate,endTouch,nativeTouchMask,(inputCopy.mouseCap & kMouseButtonMask)!=0);
        const auto applyMask = changes | automate | endTouch;
        for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
            if (applyMask.test (i))
            {
                uiSliderOverrideValues[(size_t) i] = vmSliders[(size_t) i];
                uiSliderOverrideMask.set (i);
            }
        if (applyMask.any())
        {
            queueSliderApply (vmSliders, applyMask, automate, endTouch);
            for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
                if (applyMask.test (i)) nativeUiPreviewSeq[(size_t) i] = processor.nativeGfxPreviewSequence (i);
        }
        for (int i = 0; i < DSPJSFX_NATIVE_GFX_COMMAND_COUNT; ++i)
            processor.setNativeGfxCommand (i, nativeFrame->state.vars[DSPJSFX_NATIVE_GFX_COMMAND_INDICES[i]], commandEpoch);
        renderer.finish (nativeFrame->commands);
#if DSPJSFX_NATIVE_GFX_LEGACY
        legacyLock.unlock(); // Image-bank execution also precedes lifecycle reset.
#endif
        if(nativeFrame->cursorPending) {
            const auto& name=nativeFrame->cursorName;
            juce::Image custom;
            if(name.endsWithIgnoreCase(".png"))custom=jsfx_gfx_resources::cursorImage(name);
            { const std::lock_guard<std::mutex> lock(cursorMutex);
              pendingCursorId=nativeFrame->cursorResource;pendingCursorName=name;pendingCursorImage=custom;cursorPending=true; }
            nativeFrame->cursorPending=false;triggerAsyncUpdate();
        }
        publishCanvas();
       #if ZA_SAMPLE_GFX_TEST_RUNNER
        processor.captureNativeGfxUi (*nativeFrame); // Diagnostics correspond to the presented image.
       #endif
        if (hasQueuedMouseFramesRemaining) notifyWorker();
    }
   #endif

    void renderWorkerFrame()
    {
       #if DSPJSFX_HAS_NATIVE_GFX
        renderNativeWorkerFrame();
        return;
       #endif
        if (! hasGfxFlag || ! gfxCompiledOkFlag || interp == nullptr || interp->hasRuntimeFaulted())
            return;

        double pixelScale = 1.0;
       #if JUCE_MAC
        if (interp->wantsRetina()) pixelScale = displayScale.load (std::memory_order_acquire);
       #endif
        framebufferPixelScale.store (pixelScale, std::memory_order_release);
        const int w = juce::jlimit (1, 8192, (int) std::ceil (targetWidth.load (std::memory_order_acquire) * pixelScale));
        const int h = juce::jlimit (1, 8192, (int) std::ceil (targetHeight.load (std::memory_order_acquire) * pixelScale));

        // Acquire before consuming input/executing @gfx: never drop a resize or
        // UI command just because the presentation buffers are still leased.
        if (! framePool.acquire (workerCanvas, w, h))
            return;
        // Pin before consuming queued input; a contested snapshot must not lose a wheel/button event.
        int snapIdx = -1;
        const auto* snap = processor.beginGfxSnapshotRead (snapIdx);
        if (snap == nullptr)
            return;
        struct ReleaseSnapshot
        {
            JSFXJuceProcessor& owner;
            int index;
            ~ReleaseSnapshot() { owner.endGfxSnapshotRead (index); }
        } releaseSnapshot { processor, snapIdx };

        SharedInputState inputCopy;
        bool hasQueuedMouseFramesRemaining = false;
        {
            const std::lock_guard<std::mutex> lock (inputMutex);

            if (! sharedInput.mouseFrames.empty())
            {
                const auto frame = sharedInput.mouseFrames.front();
                sharedInput.mouseFrames.pop_front();

                inputCopy.mouseX = frame.mouseX;
                inputCopy.mouseY = frame.mouseY;
                inputCopy.mouseCap = frame.mouseCap;
                inputCopy.pendingWheel = frame.pendingWheel;
                inputCopy.pendingHWheel = frame.pendingHWheel;
                inputCopy.captureStateWrites = frame.captureStateWrites;
                if (!frame.droppedFiles.empty())
                    inputCopy.droppedFileBatches.push_back (frame.droppedFiles);
            }
            else
            {
                inputCopy.mouseX = sharedInput.mouseX;
                inputCopy.mouseY = sharedInput.mouseY;
                inputCopy.mouseCap = sharedInput.mouseCap;
                inputCopy.pendingWheel = sharedInput.pendingWheel;
                inputCopy.pendingHWheel = sharedInput.pendingHWheel;
                inputCopy.captureStateWrites = false;
            }

            inputCopy.captureStateWrites = inputCopy.captureStateWrites || sharedInput.captureStateWrites;
            inputCopy.clearKeys = sharedInput.clearKeys;
            inputCopy.keyEvents.swap (sharedInput.keyEvents);

            sharedInput.pendingWheel = 0.0f;
            sharedInput.pendingHWheel = 0.0f;
            sharedInput.captureStateWrites = false;
            sharedInput.clearKeys = false;
            hasQueuedMouseFramesRemaining = ! sharedInput.mouseFrames.empty();
        }

        if (inputCopy.clearKeys)
            interp->clearKeys();

        for (const auto& evt : inputCopy.keyEvents)
        {
            if (evt.enqueueChar)
                interp->pushKey (evt.jsfxCode);
            interp->setKeyDown (evt.jsfxCode, evt.keyDown);
            if (evt.rawKeyCode > 0 && evt.rawKeyCode < 128)
            {
                interp->setKeyDown (evt.rawKeyCode, evt.keyDown);
                if (evt.rawKeyCode >= 'A' && evt.rawKeyCode <= 'Z')
                    interp->setKeyDown (evt.rawKeyCode + ('a' - 'A'), evt.keyDown);
            }
        }

        for (auto& batch : inputCopy.droppedFileBatches)
            interp->addDroppedFiles (std::move (batch));
        interp->setHostWindowState (windowFocused.load (std::memory_order_acquire),
                                   windowVisible.load (std::memory_order_acquire),
                                   displayScale.load (std::memory_order_acquire));

        const auto sameDouble = [] (double a, double b) noexcept
        {
            if (a == b)
                return true;
            if (std::isnan (a) && std::isnan (b))
                return true;
            return std::abs (a - b) <= 1.0e-12;
        };

        if (preserveVmStateUntilNewSnapshot && snap->sequence != preserveVmStateSnapshotSequence)
        {
            preserveVmStateUntilNewSnapshot = false;
            preserveVmStateSnapshotSequence = 0;
        }

        const int currentMouseButtons = inputCopy.mouseCap & kMouseButtonMask;
        const bool mouseButtonEdgeThisFrame = currentMouseButtons != lastRenderMouseButtons;
        lastRenderMouseButtons = currentMouseButtons;

        std::memcpy (effectiveSliders.data(), snap->sliders.data(), sizeof (double) * DSPJSFX_MAX_SLIDERS);

        for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! uiSliderOverrideMask.test (i))
                continue;

            if (sameDouble (snap->sliders[(size_t) i], uiSliderOverrideValues[(size_t) i]))
                uiSliderOverrideMask.reset (i);
            else
                effectiveSliders[(size_t) i] = uiSliderOverrideValues[(size_t) i];
        }

        std::array<jsfx_gfx::MemSpanView, kMaxGfxMemSpans> snapMemSpans {};

        const auto hostTrackNameSnapshot = processor.getHostTrackNameSnapshot();

        jsfx_gfx::Interpreter::Snapshot s;
        s.sliders = effectiveSliders.data();
        s.slidersCount = DSPJSFX_MAX_SLIDERS;
        s.vars = snap->vars.data();
        s.varsCount = snap->varsCount;
        s.logicalMemN = snap->logicalMemN;
        s.srate = snap->srate;
        s.samplesblock = snap->samplesblock;
        s.hostTrackName = hostTrackNameSnapshot.first;
        s.hostTrackNameSeq = hostTrackNameSnapshot.second;
        s.memSpans = nullptr;
        s.memSpanCount = 0;
        s.mem = nullptr;
        s.memN = 0;

        for (int i = 0; i < snap->memSpanCount && s.memSpanCount < (int) snapMemSpans.size(); ++i)
        {
            const auto& srcSpan = snap->memSpans[(size_t) i];
            if (srcSpan.count <= 0 || srcSpan.data.empty())
                continue;

            auto& dstSpan = snapMemSpans[(size_t) s.memSpanCount];
            dstSpan.base = srcSpan.base;
            dstSpan.count = srcSpan.count;
            dstSpan.data = srcSpan.data.data();
            ++s.memSpanCount;
        }

        if (s.memSpanCount > 0)
        {
            s.memSpans = snapMemSpans.data();

            if (snapMemSpans[0].base == 0)
            {
                s.mem = snapMemSpans[0].data;
                s.memN = snapMemSpans[0].count;
            }
        }

        // Preserve VM-owned interaction state only on button-transition frames,
        // and while waiting for the audio snapshot to catch up with UI-authored
        // writes. Holding a button down no longer blocks vars/mem refresh.
        const bool preserveVmStateThisFrame = preserveVmStateUntilNewSnapshot || mouseButtonEdgeThisFrame;
        if (preserveVmStateThisFrame)
        {
            s.vars = nullptr;
            if (! processor.usesExplicitGfxSync() || snap->writableMemRangeCount > 0)
            {
                s.memSpans = nullptr;
                s.mem = nullptr;
                s.memN = 0;
            }
        }

        // Ordinary AUTO mirroring can cover many megabytes, so preserve the
        // historical event-driven diff policy for those static ranges. Native
        // file_mem() ranges are different: REAPER exposes them as genuinely
        // shared mem[] and @gfx code may mutate them without a mouse event.
        // Keep those dynamic ranges under continuous writeback observation.
        std::array<jsfx_gfx::MemRange, 64> persistentFileRanges {};
        const int persistentFileRangeCount = interp->copyPersistentFileMemRanges (
            persistentFileRanges.data(), (int) persistentFileRanges.size());
        const bool captureStaticStateWrites = inputCopy.captureStateWrites
            || (processor.usesExplicitGfxSync() && snap->writableMemRangeCount > 0);
        const bool captureStateWrites = captureStaticStateWrites || persistentFileRangeCount > 0;
        bool wrotePersistentVmState = false;
        const bool profiling = processor.isGfxProfilingEnabled();
        interp->setProfilingEnabled (profiling);
        interp->setMouse (inputCopy.mouseX, inputCopy.mouseY, inputCopy.mouseCap,
                          inputCopy.pendingWheel, inputCopy.pendingHWheel);
        const bool resetCanvas = canvasResetRequested.exchange (false, std::memory_order_acq_rel)
            || lastCanvasWidth != w || lastCanvasHeight != h;
        lastCanvasWidth = w;
        lastCanvasHeight = h;
        bool historyCopied = false;
        if (resetCanvas) workerCanvas.clear (workerCanvas.getBounds(), juce::Colours::black);
        jsfx_gfx::GfxRenderSession renderSession (workerCanvas, [&] (bool discard)
        {
            if (! resetCanvas && ! discard)
            {
                const auto previous = framePool.front();
                historyCopied = jsfx_gfx::copyFramebufferHistory (workerCanvas, previous);
                if (! historyCopied) workerCanvas.clear (workerCanvas.getBounds(), juce::Colours::black);
            }
        });
        auto renderBinding = interp->bindRenderer (renderSession);
        interp->prepareFrame (w, h, s);

        // Read baselines AFTER init/snapshot sync and BEFORE @gfx execution.
        // This covers FROM_GFX-only regions too, without copying read-only data
        // or mistaking a DSP refresh for a GFX-authored write.
        double diffStart = profiling ? juce::Time::getMillisecondCounterHiRes() : 0.0;
        double diffMilliseconds = 0.0;
        memDiffSpanCount = 0;
        if (captureStateWrites)
        {
            if (captureStaticStateWrites)
            {
                varsBefore.resize ((size_t) snap->varsCount);
                if (! varsBefore.empty()) interp->readVars (varsBefore.data(), (int) varsBefore.size());
            }
            else
            {
                varsBefore.clear();
                varsAfter.clear();
            }

            // Static FROM_GFX mirror ranges plus ranges populated by native
            // GFX file_mem(). The latter are dynamic and cannot be known to the
            // AOT metadata pass ahead of time. Merge overlap/adjacency before
            // taking baselines so one large sample buffer remains one diff.
            std::array<jsfx_gfx::MemRange, kMaxGfxMemSpans * 2> diffRanges {};
            int diffRangeCount = 0;
            const auto addDiffRange = [&] (int64_t base, int count)
            {
                if (base < 0 || count <= 0) return;
                int64_t end = base + (int64_t) count;
                if (end <= base) return;
                for (int r = 0; r < diffRangeCount; ++r)
                {
                    auto& existing = diffRanges[(size_t) r];
                    const int64_t existingEnd = existing.base + (int64_t) existing.count;
                    if (end < existing.base || base > existingEnd) continue;
                    const int64_t mergedBase = std::min (base, existing.base);
                    const int64_t mergedEnd = std::max (end, existingEnd);
                    existing.base = mergedBase;
                    existing.count = (int) std::min<int64_t> (mergedEnd - mergedBase,
                                                              std::numeric_limits<int>::max());
                    return;
                }
                if (diffRangeCount < (int) diffRanges.size())
                    diffRanges[(size_t) diffRangeCount++] = { base, count };
            };

            if (captureStaticStateWrites)
            {
                for (int i = 0; i < snap->writableMemRangeCount; ++i)
                {
                    const auto& range = snap->writableMemRanges[(size_t) i];
                    addDiffRange (range.base, range.count);
                }
            }

            for (int i = 0; i < persistentFileRangeCount; ++i)
                addDiffRange (persistentFileRanges[(size_t) i].base,
                              persistentFileRanges[(size_t) i].count);

            for (int i = 0; i < diffRangeCount && memDiffSpanCount < (int) memDiffSpans.size(); ++i)
            {
                const auto& range = diffRanges[(size_t) i];
                if (range.count <= 0) continue;
                auto& diff = memDiffSpans[(size_t) memDiffSpanCount++];
                diff.base = range.base;
                diff.before.resize ((size_t) range.count);
                interp->readMemRange (diff.base, diff.before.data(), range.count);
            }
        }
        else
        {
            varsBefore.clear();
            varsAfter.clear();
        }
        if (profiling) diffMilliseconds = juce::Time::getMillisecondCounterHiRes() - diffStart;
        if (! interp->executeFrame())
        {
            juce::Logger::writeToLog ("[ZA GFX] " + processor.getName()
                                      + ": " + interp->getLastError());
            return;
        }
        if (profiling) diffStart = juce::Time::getMillisecondCounterHiRes();

        // file_mem() is a host-side bulk write into EEL RAM. Publish it even
        // when no mouse/slider interaction requested ordinary state diffing
        // (for example a resource loaded from @init).
        std::array<jsfx_gfx::MemRange, 64> forcedRanges {};
        const int forcedRangeCount = interp->copyForcedDirtyMemRanges (forcedRanges.data(),
                                                                       (int) forcedRanges.size());
        for (int ri = 0; ri < forcedRangeCount; ++ri)
        {
            const auto& range = forcedRanges[(size_t) ri];
            if (range.base < 0 || range.count <= 0) continue;
            forcedMemScratch.resize ((size_t) range.count);
            interp->readMemRange (range.base, forcedMemScratch.data(), range.count);
            if (processor.enqueueGfxMemSpanWrite (range.base, forcedMemScratch.data(), range.count, true))
                wrotePersistentVmState = true;
        }

        if (captureStateWrites)
        {
            if (captureStaticStateWrites)
            {
                varsAfter = varsBefore;
                if (! varsAfter.empty())
                    interp->readVars (varsAfter.data(), (int) varsAfter.size());

                // Variables are few compared with sample RAM; retain the scalar
                // queue for them but remove the old shared 2048-write budget that
                // caused large memory edits to be silently truncated.
                for (int i = 0; i < (int) varsAfter.size(); ++i)
                {
                    if ((getJsfxGfxVarFlags (i) & DSPJSFX_GFX_VAR_FLAG_FROM_GFX) == 0u) continue;
                    const double a = varsBefore[(size_t) i];
                    const double b = varsAfter[(size_t) i];
                    if (a == b || (std::isnan (a) && std::isnan (b)) || ! std::isfinite (b)) continue;
                    wrotePersistentVmState = true;
                    processor.enqueueGfxVarWrite (i, b);
                }
            }

            // Coalesce adjacent changed memory cells into span transactions. A
            // 500k-sample crop/reset/load therefore becomes one/few queue items
            // instead of hundreds of thousands of scalar writes.
            for (int si = 0; si < memDiffSpanCount; ++si)
            {
                auto& diff = memDiffSpans[(size_t) si];
                if ((int) diff.after.size() != (int) diff.before.size())
                    diff.after.resize (diff.before.size());
                if (diff.after.empty()) continue;

                interp->readMemRange (diff.base, diff.after.data(), (int) diff.after.size());
                int runStart = -1;
                const int n = (int) diff.after.size();
                for (int i = 0; i <= n; ++i)
                {
                    bool changed = false;
                    if (i < n)
                    {
                        const double a = diff.before[(size_t) i];
                        const double b = diff.after[(size_t) i];
                        changed = std::isfinite (b) && a != b && !(std::isnan (a) && std::isnan (b));
                    }

                    if (changed && runStart < 0)
                        runStart = i;
                    else if (! changed && runStart >= 0)
                    {
                        const int runCount = i - runStart;
                        if (runCount > 0
                            && processor.enqueueGfxMemSpanWrite (diff.base + runStart,
                                                                 diff.after.data() + runStart,
                                                                 runCount,
                                                                 false))
                            wrotePersistentVmState = true;
                        runStart = -1;
                    }
                }
            }
        }

        if (profiling) diffMilliseconds += juce::Time::getMillisecondCounterHiRes() - diffStart;
        interp->readSliders (vmSliders.data(), DSPJSFX_MAX_SLIDERS);

        const SliderMask mChange  = interp->popSliderChangeMask();
        const SliderMask mAuto    = interp->popSliderAutomateMask();
        const SliderMask mAutoEnd = interp->popSliderAutomateEndMask();
        const SliderMask notifyMask = mChange | mAuto | mAutoEnd;

        if (notifyMask.any() && ! varsAfter.empty())
        {
            processor.syncGfxSliderAliasVarsToSliderValues (vmSliders.data(),
                                                            DSPJSFX_MAX_SLIDERS,
                                                            varsBefore.empty() ? nullptr : varsBefore.data(),
                                                            varsAfter.data(),
                                                            (int) varsAfter.size(),
                                                            effectiveSliders.data(),
                                                            notifyMask);
        }

        SliderMask diffMask;
        for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
        {
            if (! sameDouble (vmSliders[(size_t) i], effectiveSliders[(size_t) i]))
                diffMask.set (i);
        }

        const SliderMask applyMask = diffMask | notifyMask;
        if (applyMask.any())
        {
            for (int i = 0; i < DSPJSFX_MAX_SLIDERS; ++i)
            {
                if (! applyMask.test (i))
                    continue;

                uiSliderOverrideValues[(size_t) i] = vmSliders[(size_t) i];
                uiSliderOverrideMask.set (i);
            }

            wrotePersistentVmState = true;
            queueSliderApply (vmSliders, applyMask, mAuto, mAutoEnd);
        }

        if (wrotePersistentVmState)
        {
            preserveVmStateUntilNewSnapshot = true;
            preserveVmStateSnapshotSequence = snap->sequence;
        }

        const double rasterStart = profiling ? juce::Time::getMillisecondCounterHiRes() : 0.0;
        const auto renderStats = renderSession.finish (interp->getCommands());
        const double rasterMilliseconds = profiling ? juce::Time::getMillisecondCounterHiRes() - rasterStart : 0.0;
        const double publishStart = profiling ? juce::Time::getMillisecondCounterHiRes() : 0.0;
        if (resetCanvas || renderStats.mainFramebufferChanged)
            publishCanvas(); // An offscreen-only/empty frame leaves the displayed canvas intact.
        const double publishMilliseconds = profiling ? juce::Time::getMillisecondCounterHiRes() - publishStart : 0.0;
       #if ZA_SAMPLE_GFX_TEST_RUNNER
        sampleBaselineFrames.fetch_add (1, std::memory_order_release);
       #endif
        if (profiling)
            recordGfxProfile (*snap, renderStats, diffMilliseconds, rasterMilliseconds,
                              publishMilliseconds, historyCopied);


        int cursorId = 0;
        juce::String cursorName;
        if (interp->takeCursorRequest (cursorId, cursorName))
        {
            juce::Image custom;
            if (cursorName.endsWithIgnoreCase (".png")) custom = jsfx_gfx_resources::cursorImage (cursorName);
            {
                const std::lock_guard<std::mutex> lock (cursorMutex);
                pendingCursorId = cursorId;
                pendingCursorName = cursorName;
                pendingCursorImage = custom;
                cursorPending = true;
            }
            triggerAsyncUpdate();
        }
       #if JUCE_MAC
        if (interp->wantsRetina() && pixelScale != displayScale.load (std::memory_order_acquire))
            notifyWorker(); // @init may have opted in on this frame.
       #endif

        if (hasQueuedMouseFramesRemaining)
            notifyWorker();
    }

    void recordGfxProfile (const JSFXJuceProcessor::GfxSnapshot& snap,
                           const jsfx_gfx::GfxRenderStats& render,
                           double diffMs, double rasterMs, double publishMs, bool historyCopied)
    {
        ++profileFrames;
        profileSnapshotMs += snap.copyMilliseconds;
        profileSyncMs += interp->getSyncMilliseconds();
        profileGfxMs += interp->getGfxMilliseconds();
        profileDiffMs += diffMs;
        profileRasterMs += rasterMs;
        profilePublishMs += publishMs;
        profileCommands += render.commandCount;
        profileContexts += render.graphicsContexts;
        profileSurfaceMaps += render.surfaceMaps;
        profileTextMisses += render.textCacheMisses;
        profileReadbacks += render.readbacks;
        profileSelfCopies += render.selfBlitCopies;
        if (historyCopied) ++profileHistoryCopies;
        for (int i = 0; i < snap.memSpanCount; ++i)
            profileSnapshotBytes += (uint64_t) snap.memSpans[(size_t) i].count * sizeof (double);
        for (int i = 0; i < memDiffSpanCount; ++i)
            profileDiffBytes += (uint64_t) memDiffSpans[(size_t) i].before.size() * sizeof (double);
        if (profileFrames < 60) return;
        const double n = (double) profileFrames;
        juce::String line = "[ZA GFX] " + processor.getName()
            + " ms(avg): snapshot=" + juce::String (profileSnapshotMs / n, 3)
            + " sync=" + juce::String (profileSyncMs / n, 3)
            + " eel=" + juce::String (profileGfxMs / n, 3)
            + " diff=" + juce::String (profileDiffMs / n, 3)
            + " raster=" + juce::String (profileRasterMs / n, 3)
            + " publish=" + juce::String (profilePublishMs / n, 3)
            + " | KiB/frame: snapshot=" + juce::String ((double) profileSnapshotBytes / n / 1024.0, 1)
            + " writable-scan=" + juce::String ((double) profileDiffBytes / n / 1024.0, 1)
            + " | commands=" + juce::String ((double) profileCommands / n, 1)
            + " glyph-contexts=" + juce::String ((double) profileContexts / n, 1)
            + " surface-maps=" + juce::String ((double) profileSurfaceMaps / n, 1)
            + " glyph-misses=" + juce::String ((double) profileTextMisses / n, 1)
            + " pixel-reads=" + juce::String ((double) profileReadbacks / n, 1)
            + " history-copies=" + juce::String ((int) profileHistoryCopies)
            + " self-blit-copies=" + juce::String ((int) profileSelfCopies);
        juce::Logger::writeToLog (line);
        profileFrames = profileHistoryCopies = profileSelfCopies = 0;
        profileSnapshotMs = profileSyncMs = profileGfxMs = profileDiffMs = profileRasterMs = profilePublishMs = 0.0;
        profileCommands = profileContexts = profileSnapshotBytes = profileDiffBytes = 0;
        profileSurfaceMaps = profileTextMisses = profileReadbacks = 0;
    }

    static int packCC(const char* s) noexcept
    {
        uint32_t v = 0;
        for (int i = 0; i < 4 && s[i] != 0; ++i)
            v |= (uint32_t) (uint8_t) s[i] << (8u * (uint32_t)i);
        return (int) v;
    }

    static juce::MouseCursor standardGfxCursor (int id, const juce::String& customName)
    {
        const auto name = customName.toLowerCase();
        using C = juce::MouseCursor;
        if (name == "arrow") return C (C::NormalCursor);
        if (name == "hand" || name == "pointing_hand") return C (C::PointingHandCursor);
        if (name == "ibeam" || name == "text") return C (C::IBeamCursor);
        if (name == "cross" || name == "crosshair") return C (C::CrosshairCursor);
        if (name == "wait" || name == "busy") return C (C::WaitCursor);
        if (name == "sizeall" || name == "move") return C (C::UpDownLeftRightResizeCursor);
        if (name == "sizens") return C (C::UpDownResizeCursor);
        if (name == "sizewe") return C (C::LeftRightResizeCursor);
        switch (id)
        {
            case 32513: return C (C::IBeamCursor);
            case 32514: case 32650: return C (C::WaitCursor);
            case 32515: return C (C::CrosshairCursor);
            case 32642: return C (C::TopLeftCornerResizeCursor);
            case 32643: return C (C::TopRightCornerResizeCursor);
            case 32644: return C (C::LeftRightResizeCursor);
            case 32645: return C (C::UpDownResizeCursor);
            case 32646: return C (C::UpDownLeftRightResizeCursor);
            case 32648: return C (C::NormalCursor); // Portable fallback; REAPER-private resources are unavailable.
            case 32649: return C (C::PointingHandCursor);
            default: return C (C::NormalCursor);
        }
    }

    static int juceKeyPressToJsfx(const juce::KeyPress& key){return jsfx_input::keyPressToJsfx(key);}

    static constexpr int kMouseButtonMask = 1 | 2 | 64;

    void enqueueMouseFrameLocked (const MouseStateFrame& frame)
    {
        sharedInput.mouseX = frame.mouseX;
        sharedInput.mouseY = frame.mouseY;
        sharedInput.mouseCap = frame.mouseCap;

        // Continuous input should catch up to the newest state instead of
        // replaying every stale move/drag/wheel event. Only discrete button
        // transitions keep strict FIFO ordering.
        if (frame.preserveOrdering)
        {
            if (! sharedInput.mouseFrames.empty())
            {
                const auto& back = sharedInput.mouseFrames.back();
                const bool staleTail = ! back.preserveOrdering
                                     && ! back.captureStateWrites
                                     && back.pendingWheel == 0.0f
                                     && back.pendingHWheel == 0.0f;
                if (staleTail)
                    sharedInput.mouseFrames.pop_back();
            }

            sharedInput.mouseFrames.push_back (frame);
        }
        else
        {
            if (! sharedInput.mouseFrames.empty())
            {
                auto& back = sharedInput.mouseFrames.back();
                if (! back.preserveOrdering)
                {
                    back.mouseX = frame.mouseX;
                    back.mouseY = frame.mouseY;
                    back.mouseCap = frame.mouseCap;
                    back.pendingWheel += frame.pendingWheel;
                    back.pendingHWheel += frame.pendingHWheel;
                    back.captureStateWrites = back.captureStateWrites || frame.captureStateWrites;
                    return;
                }
            }

            sharedInput.mouseFrames.push_back (frame);
        }

        constexpr size_t kMaxQueuedMouseFrames = 512;
        if (sharedInput.mouseFrames.size() > kMaxQueuedMouseFrames)
            sharedInput.mouseFrames.pop_front();
    }

    void updateMouseCapFromModifiers (const juce::ModifierKeys& mods)
    {
        int cap = 0;

        if (mods.isLeftButtonDown())   cap |= 1;
        if (mods.isRightButtonDown())  cap |= 2;
        if (mods.isMiddleButtonDown()) cap |= 64;

        if (mods.isShiftDown()) cap |= 8;
        if (mods.isAltDown())   cap |= 16;

       #if JUCE_MAC
        if (mods.isCommandDown()) cap |= 4;
        if (mods.isCtrlDown())    cap |= 32;
       #else
        if (mods.isCtrlDown())    cap |= 4;
        if (mods.isCommandDown()) cap |= 32;
       #endif

        mouseCap = cap;
    }

    JSFXJuceProcessor& processor;

    std::atomic<bool> windowFocused { false }, windowVisible { false };
    std::atomic<double> displayScale { 1.0 }, framebufferPixelScale { 1.0 };
    std::mutex cursorMutex;
    bool cursorPending = false;
    int pendingCursorId = 32512;
    juce::String pendingCursorName;
    juce::Image pendingCursorImage;
   #if DSPJSFX_HAS_NATIVE_GFX
    std::shared_ptr<jsfx_native_gfx::Frame> nativeFrame;
#if DSPJSFX_NATIVE_GFX_LEGACY
    std::atomic<bool> legacyReleaseFramePending { false };
    std::atomic<bool> legacyHasExecutedFrame { false };
#endif
    std::atomic<bool> nativeRuntimeFault { false };
    std::array<uint32_t, DSPJSFX_MAX_SLIDERS> nativeUiPreviewSeq {};
    SliderMask nativeTouchMask;
    uint64_t nativeUiRuntimeEpoch = 0;
   #endif
    std::unique_ptr<jsfx_gfx::Interpreter> interp;
    bool hasGfxFlag = false;
    bool gfxCompiledOkFlag = false;
    bool nativeFileDropConsumer = false;
    juce::String gfxLastError;
    int gfxPrefW = 0;
    int gfxPrefH = 0;
    ResizePolicy gfxResizePolicy = ResizePolicy::responsive;

    GfxMenuOverlay menuOverlay;
    MenuBridge menuBridge;

    std::thread workerThread;
    std::mutex workerWaitMutex;
    std::condition_variable workerCv;
    std::atomic<bool> stopWorkerFlag { false };
    std::atomic<bool> workerWakeFlag { false };

    std::mutex inputMutex;
    SharedInputState sharedInput;
    std::unordered_map<uint32_t, int> trackedKeys;
    int mouseCap = 0;

    jsfx_gfx::FramePool framePool;
    int lastCanvasWidth = 0;
    int lastCanvasHeight = 0;
    std::atomic<bool> repaintPending { false };
    std::atomic<int> targetWidth { 1 };
    std::atomic<int> targetHeight { 1 };
    std::atomic<bool> canvasResetRequested { false };

    std::mutex pendingSliderMutex;
    PendingSliderApply pendingSliderApply;

    juce::Image workerCanvas;
    unsigned int profileFrames = 0, profileHistoryCopies = 0, profileSelfCopies = 0;
    double profileSnapshotMs = 0.0, profileSyncMs = 0.0, profileGfxMs = 0.0;
    double profileDiffMs = 0.0, profileRasterMs = 0.0, profilePublishMs = 0.0;
    uint64_t profileCommands = 0, profileContexts = 0, profileSnapshotBytes = 0, profileDiffBytes = 0;
    uint64_t profileSurfaceMaps = 0, profileTextMisses = 0, profileReadbacks = 0;
    std::vector<double> varsBefore;
    std::vector<double> varsAfter;
    std::array<MemDiffSpan, kMaxGfxMemSpans * 2> memDiffSpans {};
    int memDiffSpanCount = 0;
    std::vector<double> forcedMemScratch;

    std::array<double, DSPJSFX_MAX_SLIDERS> vmSliders {};
    std::array<double, DSPJSFX_MAX_SLIDERS> effectiveSliders {};
    std::array<double, DSPJSFX_MAX_SLIDERS> uiSliderOverrideValues {};
    SliderMask uiSliderOverrideMask {};
    bool preserveVmStateUntilNewSnapshot = false;
    uint64_t preserveVmStateSnapshotSequence = 0;
    int lastRenderMouseButtons = 0;
};

// HELP show/hide
    // -----------------------
    void showHelp()
    {
       #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
        hideCorrectnessMonitor();
       #endif

        auto markdown = processor.getEmbeddedReadmeMarkdown().trim();
        if (markdown.isEmpty())
            markdown = za::pluginui::fallbackReadmeMarkdown (processor.getName());

        auto title = za::pluginui::firstMarkdownHeading (markdown).trim();
        if (title.isEmpty())
            title = processor.getName();

        helpOverlay.setHeaderTitle (title);
        helpOverlay.setMarkdownText (markdown);
        helpOverlay.setVisible (true);
        helpOverlay.toFront (true);
        helpOverlay.grabKeyboardFocus();
        resized();
    }

    void hideHelp()
    {
        helpOverlay.setVisible (false);
    }

   #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
    void showCorrectnessMonitor()
    {
        hideHelp();
        correctnessOverlay.refreshNow();
        correctnessOverlay.setVisible (true);
        correctnessOverlay.toFront (true);
        correctnessOverlay.grabKeyboardFocus();
        resized();
    }

    void hideCorrectnessMonitor()
    {
        correctnessOverlay.setVisible (false);
    }
   #endif

    JSFXJuceProcessor& processor;
    FilteredPanel genericEditor;
    juce::Viewport controlsViewport;

    int genericPrefW = 0;
    int genericPrefH = 0;

    GfxView gfxView;
    SmartIdleBadge smartIdleBadge;

    juce::ComboBox oversamplingBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> oversamplingAttach;

    juce::TextButton helpButton;
    za::pluginui::MarkdownHelpOverlay helpOverlay;
    za::fileimport::ImportLandingPad importLandingPad;
   #if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK
    juce::TextButton debugButton;
    CorrectnessOverlay correctnessOverlay { processor };
   #endif
    static constexpr int kTooltipDelayMs = 1000;
    SmartTooltipWindow tooltipWindow;
};


#if DSPJSFX_HAS_NATIVE_GFX && ZA_NATIVE_GFX_TEST_RUNNER
// Test-only access to the same pinned snapshots used by the native worker.
extern "C" bool za_native_gfx_snapshot (juce::AudioProcessor* base, const char* name,
                                       double* value, int* spans, size_t* capacity)
{
    auto& p = *static_cast<JSFXJuceProcessor*> (base);
#if DSPJSFX_NATIVE_GFX_LEGACY
    const std::lock_guard<std::mutex> lock(p.legacyGfxMutex());
    auto& state = p.legacyScriptState();
    const int slot = jsfx_native_gfx::Frame::findIndex(name);
    const int alias = slot >= 0 ? DSPJSFX_LEGACY_SLIDER_ALIASES[slot] : -1;
    *value = slot < 0 ? 0.0 : alias >= 0 ? double(state.sliders[alias]) : double(state.vars[slot]);
    *spans = 0;
    *capacity = 0;
    return slot >= 0;
#else
    int index = -1;
    const auto* snapshot = p.beginGfxSnapshotRead (index);
    if (! snapshot) return false;
    const int slot = jsfx_native_gfx::Frame::findIndex (name);
    *value = slot >= 0 && slot < snapshot->varsCount ? snapshot->vars[(size_t) slot] : 0;
    *spans = snapshot->memSpanCount;
    *capacity = 0;
    for (const auto& span : snapshot->memSpans) *capacity += span.data.capacity();
    p.endGfxSnapshotRead (index);
    return true;
#endif
}
extern "C" void za_smart_idle_mode(juce::AudioProcessor* base,int mode) {
    static_cast<JSFXJuceProcessor*>(base)->setSmartIdleUserOverrideModeForUi(mode);
}
extern "C" bool za_smart_idle_sleeping(juce::AudioProcessor* base) {
    return static_cast<JSFXJuceProcessor*>(base)->isSmartIdleSleepingForTest();
}
extern "C" void za_native_gfx_faults(juce::AudioProcessor* base,uint32_t* dsp,uint32_t* gfx) {
#if DSPJSFX_NATIVE_GFX_LEGACY
    auto& p=*static_cast<JSFXJuceProcessor*>(base);
    const std::lock_guard<std::mutex> lock(p.legacyGfxMutex());
    auto frame=p.legacyGraphics();
    *dsp=p.legacyScriptState().memoryFault;
    *gfx=frame?(frame->state.memoryFault || frame->memoryFault):0;
#else
    *dsp=0;*gfx=0;
#endif
}
extern "C" void za_native_faust_calls(juce::AudioProcessor* base,uint64_t* blocks,uint64_t* scalars,uint64_t* frames) {
    *blocks=0;*scalars=0;*frames=0;
#if DSPJSFX_HAS_FAUST && DSPJSFX_NATIVE_GFX_LEGACY
    auto& p=*static_cast<JSFXJuceProcessor*>(base);
    const std::lock_guard<std::mutex> lock(p.legacyGfxMutex());
    const auto* engine=static_cast<jsfx_faust::Engine*>(p.legacyScriptState().faustContext);
    if(engine){*blocks=engine->blockCalls;*scalars=engine->scalarCalls;*frames=engine->processedFrames;}
#endif
}
extern "C" void za_native_gfx_files(juce::AudioProcessor* base,uint64_t* opens,uint64_t* values) {
#if DSPJSFX_NATIVE_GFX_LEGACY
    auto& p=*static_cast<JSFXJuceProcessor*>(base);
    const std::lock_guard<std::mutex> lock(p.legacyGfxMutex());
    auto frame=p.legacyGraphics();
    *opens=frame?frame->testFileOpens:0;*values=frame?frame->testFileValues:0;
#else
    *opens=0;*values=0;
#endif
}
extern "C" uint64_t za_native_gfx_frames()
{
    return jsfx_native_gfx::publishedFrames.load (std::memory_order_acquire);
}
extern "C" int za_native_gfx_heap_copy(juce::AudioProcessor* base,const char* name,double* out,int count)
{
#if DSPJSFX_NATIVE_GFX_LEGACY
    auto& p=*static_cast<JSFXJuceProcessor*>(base);
    const std::lock_guard<std::mutex> lock(p.legacyGfxMutex());
    auto& state=p.legacyScriptState();
    const int slot=jsfx_native_gfx::Frame::findIndex(name);
    const int64_t address=slot<0 ? -1 : int64_t(double(state.vars[slot]));
    if(address<0 || count<0 || address+count>state.memN) return 0;
    for(int i=0;i<count;++i) out[i]=double(state.mem[address+i]);
    return count;
#else
    juce::ignoreUnused(base,name,out,count);return 0;
#endif
}
extern "C" double za_native_gfx_heap_value(juce::AudioProcessor* base, const char* name, int offset)
{
#if DSPJSFX_NATIVE_GFX_LEGACY
    auto& p = *static_cast<JSFXJuceProcessor*>(base);
    const std::lock_guard<std::mutex> lock(p.legacyGfxMutex());
    auto& state = p.legacyScriptState();
    const int slot = jsfx_native_gfx::Frame::findIndex(name);
    const int64_t address = slot < 0 ? -1 : int64_t(double(state.vars[slot])) + offset;
    return address >= 0 && address < state.memN ? double(state.mem[address]) : 0;
#else
    juce::ignoreUnused(base, name, offset); return 0;
#endif
}
#endif

#if ZA_SAMPLE_GFX_TEST_RUNNER
// Test-only baseline access. Read pinned snapshots, never the live DSP heap.
extern "C" bool za_sample_gfx_snapshot (juce::AudioProcessor* base, const char* name,
                                       double* value, size_t* copiedCells, size_t* capacity,
                                       int64_t* logicalCells)
{
    auto& p = *static_cast<JSFXJuceProcessor*> (base);
#if DSPJSFX_NATIVE_GFX_LEGACY
    const std::lock_guard<std::mutex> lock(p.legacyGfxMutex());
    auto& state = p.legacyScriptState();
    const int slot = jsfx_native_gfx::Frame::findIndex(name);
    const int alias = slot >= 0 ? DSPJSFX_LEGACY_SLIDER_ALIASES[slot] : -1;
    *value = slot < 0 ? 0.0 : alias >= 0 ? double(state.sliders[alias]) : double(state.vars[slot]);
    *copiedCells = *capacity = 0;
    *logicalCells = state.memN;
    return slot >= 0;
#else
    int index = -1;
    const auto* snapshot = p.beginGfxSnapshotRead (index);
    if (! snapshot) return false;
    int slot = -1;
    for (const auto& var : DSPJSFX_VARS)
        if (std::strcmp (var.name, name) == 0) { slot = var.index; break; }
    *value = slot >= 0 && slot < snapshot->varsCount ? snapshot->vars[(size_t) slot] : 0;
    *copiedCells = *capacity = 0;
    for (const auto& span : snapshot->memSpans)
    {
        *copiedCells += (size_t) span.count;
        *capacity += span.data.capacity();
    }
   #if DSPJSFX_HAS_NATIVE_GFX
    *copiedCells = snapshot->nativePublication.copiedCells;
    *capacity = snapshot->nativePublication.capacity();
   #endif
    *logicalCells = snapshot->logicalMemN;
    p.endGfxSnapshotRead (index);
    return slot >= 0;
#endif
}
extern "C" uint64_t za_sample_gfx_frames()
{
   #if DSPJSFX_HAS_NATIVE_GFX
    return jsfx_native_gfx::publishedFrames.load (std::memory_order_acquire);
   #else
    return sampleBaselineFrames.load (std::memory_order_acquire);
   #endif
}
extern "C" void za_sample_gfx_copy_stats (juce::AudioProcessor* base, uint64_t* count, double* total, double* maximum)
{
    static_cast<JSFXJuceProcessor*> (base)->sampleCopyStats (*count, *total, *maximum);
}
extern "C" bool za_sample_gfx_native() { return DSPJSFX_HAS_NATIVE_GFX != 0; }
extern "C" bool za_sample_gfx_legacy() { return DSPJSFX_NATIVE_GFX_LEGACY != 0; }
extern "C" double za_sample_gfx_ui (juce::AudioProcessor* base, const char* name)
{
   #if DSPJSFX_HAS_NATIVE_GFX
    return static_cast<JSFXJuceProcessor*> (base)->nativeGfxUiValue (name);
   #else
    juce::ignoreUnused (base, name); return 0;
   #endif
}
extern "C" bool za_sample_gfx_faulted (juce::AudioProcessor* base)
{
   #if DSPJSFX_HAS_NATIVE_GFX
    return static_cast<JSFXJuceProcessor*> (base)->nativeGfxFaulted();
   #else
    juce::ignoreUnused (base); return false;
   #endif
}
extern "C" void za_sample_gfx_load_bank (juce::AudioProcessor* base, const char* const* paths, int count)
{
    std::vector<juce::File> files;
    for (int i = 0; i < count; ++i) files.emplace_back (juce::String::fromUTF8 (paths[i]));
    static_cast<JSFXJuceProcessor*> (base)->setFileSlotPaths (0, files);
}
#endif

juce::AudioProcessorEditor* JSFXJuceProcessor::createEditor()
{
    return new JSFXJuceEditor (*this);
}



// -----------------------------------------------------------------------------
// DSP-JSFX runtime: file_*() entry points
// -----------------------------------------------------------------------------
//
// These are called by the AOT-compiled DSP-JSFX module. They dispatch to the
// owning JSFXJuceProcessor instance via its lifetime-bound state context.
// -----------------------------------------------------------------------------

namespace
{
static inline JSFXJuceProcessor* ownerFromState (DSPJSFX_State* state) noexcept
{
    if (state == nullptr)
        return nullptr;

    return static_cast<JSFXJuceProcessor*> (state->hostOwner);
}
} // namespace

#if DSPJSFX_NATIVE_GFX_LEGACY
extern "C" double jsfx_file_string(DSPJSFX_State* state,double handle,double dest)
{
    double* args[]={&handle,&dest};
    return jsfx_native_gfx_dispatch(state,DSPJSFX_FILE_STRING,args,2);
}
#endif

#define JSFX_FILE_RUNTIME_OWNER(state) ownerFromState(state)
#include "JsfxFileBuiltins.h"
#undef JSFX_FILE_RUNTIME_OWNER

// -----------------------------------------------------------------------------
// DSP-JSFX runtime: sample_pool_*() entry points
// -----------------------------------------------------------------------------

#define JSFX_SAMPLE_POOL_OWNER(state) ownerFromState(state)
#include "JsfxSamplePoolBuiltins.h"
#undef JSFX_SAMPLE_POOL_OWNER

#if ZA_NATIVE_GFX_TEST_RUNNER
static juce::AudioProcessor* za_latest_test_processor=nullptr;
extern "C" juce::AudioProcessor* za_test_latest_processor(){return za_latest_test_processor;}
extern "C" void za_test_gfx_button(juce::AudioProcessor* p,int slider,double value){static_cast<JSFXJuceProcessor*>(p)->testGfxButtonChange(slider-1,value);}
#endif
namespace za::jsfx {
static void validateProductionExports() {
#define JSFX_RUNTIME_EXPORT(name) static_assert(std::is_pointer_v<decltype(&name)>);
#include "JsfxRuntimeExports.inc"
#undef JSFX_RUNTIME_EXPORT
}
}
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    auto* processor = new JSFXJuceProcessor();
#if ZA_NATIVE_GFX_TEST_RUNNER
    za_latest_test_processor=processor;
#endif
    return processor;
}
