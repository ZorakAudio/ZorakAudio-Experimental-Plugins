#pragma once

#if defined(ZA_JSFX_CORRECTNESS_CHECK) && ZA_JSFX_CORRECTNESS_CHECK

#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

class JSFXJuceProcessor;

extern "C" double jsfx_file_open (DSPJSFX_State* state, double indexOrSlot, double mode);
extern "C" double jsfx_file_close (DSPJSFX_State* state, double handle);
extern "C" double jsfx_file_rewind (DSPJSFX_State* state, double handle);
extern "C" double jsfx_file_seek (DSPJSFX_State* state, double handle, double offset);
extern "C" double jsfx_file_avail (DSPJSFX_State* state, double handle);
extern "C" double jsfx_file_text (DSPJSFX_State* state, double handle);
extern "C" double jsfx_file_riff (DSPJSFX_State* state, double handle, double* outNch, double* outSr);
extern "C" double jsfx_file_var (DSPJSFX_State* state, double handle, double* outVar);
extern "C" double jsfx_file_mem (DSPJSFX_State* state, double handle, double destIndex, double length);

namespace jsfx_correctness
{
static constexpr double kAudioCompareEpsilon = 1.0e-5;
static constexpr double kScalarCompareEpsilon = 1.0e-8;
static constexpr int kRingChannels = 2;
static constexpr double kRingSeconds = 30.0;
static constexpr int kMemPageDoubles = 1024;

static inline bool nearlyEqual (double a, double b, double eps) noexcept
{
    if (std::isnan (a) != std::isnan (b))
        return false;
    if (std::isnan (a) && std::isnan (b))
        return true;
    if (std::isinf (a) || std::isinf (b))
        return a == b;
    return std::abs (a - b) <= eps;
}

static inline juce::String sanitiseFileComponent (juce::String s)
{
    s = s.trim();
    if (s.isEmpty())
        return "jsfx";

    static constexpr const char* invalid = "\\/:*?\"<>|";
    for (const char* p = invalid; *p != 0; ++p)
        s = s.replaceCharacter ((juce::juce_wchar) *p, '_');

    s = s.retainCharacters ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ .()");
    s = s.trim();
    return s.isEmpty() ? juce::String ("jsfx") : s;
}

static inline juce::String jsonEscape (const juce::String& in)
{
    juce::String out;
    out.preallocateBytes (juce::jmax<size_t> ((size_t) 32, (size_t) in.getNumBytesAsUTF8() + (size_t) 16));

    for (int i = 0; i < in.length(); ++i)
    {
        const juce::juce_wchar c = in[i];
        switch (c)
        {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if ((uint32_t) c < 0x20u)
                    out << "?";
                else
                    out << juce::String::charToString (c);
                break;
        }
    }

    return out;
}

enum class MonitorAudioMode : int
{
    Compiled = 0,
    Shadow = 1,
    Delta = 2,
};

static inline const char* monitorModeName (MonitorAudioMode mode) noexcept
{
    switch (mode)
    {
        case MonitorAudioMode::Shadow:   return "Shadow EEL2";
        case MonitorAudioMode::Delta:    return "Delta";
        case MonitorAudioMode::Compiled:
        default:                         return "Compiled DSP-JSFX";
    }
}

class ShadowVm final : public jsfx_gfx::GfxVm
{
public:
    struct PendingSliderMasks
    {
        uint64_t change = 0;
        uint64_t automate = 0;
        uint64_t automateEnd = 0;
    };

    explicit ShadowVm (JSFXJuceProcessor* ownerIn)
        : owner (ownerIn)
    {
        static std::once_flag s_once;
        std::call_once (s_once, []()
        {
            NSEEL_addfunc_varparm_ex ("dsp_spl",   1, 0, NSEEL_PProc_THIS, &ShadowVm::eel_dsp_spl,   nullptr);
            NSEEL_addfunc_varparm_ex ("midirecv",  4, 1, NSEEL_PProc_THIS, &ShadowVm::eel_midirecv,  nullptr);
            NSEEL_addfunc_varparm_ex ("midisend",  4, 1, NSEEL_PProc_THIS, &ShadowVm::eel_midisend,  nullptr);
            NSEEL_addfunc_varparm_ex ("file_open",   1, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_open,   nullptr);
            NSEEL_addfunc_varparm_ex ("file_close",  1, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_close,  nullptr);
            NSEEL_addfunc_varparm_ex ("file_rewind", 1, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_rewind, nullptr);
            NSEEL_addfunc_varparm_ex ("file_seek",   2, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_seek,   nullptr);
            NSEEL_addfunc_varparm_ex ("file_avail",  1, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_avail,  nullptr);
            NSEEL_addfunc_varparm_ex ("file_text",   1, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_text,   nullptr);
            NSEEL_addfunc_varparm_ex ("file_riff",   3, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_riff,   nullptr);
            NSEEL_addfunc_varparm_ex ("file_var",    2, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_var,    nullptr);
            NSEEL_addfunc_varparm_ex ("file_mem",    3, 1, NSEEL_PProc_THIS, &ShadowVm::eel_file_mem,    nullptr);
            NSEEL_addfunc_varparm_ex ("memset",      3, 1, NSEEL_PProc_THIS, &ShadowVm::eel_memset,      nullptr);
        });

        bindSliderPtrs();
        bindUserVars (DSPJSFX_VARS, (int) DSPJSFX_VARS_COUNT);
        bindSplPtrs();
        bindSliderAliasPtrs();

        if (m_vm != nullptr)
            NSEEL_VM_SetWriteTrace (m_vm, &ShadowVm::writeTraceThunk, this);

        std::memset (&bridgeState, 0, sizeof (bridgeState));
        bridgeState.srate = 44100.0;
        bridgeState.currentSampleRate = 44100.0;

        gFileOwner[&bridgeState] = owner;

        sections = jsfx_gfx::extractJsfxSections (kJsfxSourceText);
        compileSections();
    }

    ~ShadowVm() override
    {
        gFileOwner.erase (&bridgeState);
        gMemOwner.erase (&bridgeState);
        gMemSize.erase (&bridgeState);
        gMemUsed.erase (&bridgeState);

        if (bridgeState.mem != nullptr)
        {
            std::free (bridgeState.mem);
            bridgeState.mem = nullptr;
            bridgeState.memN = 0;
        }
    }

    bool isReady() const noexcept { return ready; }
    bool freembufIsNoop() const noexcept override { return true; }
    const juce::String& getLastError() const noexcept { return lastError; }

    void ensureRamSize (int64_t needed)
    {
        if (m_vm == nullptr || needed <= 0)
            return;

        needed = juce::jlimit<int64_t> ((int64_t) 0,
                                        (int64_t) std::numeric_limits<int>::max(),
                                        needed);
        if (needed <= (int64_t) memSize)
            return;

        NSEEL_VM_setramsize (m_vm, (unsigned int) needed);
        memSize = (int) needed;
    }

    void syncHostSlidersAndAliases (const double* sliders, int count)
    {
        syncSliders (sliders, count);
        syncAliasVarsFromCurrentSliders();
    }

    void syncAliasVarsFromCurrentSliders()
    {
        for (size_t i = 0; i < sliderAliasPtrs.size(); ++i)
        {
            auto* alias = sliderAliasPtrs[i];
            auto* slider = sliderPtrs[i];
            if (alias != nullptr && slider != nullptr)
                *alias = *slider;
        }
    }

    void applyExternalVarWrite (int varIndex, double value)
    {
        for (const auto& bv : boundVars)
        {
            if (bv.index == varIndex && bv.ptr != nullptr)
            {
                *bv.ptr = (EEL_F) value;
                return;
            }
        }
    }

    void applyExternalMemWrite (int memIndex, double value)
    {
        if (memIndex < 0)
            return;

        ensureRamSize ((int64_t) memIndex + 1);
        int validCount = 0;
        if (auto* ptr = NSEEL_VM_getramptr (m_vm, (unsigned int) memIndex, &validCount))
        {
            if (validCount > 0)
                ptr[0] = (EEL_F) value;
        }

        highestTouchedMem = std::max<int64_t> (highestTouchedMem, (int64_t) memIndex + 1);
    }

    void beginBlock (const DSPJSFX_State& aotState, int numSamples)
    {
        bridgeState.srate = aotState.srate;
        bridgeState.currentSampleRate = aotState.currentSampleRate;
        bridgeState.currentBlockSize = numSamples;
        bridgeState.samplesblock = (double) numSamples;

        setTiming (aotState.srate, (double) numSamples);

        midiInQueue.clear();
        midiOutQueue.clear();
        midiInReadIndex = 0;
        if (aotState.midiIn != nullptr && aotState.midiInCount > 0)
            midiInQueue.assign (aotState.midiIn, aotState.midiIn + aotState.midiInCount);

        (void) popSliderChangeMask();
        (void) popSliderAutomateMask();
        (void) popSliderAutomateEndMask();
        (void) popUndoPointRequested();

        ensureRamSize (juce::jmax<int64_t> ((int64_t) 65536, aotState.memN));
    }

    void runInit()      { execute (codeInit); }
    void runSlider()    { execute (codeSlider); }
    void runBlock()     { execute (codeBlock); }
    void runSample()    { execute (codeSample); }
    void runSerialize() { execute (codeSerialize); }

    void setInputSample (const float* const* inputs, int numCh, int sampleIndex)
    {
        const int n = juce::jlimit (0, 64, numCh);
        for (int ch = 0; ch < n; ++ch)
        {
            auto* ptr = splPtrs[(size_t) ch];
            if (ptr != nullptr)
                *ptr = (inputs != nullptr && inputs[ch] != nullptr) ? (EEL_F) inputs[ch][sampleIndex] : 0.0;
        }
    }

    void readOutputSample (double* dst, int numCh) const
    {
        if (dst == nullptr)
            return;

        const int n = juce::jlimit (0, 64, numCh);
        for (int ch = 0; ch < n; ++ch)
            dst[ch] = getOutputValue (ch);
    }

    double getOutputValue (int ch) const
    {
        if (ch < 0 || ch >= 64)
            return 0.0;

        auto* ptr = splPtrs[(size_t) ch];
        return ptr != nullptr ? (double) *ptr : 0.0;
    }

    const std::vector<DSPJSFX_MidiEvent>& getMidiOut() const noexcept { return midiOutQueue; }

    PendingSliderMasks peekPendingSliderMasks() const noexcept
    {
        PendingSliderMasks masks;
        masks.change = sliderChangeMask;
        masks.automate = sliderAutomateMask;
        masks.automateEnd = sliderAutomateEndMask;
        return masks;
    }

    PendingSliderMasks consumePendingSliderMasks()
    {
        const PendingSliderMasks masks = peekPendingSliderMasks();
        (void) popSliderChangeMask();
        (void) popSliderAutomateMask();
        (void) popSliderAutomateEndMask();
        return masks;
    }

    int64_t logicalMemUsedForCompare() const noexcept
    {
        return std::max<int64_t> ((int64_t) 0, highestTouchedMem);
    }

private:
    void execute (NSEEL_CODEHANDLE h)
    {
        if (h != nullptr)
            NSEEL_code_execute (h);
    }

    void bindSplPtrs()
    {
        for (int i = 0; i < 64; ++i)
        {
            const std::string name = std::string ("spl") + std::to_string (i);
            splPtrs[(size_t) i] = get_var (name.c_str());
            if (splPtrs[(size_t) i] != nullptr)
                *splPtrs[(size_t) i] = 0.0;
        }
    }

    void bindSliderAliasPtrs()
    {
        const auto decls = parseJsfxSliderDecls (kJsfxSourceText, nullptr);
        sliderAliasPtrs.fill (nullptr);

        for (const auto& decl : decls)
        {
            if (decl.index0 < 0 || decl.index0 >= 64 || decl.varName.isEmpty())
                continue;

            sliderAliasPtrs[(size_t) decl.index0] = get_var (decl.varName.toRawUTF8());
        }
    }

    static std::string renameFunctionCalls (const std::string& in, const char* from, const char* to)
    {
        std::string out;
        out.reserve (in.size() + 64);

        const size_t fromLen = std::strlen (from);
        bool inLineComment = false;
        bool inBlockComment = false;
        bool inString = false;
        char quote = 0;

        for (size_t i = 0; i < in.size();)
        {
            const char c = in[i];
            const char next = (i + 1 < in.size() ? in[i + 1] : '\0');

            if (inLineComment)
            {
                out.push_back (c);
                if (c == '\n')
                    inLineComment = false;
                ++i;
                continue;
            }

            if (inBlockComment)
            {
                out.push_back (c);
                if (c == '*' && next == '/')
                {
                    out.push_back (next);
                    i += 2;
                    inBlockComment = false;
                }
                else
                {
                    ++i;
                }
                continue;
            }

            if (inString)
            {
                out.push_back (c);
                if (c == '\\' && i + 1 < in.size())
                {
                    out.push_back (in[i + 1]);
                    i += 2;
                    continue;
                }
                if (c == quote)
                    inString = false;
                ++i;
                continue;
            }

            if (c == '/' && next == '/')
            {
                out.push_back (c);
                out.push_back (next);
                i += 2;
                inLineComment = true;
                continue;
            }

            if (c == '/' && next == '*')
            {
                out.push_back (c);
                out.push_back (next);
                i += 2;
                inBlockComment = true;
                continue;
            }

            if (c == '\'' || c == '"')
            {
                out.push_back (c);
                quote = c;
                inString = true;
                ++i;
                continue;
            }

            if (i + fromLen < in.size() && in.compare (i, fromLen, from) == 0)
            {
                const char prev = (i > 0 ? in[i - 1] : '\0');
                const char after = (i + fromLen < in.size() ? in[i + fromLen] : '\0');
                if (! jsfx_gfx::isIdentChar (prev) && after == '(')
                {
                    out.append (to);
                    i += fromLen;
                    continue;
                }
            }

            out.push_back (c);
            ++i;
        }

        return out;
    }

    std::string preprocessSection (const std::string& code) const
    {
        auto out = jsfx_gfx::preprocessJsfxForPortableEel (code.empty() ? std::string ("0;") : code);
        out = renameFunctionCalls (out, "spl", "dsp_spl");
        return out;
    }

    void compileSections()
    {
        const char* err = nullptr;

        const auto compileOne = [this, &err] (const std::string& code, juce::String& errOut) -> NSEEL_CODEHANDLE
        {
            err = nullptr;
            const std::string pre = preprocessSection (code);
            NSEEL_CODEHANDLE h = compile_code (pre.empty() ? "0;" : pre.c_str(), &err);
            if (h == nullptr)
                errOut = (err != nullptr ? err : "Unknown EEL compile error");
            return h;
        };

        juce::String errText;
        codeInit = compileOne (sections.init, errText);
        if (errText.isNotEmpty()) { lastError = "@init: " + errText; return; }

        codeSlider = compileOne (sections.slider.empty() ? std::string ("0;") : sections.slider, errText);
        if (errText.isNotEmpty()) { lastError = "@slider: " + errText; return; }

        codeBlock = compileOne (sections.block.empty() ? std::string ("0;") : sections.block, errText);
        if (errText.isNotEmpty()) { lastError = "@block: " + errText; return; }

        codeSample = compileOne (sections.sample.empty() ? std::string ("0;") : sections.sample, errText);
        if (errText.isNotEmpty()) { lastError = "@sample: " + errText; return; }

        codeSerialize = compileOne (sections.serialize.empty() ? std::string ("0;") : sections.serialize, errText);
        if (errText.isNotEmpty()) { lastError = "@serialize: " + errText; return; }

        ready = true;
    }

    static void writeTraceThunk (void* userctx, EEL_F* addr, unsigned int count)
    {
        auto* self = static_cast<ShadowVm*> (userctx);
        if (self == nullptr || self->m_vm == nullptr || addr == nullptr || count == 0)
            return;

        unsigned int index = 0;
        if (NSEEL_VM_GetRAMIndexForPtr (self->m_vm, addr, &index, nullptr))
            self->highestTouchedMem = std::max<int64_t> (self->highestTouchedMem, (int64_t) index + (int64_t) count);
    }

    static EEL_F NSEEL_CGEN_CALL eel_dsp_spl (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 1)
            return 0.0;

        const int idx = (int) jsfx_gfx::jsfxTruncIndexLikeAot ((double) *parms[0]);
        if (idx < 0 || idx >= 64)
            return (np >= 2) ? *parms[1] : 0.0;

        auto* ptr = self->splPtrs[(size_t) idx];
        if (ptr == nullptr)
            return (np >= 2) ? *parms[1] : 0.0;

        if (np >= 2)
        {
            *ptr = *parms[1];
            return *ptr;
        }

        return *ptr;
    }

    static EEL_F NSEEL_CGEN_CALL eel_midirecv (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 4)
            return 0.0;

        if (self->midiInReadIndex < 0 || self->midiInReadIndex >= (int) self->midiInQueue.size())
            return 0.0;

        const auto& ev = self->midiInQueue[(size_t) self->midiInReadIndex++];
        *parms[0] = (EEL_F) ev.sampleOffset;
        *parms[1] = (EEL_F) ev.msg1;
        *parms[2] = (EEL_F) ev.msg2;
        *parms[3] = (EEL_F) ev.msg3;
        return 1.0;
    }

    static EEL_F NSEEL_CGEN_CALL eel_midisend (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 4)
            return 0.0;

        DSPJSFX_MidiEvent ev {};
        ev.sampleOffset = jsfxClampMidiOffset (&self->bridgeState, (double) *parms[0]);
        ev.msg1 = jsfxClampMidiByte ((double) *parms[1]);
        ev.msg2 = jsfxClampMidiByte ((double) *parms[2]);
        ev.msg3 = jsfxClampMidiByte ((double) *parms[3]);
        self->midiOutQueue.push_back (ev);
        return 1.0;
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_open (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 1)
            return -1.0;

        const double indexOrSlot = (double) *parms[0];
        const double mode = (np >= 2) ? (double) *parms[1] : 0.0;
        return (EEL_F) jsfx_file_open (&self->bridgeState, indexOrSlot, mode);
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_close (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 1)
            return 0.0;
        return (EEL_F) jsfx_file_close (&self->bridgeState, (double) *parms[0]);
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_rewind (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 1)
            return 0.0;
        return (EEL_F) jsfx_file_rewind (&self->bridgeState, (double) *parms[0]);
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_seek (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 2)
            return 0.0;
        return (EEL_F) jsfx_file_seek (&self->bridgeState, (double) *parms[0], (double) *parms[1]);
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_avail (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 1)
            return 0.0;
        return (EEL_F) jsfx_file_avail (&self->bridgeState, (double) *parms[0]);
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_text (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 1)
            return 0.0;
        return (EEL_F) jsfx_file_text (&self->bridgeState, (double) *parms[0]);
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_riff (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 3)
            return 0.0;

        double nch = 0.0;
        double sr = 0.0;
        const double ret = jsfx_file_riff (&self->bridgeState, (double) *parms[0], &nch, &sr);
        *parms[1] = (EEL_F) nch;
        *parms[2] = (EEL_F) sr;
        return (EEL_F) ret;
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_var (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 2)
            return 0.0;

        double outVar = 0.0;
        const double ret = jsfx_file_var (&self->bridgeState, (double) *parms[0], &outVar);
        *parms[1] = (EEL_F) outVar;
        return (EEL_F) ret;
    }

    static EEL_F NSEEL_CGEN_CALL eel_file_mem (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 3)
            return 0.0;

        const double handle = (double) *parms[0];
        const double destIndex = (double) *parms[1];
        const double length = (double) *parms[2];
        const double ret = jsfx_file_mem (&self->bridgeState, handle, destIndex, length);

        const int64_t dst = std::max<int64_t> ((int64_t) 0, (int64_t) (destIndex + 1.0e-5));
        const int copied = (int) std::max<int64_t> ((int64_t) 0, (int64_t) std::llround (ret));
        if (copied > 0 && self->bridgeState.mem != nullptr && dst < self->bridgeState.memN)
        {
            self->ensureRamSize (self->bridgeState.memN);
            self->syncMemRange (self->bridgeState.mem + dst, dst, copied);
            self->highestTouchedMem = std::max<int64_t> (self->highestTouchedMem, dst + (int64_t) copied);
        }

        return (EEL_F) ret;
    }

    static EEL_F NSEEL_CGEN_CALL eel_memset (void* opaque, INT_PTR np, EEL_F** parms)
    {
        auto* self = static_cast<ShadowVm*> (opaque);
        if (self == nullptr || np < 3)
            return 0.0;

        int64_t dst = (int64_t) ((double) *parms[0] + 1.0e-5);
        int64_t len = (int64_t) ((double) *parms[2] + 1.0e-5);
        if (dst < 0) dst = 0;
        if (len <= 0) return (EEL_F) dst;

        self->ensureRamSize (dst + len);
        const EEL_F value = *parms[1];

        int64_t pos64 = dst;
        int64_t remaining = len;
        while (remaining > 0)
        {
            if (pos64 > (int64_t) std::numeric_limits<unsigned int>::max())
                break;

            int validCount = 0;
            EEL_F* ptr = NSEEL_VM_getramptr (self->m_vm, (unsigned int) pos64, &validCount);
            if (ptr == nullptr || validCount <= 0)
                break;

            const int n = (int) std::min<int64_t> ((int64_t) validCount, remaining);
            for (int i = 0; i < n; ++i)
                ptr[i] = value;

            pos64 += (int64_t) n;
            remaining -= (int64_t) n;
        }

        self->highestTouchedMem = std::max<int64_t> (self->highestTouchedMem, dst + len);
        return (EEL_F) dst;
    }

    JSFXJuceProcessor* owner = nullptr;
    jsfx_gfx::JsfxSections sections;
    NSEEL_CODEHANDLE codeInit = nullptr;
    NSEEL_CODEHANDLE codeSlider = nullptr;
    NSEEL_CODEHANDLE codeBlock = nullptr;
    NSEEL_CODEHANDLE codeSample = nullptr;
    NSEEL_CODEHANDLE codeSerialize = nullptr;
    bool ready = false;
    juce::String lastError;

    std::array<EEL_F*, 64> splPtrs {};
    std::array<EEL_F*, 64> sliderAliasPtrs {};

    std::vector<DSPJSFX_MidiEvent> midiInQueue;
    std::vector<DSPJSFX_MidiEvent> midiOutQueue;
    int midiInReadIndex = 0;

    DSPJSFX_State bridgeState {};
    int64_t highestTouchedMem = 0;
};

class Runtime final
{
public:
    explicit Runtime (JSFXJuceProcessor* ownerIn)
        : owner (ownerIn)
    {
        initVarFirstWriteStages();
        resizeRingForSampleRate (44100.0);
    }

    void resetAndPrime (const DSPJSFX_State& aotState)
    {
        clear();
        shadow = std::make_unique<ShadowVm> (owner);

        if (shadow == nullptr || ! shadow->isReady())
        {
            std::lock_guard<std::mutex> lk (detailMutex);
            compileError = (shadow != nullptr ? shadow->getLastError() : juce::String ("Failed to create shadow VM"));
            return;
        }

        shadow->ensureRamSize (juce::jmax<int64_t> ((int64_t) 65536, aotState.memN));
        shadow->syncHostSlidersAndAliases (aotState.sliders, 64);
        shadow->setTiming (aotState.srate, 0.0);
        shadow->runInit();
        shadow->syncAliasVarsFromCurrentSliders();
        shadow->runSlider();
    }

    bool isReady() const noexcept
    {
        return shadow != nullptr && shadow->isReady();
    }

    ShadowVm* getShadow() noexcept { return shadow.get(); }
    const ShadowVm* getShadow() const noexcept { return shadow.get(); }

    MonitorAudioMode getMonitorMode() const noexcept
    {
        return (MonitorAudioMode) monitorMode.load (std::memory_order_acquire);
    }

    void setMonitorMode (MonitorAudioMode mode) noexcept
    {
        monitorMode.store ((int) mode, std::memory_order_release);
    }

    bool getFreezeOnFirstMismatch() const noexcept
    {
        return freezeOnFirstMismatch.load (std::memory_order_acquire);
    }

    void setFreezeOnFirstMismatch (bool shouldFreeze) noexcept
    {
        freezeOnFirstMismatch.store (shouldFreeze, std::memory_order_release);
    }

    void clear()
    {
        frozen.store (false, std::memory_order_release);
        mismatchLatched.store (false, std::memory_order_release);
        blocksCompared.store (0, std::memory_order_release);
        scalarComparisons.store (0, std::memory_order_release);
        scalarMismatches.store (0, std::memory_order_release);
        sumSquaredDelta.store (0.0, std::memory_order_release);
        maxAbsDelta.store (0.0, std::memory_order_release);
        totalRingFrames.store (0, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lk (detailMutex);
            compileError.clear();
            firstMismatchStage.clear();
            firstMismatchDetail.clear();
            firstMismatchPages.clear();
            lastExportPath.clear();
            firstMismatchBlock = -1;
            firstMismatchSample = -1;
            firstMismatchChannel = -1;
            firstMismatchCompiled = 0.0;
            firstMismatchShadow = 0.0;
            firstMismatchAbsDelta = 0.0;
        }

        resizeRingForSampleRate (ringSampleRate.load (std::memory_order_acquire));
    }

    void setRingSampleRate (double sr)
    {
        if (sr <= 0.0)
            return;
        ringSampleRate.store (sr, std::memory_order_release);
        resizeRingForSampleRate (sr);
    }

    void applyExternalVarWrite (int varIndex, double value) noexcept
    {
        if (shadow != nullptr && shadow->isReady())
            shadow->applyExternalVarWrite (varIndex, value);
    }

    void applyExternalMemWrite (int memIndex, double value) noexcept
    {
        if (shadow != nullptr && shadow->isReady())
            shadow->applyExternalMemWrite (memIndex, value);
    }

    int getNextBlockIndex() const noexcept
    {
        return (int) blocksCompared.load (std::memory_order_acquire);
    }

    void observeAudioFrame (const DSPJSFX_State& aotState, int blockIndex, int sampleIndex, int numCh)
    {
        if (! isReady() || (isFrozen() && mismatchLatched.load (std::memory_order_acquire)))
            return;

        std::array<double, 64> shadowFrame {};
        shadow->readOutputSample (shadowFrame.data(), numCh);

        uint64_t mismatches = 0;
        for (int ch = 0; ch < numCh; ++ch)
        {
            const double compiled = (double) (float) aotState.spl[ch];
            const double ref = (double) (float) shadowFrame[(size_t) ch];
            const double delta = compiled - ref;
            const double absDelta = std::abs (delta);

            scalarComparisons.fetch_add (1, std::memory_order_relaxed);
            addToAtomicDouble (sumSquaredDelta, delta * delta);
            updateMaxAbsDelta (absDelta);

            if (! nearlyEqual (compiled, ref, kAudioCompareEpsilon))
            {
                ++mismatches;
                latchMismatch ("@sample", blockIndex, sampleIndex, ch, compiled, ref,
                               juce::String ("spl") + juce::String (ch));

                if (isFrozen() && mismatchLatched.load (std::memory_order_acquire))
                    break;
            }
        }

        if (mismatches > 0)
            scalarMismatches.fetch_add (mismatches, std::memory_order_relaxed);

        pushRingFrame (aotState, shadowFrame.data(), numCh);
    }

    void compareSliderAndVarState (const DSPJSFX_State& aotState, int blockIndex, int sampleIndex, const char* stage)
    {
        if (! isReady() || (isFrozen() && mismatchLatched.load (std::memory_order_acquire)))
            return;

        shadow->readSliders (shadowSliders.data(), 64);
        scalarComparisons.fetch_add (64, std::memory_order_relaxed);
        for (int i = 0; i < 64; ++i)
        {
            if (! nearlyEqual (aotState.sliders[i], shadowSliders[(size_t) i], kScalarCompareEpsilon))
            {
                scalarMismatches.fetch_add (1, std::memory_order_relaxed);
                latchMismatch (stage, blockIndex, sampleIndex, i,
                               aotState.sliders[i], shadowSliders[(size_t) i],
                               juce::String ("slider") + juce::String (i + 1));
                break;
            }
        }

        const int varCount = (int) (sizeof (aotState.vars) / sizeof (aotState.vars[0]));
        if ((int) shadowVars.size() != varCount)
            shadowVars.resize ((size_t) varCount, 0.0);
        shadow->readVars (shadowVars.data(), varCount);

        const SourceWriteStage visibleStage = visibleStageForCompareLabel (stage);
        for (int i = 0; i < varCount; ++i)
        {
            if (! shouldCompareVarAtStage (i, visibleStage))
                continue;

            scalarComparisons.fetch_add (1, std::memory_order_relaxed);
            if (! nearlyEqual (aotState.vars[i], shadowVars[(size_t) i], kScalarCompareEpsilon))
            {
                scalarMismatches.fetch_add (1, std::memory_order_relaxed);
                juce::String detail = "vars[" + juce::String (i) + "]";
                if (const char* varName = lookupVarName (i))
                    detail << " (" << varName << ")";
                latchMismatch (stage, blockIndex, sampleIndex, i,
                               aotState.vars[i], shadowVars[(size_t) i], detail);
                break;
            }
        }
    }

    void comparePendingSliderMasks (const DSPJSFX_State& aotState,
                                    const ShadowVm::PendingSliderMasks& shadowMasks,
                                    int blockIndex,
                                    const char* stage)
    {
        if (! isReady() || (isFrozen() && mismatchLatched.load (std::memory_order_acquire)))
            return;
        const uint64_t compiledChange = aotState.pendingSliderChangeMask > 0 ? (uint64_t) aotState.pendingSliderChangeMask : 0u;
        const uint64_t compiledAuto = aotState.pendingSliderAutomateMask > 0 ? (uint64_t) aotState.pendingSliderAutomateMask : 0u;
        const uint64_t compiledAutoEnd = aotState.pendingSliderAutomateEndMask > 0 ? (uint64_t) aotState.pendingSliderAutomateEndMask : 0u;

        if (compiledChange != shadowMasks.change)
        {
            scalarMismatches.fetch_add (1, std::memory_order_relaxed);
            latchMismatch (stage, blockIndex, -1, -1,
                           (double) compiledChange, (double) shadowMasks.change,
                           "pendingSliderChangeMask");
        }
        else if (compiledAuto != shadowMasks.automate)
        {
            scalarMismatches.fetch_add (1, std::memory_order_relaxed);
            latchMismatch (stage, blockIndex, -1, -1,
                           (double) compiledAuto, (double) shadowMasks.automate,
                           "pendingSliderAutomateMask");
        }
        else if (compiledAutoEnd != shadowMasks.automateEnd)
        {
            scalarMismatches.fetch_add (1, std::memory_order_relaxed);
            latchMismatch (stage, blockIndex, -1, -1,
                           (double) compiledAutoEnd, (double) shadowMasks.automateEnd,
                           "pendingSliderAutomateEndMask");
        }
    }

    void compareMidiOutput (const DSPJSFX_State& aotState, int blockIndex, const char* stage)
    {
        if (! isReady() || (isFrozen() && mismatchLatched.load (std::memory_order_acquire)))
            return;

        auto* compiledState = const_cast<DSPJSFX_State*> (&aotState);
        if (compiledState->midiOut != nullptr && compiledState->midiOutCount > 0
            && ! midiEventsAlreadySortedByOffset (compiledState->midiOut, compiledState->midiOutCount))
        {
            stableSortMidiEventsByOffset (compiledState->midiOut, compiledState->midiOutCount);
        }

        shadowMidiScratch = shadow->getMidiOut();
        if (! shadowMidiScratch.empty() && ! midiEventsAlreadySortedByOffset (shadowMidiScratch.data(), (int) shadowMidiScratch.size()))
            stableSortMidiEventsByOffset (shadowMidiScratch.data(), (int) shadowMidiScratch.size());

        if (aotState.midiOutCount != (int32_t) shadowMidiScratch.size())
        {
            scalarMismatches.fetch_add (1, std::memory_order_relaxed);
            latchMismatch (stage, blockIndex, -1, -1,
                           (double) aotState.midiOutCount, (double) shadowMidiScratch.size(),
                           "midiOutCount");
            return;
        }

        for (int i = 0; i < aotState.midiOutCount; ++i)
        {
            const auto& a = aotState.midiOut[i];
            const auto& b = shadowMidiScratch[(size_t) i];
            if (a.sampleOffset != b.sampleOffset || a.msg1 != b.msg1 || a.msg2 != b.msg2 || a.msg3 != b.msg3)
            {
                scalarMismatches.fetch_add (1, std::memory_order_relaxed);
                juce::String detail = "midiOut[" + juce::String (i) + "]";
                latchMismatch (stage, blockIndex, -1, i,
                               (double) a.sampleOffset * 1.0e9 + (double) a.msg1 * 1.0e6 + (double) a.msg2 * 1.0e3 + (double) a.msg3,
                               (double) b.sampleOffset * 1.0e9 + (double) b.msg1 * 1.0e6 + (double) b.msg2 * 1.0e3 + (double) b.msg3,
                               detail);
                return;
            }
        }
    }

    void compareMemoryPages (const DSPJSFX_State& aotState, int blockIndex, int sampleIndex, const char* stage)
    {
        if (! isReady() || aotState.mem == nullptr
            || (isFrozen() && mismatchLatched.load (std::memory_order_acquire)))
            return;

        const int64_t compareUsed = std::max<int64_t> (getTrackedJsfxMemUsed (const_cast<DSPJSFX_State*> (&aotState)),
                                                       shadow->logicalMemUsedForCompare());
        if (compareUsed <= 0)
            return;

        if ((int) memA.size() < kMemPageDoubles)
        {
            memA.resize ((size_t) kMemPageDoubles, 0.0);
            memB.resize ((size_t) kMemPageDoubles, 0.0);
        }

        for (int64_t base = 0, page = 0; base < compareUsed; base += kMemPageDoubles, ++page)
        {
            const int count = (int) std::min<int64_t> ((int64_t) kMemPageDoubles, compareUsed - base);
            std::fill (memA.begin(), memA.begin() + count, 0.0);
            std::fill (memB.begin(), memB.begin() + count, 0.0);

            if (base < aotState.memN)
            {
                const int aCount = (int) std::min<int64_t> ((int64_t) count, aotState.memN - base);
                std::memcpy (memA.data(), aotState.mem + base, (size_t) aCount * sizeof (double));
            }

            shadow->readMemRange (base, memB.data(), count);

            for (int i = 0; i < count; ++i)
            {
                if (! nearlyEqual (memA[(size_t) i], memB[(size_t) i], kScalarCompareEpsilon))
                {
                    scalarMismatches.fetch_add (1, std::memory_order_relaxed);
                    latchMismatch (stage, blockIndex, sampleIndex, (int) (base + i),
                                   memA[(size_t) i], memB[(size_t) i],
                                   "mem[" + juce::String ((int) (base + i)) + "]",
                                   juce::String (page));
                    return;
                }
            }
        }
    }

    void noteBlockCompared() noexcept
    {
        blocksCompared.fetch_add (1, std::memory_order_relaxed);
    }

    float selectOutputSample (int channel, double compiled, double shadowSample, int numCh) const noexcept
    {
        juce::ignoreUnused (channel, numCh);

        switch (getMonitorMode())
        {
            case MonitorAudioMode::Shadow: return (float) shadowSample;
            case MonitorAudioMode::Delta:  return (float) (compiled - shadowSample);
            case MonitorAudioMode::Compiled:
            default:                       return (float) compiled;
        }
    }

    juce::String getStatusText() const
    {
        juce::String compileErr;
        juce::String mismatchStage;
        juce::String mismatchDetail;
        juce::String mismatchPages;
        juce::String exportPath;
        int mismatchBlock = -1;
        int mismatchSample = -1;
        int mismatchIndex = -1;
        double mismatchCompiled = 0.0;
        double mismatchShadow = 0.0;
        double mismatchAbs = 0.0;

        {
            std::lock_guard<std::mutex> lk (detailMutex);
            compileErr = compileError;
            mismatchStage = firstMismatchStage;
            mismatchDetail = firstMismatchDetail;
            mismatchPages = firstMismatchPages;
            exportPath = lastExportPath;
            mismatchBlock = firstMismatchBlock;
            mismatchSample = firstMismatchSample;
            mismatchIndex = firstMismatchChannel;
            mismatchCompiled = firstMismatchCompiled;
            mismatchShadow = firstMismatchShadow;
            mismatchAbs = firstMismatchAbsDelta;
        }

        const auto mode = getMonitorMode();
        const uint64_t blocks = blocksCompared.load (std::memory_order_acquire);
        const uint64_t comps = scalarComparisons.load (std::memory_order_acquire);
        const uint64_t mism = scalarMismatches.load (std::memory_order_acquire);
        const double sumSq = sumSquaredDelta.load (std::memory_order_acquire);
        const double maxD = maxAbsDelta.load (std::memory_order_acquire);
        const double rms = comps > 0 ? std::sqrt (sumSq / (double) comps) : 0.0;

        juce::String out;
        out << "shadow VM: ";
        if (! compileErr.isEmpty() && ! isReady())
            out << "compile failed\n";
        else
            out << (isReady() ? "ready\n" : "not primed\n");

        out << "monitor: " << monitorModeName (mode) << "\n";
        out << "freeze on mismatch: " << (getFreezeOnFirstMismatch() ? "on" : "off") << "\n";
        out << "frozen: " << (isFrozen() ? "yes" : "no") << "\n";
        out << "blocks compared: " << juce::String ((juce::uint64) blocks) << "\n";
        out << "scalar comparisons: " << juce::String ((juce::uint64) comps) << "\n";
        out << "scalar mismatches: " << juce::String ((juce::uint64) mism) << "\n";
        out << "max |delta|: " << juce::String (maxD, 9) << "\n";
        out << "RMS delta: " << juce::String (rms, 9) << "\n";

        if (! compileErr.isEmpty() && ! isReady())
            out << "compile error: " << compileErr << "\n";

        if (mismatchLatched.load (std::memory_order_acquire))
        {
            out << "first mismatch: " << mismatchStage
                << " | block " << juce::String (mismatchBlock)
                << " | sample " << juce::String (mismatchSample)
                << " | idx " << juce::String (mismatchIndex) << "\n";
            out << "detail: " << mismatchDetail << "\n";
            out << "compiled: " << juce::String (mismatchCompiled, 9)
                << " | shadow: " << juce::String (mismatchShadow, 9)
                << " | |delta|: " << juce::String (mismatchAbs, 9) << "\n";
            if (mismatchPages.isNotEmpty())
                out << "changed pages: " << mismatchPages << "\n";
        }

        if (exportPath.isNotEmpty())
            out << "last export: " << exportPath << "\n";

        return out;
    }

    juce::String exportBundle (const juce::String& pluginName) const
    {
        int capacity = 0;
        int64_t totalFramesNow = 0;
        int framesToCopy = 0;
        std::array<std::vector<float>, kRingChannels> compiledCopy;
        std::array<std::vector<float>, kRingChannels> shadowCopy;
        std::array<std::vector<float>, kRingChannels> deltaCopy;

        {
            std::lock_guard<std::mutex> ringLock (ringMutex);
            capacity = ringCapacityFrames.load (std::memory_order_acquire);
            if (capacity <= 0)
                return "Correctness ring buffer is empty";

            totalFramesNow = totalRingFrames.load (std::memory_order_acquire);
            framesToCopy = (int) std::min<int64_t> ((int64_t) capacity, totalFramesNow);
            if (framesToCopy <= 0)
                return "Correctness ring buffer is empty";

            const int64_t start = juce::jmax<int64_t> ((int64_t) 0, totalFramesNow - framesToCopy);
            for (int ch = 0; ch < kRingChannels; ++ch)
            {
                compiledCopy[(size_t) ch].resize ((size_t) framesToCopy, 0.0f);
                shadowCopy[(size_t) ch].resize ((size_t) framesToCopy, 0.0f);
                deltaCopy[(size_t) ch].resize ((size_t) framesToCopy, 0.0f);

                for (int i = 0; i < framesToCopy; ++i)
                {
                    const int idx = (int) ((start + i) % capacity);
                    if (idx >= 0)
                    {
                        if (idx < (int) ringCompiled[(size_t) ch].size()) compiledCopy[(size_t) ch][(size_t) i] = ringCompiled[(size_t) ch][(size_t) idx];
                        if (idx < (int) ringShadow[(size_t) ch].size())   shadowCopy[(size_t) ch][(size_t) i] = ringShadow[(size_t) ch][(size_t) idx];
                        if (idx < (int) ringDelta[(size_t) ch].size())    deltaCopy[(size_t) ch][(size_t) i] = ringDelta[(size_t) ch][(size_t) idx];
                    }
                }
            }
        }

        const juce::String safeName = sanitiseFileComponent (pluginName);
        juce::File root = juce::File::getSpecialLocation (juce::File::userDocumentsDirectory)
                              .getChildFile ("ZorakAudio-Correctness")
                              .getChildFile (safeName + "-" + juce::Time::getCurrentTime().formatted ("%Y%m%d-%H%M%S"));
        root.createDirectory();

        auto writeWav = [&root, framesToCopy] (const juce::String& name,
                                               const std::array<std::vector<float>, kRingChannels>& data,
                                               double sr) -> bool
        {
            juce::AudioBuffer<float> buf (kRingChannels, framesToCopy);
            for (int ch = 0; ch < kRingChannels; ++ch)
            {
                auto* dst = buf.getWritePointer (ch);
                const auto& src = data[(size_t) ch];
                for (int i = 0; i < framesToCopy; ++i)
                    dst[i] = (i >= 0 && i < (int) src.size()) ? src[(size_t) i] : 0.0f;
            }

            juce::WavAudioFormat wav;
            auto outFile = root.getChildFile (name + ".wav");
            auto stream = std::unique_ptr<juce::FileOutputStream> (outFile.createOutputStream());
            if (stream == nullptr)
                return false;

            auto writer = std::unique_ptr<juce::AudioFormatWriter> (wav.createWriterFor (stream.get(), sr, (unsigned int) kRingChannels, 24, {}, 0));
            if (writer == nullptr)
                return false;

            stream.release();
            return writer->writeFromAudioSampleBuffer (buf, 0, framesToCopy);
        };

        const double sr = ringSampleRate.load (std::memory_order_acquire);
        const bool okA = writeWav ("compiled", compiledCopy, sr);
        const bool okB = writeWav ("shadow", shadowCopy, sr);
        const bool okC = writeWav ("delta", deltaCopy, sr);

        juce::String mismatchStage;
        juce::String mismatchDetail;
        juce::String mismatchPages;
        int mismatchBlock = -1;
        int mismatchSample = -1;
        int mismatchIndex = -1;
        double mismatchCompiled = 0.0;
        double mismatchShadow = 0.0;
        double mismatchAbs = 0.0;
        {
            std::lock_guard<std::mutex> lk (detailMutex);
            mismatchStage = firstMismatchStage;
            mismatchDetail = firstMismatchDetail;
            mismatchPages = firstMismatchPages;
            mismatchBlock = firstMismatchBlock;
            mismatchSample = firstMismatchSample;
            mismatchIndex = firstMismatchChannel;
            mismatchCompiled = firstMismatchCompiled;
            mismatchShadow = firstMismatchShadow;
            mismatchAbs = firstMismatchAbsDelta;
        }

        juce::String json;
        json << "{\n";
        json << "  \"plugin\": \"" << jsonEscape (safeName) << "\",\n";
        json << "  \"sampleRate\": " << juce::String (sr, 3) << ",\n";
        json << "  \"frames\": " << juce::String (framesToCopy) << ",\n";
        json << "  \"monitorMode\": \"" << jsonEscape (monitorModeName (getMonitorMode())) << "\",\n";
        json << "  \"freezeOnFirstMismatch\": " << (getFreezeOnFirstMismatch() ? "true" : "false") << ",\n";
        json << "  \"frozen\": " << (isFrozen() ? "true" : "false") << ",\n";
        json << "  \"blocksCompared\": " << juce::String ((juce::uint64) blocksCompared.load (std::memory_order_acquire)) << ",\n";
        json << "  \"scalarComparisons\": " << juce::String ((juce::uint64) scalarComparisons.load (std::memory_order_acquire)) << ",\n";
        json << "  \"scalarMismatches\": " << juce::String ((juce::uint64) scalarMismatches.load (std::memory_order_acquire)) << ",\n";
        json << "  \"maxAbsDelta\": " << juce::String (maxAbsDelta.load (std::memory_order_acquire), 12) << ",\n";
        json << "  \"sumSquaredDelta\": " << juce::String (sumSquaredDelta.load (std::memory_order_acquire), 12) << ",\n";
        json << "  \"firstMismatchStage\": \"" << jsonEscape (mismatchStage) << "\",\n";
        json << "  \"firstMismatchDetail\": \"" << jsonEscape (mismatchDetail) << "\",\n";
        json << "  \"firstMismatchPages\": \"" << jsonEscape (mismatchPages) << "\",\n";
        json << "  \"firstMismatchBlock\": " << juce::String (mismatchBlock) << ",\n";
        json << "  \"firstMismatchSample\": " << juce::String (mismatchSample) << ",\n";
        json << "  \"firstMismatchIndex\": " << juce::String (mismatchIndex) << ",\n";
        json << "  \"firstMismatchCompiled\": " << juce::String (mismatchCompiled, 12) << ",\n";
        json << "  \"firstMismatchShadow\": " << juce::String (mismatchShadow, 12) << ",\n";
        json << "  \"firstMismatchAbsDelta\": " << juce::String (mismatchAbs, 12) << "\n";
        json << "}\n";

        root.getChildFile ("report.json").replaceWithText (json);

        {
            std::lock_guard<std::mutex> lk (detailMutex);
            lastExportPath = root.getFullPathName();
        }

        if (okA && okB && okC)
            return root.getFullPathName();
        return "Export incomplete: " + root.getFullPathName();
    }

private:
    static void addToAtomicDouble (std::atomic<double>& target, double value) noexcept
    {
        double cur = target.load (std::memory_order_relaxed);
        while (! target.compare_exchange_weak (cur, cur + value,
                                               std::memory_order_release,
                                               std::memory_order_relaxed))
        {
        }
    }

    static const char* lookupVarName (int varIndex) noexcept
    {
        for (int i = 0; i < DSPJSFX_VARS_COUNT; ++i)
        {
            if (DSPJSFX_VARS[i].index == varIndex && DSPJSFX_VARS[i].name != nullptr && *DSPJSFX_VARS[i].name != 0)
                return DSPJSFX_VARS[i].name;
        }
        return nullptr;
    }
    enum class SourceWriteStage : uint8_t
    {
        Unknown = 0,
        Init,
        Slider,
        Block,
        Sample,
    };

    static bool isIdentStartChar (char c) noexcept
    {
        const unsigned char uc = (unsigned char) c;
        return std::isalpha (uc) || c == '_' || c == '$' || c == '#';
    }

    static bool isIdentChar (char c) noexcept
    {
        const unsigned char uc = (unsigned char) c;
        return std::isalnum (uc) || c == '_' || c == '$' || c == '#' || c == '.';
    }

    static bool isAssignmentOperatorAt (const std::string& text, size_t pos) noexcept
    {
        if (pos >= text.size())
            return false;

        const char c = text[pos];
        if (c == '=')
        {
            const char prev = (pos > 0 ? text[pos - 1] : '\0');
            const char next = (pos + 1 < text.size() ? text[pos + 1] : '\0');
            return prev != '=' && prev != '!' && prev != '<' && prev != '>' && next != '=';
        }

        if (pos + 1 >= text.size())
            return false;

        const char next = text[pos + 1];
        return ((c == '+' || c == '-' || c == '*' || c == '/' || c == '%' || c == '&' || c == '|' || c == '^')
                 && next == '=');
    }

    static SourceWriteStage visibleStageForCompareLabel (const char* stage) noexcept
    {
        if (stage == nullptr)
            return SourceWriteStage::Sample;

        const std::string s (stage);
        if (s == "@prepare" || s == "pre-@block" || s == "pre-@block/@slider")
            return SourceWriteStage::Slider;
        if (s == "@block" || s == "@slider")
            return SourceWriteStage::Block;
        if (s == "@sample")
            return SourceWriteStage::Sample;
        return SourceWriteStage::Sample;
    }

    void scanSectionForFirstWrites (const std::string& code, SourceWriteStage stage,
                                    const std::unordered_map<std::string, int>& nameToVar) noexcept
    {
        bool inLineComment = false;
        bool inBlockComment = false;
        bool inString = false;
        char quote = 0;

        for (size_t i = 0; i < code.size();)
        {
            const char c = code[i];
            const char next = (i + 1 < code.size() ? code[i + 1] : '\0');

            if (inLineComment)
            {
                if (c == '\n')
                    inLineComment = false;
                ++i;
                continue;
            }

            if (inBlockComment)
            {
                if (c == '*' && next == '/')
                {
                    i += 2;
                    inBlockComment = false;
                }
                else
                {
                    ++i;
                }
                continue;
            }

            if (inString)
            {
                if (c == '\\' && i + 1 < code.size())
                {
                    i += 2;
                    continue;
                }
                if (c == quote)
                    inString = false;
                ++i;
                continue;
            }

            if (c == '/' && next == '/')
            {
                i += 2;
                inLineComment = true;
                continue;
            }

            if (c == '/' && next == '*')
            {
                i += 2;
                inBlockComment = true;
                continue;
            }

            if (c == '"' || c == '\'')
            {
                inString = true;
                quote = c;
                ++i;
                continue;
            }

            if (! isIdentStartChar (c))
            {
                ++i;
                continue;
            }

            const size_t start = i;
            ++i;
            while (i < code.size() && isIdentChar (code[i]))
                ++i;

            const std::string ident = code.substr (start, i - start);
            auto it = nameToVar.find (ident);
            if (it == nameToVar.end())
                continue;

            size_t j = i;
            while (j < code.size() && (code[j] == ' ' || code[j] == '\t' || code[j] == '\r' || code[j] == '\n'))
                ++j;

            if (! isAssignmentOperatorAt (code, j))
                continue;

            const int varIndex = it->second;
            if (varIndex < 0 || varIndex >= (int) varFirstWriteStage.size())
                continue;

            if (varFirstWriteStage[(size_t) varIndex] == SourceWriteStage::Unknown)
                varFirstWriteStage[(size_t) varIndex] = stage;
        }
    }

    void initVarFirstWriteStages()
    {
        varFirstWriteStage.clear();

        if (DSPJSFX_VARS_COUNT <= 0)
            return;

        std::unordered_map<std::string, int> nameToVar;
        nameToVar.reserve ((size_t) DSPJSFX_VARS_COUNT);
        int maxIndex = -1;
        for (int i = 0; i < DSPJSFX_VARS_COUNT; ++i)
        {
            if (DSPJSFX_VARS[i].name == nullptr || *DSPJSFX_VARS[i].name == 0 || DSPJSFX_VARS[i].index < 0)
                continue;

            nameToVar.emplace (DSPJSFX_VARS[i].name, DSPJSFX_VARS[i].index);
            maxIndex = std::max (maxIndex, DSPJSFX_VARS[i].index);
        }

        if (maxIndex < 0)
            return;

        varFirstWriteStage.assign ((size_t) (maxIndex + 1), SourceWriteStage::Unknown);

        const auto sections = jsfx_gfx::extractJsfxSections (kJsfxSourceText);
        scanSectionForFirstWrites (sections.init,   SourceWriteStage::Init,   nameToVar);
        scanSectionForFirstWrites (sections.slider, SourceWriteStage::Slider, nameToVar);
        scanSectionForFirstWrites (sections.block,  SourceWriteStage::Block,  nameToVar);
        scanSectionForFirstWrites (sections.sample, SourceWriteStage::Sample, nameToVar);
    }

    bool shouldCompareVarAtStage (int varIndex, SourceWriteStage visibleStage) const noexcept
    {
        if (visibleStage == SourceWriteStage::Sample)
            return true;
        if (varIndex < 0 || varIndex >= (int) varFirstWriteStage.size())
            return true;

        const SourceWriteStage first = varFirstWriteStage[(size_t) varIndex];
        if (first == SourceWriteStage::Unknown)
            return true;

        return (int) first <= (int) visibleStage;
    }

    bool isFrozen() const noexcept
    {
        return frozen.load (std::memory_order_acquire);
    }

    void resizeRingForSampleRate (double sr)
    {
        const int newCap = juce::jmax (256, (int) std::llround (sr * kRingSeconds));
        std::lock_guard<std::mutex> lk (ringMutex);
        ringCapacityFrames.store (newCap, std::memory_order_release);
        for (int ch = 0; ch < kRingChannels; ++ch)
        {
            ringCompiled[(size_t) ch].assign ((size_t) newCap, 0.0f);
            ringShadow[(size_t) ch].assign ((size_t) newCap, 0.0f);
            ringDelta[(size_t) ch].assign ((size_t) newCap, 0.0f);
        }
    }

    void pushRingFrame (const DSPJSFX_State& aotState, const double* shadowFrame, int numCh)
    {
        if (isFrozen() && mismatchLatched.load (std::memory_order_acquire))
            return;

        std::lock_guard<std::mutex> lk (ringMutex);

        const int capacity = ringCapacityFrames.load (std::memory_order_acquire);
        if (capacity <= 0)
            return;

        const int64_t frameIndex = totalRingFrames.fetch_add (1, std::memory_order_acq_rel);
        const int pos = (int) (frameIndex % capacity);

        const double compiledL = numCh > 0 ? (double) (float) aotState.spl[0] : 0.0;
        const double compiledR = numCh > 1 ? (double) (float) aotState.spl[1] : compiledL;
        const double shadowL = numCh > 0 ? (double) (float) shadowFrame[0] : 0.0;
        const double shadowR = numCh > 1 ? (double) (float) shadowFrame[1] : shadowL;

        ringCompiled[0][(size_t) pos] = (float) compiledL;
        ringCompiled[1][(size_t) pos] = (float) compiledR;
        ringShadow[0][(size_t) pos] = (float) shadowL;
        ringShadow[1][(size_t) pos] = (float) shadowR;
        ringDelta[0][(size_t) pos] = (float) (compiledL - shadowL);
        ringDelta[1][(size_t) pos] = (float) (compiledR - shadowR);
    }

    void updateMaxAbsDelta (double candidate) noexcept
    {
        double cur = maxAbsDelta.load (std::memory_order_relaxed);
        while (candidate > cur && ! maxAbsDelta.compare_exchange_weak (cur, candidate,
                                                                        std::memory_order_release,
                                                                        std::memory_order_relaxed))
        {
        }
    }

    void latchMismatch (const char* stage,
                        int blockIndex,
                        int sampleIndex,
                        int channelOrIndex,
                        double compiled,
                        double shadowValue,
                        const juce::String& detail,
                        const juce::String& pages = {})
    {
        bool expected = false;
        if (! mismatchLatched.compare_exchange_strong (expected, true, std::memory_order_acq_rel))
            return;

        std::lock_guard<std::mutex> lk (detailMutex);
        firstMismatchStage = stage;
        firstMismatchDetail = detail;
        firstMismatchPages = pages;
        firstMismatchBlock = blockIndex;
        firstMismatchSample = sampleIndex;
        firstMismatchChannel = channelOrIndex;
        firstMismatchCompiled = compiled;
        firstMismatchShadow = shadowValue;
        firstMismatchAbsDelta = std::abs (compiled - shadowValue);

        if (freezeOnFirstMismatch.load (std::memory_order_acquire))
            frozen.store (true, std::memory_order_release);
    }

    JSFXJuceProcessor* owner = nullptr;
    std::unique_ptr<ShadowVm> shadow;

    std::atomic<int> monitorMode { (int) MonitorAudioMode::Compiled };
    std::atomic<bool> freezeOnFirstMismatch { true };
    std::atomic<bool> frozen { false };
    std::atomic<bool> mismatchLatched { false };
    std::atomic<uint64_t> blocksCompared { 0 };
    std::atomic<uint64_t> scalarComparisons { 0 };
    std::atomic<uint64_t> scalarMismatches { 0 };
    std::atomic<double> sumSquaredDelta { 0.0 };
    std::atomic<double> maxAbsDelta { 0.0 };
    std::atomic<int> ringCapacityFrames { 0 };
    std::atomic<int64_t> totalRingFrames { 0 };
    std::atomic<double> ringSampleRate { 44100.0 };

    mutable std::mutex detailMutex;
    mutable std::mutex ringMutex;
    juce::String compileError;
    juce::String firstMismatchStage;
    juce::String firstMismatchDetail;
    juce::String firstMismatchPages;
    mutable juce::String lastExportPath;
    int firstMismatchBlock = -1;
    int firstMismatchSample = -1;
    int firstMismatchChannel = -1;
    double firstMismatchCompiled = 0.0;
    double firstMismatchShadow = 0.0;
    double firstMismatchAbsDelta = 0.0;

    std::array<double, 64> shadowSliders {};
    std::vector<double> shadowVars;
    std::vector<SourceWriteStage> varFirstWriteStage;
    mutable std::array<std::vector<float>, kRingChannels> ringCompiled;
    mutable std::array<std::vector<float>, kRingChannels> ringShadow;
    mutable std::array<std::vector<float>, kRingChannels> ringDelta;
    std::vector<double> memA;
    std::vector<double> memB;
    std::vector<DSPJSFX_MidiEvent> shadowMidiScratch;
};

} // namespace jsfx_correctness

#endif // ZA_JSFX_CORRECTNESS_CHECK
