// SPDX-License-Identifier: Zlib
#pragma once
#include <regex>
#include <functional>
#include <cerrno>
#include "JsfxStateVariables.h"
#if DSPJSFX_HAS_TASKS
#include "JsfxTasks.h"
#endif
namespace za::jsfx {
struct IdleTopology {int inputs=0,outputs=0;bool acceptsMidi=false,producesMidi=false,fileWake=false;};
class IdleRuntime {
public:
    virtual ~IdleRuntime()=default;
    virtual bool idleIsNonRealtime()const noexcept {return false;}
    DSPJSFX_State* idleState=nullptr;
    IdleTopology idleTopology;
    std::function<int(const char*)> idleIndexLookup;
    enum class SmartIdleMode : int32_t
    {
        Auto         = 0,
        InputDriven  = 1,
        EventDriven  = 2,
        FreeRunning  = 3,
        AlwaysAwake  = 4,
        Cooperative  = 5,
    };

    struct SmartIdleConfig
    {
        SmartIdleMode optionMode = SmartIdleMode::Auto;
        SmartIdleMode inferredMode = SmartIdleMode::AlwaysAwake;
        double holdMs = 250.0;
        double tailMs = 0.0;
        float inputThreshold = 0.00001584893192461114f;  // -96 dB
        float outputThreshold = 0.00000316227766016838f; // -110 dB
        int64_t holdSamples = 0;
        int64_t tailSamples = 0;
        int keepAwakeVarIndex = -1;
        int modeVarIndex = -1;
        int sleepReadyVarIndex = -1;
    };

    struct SmartIdleRuntimeState
    {
        int lastHostBlockSize = 0;
        bool sleeping = false;
        int64_t quietSamples = 0;
        float lastInputPeak = 0.0f;
        float lastOutputPeak = 0.0f;
        SmartIdleMode lastEffectiveMode = SmartIdleMode::AlwaysAwake;
    };

    static constexpr double kDefaultSmartIdleHoldMs = 250.0;
    static constexpr double kDefaultSmartIdleTailMs = 0.0;
    static constexpr double kDefaultSmartIdleInputDb = -96.0;
    static constexpr double kDefaultSmartIdleOutputDb = -110.0;

    static std::string lowerAscii (std::string text)
    {
        std::transform (text.begin(), text.end(), text.begin(), [] (unsigned char c)
        {
            return (char) std::tolower (c);
        });
        return text;
    }

    static bool tryParseDoubleStrict (const std::string& text, double& out) noexcept
    {
        const std::string trimmed = trimAscii (text);
        if (trimmed.empty())
            return false;

        char* end = nullptr;
        errno = 0;
        const double parsed = std::strtod (trimmed.c_str(), &end);
        if (end == trimmed.c_str())
            return false;

        while (end != nullptr && *end != '\0' && std::isspace ((unsigned char) *end))
            ++end;

        if (end == nullptr || *end != '\0' || errno == ERANGE || ! std::isfinite (parsed))
            return false;

        out = parsed;
        return true;
    }

    static float decibelsToLinearAmplitude (double db) noexcept
    {
        const double clampedDb = juce::jlimit (-300.0, 0.0, db);
        return (float) std::pow (10.0, clampedDb / 20.0);
    }

    static std::unordered_map<std::string, std::string> parseRuntimeJsfxOptions (const char* jsfxText)
    {
        std::unordered_map<std::string, std::string> opts;
        if (jsfxText == nullptr)
            return opts;

        std::string textBody (jsfxText);
        size_t start = 0;
        const std::regex reOptions (R"(^\s*options\s*:\s*(.*)$)", std::regex::ECMAScript | std::regex::icase);

        while (start < textBody.size())
        {
            size_t end = textBody.find_first_of ("\r\n", start);
            if (end == std::string::npos)
                end = textBody.size();

            std::string line = textBody.substr (start, end - start);

            size_t next = end;
            while (next < textBody.size() && (textBody[next] == '\r' || textBody[next] == '\n'))
                ++next;
            start = next;

            std::smatch m;
            if (! std::regex_match (line, m, reOptions))
                continue;

            const std::string payload = m[1].str();
            std::string token;

            auto flushToken = [&opts] (std::string tok)
            {
                tok = trimAscii (std::move (tok));
                if (tok.empty())
                    return;

                const auto eq = tok.find ('=');
                if (eq == std::string::npos)
                    return;

                auto key = lowerAscii (trimAscii (tok.substr (0, eq)));
                auto value = trimAscii (tok.substr (eq + 1));
                if (! key.empty())
                    opts[key] = value;
            };

            for (char c : payload)
            {
                if (std::isspace ((unsigned char) c) || c == ',')
                {
                    flushToken (token);
                    token.clear();
                }
                else
                {
                    token.push_back (c);
                }
            }

            flushToken (token);
        }

        return opts;
    }

    static SmartIdleMode parseSmartIdleModeToken (std::string token) noexcept
    {
        token = lowerAscii (trimAscii (std::move (token)));
        if (token.empty() || token == "auto" || token == "default")
            return SmartIdleMode::Auto;
        if (token == "1" || token == "input" || token == "audio" || token == "silence" || token == "sleep_on_silence" || token == "sleep-on-silence" || token == "inputdriven" || token == "input_driven" || token == "input-driven")
            return SmartIdleMode::InputDriven;
        if (token == "2" || token == "event" || token == "events" || token == "midi" || token == "sleep_on_events" || token == "sleep-on-events" || token == "eventdriven" || token == "event_driven" || token == "event-driven")
            return SmartIdleMode::EventDriven;
        if (token == "3" || token == "free" || token == "generator" || token == "freerunning" || token == "free_running" || token == "free-running")
            return SmartIdleMode::FreeRunning;
        if (token == "4" || token == "always" || token == "awake" || token == "neversleep" || token == "never_sleep" || token == "never-sleep" || token == "alwaysawake" || token == "always_awake" || token == "always-awake" || token == "off" || token == "never" || token == "disabled")
            return SmartIdleMode::AlwaysAwake;
        return SmartIdleMode::Auto;
    }

    static SmartIdleMode parseSmartIdleModeValue (double raw) noexcept
    {
        if (! std::isfinite (raw))
            return SmartIdleMode::Auto;

        const int mode = (int) std::llround (raw);
        switch (mode)
        {
            case 1:  return SmartIdleMode::InputDriven;
            case 2:  return SmartIdleMode::EventDriven;
            case 3:  return SmartIdleMode::FreeRunning;
            case 4:  return SmartIdleMode::AlwaysAwake;
            case 5:  return SmartIdleMode::Cooperative;
            default: return SmartIdleMode::Auto;
        }
    }


    static SmartIdleMode storedSmartIdleModeToEnum (int raw) noexcept
    {
        switch (raw)
        {
            case 1:  return SmartIdleMode::InputDriven;
            case 2:  return SmartIdleMode::EventDriven;
            case 3:  return SmartIdleMode::FreeRunning;
            case 4:  return SmartIdleMode::AlwaysAwake;
            case 5:  return SmartIdleMode::Cooperative;
            default: return SmartIdleMode::Auto;
        }
    }

    static juce::String smartIdleModeDisplayName (SmartIdleMode mode)
    {
        switch (mode)
        {
            case SmartIdleMode::Cooperative: return "Plugin signal";
            case SmartIdleMode::InputDriven: return "Input-driven";
            case SmartIdleMode::EventDriven: return "Event-driven";
            case SmartIdleMode::FreeRunning: return "Free-running";
            case SmartIdleMode::AlwaysAwake: return "Always awake";
            case SmartIdleMode::Auto:
            default:                         return "Auto";
        }
    }

    static juce::String smartIdleModeShortName (SmartIdleMode mode)
    {
        switch (mode)
        {
            case SmartIdleMode::Cooperative: return "PLUGIN";
            case SmartIdleMode::InputDriven: return "INPUT";
            case SmartIdleMode::EventDriven: return "EVENT";
            case SmartIdleMode::FreeRunning: return "FREE";
            case SmartIdleMode::AlwaysAwake: return "ALWAYS";
            case SmartIdleMode::Auto:
            default:                         return "AUTO";
        }
    }

    static juce::String formatSmartIdlePeakDbText (float peak)
    {
        if (! std::isfinite (peak) || peak <= 0.0f)
            return "-inf dBFS";

        const double db = 20.0 * std::log10 ((double) peak);
        if (! std::isfinite (db) || db <= -300.0)
            return "-inf dBFS";

        return juce::String (db, 1) + " dBFS";
    }

    static bool isSmartIdleSleepEligible (SmartIdleMode mode) noexcept
    {
        return mode == SmartIdleMode::InputDriven || mode == SmartIdleMode::EventDriven || mode == SmartIdleMode::Cooperative;
    }

    SmartIdleMode inferSmartIdleModeFromTopology() const noexcept
    {
        const bool hasAudioInputs = idleTopology.inputs>0;
        const bool hasAudioOutputs = idleTopology.outputs>0;
        const bool acceptsMidiInput = idleTopology.acceptsMidi;
        const bool producesMidiOutput = idleTopology.producesMidi;
        const bool hasFileWakeSources = idleTopology.fileWake;

        if (hasAudioInputs)
            return SmartIdleMode::InputDriven;

        if (acceptsMidiInput || hasFileWakeSources)
            return SmartIdleMode::EventDriven;

        if (hasAudioOutputs || producesMidiOutput)
            return SmartIdleMode::FreeRunning;

        return SmartIdleMode::AlwaysAwake;
    }

    void updateSmartIdleSampleRate (double sampleRate) noexcept
    {
        const double sr = std::max (1.0, sampleRate);
        smartIdleConfig.holdSamples = (int64_t) std::llround (juce::jmax (0.0, smartIdleConfig.holdMs) * 0.001 * sr);
        smartIdleConfig.tailSamples = (int64_t) std::llround (juce::jmax (0.0, smartIdleConfig.tailMs) * 0.001 * sr);
        smartIdleUiSampleRate.store (sr, std::memory_order_relaxed);
    }

    void publishSmartIdleUiConfigSnapshot() noexcept
    {
        smartIdleUiOptionMode.store ((int) smartIdleConfig.optionMode, std::memory_order_relaxed);
        smartIdleUiInferredMode.store ((int) smartIdleConfig.inferredMode, std::memory_order_relaxed);
        smartIdleUiUserOverrideMode.store ((int) storedSmartIdleModeToEnum (smartIdleUserOverrideMode.load (std::memory_order_acquire)), std::memory_order_relaxed);
        smartIdleUiHoldMs.store (smartIdleConfig.holdMs, std::memory_order_relaxed);
        smartIdleUiTailMs.store (smartIdleConfig.tailMs, std::memory_order_relaxed);
    }

    void publishSmartIdleUiRuntimeSnapshot (SmartIdleMode effectiveMode,
                                            bool sleeping,
                                            float inputPeak,
                                            float outputPeak,
                                            int64_t quietSamples) noexcept
    {
        smartIdleUiEffectiveMode.store ((int) effectiveMode, std::memory_order_relaxed);
        smartIdleUiSleeping.store (sleeping, std::memory_order_relaxed);
        smartIdleUiInputPeak.store (inputPeak, std::memory_order_relaxed);
        smartIdleUiOutputPeak.store (outputPeak, std::memory_order_relaxed);
        smartIdleUiQuietSamples.store (quietSamples, std::memory_order_relaxed);
    }

    void resetSmartIdleRuntime() noexcept
    {
        smartIdleRuntime = SmartIdleRuntimeState {};
        publishSmartIdleUiRuntimeSnapshot (resolveSmartIdleModeForCurrentState(), false, 0.0f, 0.0f, 0);
    }

    void initialiseIdle(DSPJSFX_State& state,const char* source,IdleTopology topology,std::function<int(const char*)> lookup)
    {
        idleState=&state;idleTopology=topology;idleIndexLookup=std::move(lookup);
        smartIdleConfig = SmartIdleConfig {};
        smartIdleConfig.inferredMode = inferSmartIdleModeFromTopology();
        smartIdleConfig.keepAwakeVarIndex = idleIndexLookup ("za_keep_awake");
        smartIdleConfig.modeVarIndex = idleIndexLookup ("za_idle_mode");
        smartIdleConfig.sleepReadyVarIndex = idleIndexLookup ("za_sleep_ready");

        const auto opts = parseRuntimeJsfxOptions (source);

        if (auto it = opts.find ("idle"); it != opts.end())
            smartIdleConfig.optionMode = parseSmartIdleModeToken (it->second);

        if (auto it = opts.find ("idle_hold_ms"); it != opts.end())
        {
            double parsed = kDefaultSmartIdleHoldMs;
            if (tryParseDoubleStrict (it->second, parsed) && parsed >= 0.0)
                smartIdleConfig.holdMs = parsed;
        }

        if (auto it = opts.find ("tail_ms"); it != opts.end())
        {
            double parsed = kDefaultSmartIdleTailMs;
            if (tryParseDoubleStrict (it->second, parsed) && parsed >= 0.0)
                smartIdleConfig.tailMs = parsed;
        }

        if (auto it = opts.find ("idle_in_db"); it != opts.end())
        {
            double parsed = kDefaultSmartIdleInputDb;
            if (tryParseDoubleStrict (it->second, parsed))
                smartIdleConfig.inputThreshold = decibelsToLinearAmplitude (parsed);
        }

        if (auto it = opts.find ("idle_out_db"); it != opts.end())
        {
            double parsed = kDefaultSmartIdleOutputDb;
            if (tryParseDoubleStrict (it->second, parsed))
                smartIdleConfig.outputThreshold = decibelsToLinearAmplitude (parsed);
        }

        smartIdleTailLengthSeconds = juce::jmax (0.0, smartIdleConfig.tailMs) * 0.001;
        publishSmartIdleUiConfigSnapshot();
        updateSmartIdleSampleRate (idleState->srate);
        resetSmartIdleRuntime();
    }

    SmartIdleMode getConfiguredSmartIdleMode() const noexcept
    {
        return smartIdleConfig.optionMode == SmartIdleMode::Auto
                 ? smartIdleConfig.inferredMode
                 : smartIdleConfig.optionMode;
    }

    SmartIdleMode resolveSmartIdleModeForCurrentState() const noexcept
    {
        if (idleIsNonRealtime()) return SmartIdleMode::AlwaysAwake;
        SmartIdleMode resolved = smartIdleConfig.sleepReadyVarIndex >= 0 ? SmartIdleMode::Cooperative : getConfiguredSmartIdleMode();

        const int modeVarIndex = smartIdleConfig.modeVarIndex;
        const int varsCap = (int) za::jsfx::variableCount(*idleState);
        if (modeVarIndex >= 0 && modeVarIndex < varsCap)
        {
            const SmartIdleMode runtimeMode = parseSmartIdleModeValue (idleState->vars[(size_t) modeVarIndex]);
            if (runtimeMode != SmartIdleMode::Auto)
                resolved = runtimeMode;
        }

        const SmartIdleMode userOverride = storedSmartIdleModeToEnum (smartIdleUserOverrideMode.load (std::memory_order_acquire));
        if (userOverride != SmartIdleMode::Auto)
            resolved = userOverride;

        return resolved;
    }

    bool getSmartIdleKeepAwakeFlag() const noexcept
    {
#if DSPJSFX_HAS_TASKS
        if (auto* tasks=static_cast<jsfx_tasks::Runtime*>(idleState->taskContext);tasks && tasks->hasOutstandingWorkForIdle())return true;
#endif
        const int keepAwakeVarIndex = smartIdleConfig.keepAwakeVarIndex;
        const int varsCap = (int) za::jsfx::variableCount(*idleState);
        if (keepAwakeVarIndex < 0 || keepAwakeVarIndex >= varsCap)
            return false;

        const double raw = idleState->vars[(size_t) keepAwakeVarIndex];
        return std::isfinite (raw) && std::abs (raw) > 0.5;
    }

    bool getSmartIdleSleepReadyFlag() const noexcept {
        const int index=smartIdleConfig.sleepReadyVarIndex;
        if(index<0 || index>=(int)za::jsfx::variableCount(*idleState))return false;
        const double value=double(idleState->vars[index]);
        return std::isfinite(value) && value==1.0;
    }
    void clearSmartIdleSleepReadyFlag() noexcept {
        const int index=smartIdleConfig.sleepReadyVarIndex;
        if(index>=0 && index<(int)za::jsfx::variableCount(*idleState))idleState->vars[index]=0.0;
    }

    template <typename SampleType>
    static bool channelPointersExceedThreshold (SampleType* const* channels,
                                                int numChannels,
                                                int numSamples,
                                                float threshold,
                                                float& outPeak) noexcept
    {
        outPeak = 0.0f;
        if (channels == nullptr || numChannels <= 0 || numSamples <= 0)
            return false;

        const float useThreshold = juce::jmax (0.0f, threshold);

        for (int ch = 0; ch < numChannels; ++ch)
        {
            const SampleType* src = channels[(size_t) ch];
            if (src == nullptr)
                continue;

            for (int i = 0; i < numSamples; ++i)
            {
                const float mag = std::abs ((float) src[i]);
                if (!std::isfinite(mag))return true;
                if (mag > outPeak)
                    outPeak = mag;
                if (mag > useThreshold)
                    return true;
            }
        }

        return false;
    }

    static void clearWritableChannelPointers (float* const* channels, int numChannels, int numSamples) noexcept
    {
        if (channels == nullptr || numChannels <= 0 || numSamples <= 0)
            return;

        for (int ch = 0; ch < numChannels; ++ch)
        {
            float* dst = channels[(size_t) ch];
            if (dst != nullptr)
                juce::FloatVectorOperations::clear (dst, numSamples);
        }
    }

    void noteSmartIdlePostBlock (SmartIdleMode mode,
                                 bool wakeEvent,
                                 bool outputActive,
                                 bool keepAwake,
                                 bool midiOutputActive,
                                 bool dspSliderActivity,
                                 int numSamples) noexcept
    {
        smartIdleRuntime.lastEffectiveMode = mode;

        if (! isSmartIdleSleepEligible (mode))
        {
            smartIdleRuntime.sleeping = false;
            smartIdleRuntime.quietSamples = 0;
            return;
        }

        if (keepAwake || wakeEvent || outputActive || midiOutputActive || dspSliderActivity)
        {
            smartIdleRuntime.sleeping = false;
            smartIdleRuntime.quietSamples = 0;
            return;
        }

        if (mode==SmartIdleMode::Cooperative) {
            smartIdleRuntime.quietSamples=0;
            smartIdleRuntime.sleeping=true;
        } else {
            smartIdleRuntime.quietSamples += std::max<int64_t>(0,numSamples);
            smartIdleRuntime.sleeping = smartIdleRuntime.quietSamples >= std::max(smartIdleConfig.holdSamples,smartIdleConfig.tailSamples);
        }
    }

    struct IdleBlock {SmartIdleMode mode;bool wake=false,skip=false;float inputPeak=0;};
    IdleBlock beginIdleBlock(int hostSamples,const float* const* inputs,int inputChannels,bool externalWake) noexcept {
        IdleBlock block{resolveSmartIdleModeForCurrentState()};
        smartIdleRuntime.lastEffectiveMode=block.mode;
        const bool sizeChanged=smartIdleRuntime.lastHostBlockSize!=hostSamples;
        smartIdleRuntime.lastHostBlockSize=hostSamples;
        const bool inputActive=isSmartIdleSleepEligible(block.mode) && channelPointersExceedThreshold(inputs,inputChannels,hostSamples,0.0f,block.inputPeak);
        smartIdleRuntime.lastInputPeak=block.inputPeak;
        block.wake=externalWake || sizeChanged || inputActive || getSmartIdleKeepAwakeFlag() || (block.mode==SmartIdleMode::Cooperative && !getSmartIdleSleepReadyFlag());
        if(!isSmartIdleSleepEligible(block.mode) || block.wake){smartIdleRuntime.sleeping=false;smartIdleRuntime.quietSamples=0;}
        else block.skip=smartIdleRuntime.sleeping;
        if(!block.skip)clearSmartIdleSleepReadyFlag();
        return block;
    }
    void finishIdleBlock(const IdleBlock& block,float* const* outputs,int outputChannels,int hostSamples,int processSamples,bool outgoingMidi,bool sliderActivity) noexcept {
        if(block.skip){
            smartIdleRuntime.lastOutputPeak=0;
            publishSmartIdleUiRuntimeSnapshot(block.mode,true,block.inputPeak,0,smartIdleRuntime.quietSamples);
            return;
        }
        auto mode=resolveSmartIdleModeForCurrentState();float peak=0;
        const bool active=channelPointersExceedThreshold(outputs,outputChannels,hostSamples,mode==SmartIdleMode::Cooperative?0.0f:smartIdleConfig.outputThreshold,peak);
        smartIdleRuntime.lastOutputPeak=peak;
        bool keep=getSmartIdleKeepAwakeFlag() || (mode==SmartIdleMode::Cooperative && !getSmartIdleSleepReadyFlag());
        noteSmartIdlePostBlock(mode,block.wake,active,keep,outgoingMidi,sliderActivity,processSamples);
        publishSmartIdleUiRuntimeSnapshot(mode,smartIdleRuntime.sleeping,block.inputPeak,peak,smartIdleRuntime.quietSamples);
    }

    double smartIdleTailLengthSeconds = 0.0;
    SmartIdleConfig smartIdleConfig {};
    SmartIdleRuntimeState smartIdleRuntime {};
    std::atomic<bool> smartIdleUiSleeping { false };
    std::atomic<int> smartIdleUserOverrideMode { (int) SmartIdleMode::Auto };
    std::atomic<int> smartIdleUiOptionMode { (int) SmartIdleMode::Auto };
    std::atomic<int> smartIdleUiInferredMode { (int) SmartIdleMode::AlwaysAwake };
    std::atomic<int> smartIdleUiUserOverrideMode { (int) SmartIdleMode::Auto };
    std::atomic<int> smartIdleUiEffectiveMode { (int) SmartIdleMode::AlwaysAwake };
    std::atomic<float> smartIdleUiInputPeak { 0.0f };
    std::atomic<float> smartIdleUiOutputPeak { 0.0f };
    std::atomic<int64_t> smartIdleUiQuietSamples { 0 };
    std::atomic<double> smartIdleUiSampleRate { 44100.0 };
    std::atomic<double> smartIdleUiHoldMs { kDefaultSmartIdleHoldMs };
    std::atomic<double> smartIdleUiTailMs { kDefaultSmartIdleTailMs };

};
}
