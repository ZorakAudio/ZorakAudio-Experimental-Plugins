#include <cmath>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include "faust_support_min.h"
#include "FaustDSP.h"

namespace
{
    // Remove Faust-style metadata like: "Gain [unit:dB][scale:log]"
    static juce::String stripLabelAndCollectMeta (const juce::String& in,
                                                  std::map<std::string, std::string>* metaOut)
    {
        juce::String label = in;
        int start = 0;

        while ((start = label.indexOfChar ('[')) >= 0)
        {
            const int end = label.indexOfChar (start, ']');
            if (end < 0) break;

            const auto inside = label.substring (start + 1, end).trim();
            const int colon = inside.indexOfChar (':');

            if (metaOut != nullptr && colon > 0)
            {
                const auto k = inside.substring (0, colon).trim().toStdString();
                const auto v = inside.substring (colon + 1).trim().toStdString();
                (*metaOut)[k] = v;
            }

            label = (label.substring (0, start) + label.substring (end + 1)).trim();
        }

        return label.trim();
    }

    static juce::String sanitizeId (const juce::String& s)
    {
        juce::String out;
        for (auto c : s)
        {
            if (juce::CharacterFunctions::isLetterOrDigit (c)) out << c;
            else if (c == ' ' || c == '-' || c == '_') out << '_';
        }
        if (out.isEmpty()) out = "param";
        return out;
    }

    static inline bool isNearlyInteger (float v)
    {
        const float r = std::round (v);
        return std::abs (v - r) <= 1.0e-6f;
    }

    // Derive a sensible fixed-decimal display from a FAUST step size.
    // Example: 1 -> 0 decimals, 0.1 -> 1, 0.01 -> 2, 0.005 -> 3, etc.
    static inline int decimalsFromStep (float step)
    {
        if (step <= 0.0f) return 3;

        int d = 0;
        float s = step;

        while (d < 6 && s < 1.0f && ! isNearlyInteger (s))
        {
            s *= 10.0f;
            ++d;
        }

        return juce::jlimit (0, 6, d);
    }

    static bool parseFaustMenuStyle (const std::map<std::string, std::string>& meta,
                                     juce::StringArray& outChoices)
    {
        outChoices.clear();

        auto it = meta.find ("style");
        if (it == meta.end())
            return false;

        juce::String s = juce::String::fromUTF8 (it->second.c_str()).trim();

        // Expect: menu{...}
        if (! s.startsWithIgnoreCase ("menu"))
            return false;

        const int l = s.indexOfChar ('{');
        const int r = s.lastIndexOfChar ('}');
        if (l < 0 || r <= l)
            return false;

        juce::String body = s.substring (l + 1, r).trim();

        // Split on ';' (FAUST common) else ','.
        juce::StringArray items;
        if (body.containsChar (';')) items.addTokens (body, ";", "");
        else                         items.addTokens (body, ",", "");

        std::vector<std::pair<int, juce::String>> indexed;
        int implicitIndex = 0;

        auto stripQuotes = [] (juce::String t)
        {
            t = t.trim();
            if ((t.startsWithChar ('\'') && t.endsWithChar ('\'')) ||
                (t.startsWithChar ('"')  && t.endsWithChar ('"')))
                t = t.substring (1, t.length() - 1);
            return t.trim();
        };

        for (auto item : items)
        {
            item = item.trim();
            if (item.isEmpty()) continue;

            const int colon = item.indexOfChar (':');
            if (colon > 0)
            {
                auto name = stripQuotes (item.substring (0, colon));
                auto idxS = item.substring (colon + 1).trim();
                const int idx = idxS.getIntValue();
                indexed.push_back ({ idx, name });
            }
            else
            {
                indexed.push_back ({ implicitIndex++, stripQuotes (item) });
            }
        }

        if (indexed.empty())
            return false;

        std::sort (indexed.begin(), indexed.end(),
                   [] (auto& a, auto& b) { return a.first < b.first; });

        for (auto& kv : indexed)
            outChoices.add (kv.second);

        return outChoices.size() > 0;
    }


    struct FaustParam
    {
        juce::String id;
        juce::String name;
        FAUSTFLOAT* zone = nullptr;

        float init = 0.0f;
        float min  = 0.0f;
        float max  = 1.0f;
        float step = 0.0f;

        bool isDiscrete01 = false;
        bool isNumEntry   = false; // distinguish nentry vs slider for nicer text/format
        std::map<std::string, std::string> meta;
    };

    struct ParamCollectorUI : public UI
    {
        std::vector<juce::String> groupStack;
        juce::String rootGroup; // first/top-level group (often plugin name)
        std::vector<FaustParam> params;

        // Faust can attach metadata via UI::declare(zone, key, value).
        // We collect it per-zone and merge it into the eventual parameter metadata.
        std::map<FAUSTFLOAT*, std::map<std::string, std::string>> declaredMeta;
        std::map<std::string, std::string> globalMeta; // zone=nullptr declarations (e.g. latency_samples)


        // Faust UI layout
        void openTabBox (const char* label) override         { pushGroup (label); }
        void openHorizontalBox (const char* label) override  { pushGroup (label); }
        void openVerticalBox (const char* label) override    { pushGroup (label); }
        void closeBox() override                             { popGroup(); }

        // Active widgets
        void addButton (const char* label, FAUSTFLOAT* zone) override
        {
            addParam (label, zone, 0.0f, 0.0f, 1.0f, 1.0f, true);
        }

        void addCheckButton (const char* label, FAUSTFLOAT* zone) override
        {
            addParam (label, zone, 0.0f, 0.0f, 1.0f, 1.0f, true);
        }

        void addVerticalSlider (const char* label, FAUSTFLOAT* zone,
                                FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) override
        {
            addParam (label, zone, (float) init, (float) min, (float) max, (float) step, false);
        }

        void addHorizontalSlider (const char* label, FAUSTFLOAT* zone,
                                  FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) override
        {
            addParam (label, zone, (float) init, (float) min, (float) max, (float) step, false);
        }

        void addNumEntry (const char* label, FAUSTFLOAT* zone,
                          FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) override
        {
            addParam (label, zone, (float) init, (float) min, (float) max, (float) step, false, true);
        }

        // Passive widgets (ignored for parameters)
        void addHorizontalBargraph (const char*, FAUSTFLOAT*, FAUSTFLOAT, FAUSTFLOAT) override {}
        void addVerticalBargraph   (const char*, FAUSTFLOAT*, FAUSTFLOAT, FAUSTFLOAT) override {}

        // Soundfiles (ignored)
        // void addSoundfile (const char*, const char*, Soundfile**) override {}

        // Metadata hook (FAUST uses this heavily; labels alone are not enough).
        void declare (FAUSTFLOAT* zone, const char* key, const char* value) override
        {
            if (key == nullptr || value == nullptr) return;

            const std::string k (key);
            const std::string v (value);

            if (zone == nullptr)
            {
                // Global metadata like declare latency_samples "N"
                globalMeta[k] = v;
                return;
            }

            // Per-zone metadata
            declaredMeta[zone][k] = v;
        }


    private:
        void pushGroup (const char* raw)
        {
            std::map<std::string, std::string> meta;
            const auto clean = stripLabelAndCollectMeta (raw, &meta);
            if (groupStack.empty()) rootGroup = clean;
            groupStack.push_back (clean);
        }

        void popGroup()
        {
            if (!groupStack.empty())
                groupStack.pop_back();
        }

        juce::String makeId (const juce::String& leaf) const
        {
            juce::StringArray parts;
            for (auto& g : groupStack)
                if (g.isNotEmpty())
                    parts.add (sanitizeId (g));

            parts.add (sanitizeId (leaf));
            return parts.joinIntoString ("__");
        }

        void addParam (const char* rawLabel, FAUSTFLOAT* zone,
                       float init, float min, float max, float step,
                       bool discrete01,
                       bool isNumEntry = false)
        {
            FaustParam p;
            p.meta.clear();
            juce::String raw = juce::String::fromUTF8 (rawLabel);  // keeps σ, →, etc.
            auto leaf = stripLabelAndCollectMeta (raw, &p.meta);

            juce::String path;
            for (auto& g : groupStack)
                if (g.isNotEmpty())
                    path << g << "/";

            path << leaf;

            // Drop the top-level group (usually the plugin name) from the *display* label.
            if (rootGroup.isNotEmpty() && path.startsWith (rootGroup + "/"))
                path = path.substring (rootGroup.length() + 1);

            p.name = path;

            // IDs stay stable and host-safe; use the leaf name + group stack (not the full display path).
            p.id   = makeId (leaf);

            p.zone = zone;
            p.init = init;
            p.min  = min;
            p.max  = max;
            p.step = step;
            p.isDiscrete01 = discrete01;
            p.isNumEntry   = isNumEntry;

            // Merge FAUST UI::declare() metadata (if any)
            if (zone != nullptr)
            {
                const auto it = declaredMeta.find (zone);
                if (it != declaredMeta.end())
                {
                    for (const auto& kv : it->second)
                        p.meta[kv.first] = kv.second;
                }
            }

            params.push_back (p);
        }
    };
}

class FaustJuceProcessor final : public juce::AudioProcessor
{
    using juce::AudioProcessor::processBlock;

public:
    FaustJuceProcessor() : juce::AudioProcessor (makeBusesFromFaust())
    {
        dsp = std::make_unique<mydsp>();

        // Collect parameters from Faust UI
        dsp->buildUserInterface (&ui);
        faustParams = ui.params;
        // Pull declared latency from FAUST metadata (declare latency_samples "N")
        if (auto it = ui.globalMeta.find ("latency_samples"); it != ui.globalMeta.end())
        {
            faustLatencySamples = std::max (0, std::atoi (it->second.c_str()));
            setLatencySamples (faustLatencySamples);
            updateHostDisplay(); // helps some hosts refresh PDC immediately
        }


        juce::AudioProcessorValueTreeState::ParameterLayout layout;

        for (const auto& p : faustParams)
        {
            // Unit: prefer UI::declare metadata, fall back to bracket metadata in label.
            juce::String unit;
            if (auto it = p.meta.find ("unit"); it != p.meta.end())
                unit = juce::String (it->second);

            // If FAUST declares a menu style (e.g. [style:menu{'Output':0;'Delta':1}]),
            // expose it as a CHOICE so hosts show real labels instead of numbers.
            juce::StringArray menuChoices;
            if (p.isNumEntry && parseFaustMenuStyle (p.meta, menuChoices))
            {
                const int maxIndex = menuChoices.size() - 1;
                const int defIndex = juce::jlimit (0, maxIndex, (int) std::llround (p.init));

                layout.add (std::make_unique<juce::AudioParameterChoice> (
                    p.id, p.name, menuChoices, defIndex));
                continue;
            }


            // Discrete 0/1 becomes a proper boolean parameter (nicer host UI + automation).
            if (p.isDiscrete01)
            {
                layout.add (std::make_unique<juce::AudioParameterBool> (p.id, p.name, (p.init >= 0.5f)));
                continue;
            }

            // If this is an nentry that is effectively integer-stepped, expose it as an int.
            const bool integerStepped = (p.step > 0.0f && isNearlyInteger (p.step)
                                         && isNearlyInteger (p.min) && isNearlyInteger (p.max)
                                         && isNearlyInteger (p.init));

            if (p.isNumEntry && integerStepped)
            {
                layout.add (std::make_unique<juce::AudioParameterInt> (
                    p.id,
                    p.name,
                    (int) std::llround (p.min),
                    (int) std::llround (p.max),
                    (int) std::llround (p.init)));
                continue;
            }

            const auto range = juce::NormalisableRange<float> (p.min, p.max, p.step);
            const int decimals = decimalsFromStep (p.step);

            auto valueToText = [unit, decimals] (float v, int) -> juce::String
            {
                juce::String s (v, decimals);
                if (unit.isNotEmpty()) s << " " << unit;
                return s;
            };

            auto textToValue = [unit] (const juce::String& text) -> float
            {
                auto t = text.trim();
                if (unit.isNotEmpty() && t.endsWithIgnoreCase (unit))
                    t = t.dropLastCharacters (unit.length()).trim();
                return (float) t.getDoubleValue();
            };

            layout.add (std::make_unique<juce::AudioParameterFloat> (
                p.id,
                p.name,
                range,
                p.init,
                unit,
                juce::AudioProcessorParameter::genericParameter,
                valueToText,
                textToValue));
        }

        apvts = std::make_unique<juce::AudioProcessorValueTreeState> (*this, nullptr, "PARAMS", std::move (layout));
    }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void prepareToPlay (double sampleRate, int /*samplesPerBlock*/) override
    {
        dsp->init ((int) sampleRate);
        setLatencySamples (faustLatencySamples);
        setLatencySamples (faustLatencySamples);
        updateHostDisplay();

        inputPtrs.resize ((size_t) dsp->getNumInputs());
        outputPtrs.resize ((size_t) dsp->getNumOutputs());
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        const int ins  = dsp->getNumInputs();
        const int outs = dsp->getNumOutputs();

        const auto inSet  = layouts.getMainInputChannelSet();
        const auto outSet = layouts.getMainOutputChannelSet();

        if (ins > 0 && inSet.size() != ins)  return false;
        if (outs > 0 && outSet.size() != outs) return false;

        // If you want “must match in/out” for typical effects, keep this:
        if (ins > 0 && outs > 0 && inSet.size() != outSet.size())
            return false;

        return true;
    }


    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;

        // Push JUCE param values into Faust zones
        for (auto& p : faustParams)
        {
            if (auto* v = apvts->getRawParameterValue (p.id))
                *p.zone = (FAUSTFLOAT) v->load();
        }

        const int numSamples = buffer.getNumSamples();

        for (int ch = 0; ch < dsp->getNumInputs(); ++ch)
            inputPtrs[(size_t) ch] = (FAUSTFLOAT*) buffer.getReadPointer (ch);

        for (int ch = 0; ch < dsp->getNumOutputs(); ++ch)
            outputPtrs[(size_t) ch] = (FAUSTFLOAT*) buffer.getWritePointer (ch);

        dsp->compute (numSamples, inputPtrs.data(), outputPtrs.data());
    }

    juce::AudioProcessorEditor* createEditor() override
    {
        return new juce::GenericAudioProcessorEditor (*this);
    }

    bool hasEditor() const override { return true; }

    void getStateInformation (juce::MemoryBlock& destData) override
    {
        auto state = apvts->copyState();
        std::unique_ptr<juce::XmlElement> xml (state.createXml());
        copyXmlToBinary (*xml, destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xml (getXmlFromBinary (data, sizeInBytes));
        if (xml && xml->hasTagName (apvts->state.getType()))
            apvts->replaceState (juce::ValueTree::fromXml (*xml));
    }

private:
    std::unique_ptr<mydsp> dsp;
    ParamCollectorUI ui;

    std::unique_ptr<juce::AudioProcessorValueTreeState> apvts;
    std::vector<FaustParam> faustParams;

    std::vector<FAUSTFLOAT*> inputPtrs;
    std::vector<FAUSTFLOAT*> outputPtrs;
    int faustLatencySamples = 0;


    static BusesProperties makeBusesFromFaust()
    {
        mydsp d;

        const int ins  = d.getNumInputs();
        const int outs = d.getNumOutputs();

        // Use discrete sets for >2 channels so hosts can wire arbitrary multichannel.
        auto setFor = [] (int ch) -> juce::AudioChannelSet
        {
            if (ch <= 0) return juce::AudioChannelSet::disabled();
            if (ch == 1) return juce::AudioChannelSet::mono();
            if (ch == 2) return juce::AudioChannelSet::stereo();
            return juce::AudioChannelSet::discreteChannels (ch);
        };

        BusesProperties props;
        if (ins > 0)  props = props.withInput  ("Input",  setFor(ins),  true);
        if (outs > 0) props = props.withOutput ("Output", setFor(outs), true);

        return props;
    }

};

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new FaustJuceProcessor();
}
